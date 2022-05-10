import math
import os

from transformers import AutoTokenizer, EarlyStoppingCallback
from datasets import load_metric, load_from_disk, IterableDataset, load_dataset, Dataset, DatasetDict
from transformers import TrainingArguments, Trainer, AutoModelForMaskedLM
import logging
import argparse

# Train num_rows: 5093504
# Validation num_rows: 565200
from transformers.trainer_utils import get_last_checkpoint

from collators import DataCollatorForWholeKeywordMask

parser = argparse.ArgumentParser(description="MLM pre-training args")

parser.add_argument("--model_ckpt", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
parser.add_argument("--output_dir", default="ckpt")
parser.add_argument("--num_keywords", type=int, default=5000)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--effective_batch_size", type=int, default=8000)
parser.add_argument("--logging_steps", type=int, default=5)
parser.add_argument("--dataset_path", default='/media/evo870/data/kw_pubmed')
parser.add_argument("--num_eval_examples", type=int, default=10_000)
parser.add_argument("--evals_per_epoch",type=int,  default=10)
parser.add_argument("--learning_rate",  type=float, default=3e-4)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--disable_tqdm", action="store_true")
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--do_eval", action="store_true")
parser.add_argument("--resume_from_checkpoint", action="store_true")
parser.add_argument("--hf_token", type=str, default=None)
parser.add_argument("model_id")

def main(args):
    model_ckpt = args.model_ckpt#"microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    output_dir = args.output_dir
    # keyword_file_path = "keywords.txt"
    num_keywords = args.num_keywords
    batch_size = args.batch_size
    effective_batch_size = args.effective_batch_size
    gradient_accumulation_steps = effective_batch_size // batch_size
    logging_steps = args.logging_steps
    dataset_path = args.dataset_path
    num_eval_examples = args.num_eval_examples
    evals_per_epoch = args.evals_per_epoch
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    disable_tqdm = args.disable_tqdm
    do_train = args.do_train
    do_eval  = args.do_eval
    resume_from_checkpoint = args.resume_from_checkpoint
    model_id = args.model_id
    token = args.hf_token

    logging.info("Loading dataset")
    # Uncomment to use the dataset builder script with streaming from the raw files
    # dataset = load_dataset('enoriega/keyword_pubmed', "sentence", data_dir='/media/evo870/data/keyword_data_files', streaming=True).with_format("torch")

    # Uncomment to use the arrow-preprocessed dataset without streaming
    dataset = load_from_disk(dataset_path)

    logging.info("Tokienizing")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=512,
                         return_special_tokens_mask=True)


    # Tokenize the inputs
    dataset = dataset.map(tokenize, batched=True)

    train_dataset = dataset['train'].filter(lambda e: e['keyword_rank'] < num_keywords)

    # In case of a datasetdict use select
    try:
        eval_dataset = dataset['validation'].select(range(num_eval_examples))
        eval_steps = len(train_dataset)  // effective_batch_size // evals_per_epoch

    # In case of an iterable dataset, use take
    except:
        eval_dataset = dataset['validation'].take(num_eval_examples)
        eval_steps = 5093504 // effective_batch_size // evals_per_epoch  # We know the dataset is this size when using all keywords


    data_collator = DataCollatorForWholeKeywordMask(tokenizer=tokenizer,
                                                    mlm_probability=0.15, return_tensors='pt')


    # Hub args
    hub_args = {}
    if token:
        hub_args['push_to_hub'] = True
        hub_args['push_to_hub_model_id'] = model_id
        hub_args['push_to_hub_token'] = token

    # training arguments setting
    # noinspection PyTypeChecker
    training_args = TrainingArguments(output_dir=output_dir,
                                      do_train=do_train,
                                      do_eval=do_eval,
                                      resume_from_checkpoint=resume_from_checkpoint,
                                      per_device_train_batch_size=batch_size,
                                      per_gpu_train_batch_size=batch_size,
                                      per_gpu_eval_batch_size=batch_size,
                                      logging_strategy="steps",
                                      logging_steps=logging_steps,
                                      evaluation_strategy="steps",
                                      eval_steps=eval_steps,
                                      eval_accumulation_steps=100,  # This is arbitrary, maybe tune later
                                      load_best_model_at_end=True,
                                      metric_for_best_model= "eval_accuracy",
                                      greater_is_better= True,
                                      # dataloader_num_workers=1,
                                      save_strategy="steps",
                                      save_steps=eval_steps,  # Save after each evaluation
                                      save_total_limit=10,
                                      num_train_epochs=num_epochs,
                                      fp16=True,
                                      gradient_accumulation_steps=gradient_accumulation_steps,
                                      log_level="error",
                                      disable_tqdm=disable_tqdm,
                                      learning_rate=learning_rate,
                                      remove_unused_columns=False,  # Necessary for our custom collation code
                                      **hub_args
                                      )




    print(training_args)

    # Metric
    metric = load_metric("accuracy")


    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)

    def load_checkpoint(train_args:TrainingArguments):
        if train_args.do_train or not os.path.exists(training_args.output_dir):
            return AutoModelForMaskedLM.from_pretrained(model_ckpt)
        else:
            return AutoModelForMaskedLM.from_pretrained(get_last_checkpoint(training_args.output_dir))


    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logging.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # trainer setting
    trainer = Trainer(model=load_checkpoint(training_args),
                      tokenizer=tokenizer,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
                      preprocess_logits_for_metrics=preprocess_logits_for_metrics)

    if training_args.do_train:

        logging.info("Train")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        # trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics(split="train", metrics=metrics)
        trainer.save_metrics(split="train", metrics=metrics)
        if token:
            trainer.save_state()
            trainer.push_to_hub("Finished training")

    if training_args.do_eval:

        logging.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


