import warnings
from typing import List, Union, Any, Dict

from transformers import DataCollatorForWholeWordMask, BatchEncoding, BertTokenizer, BertTokenizerFast
from transformers.data.data_collator import _torch_collate_batch, tolist


class DataCollatorForWholeKeywordMask(DataCollatorForWholeWordMask):

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            # This is the change introduced by Enrique
            mask_labels.append(self._whole_word_mask(ref_tokens, e["keyword_indices"]))
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}

    def _whole_word_mask(self, input_tokens: List[str], keyword_indices:List[int], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        # random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(keyword_indices) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        keyword_indices = set(keyword_indices)
        for word_ix, index_set in enumerate(cand_indexes):
            if len(masked_lms) >= num_to_predict:
                break

            if word_ix in keyword_indices:
                # If adding a whole-word mask would exceed the maximum number of
                # predictions, then just skip this candidate.
                if len(masked_lms) + len(index_set) > num_to_predict:
                    continue

                for index in index_set:
                    covered_indexes.add(index)
                    masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]

        return mask_labels
