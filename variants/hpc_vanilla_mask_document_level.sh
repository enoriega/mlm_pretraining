#!/bin/bash
# Submit this through slurm

### Optional. Set the job name
#SBATCH --job-name=keyword_mlm_vanilla_doc
### Optional. Set the output filename.
### SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=%x-%A_%a.out
### REQUIRED. Specify the PI group for this job
#SBATCH --account=msurdenau
### Optional. Request email when job begins and ends
### SBATCH --mail-type=ALL
### Optional. Specify email address to use for notification
### SBATCH --mail-user=enoriega@email.arizona.edu
### REQUIRED. Set the partition for your job.
#SBATCH --partition=standard
### REQUIRED. Set the number of cores that will be used for this job.
#SBATCH --ntasks=2
### REQUIRED. Set the number of nodes
#SBATCH --nodes=1
### REQUIRED. Set the memory required for this job.
#SBATCH --mem=8gb
# GPU
#SBATCH --gres=gpu:1
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=48:00:00

# Loop through the grid until finding the parameters' index
LR=0.0003
NUM_KW=10000


MODEL_ID="kw_pubmed_vanilla_document_${NUM_KW}_${LR}"
OUT_DIR="/xdisk/msurdeanu/enoriega/kw_pubmed/$MODEL_ID"
TOKEN=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
BATCH_SIZE=32

echo "$MODEL_ID"

python pre_train_mlm.py \
            --dataset_mode document \
            --mlm_type vanilla \
            --output_dir "$OUT_DIR"  \
            --hf_token $TOKEN \
            --batch_size $BATCH_SIZE \
            --do_train \
            --do_eval \
            --disable_tqdm \
            --num_keywords "$NUM_KW" \
            --learning_rate "$LR"
            $MODEL_ID


