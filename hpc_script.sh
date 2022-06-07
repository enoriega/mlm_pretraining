#!/bin/bash
# Submit this through slurm

### Optional. Set the job name
#SBATCH --job-name=keyword_mlm
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
#SBATCH --ntasks=1
### REQUIRED. Set the number of nodes
#SBATCH --nodes=1
### REQUIRED. Set the memory required for this job.
#SBATCH --mem=8gb
# GPU
#SBATCH --gres=gpu:1
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=10:00:00
## ARRAY config
#SBATCH --array=0-8

IX=$SLURM_ARRAY_TASK_ID

# Loop through the grid until finding the parameters' index
LRS=(0.0003  0.00006  0.000006)
NUM_KWS=( 1000 5000 10000 )

COUNT=0
for NUM_KW in "${NUM_KWS[@]}"
do
  for LR in "${LRS[@]}"
  do
    if [[ $COUNT -eq $IX ]]
    then
      break 2
    fi
    ((COUNT++))
  done
done


MODEL_ID="kw_pubmed_${NUM_KW}_${LR}"
OUT_DIR="/xdisk/msurdeanu/enoriega/kw_pubmed/$MODEL_ID"
TOKEN=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
BATCH_SIZE=32

echo "$MODEL_ID"

python pre_train_mlm.py \
            --dataset_path "$HOME/kw_dataset" \
            --output_dir "$OUT_DIR"  \
            --hf_token $TOKEN \
            --batch_size $BATCH_SIZE \
            --do_train \
            --do_eval \
            --disable_tqdm \
            --num_keywords "$NUM_KW" \
            --learning_rate "$LR"
            $MODEL_ID
