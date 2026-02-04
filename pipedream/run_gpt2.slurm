#!/bin/bash
#SBATCH --job-name=gpt2-shakespeare
#SBATCH --partition=gpunodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# -----------------------
# Environment
# -----------------------
module load cuda
module load python

# activate your venv if you use one
# source ~/venvs/gpt2/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

# (helps debugging if something hangs)
export NCCL_DEBUG=INFO

# -----------------------
# Run fine-tuning
# -----------------------
accelerate launch \
  transformers/examples/pytorch/language-modeling/run_clm.py \
  --model_name_or_path gpt2 \
  --train_file shakespeare.txt \
  --validation_file shakespeare.txt \
  --do_train \
  --do_eval \
  --block_size 512 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --learning_rate 5e-5 \
  --num_train_epochs 20 \
  --weight_decay 0.01 \
  --fp16 \
  --logging_steps 50 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --output_dir out-gpt2-shakespeare \
  --overwrite_output_dir \
  --deepspeed ds_config.json
