#!/bin/bash
#SBATCH --job-name=test_t-projection
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=test_t-projection.out.txt
#SBATCH --error=test_t-projection.err.txt

source /ikerlariak/igarcia945/envs/pytorch2/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS="true"


accelerate launch --mixed_precision bf16 seq2seq_t5.py \
  --train_tsv test_data/en.europarl.conll \
  --dev_tsv test_data/en.europarl.conll \
  --test_source_tsv test_data/en.europarl.conll \
  --test_target_tsv test_data/es.europarl.conll \
  --num_beams 10 \
  --num_return_sequences 10 \
  --model_name_or_path google/mt5-large \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --per_device_test_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --num_train_epochs 400 \
  --output_dir test_data/mt5-large \
  --seed 42 \
  --eval_every 50 \
  --max_source_length 256 \
  --max_target_length 256 \
  --lr_scheduler_type cosine \
  --num_warmup_steps 500 \
  --project_name "T-Projection-europarl"

python3 calculate_scores_nmts.py \
  --jsonl_path test_data/mt5-large/es.europarl.jsonl \
  --model_name_or_path facebook/m2m100_418M \
  --output_path test_data/mt5-large/es.europarl.json \
  --source_lang en \
  --target_lang es \
  --normalize \
  --both_directions

python3 label_projection.py \
  --jsonl_path test_data/mt5-large/es.europarl.jsonl \
  --dictionary_path test_data/mt5-large/es.europarl.json \
  --output_path test_data/mt5-large/en2es.europarl.conll \
  --gold_tsv test_data/es.europarl.gold
