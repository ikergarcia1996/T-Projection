# This example uses google/mt5-xl model to generate translation candidates
# and facebook/m2m100-12B-last-ckpt to score the candidates. This is the best
# configuration for performance. Although as shown in the paper, you can get
# very competitive results with smaller models. If you do not have a GPU
# with a lot of VRAM, you can use the following models:
# Candidate Generation: google/mt5-large
# Candidate Scoring: facebook/m2m100_418M facebook/m2m100_1.2B facebook/nllb-200-1.3B facebook/nllb-200-3.3B
# See Appendix B: https://arxiv.org/abs/2212.10548

# In the paper we generate 100 candidates for each source sentence.
# As shown in Appendix A (https://arxiv.org/abs/2212.10548), this is excessive, 25 candidates is enough.
# Any value from 15 to 25 will get you good results. Adjust --num_return_sequences and --num_beams for
# your hardware specs. Using a lower value will reduce VRAM usage and run faster.


# 1) Train mT5 and generate projection candidates (15 per source entity)

accelerate launch --mixed_precision bf16 seq2seq_t5.py \
--train_tsv datasets/data/en.neoplasm.train.dev.test.tsv \
--dev_tsv datasets/data/en.neoplasm.train.tsv \
--test_source_tsv \
  datasets/data/en.neoplasm.train.tsv \
--test_target_tsv \
  datasets/data/es.neoplasm.train.tsv \
--num_beams 15 \
--num_return_sequences 15 \
--model_name_or_path google/mt5-xl \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 1 \
--per_device_test_batch_size 1 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-4 \
--num_train_epochs 4 \
--output_dir results/absrct/ \
--seed 42 \
--eval_every 1 \
--max_source_length 256 \
--max_target_length 256 \
--lr_scheduler_type cosine \
--num_warmup_steps 500 \
--project_name "T5NMTS"



for lang in es
do
  # 2) Compute translation probabilities for each candidate

  python3 calculate_scores_nmts.py \
  --jsonl_path results/absrct/"$lang".absrct.train.jsonl \
  --model_name_or_path facebook/m2m100-12B-last-ckpt \
  --output_path results/absrct/m2m100-12B.en2"$lang".train.json \
  --source_lang en \
  --target_lang "$lang" \
  --normalize \
  --both_directions

  # 3) Label projection

  python3 label_projection.py \
  --jsonl_path results/absrct/"$lang".absrct.train.jsonl \
  --dictionary_path results/absrct/m2m100-12B.en2"$lang".train.json \
  --output_path results/absrct/en2"$lang".train.tsv \
  --gold_tsv datasets/data/"$lang".absrct.train.tsv

done

