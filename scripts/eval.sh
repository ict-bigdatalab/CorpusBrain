#!/bin/bash

MODEL=$1
DATASET=$2

python scripts/evaluate_kilt_dataset.py \
    models/$1 \
    data/kilt/$2.jsonl \
    predictions/$1_$2.jsonl \
    --batch_size 512 \
    --beams 10 \
    --max_len_a 384 \
    --max_len_b 15 \
    --trie data/kilt_titles_trie_dict.pkl \
    --checkpoint_file checkpoint_best.pt \
