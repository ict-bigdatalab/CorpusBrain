#!/bin/bash

# Copyright (c) Chriskuei.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Download the model.
mkdir models && cd models
wget http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar -xzvf bart.large.tar.gz

# Download the corpus.
cd ..
mkdir data && cd data
wget http://dl.fbaipublicfiles.com/GENRE/kilt_titles_trie_dict.pkl
wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json

# Filter the corpus.
python utils/filter_corpus.py kilt_knowledgesource.json

# Preprocess the corpus.
mkdir knowledge
python utils/preprocess_corpus.py kilt_knowledgesource.json
