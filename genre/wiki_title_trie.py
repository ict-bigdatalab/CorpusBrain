# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from genre.trie import DummyTrieMention, Trie


def get_prefix_allowed_tokens_fn(
    model,
    split_token="|",
    title_trie: Trie = None,
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        lambda x: model.encode(x).tolist(),
        lambda x: model.decode(torch.tensor(x)),
        model.model.decoder.dictionary.bos(),
        model.model.decoder.dictionary.pad(),
        model.model.decoder.dictionary.eos(),
        len(model.model.decoder.dictionary),
        split_token,
        title_trie,
    )


def _get_end_to_end_prefix_allowed_tokens_fn(
    encode_fn,
    decode_fn,
    bos_token_id,
    pad_token_id,
    eos_token_id,
    vocabulary_length,
    split_token="|",
    title_trie: Trie = None,
):

    codes = {
        n: encode_fn(" {}".format(c))[1]
        for n, c in zip(
            (
                "split_token",
                "BOS",
            ),
            (
                split_token,
                bos_token_id,
            ),
        )
    }
    codes["EOS"] = eos_token_id
    # print(codes)

    if title_trie is None:
        title_trie = DummyTrieMention(
            [
                i
                for i in range(vocabulary_length)
                if i
                not in (
                    bos_token_id,
                    pad_token_id,
                )
            ]
        )

    def prefix_allowed_tokens_fn(batch_id, sent):

        sent = sent.tolist()
        available_tokens = get_trie_title(sent)
        return available_tokens

    def get_trie_title(sent):

        pointer_start, _ = get_pointer_title(sent)
        if pointer_start + 1 < len(sent):
            title_next = title_trie.get([2] + sent[pointer_start+1:])
        else:
            title_next = title_trie.get([2])

        if codes["EOS"] in title_next:
            title_next.append(codes["split_token"])
        return title_next

    def get_pointer_title(sent):
        pointer_end = -1
        pointer_start = 0
        for i, e in enumerate(sent):
            if e == codes["split_token"]:
                pointer_start = i

        return pointer_start, pointer_end

    return prefix_allowed_tokens_fn
