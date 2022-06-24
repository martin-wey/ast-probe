import numpy as np
import torch

from .utils import match_tokenized_to_untokenized_roberta


def collator_fn(batch, tokenizer):
    tokens = [b['code_tokens'] for b in batch]
    cs = [b['c'] for b in batch]
    ds = [b['d'] for b in batch]
    us = [b['u'] for b in batch]

    # generate inputs and attention masks
    all_inputs = []
    all_attentions = []
    all_mappings = []
    for untokenized_sent in tokens:
        to_convert, mapping = match_tokenized_to_untokenized_roberta(untokenized_sent, tokenizer)
        inputs = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + to_convert + [tokenizer.sep_token])
        masks = [1] * len(inputs)
        all_inputs.append(inputs)
        all_attentions.append(masks)
        all_mappings.append({x: [l + 1 for l in y] for x, y in mapping.items()})

    max_len_subtokens = np.max([len(m) for m in all_attentions])
    # pad sequences
    all_inputs = torch.tensor(
        [inputs + tokenizer.convert_tokens_to_ids([tokenizer.pad_token] * (max_len_subtokens - len(inputs)))
         for inputs in all_inputs])
    all_attentions = torch.tensor([mask + ([0] * (max_len_subtokens - len(mask)))
                                   for mask in all_attentions])

    batch_len_tokens = [len(m) for m in tokens]
    max_len_tokens = np.max(batch_len_tokens)

    cs = torch.tensor([c + [-1] * (max_len_tokens - 1 - len(c)) for c in cs])
    ds = torch.tensor([d + [-1] * (max_len_tokens - 1 - len(d)) for d in ds])
    us = torch.tensor([u + [-1] * (max_len_tokens - len(u)) for u in us])

    # generate token alignment
    alignment = []
    for mapping in all_mappings:
        j = 0
        indices = []
        for i in range(len(mapping)):
            indices += [j]*len(mapping[i])
            j += 1
        indices += [j]*(max_len_subtokens - 1 - len(indices))
        alignment.append(indices)
    alignment = torch.tensor(alignment)
    
    return all_inputs, all_attentions, ds, cs, us, torch.tensor(batch_len_tokens), alignment
