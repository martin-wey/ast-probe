import torch
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import scatter_mean


def get_embeddings(all_inputs, all_attentions, model, layer, model_type):
    if model_type == 't5':
        with torch.no_grad():
            embs = model(input_ids=all_inputs, attention_mask=all_attentions)[1][layer][:, 1:, :]
    else:
        with torch.no_grad():
            embs = model(input_ids=all_inputs, attention_mask=all_attentions)[2][layer][:, 1:, :]
    return embs


def align_function(embs, align):
    seq = []
    for j, emb in enumerate(embs):
        seq.append(scatter_mean(emb, align[j], dim=0))
    # remove the last token since it corresponds to <\s> or padding to much the lens
    return pad_sequence(seq, batch_first=True)[:, :-1, :]
