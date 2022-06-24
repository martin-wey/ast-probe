import torch.nn as nn
import torch


class Probe(nn.Module):
    pass


class ParserProbe(Probe):
    def __init__(self, probe_rank, hidden_dim, number_labels_c, number_labels_u):
        print('Constructing ParserProbe')
        super(ParserProbe, self).__init__()
        self.probe_rank = probe_rank
        self.hidden_dim = hidden_dim
        self.number_vectors_c = number_labels_c
        self.number_vectors_u = number_labels_u
        self.proj = nn.Parameter(data=torch.zeros(self.hidden_dim, self.probe_rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.vectors_c = nn.Parameter(data=torch.zeros(self.probe_rank, self.number_vectors_c))
        nn.init.uniform_(self.vectors_c, -0.05, 0.05)
        self.vectors_u = nn.Parameter(data=torch.zeros(self.probe_rank, self.number_vectors_u))
        nn.init.uniform_(self.vectors_u, -0.05, 0.05)

    def forward(self, batch):
        """
        Args:
            batch: a batch of word representations of the shape
                    (batch_size, max_seq_len, representation_dim)

        Returns:
            d_pred: (batch_size, max_seq_len - 1)
            scores_c: (batch_size, max_seq_len - 1, number classes_c)
            scores_u: (batch_size, max_seq_len, number classes_u)
        """
        transformed = torch.matmul(batch, self.proj)
        shift = transformed[:, 1:, :]
        diffs = shift - transformed[:, :-1, :]
        return (diffs ** 2).sum(dim=2), torch.matmul(diffs, self.vectors_c), torch.matmul(transformed, self.vectors_u)
