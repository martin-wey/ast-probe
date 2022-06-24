import torch
import torch.nn as nn


class ParserLoss(nn.Module):
    def __init__(self, loss='l1', pretrained=False):
        super(ParserLoss, self).__init__()
        self.cs = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss = loss
        self.pretrained = pretrained

    def forward(self, d_pred, scores_c, scores_u, d_real, c_real, u_real, length_batch):
        total_sents = torch.sum(length_batch != 0).float()
        labels_1s = (d_real != -1).float()
        d_pred_masked = d_pred * labels_1s  # b x seq-1
        d_real_masked = d_real * labels_1s  # b x seq-1
        if self.loss == 'l1':
            loss_d = torch.sum(torch.abs(d_pred_masked - d_real_masked), dim=1) / (length_batch.float() - 1)
            loss_d = torch.sum(loss_d) / total_sents
        elif self.loss == 'l2':
            loss_d = torch.sum(torch.abs(d_pred_masked - d_real_masked) ** 2, dim=1) / (length_batch.float() - 1)
            loss_d = torch.sum(loss_d) / total_sents
        elif self.loss == 'rank':
            lens_d = length_batch - 1
            max_len_d = torch.max(lens_d)
            mask = torch.arange(max_len_d, device=max_len_d.device)[None, :] < lens_d[:, None]
            loss_d = rankloss(d_pred, d_real, mask, exp=False)
        loss_c = self.cs(scores_c.view(-1, scores_c.shape[2]), c_real.view(-1))
        loss_u = self.cs(scores_u.view(-1, scores_u.shape[2]), u_real.view(-1))
        if self.pretrained:
            return loss_c + loss_u
        else:
            return loss_c + loss_d + loss_u


def rankloss(input, target, mask, exp=False):
    diff = input[:, :, None] - input[:, None, :]
    target_diff = ((target[:, :, None] - target[:, None, :]) > 0).float()
    mask = mask[:, :, None] * mask[:, None, :] * target_diff
    if exp:
        loss = torch.exp(torch.relu(target_diff - diff)) - 1
    else:
        loss = torch.relu(target_diff - diff)
    loss = (loss * mask).sum() / (mask.sum() + 1e-9)
    return loss
