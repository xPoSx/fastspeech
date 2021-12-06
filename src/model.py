from torch.nn.utils.rnn import pad_sequence
from torch import nn
import torch
import math


class PositionalEncoding(nn.Module):

    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads

        self.W_O = nn.Linear(hidden_size * n_heads, hidden_size)
        self.W_Q = nn.Linear(hidden_size, hidden_size * n_heads)
        self.W_K = nn.Linear(hidden_size, hidden_size * n_heads)
        self.W_V = nn.Linear(hidden_size, hidden_size * n_heads)

    def forward(self, x):
        q = self.W_Q(x).view(x.shape[0], x.shape[1], self.n_heads, self.hidden_size).transpose(1, 2)
        k = self.W_K(x).view(x.shape[0], x.shape[1], self.n_heads, self.hidden_size).transpose(1, 2)
        v = self.W_V(x).view(x.shape[0], x.shape[1], self.n_heads, self.hidden_size).transpose(1, 2)

        res = (torch.matmul(q, k.transpose(3, 2)) / math.sqrt(self.hidden_size)).softmax(dim=-1)
        res = torch.matmul(res, v).transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        return self.W_O(res)


class FFTBlock(nn.Module):
    def __init__(self, in_feats, conv_hidden_size, kernel_size, n_heads):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, in_feats)
        self.ln1 = nn.LayerNorm(in_feats)
        self.ln2 = nn.LayerNorm(in_feats)
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_feats, out_channels=conv_hidden_size, kernel_size=kernel_size, padding=kernel_size // 2
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=conv_hidden_size, out_channels=in_feats, kernel_size=kernel_size, padding=kernel_size // 2
            )
        )

    def forward(self, x):
        res = x
        x = self.attention(self.ln1(x))
        x += res
        res = x
        x = self.ln2(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x += res
        return x


class FFT(nn.Module):
    def __init__(self, n_blocks, n_heads, in_feats, conv_hidden_size, kernel_size, dropout, max_len=5000):
        super().__init__()
        self.pe = PositionalEncoding(in_feats, dropout, max_len)
        self.blocks = nn.Sequential(*[
            FFTBlock(
                in_feats, conv_hidden_size, kernel_size, n_heads
            ) for _ in range(n_blocks)
        ])
        self.ln = nn.LayerNorm(in_feats)

    def forward(self, x):
        x = self.pe(x)
        x = self.blocks(x)
        x = self.ln(x)
        return x


class DurationPredictor(nn.Module):
    def __init__(self, in_feats, conv_hidden_size, kernel_size, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_feats, conv_hidden_size, kernel_size, padding=kernel_size // 2)
        self.ln1 = nn.LayerNorm(conv_hidden_size)
        self.conv2 = nn.Conv1d(conv_hidden_size, conv_hidden_size, kernel_size, padding=kernel_size // 2)
        self.ln2 = nn.LayerNorm(conv_hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(conv_hidden_size, 1)

    def forward(self, x):
        x = self.conv1(x.transpose(-2, -1))
        x = self.dropout(self.relu(self.ln1(x.transpose(-2, -1))))

        x = self.conv2(x.transpose(-2, -1))
        x = self.dropout(self.relu(self.ln2(x.transpose(-2, -1))))

        x = torch.exp(self.linear(x).squeeze(-1))
        return x


class LengthRegulator(nn.Module):
    def __init__(self, in_feats, conv_hidden_size, kernel_size, dropout):
        super().__init__()
        self.dp = DurationPredictor(in_feats, conv_hidden_size, kernel_size, dropout)

    def forward(self, x, durations):
        pred_dur = self.dp(x)
        res = []
        if self.training:
            cur_dur = durations.int()
        else:
            cur_dur = torch.round(pred_dur).int()

        min_dur = min(cur_dur.size(1), x.size(1))
        for i in range(x.shape[0]):
            res.append(torch.repeat_interleave(x[i, :min_dur], cur_dur[i, :min_dur], 0))
        res = pad_sequence(res, batch_first=True)
        res = res.to('cuda')
        return res, pred_dur


class FastSpeech(nn.Module):
    def __init__(self, vocab_size=51, ph_blocks=6, mel_blocks=6, hidden_size=384, n_heads=2, kernel_size=3,
                 conv_hidden_size=1536, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.fft_ph = FFT(ph_blocks, n_heads, hidden_size, conv_hidden_size, kernel_size, dropout)
        self.lr = LengthRegulator(hidden_size, hidden_size, kernel_size, dropout)
        self.fft_mel = FFT(mel_blocks, n_heads, hidden_size, conv_hidden_size, kernel_size, dropout)
        self.l = nn.Linear(hidden_size, 80)

    def forward(self, x, durations):
        x = self.emb(x)
        x = self.fft_ph(x)
        x, pred_dur = self.lr(x, durations)
        x = self.fft_mel(x)
        return self.l(x), pred_dur
