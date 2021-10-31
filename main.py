# !/usr/bin/python
# -*- coding: utf-8 -*-
# Author: MrPickles
# Date: 26/Oct/2021
# Learn implementation of TRM by pytorch
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [seq_len, batch_size, d_model]
        :return:
        """
        x += self.pe[:x.size(0), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    @staticmethod
    def forward(Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size * len_q * d_model]
        :param K: [batch_size * len_k * d_model]
        :param V: [batch_size * len_k * d_model]
        :param attn_mask: [batch_size * len_q * d_model]
        :return: self.layer_norm(output + residual), attn
        """
        residual, batch_size = Q, Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # [batch_size * n_heads * len_q * d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # [batch_size * n_heads * len_k * d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # [batch_size * n_heads * len_k * d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # [batch_size * n_heads * len_q * len_k]

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)  # [batch_size * len_q * d_model]
        return self.layer_norm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs  # [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


def get_attn_pad_mask(seq_q, seq_k):
    print("seq_q size: ", seq_q.size())
    print("seq_k size: ", seq_k.size())

    batch_size, len_q = seq_q.size()[0], seq_q.size()[1]
    batch_size, len_k = seq_k.size()[0], seq_k.size()[1]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_in, enc_self_attn_mask):
        enc_out, attn = self.enc_self_attn(enc_in, enc_in, enc_in, enc_self_attn_mask)  # enc_in is same to Q,K,V
        enc_out = self.pos_ffn(enc_out)  # enc_out: [batch_size * len_q * d_model]
        return enc_out, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_in):
        enc_out = self.src_emb(enc_in)
        enc_out = self.pos_emb(enc_out.transpose(0, 1)).transpose(0, 1)
        # enc_out.shape(): [batch_size, src_len, d_model]

        enc_self_attn_mask = get_attn_pad_mask(enc_in, enc_in)
        enc_self_attns = []
        for layer in self.layers:
            enc_out, enc_self_attn = layer(enc_out, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_out, enc_self_attns


def get_attn_subsequent_mask(seq):
    """
    :param seq: [batch_size, tgt_len]
    :return subsequence_mask: [batch_size, tgt_len, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]  # [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_in, enc_out, dec_self_attn_mask, dec_enc_attn_mask):
        dec_out, dec_self_attn = self.dec_self_attn(dec_in, dec_in, dec_in, dec_self_attn_mask)
        dec_out, dec_enc_attn = self.dec_enc_attn(dec_out, enc_out, enc_out, dec_enc_attn_mask)
        dec_out = self.pos_ffn(dec_out)
        return dec_out, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_in, enc_in, enc_out):
        dec_out = self.tgt_emb(dec_in)
        dec_out = self.pos_emb(dec_out.transpose(0, 1)).transpose(0, 1)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_in, dec_in)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_in)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_in, enc_in)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_out, dec_self_attn, dec_enc_attn = layer(dec_in, enc_out, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_out, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_in, dec_in):
        # enc_in.shape(): [batch_size, src_len]
        # dec_in.shape(): [batch_size, tgt_len]
        # attns(Q, K, V) = softmax(Q * K.transpose() / sqrt(dk)) * V
        enc_out, enc_self_attns = self.encoder(enc_in)
        dec_out, dec_self_attns, dec_enc_attns = self.decoder(dec_in, enc_out, enc_in)

        # dec_logits.shape(): [batch_size * src_vocab_size * tgt_vocab_size]
        dec_logits = self.proj(dec_out)
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns


def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


def visualization(attn):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()


if __name__ == '__main__':

    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    # Transformer Parameters
    # Padding Should be Zero
    # Create vocabulary
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5  # length of source
    tgt_len = 5  # length of target

    # Params
    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention

    # Instantiate
    model = Transformer()

    # Train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    for epoch in range(20):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    print('first head of last state enc_self_attns')
    visualization(enc_self_attns)

    print('first head of last state dec_self_attns')
    visualization(dec_self_attns)

    print('first head of last state dec_enc_attns')
    visualization(dec_enc_attns)
