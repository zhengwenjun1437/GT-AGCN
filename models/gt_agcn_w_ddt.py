import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import TextGATConv, TextGATConv_mod, TextGCNConv, TextGINConv, TransformerLayer


class GT_AGCN_W_DDT(nn.Module):
    """将AGCN中的AODT修改为DDT"""
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        self.opt = opt
        self.embedding_matrix = embedding_matrix
        self.emb = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)
        self.pos_emb = nn.Embedding(opt.pos_size, opt.pos_dim, padding_idx=0) if opt.pos_dim > 0 else None
        self.post_emb = nn.Embedding(2 * opt.max_position + 2, opt.post_dim,
                                     padding_idx=0) if opt.post_dim > 0 else None
        self.deprel_emb = nn.Embedding(opt.deprel_size + 1, opt.deprel_dim,
                                       padding_idx=0) if opt.deprel_dim > 0 else None
        in_dim = opt.embed_dim + opt.pos_dim + opt.post_dim

        if opt.use_rnn:
            self.rnn = nn.LSTM(in_dim, opt.rnn_hidden, batch_first=True,
                               dropout=opt.rnn_dropout if opt.rnn_layers > 1 else 0.0, bidirectional=opt.bidirect)
            self.rnn_drop = nn.Dropout(opt.rnn_dropout)
            self.linear_middle = nn.Linear(opt.rnn_hidden * 2 if opt.bidirect else opt.rnn_hidden, opt.hidden_dim)
        else:
            self.linear_in = nn.Linear(in_dim, opt.hidden_dim)

        self.linear_out_1 = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        self.linear_out_2 = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        self.linear_out = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)

        self.emb_drop = nn.Dropout(opt.input_dropout)
        self.ffn_dropout = opt.ffn_dropout

        self.graph_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.transformer_layers = nn.ModuleList()
        self.wa_s = nn.ModuleList()

        norm_class = None
        if opt.norm == 'ln':
            norm_class = nn.LayerNorm
        elif opt.norm == 'bn':
            norm_class = nn.BatchNorm1d

        for i in range(opt.num_layers):
            if opt.graph_conv_type == 'gat-mod':
                graph_conv = TextGATConv_mod(hidden_dim=opt.hidden_dim,  # F = H * C
                                             num_heads=opt.graph_conv_attention_heads,  # H
                                             attn_dropout=opt.graph_conv_attn_dropout,
                                             ffn_dropout=opt.ffn_dropout,
                                             edge_dim=opt.deprel_dim,
                                             norm=opt.norm)
            elif opt.graph_conv_type == 'gcn':
                graph_conv = TextGCNConv(opt.hidden_dim, opt.deprel_dim)
            elif opt.graph_conv_type == 'gin':
                graph_conv = TextGINConv(opt.hidden_dim,
                                         dropout_ratio=opt.ffn_dropout,
                                         norm=opt.norm,
                                         edge_dim=opt.deprel_dim)
            elif opt.graph_conv_type == 'gat':
                graph_conv = TextGATConv(hidden_dim=opt.hidden_dim,  # F = H * C
                                         num_heads=opt.graph_conv_attention_heads,  # H
                                         attn_dropout=opt.graph_conv_attn_dropout,
                                         ffn_dropout=opt.ffn_dropout,
                                         edge_dim=opt.deprel_dim,
                                         norm=opt.norm)
            self.graph_convs.append(graph_conv)
            self.norms.append(norm_class(opt.hidden_dim))
            self.transformer_layers.append(TransformerLayer(
                opt.hidden_dim,
                opt.attention_heads,
                attn_dropout_ratio=opt.attn_dropout,
                ffn_dropout_ratio=opt.ffn_dropout,
                norm=opt.norm))

        for j in range(opt.gcn_layers):
            linear = nn.Linear(opt.hidden_dim, opt.hidden_dim)
            self.wa_s.append(linear)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, length, adj, adj_2, adj_3, adj_dis = inputs

        maxlen = torch.max(length).item()
        tok = tok[:, :maxlen]
        pos = pos[:, :maxlen]
        deprel = deprel[:, :maxlen]
        post = post[:, :maxlen]
        mask = mask[:, :maxlen]
        adj = adj[:, :maxlen, :maxlen]
        adj_2 = adj_2[:, :maxlen, :maxlen]
        adj_2 = torch.exp(self.opt.alpha * (-1.0) * adj_2).type(torch.float32)
        src_mask = (tok != 0)

        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.opt.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.emb_drop(embs)

        if self.opt.use_rnn:
            self.rnn.flatten_parameters()
            rnn_output, (hn, cn) = self.rnn(embs)
            rnn_output = self.rnn_drop(rnn_output)
            h = self.linear_middle(rnn_output)
        else:
            h = self.linear_in(embs)

        h_dis = h.clone()
        e = self.deprel_emb(adj)

        for i in range(self.opt.num_layers):
            h0 = h
            h = self.graph_convs[i](h, adj, e)

            h = self.norms[i](h)
            h = h.relu()
            h = F.dropout(h, self.ffn_dropout, training=self.training)

            h = self.transformer_layers[i](h, src_mask)
            h = h + h0

        for j in range(self.opt.gcn_layers):
            h_dis = adj_2.bmm(h_dis)
            h_dis = self.wa_s[j](h_dis)
            h_dis = h_dis / (adj_2.sum(-1).unsqueeze(-1) + 1)
            h_dis = F.relu(h_dis)

        aspect_words_num = mask.sum(dim=1).unsqueeze(-1)
        mask = mask.unsqueeze(-1)
        out_1 = (h * mask).sum(dim=1) / aspect_words_num
        out_2 = (h_dis * mask).sum(dim=1) / aspect_words_num
        out = torch.cat((out_1, out_2), dim=-1)

        output_1 = self.linear_out_1(out_1)
        output_2 = self.linear_out_2(out_2)
        output_3 = self.linear_out(out)

        output = torch.stack((output_1, output_2, output_3), dim=-1)
        output = torch.mean(output, dim=1)

        return output






























