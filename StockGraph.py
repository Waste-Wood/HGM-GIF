import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import dgl
from module.PositionEmbedding import get_sinusoid_encoding_table
from module.Encoder import sentEncoder
from module.GATStackLayer import MultiHeadSGATLayer, MultiHeadLayer
from module.GATLayer import PositionwiseFeedForward, WSGATLayer, SWGATLayer
import torch.nn.functional as F
from NTN import NeuralTensorNetwork


def get_node_feat(G, feat, dtype):
    glist = dgl.unbatch(G)
    feature = []
    glen = []
    for g in glist:
        snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == dtype)
        feature.append(g.nodes[snode_id].data[feat])
        glen.append(len(snode_id))
    return feature, glen



class TextEncoder(nn.Module):
    def __init__(self, hps, embed):
        super().__init__()
        # self.dtype = dtype
        self.hps = hps
        
        self.pos_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.hps.max_sent_num + 1, self.hps.word_emb_dim, padding_idx=0),
            freeze=True)

        self.cnn_proj = nn.Linear(hps.word_emb_dim, hps.n_feature_size)
        self.lstm_hidden_state = hps.lstm_hidden_state
        self.lstm = nn.LSTM(hps.word_emb_dim, self.lstm_hidden_state, num_layers=hps.lstm_layers, dropout=0.1,
                            batch_first=True, bidirectional=hps.bidirectional)
        
        if hps.bidirectional:
            self.lstm_proj = nn.Linear(2*self.lstm_hidden_state, hps.n_feature_size)
        else:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state, hps.n_feature_size)
        
        self.n_gram_enc = sentEncoder(hps, embed)

        self.nfeature_proj = nn.Linear(2*hps.n_feature_size, hps.hidden_size, bias=False)
        
    
    def forward(self, graph, dtype):

        # get cnn featrue
        
        tnodes = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == dtype)

        ngram_feature = self.n_gram_enc(graph.nodes[tnodes].data["words"])
        graph.nodes[tnodes].data["embedding"] = ngram_feature
        snode_pos = graph.nodes[tnodes].data["position"].view(-1)
        position_embedding = self.pos_embedding(snode_pos)
        cnn_feature = self.cnn_proj(ngram_feature+position_embedding)

        # get lstm fearure
        features, glen = get_node_feat(graph, 'embedding', dtype)
        seq_pad = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(seq_pad, glen, batch_first=True, enforce_sorted=False)
        lstm_output, _ = self.lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        lstm_feature = self.lstm_proj(torch.cat(lstm_embedding, dim=0))

        node_feature = self.nfeature_proj(torch.cat([cnn_feature, lstm_feature], dim=1))

        graph.nodes[tnodes].data["feat"] = node_feature

        return node_feature


class WordEncoder(nn.Module):
    def __init__(self, hps, embed):
        super().__init__()
        self._embed = embed
        self._TFembed = nn.Embedding(10, hps.feat_embed_size)
    
    def forward(self, graph):
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"]==0)
        ws_edges = graph.filter_edges(lambda edges: edges.data["dtype"]==0)
        etf = graph.edges[ws_edges].data["tffrac"]
        graph.edges[ws_edges].data["tfidfembed"] = self._TFembed(etf)

        we_edges = graph.filter_edges(lambda edges: edges.data["dtype"]==1)
        etf = graph.edges[we_edges].data["tffrac"]
        graph.edges[we_edges].data["tfidfembed"] = self._TFembed(etf)

        wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]
        w_embed = self._embed(wid)  # [n_wnodes, D]
        graph.nodes[wnode_id].data["embedding"] = w_embed

        return w_embed


class ArgumentEncoder(nn.Module):
    def __init__(self, hps, embed):
        super().__init__()
        self.embed = embed
        self.atten = nn.Linear(300, 1)
        self.softmax = nn.Softmax(1)
    
    def forward(self, data, mask):
        data = torch.LongTensor(data.cpu().tolist()).cuda()
        feature = self.embed(data)
        mask = (mask - 1.0) * -10000.0
        atten_score = self.atten(feature).squeeze(2)
        atten_weight = self.softmax(atten_score + mask.cuda())
        output = torch.matmul(atten_weight.unsqueeze(1), feature).squeeze(1)
        return output


class EventEncoder(nn.Module):
    def __init__(self, hps, embed):
        super().__init__()
        self.subj_encoder = ArgumentEncoder(hps, embed)
        self.verb_encoder = ArgumentEncoder(hps, embed)
        self.obj_encoder = ArgumentEncoder(hps, embed)
        self.linear1 = nn.Linear(300, 64)
        self.linear2 = nn.Linear(300, 64)
        self.linear3 = nn.Linear(300, 64)
    
    def forward(self, graph):
        enode_id = graph.filter_nodes(lambda nodes: nodes.data['dtype']==2)
        e_data = graph.nodes[enode_id].data['data']
        subj, verb, obj = e_data[:, 0, :], e_data[:, 1, :], e_data[:, 2, :]
        subj_mask, verb_mask, obj_mask = e_data[:, 3, :], e_data[:, 4, :], e_data[:, 5, :]
        subj_feat = self.subj_encoder(subj, subj_mask)
        verb_feat = self.verb_encoder(verb, verb_mask)
        obj_feat = self.obj_encoder(obj, obj_mask)
        subj_feat = self.linear1(subj_feat)
        verb_feat = self.linear2(verb_feat)
        obj_feat = self.linear3(obj_feat)
        output = torch.tanh(subj_feat+verb_feat+obj_feat)

        del graph.ndata['data']
        graph.nodes[enode_id].data['feat'] = output
        return output


class StockGraph(nn.Module):
    def __init__(self, hps, embed):
        super().__init__()
        self.hps = hps
        self.embed = embed

        self.sent_encoder = TextEncoder(hps, embed)
        self.event_encoder = TextEncoder(hps, embed)
        self.word_encoder = WordEncoder(hps, embed)
        # self.event_encoder = EventEncoder(hps, embed)

        # self.event_proj = nn.Linear(300, 64)

        embed_size = hps.word_emb_dim

        # word -> sent and event
        self.W2T = WTWGAT(in_dim=embed_size,
                          out_dim=hps.hidden_size,
                          num_heads=hps.n_head,
                          attn_drop_out=hps.atten_dropout_prob,
                          ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                          ffn_drop_out=hps.ffn_dropout_prob,
                          feat_embed_size=hps.feat_embed_size,
                          layerType="WT")
        
        # sent, event -> word
        self.T2W = WTWGAT(in_dim=hps.hidden_size,
                          out_dim=embed_size,
                          num_heads=6,
                          attn_drop_out=hps.atten_dropout_prob,
                          ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                          ffn_drop_out=hps.ffn_dropout_prob,
                          feat_embed_size=hps.feat_embed_size,
                          layerType="TW")
        
        if hps.event2event:
            self.E2E = ESEGAT(in_dim=hps.hidden_size,
                          out_dim=hps.hidden_size,
                          num_heads=8,
                          attn_drop_out=hps.atten_dropout_prob,
                          ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                          ffn_drop_out=hps.ffn_dropout_prob,
                          feat_embed_size=hps.feat_embed_size,
                          layerType="EE")


        if hps.event2sent:
            self.S2E = ESEGAT(in_dim=hps.hidden_size,
                          out_dim=hps.hidden_size,
                          num_heads=8,
                          attn_drop_out=hps.atten_dropout_prob,
                          ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                          ffn_drop_out=hps.ffn_dropout_prob,
                          feat_embed_size=hps.feat_embed_size,
                          layerType="SE")
            
            self.E2S = ESEGAT(in_dim=hps.hidden_size,
                          out_dim=hps.hidden_size,
                          num_heads=8,
                          attn_drop_out=hps.atten_dropout_prob,
                          ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                          ffn_drop_out=hps.ffn_dropout_prob,
                          feat_embed_size=hps.feat_embed_size,
                          layerType="ES")
        
        
        
        if hps.sent2sent:
            self.S2S = ESEGAT(in_dim=hps.hidden_size,
                          out_dim=hps.hidden_size,
                          num_heads=8,
                          attn_drop_out=hps.atten_dropout_prob,
                          ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                          ffn_drop_out=hps.ffn_dropout_prob,
                          feat_embed_size=hps.feat_embed_size,
                          layerType="SS")
        
        self.n_feature = hps.hidden_size
        self.wh = nn.Linear(self.n_feature*2, 2)
        self.cp = nn.Linear(embed_size, self.n_feature)
        self.atten = nn.Linear(self.n_feature, 1)
        self.softmax = nn.Softmax(1)
        # self.event_encoder = NeuralTensorNetwork(embed, 300)
    
    def forward(self, graph):

        # init node feature
        wn_feature = self.word_encoder(graph)
        sent_feature = self.sent_encoder(graph, 1)

        text_nodes = graph.filter_nodes(lambda nodes: nodes.data["unit"]==1)
        word_nodes = graph.filter_nodes(lambda nodes: nodes.data["unit"]==0)
        event_nodes = graph.filter_nodes(lambda nodes: nodes.data["dtype"]==2)
        sent_nodes = graph.filter_nodes(lambda nodes: nodes.data["dtype"]==1)
        crop_nodes = graph.filter_nodes(lambda nodes: nodes.data["dtype"]==4)
        event_state = self.event_encoder(graph, 2)

        # for _ in range(self.hps.n_iter):
            # # Event graph update: event --> event
            # event_state = graph.nodes[event_nodes].data['feat']
            # event_state = self.E2E(graph, event_state, event_state)
            # graph.nodes[event_nodes].data['feat'] = event_state

            # # Sent graph update: sent --> sent
            # sent_state = graph.nodes[sent_nodes].data['feat']
            # sent_state = self.S2S(graph, sent_state, sent_state)
            # graph.nodes[sent_nodes].data['feat'] = sent_state

            # # Heterogeneous graph update: word <==> event, sent
            # word_state = graph.nodes[word_nodes].data['embedding']
            # text_state = graph.nodes[text_nodes].data['feat']
            # text_state = self.W2T(graph, word_state, text_state)
            # graph.nodes[text_nodes].data['feat'] = text_state
            # word_state = self.T2W(graph, word_state, text_state)
            # graph.nodes[word_nodes].data['embedding'] = word_state
            # text_state = self.W2T(graph, word_state, text_state)
            # graph.nodes[text_nodes].data['feat'] = text_state

            # # Hetergeneous graph update: event <==> sent
            # event_state = graph.nodes[event_nodes].data['feat']
            # sent_state = graph.nodes[sent_nodes].data['feat']
            # event_state = self.S2E(graph, sent_state, event_state)
            # graph.nodes[event_nodes].data['feat'] = event_state
            # sent_state = self.E2S(graph, sent_state, event_state)
            # graph.nodes[sent_nodes].data['feat'] = sent_state


        text_state = graph.nodes[text_nodes].data['feat']
        word_state = wn_feature
        # text_state = self.W2T(graph, word_state, text_state)
        graph.nodes[text_nodes].data['feat'] = text_state

        for _ in range(self.hps.n_iter):
            if self.hps.event2event:
                event_state = graph.nodes[event_nodes].data['feat']
                event_state = self.E2E(graph, event_state, event_state)
                
                graph.nodes[event_nodes].data['feat'] = event_state

            if self.hps.event2sent:
                event_state = graph.nodes[event_nodes].data['feat']
                sent_state= graph.nodes[sent_nodes].data['feat']
                event_state = self.S2E(graph, sent_state, event_state)
                sent_state = self.E2S(graph, sent_state, event_state)

                graph.nodes[sent_nodes].data['feat'] = sent_state
                graph.nodes[event_nodes].data['feat'] = event_state

            # sent, event -> word
            text_state = graph.nodes[text_nodes].data['feat']
            # word_state = self.T2W(graph, word_state, text_state)
            graph.nodes[word_nodes].data['embedding'] = word_state

            # word -> sent, event
            # text_state = self.W2T(graph, word_state, text_state)
            # text_state = F.elu(text_state) if self.hps.n_iter > 1 else text_state 
            graph.ndata['feat'][text_nodes] = text_state

        

        predict_nodes = graph.filter_nodes(lambda nodes: nodes.data['dtype']==1)
        predict_state = graph.nodes[predict_nodes].data['feat']

        glist = dgl.unbatch(graph)
        glen = []
        for g in glist:
            enodes = g.filter_nodes(lambda nodes: nodes.data["dtype"]==1)
            glen.append(len(enodes))
        attention_mask = [[1 for _ in range(length)]+[0 for _ in range(self.hps.max_event_num-length)]for length in glen]
        attention_mask = torch.FloatTensor(attention_mask)
        attention_mask = (1.0 - attention_mask) * -10000.0
        attention_mask = attention_mask.cuda()

        glen = [0] + [sum(glen[:i+1]) for i in range(len(glen))]
        tmp = []
        for i in range(1, len(glen)):
            x1 = predict_state[glen[i-1]:glen[i]][:]
            x2 = torch.FloatTensor([[0 for _  in range(self.n_feature)]for _ in range(self.hps.max_event_num-glen[i]+glen[i-1])]).cuda()
            tmp.append(torch.cat((x1, x2), 0))
        
        output = torch.cat(tuple([torch.unsqueeze(t, 0) for t in tmp]), 0)

        crop_feature = torch.mean(self.embed(graph.nodes[crop_nodes].data['crop']), 1)
        crop_feature = self.cp(crop_feature)

        # attention with crop
        atten_score = torch.matmul(crop_feature.unsqueeze(1), output.transpose(1, 2)).squeeze(1)
        atten_weight = self.softmax(atten_score+attention_mask).unsqueeze(1)
        output = torch.matmul(atten_weight, output).squeeze(1)


        ## cross attention
        # atten_score = self.atten(output).squeeze(2)
        # atten_weight = self.softmax(atten_score + attention_mask).unsqueeze(1) 
        # output = torch.matmul(atten_weight, output).squeeze(1)

        output = torch.cat((crop_feature, output), dim=-1)

        output = self.wh(output)
        return output


class WTWGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == 'WT':
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=WTLayer)
        else:
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=TWLayer)
        
        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)
        self.proj = nn.Linear(2*out_dim, out_dim)
    
    def forward(self, g, w, t):
        origin, neighbor = [t, w] if self.layerType == 'WT' else [w, t]
        h = F.elu(self.layer(g, neighbor))
        # h = h + origin
        h = self.proj(torch.cat((h, origin), -1))
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        return h


class WTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.dstfeat_fc = nn.Linear(64, out_dim, bias=False)

    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        dstfeat = self.dstfeat_fc(edges.dst["feat"])
        # wa = torch.matmul(edges.src['z'].unsqueeze(1), dstfeat.unsqueeze(2)).squeeze(1)
        z2 = torch.cat([edges.src['z'], dstfeat], dim=1) 
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}


    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}


    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    
    def forward(self, g, h):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        tnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        edges = g.filter_edges(lambda edges: (edges.src["unit"]==0) & (edges.dst["unit"]==1))

        z = self.fc(h)
        g.nodes[wnode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=edges)
        g.pull(tnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[tnode_id]


class TWLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.dstfeat_fc = nn.Linear(300, out_dim, bias=False)


    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        dstfeat = self.dstfeat_fc(edges.dst["embedding"])
        # wa = torch.matmul(edges.src['z'].unsqueeze(1), dstfeat.unsqueeze(2)).squeeze(1)
        z2 = torch.cat([edges.src['z'], dstfeat], dim=1)  # [edge_num, 2 * out_dim]
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}


    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}


    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    
    def forward(self, g, h):
        wnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        tnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == 1)
        edges = g.filter_edges(lambda edges: (edges.src["unit"]==1) & (edges.dst["unit"]==0))

        z = self.fc(h)
        g.nodes[tnode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=edges)
        g.pull(wnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[wnode_id]


class ESEGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == 'SE':
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=SELayer)
        elif layerType == 'ES':
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=ESLayer)
        elif layerType == 'EE':
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=EELayer)
        else:
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=SSLayer)
        
        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)
        # self.linear = nn.Linear(feat_embed_size, out_dim)
        self.proj = nn.Linear(2*out_dim, out_dim)
    
    
    def forward(self, g, s, e):
        origin, neighbor = [e, s] if self.layerType == 'SE' else [s, e]
        h = F.elu(self.layer(g, neighbor))
        # h = h + origin
        h = self.proj(torch.cat((h, origin), -1))
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        return h


class SELayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)
        self.dstfeat_fc = nn.Linear(64, out_dim, bias=False)

    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        dstfeat = self.dstfeat_fc(edges.dst["feat"])
        wa = torch.matmul(edges.src['z'].unsqueeze(1), dstfeat.unsqueeze(2)).squeeze(1)
        # z2 = torch.cat([edges.src['z'], dfeat, dstfeat], dim=1)  # [edge_num, 2 * out_dim]
        # wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}


    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}


    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}
    
    def forward(self, g, h):
        enode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        edges = g.filter_edges(lambda edges: (edges.src["dtype"]==1) & (edges.dst["dtype"]==2))

        z = self.fc(h)
        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=edges)
        g.pull(enode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[enode_id]


class ESLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)
        self.dstfeat_fc = nn.Linear(64, out_dim, bias=False)

    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        dstfeat = self.dstfeat_fc(edges.dst["feat"])
        wa = torch.matmul(edges.src['z'].unsqueeze(1), dstfeat.unsqueeze(2)).squeeze(1)
        # z2 = torch.cat([edges.src['z'], dfeat, dstfeat], dim=1)  # [edge_num, 2 * out_dim]
        # wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}


    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}


    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}
    
    def forward(self, g, h):
        enode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        edges = g.filter_edges(lambda edges: (edges.src["dtype"]==2) & (edges.dst["dtype"]==1))

        z = self.fc(h)
        g.nodes[enode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=edges)
        g.pull(snode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[snode_id]


class EELayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)
        self.dstfeat_fc = nn.Linear(64, out_dim, bias=False)


    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        dstfeat = self.dstfeat_fc(edges.dst["feat"])
        wa = torch.matmul(edges.src['z'].unsqueeze(1), dstfeat.unsqueeze(2)).squeeze(1)
        # z2 = torch.cat([edges.src['z'], dfeat, dstfeat], dim=1)  # [edge_num, 2 * out_dim]
        # wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}


    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}


    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}
    
    def forward(self, g, h):
        enode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        edges = g.filter_edges(lambda edges: (edges.src["dtype"]==2) & (edges.dst["dtype"]==2))

        z = self.fc(h)
        g.nodes[enode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=edges)
        g.pull(enode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[enode_id]


class SSLayer(nn.Module):
    def __init__(self, in_dim, out_dim, feat_embed_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.feat_fc = nn.Linear(feat_embed_size, out_dim, bias=False)
        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)
        self.dstfeat_fc = nn.Linear(64, out_dim, bias=False)


    def edge_attention(self, edges):
        # dfeat = self.feat_fc(edges.data["tfidfembed"])  # [edge_num, out_dim]
        dstfeat = self.dstfeat_fc(edges.dst["feat"])
        wa = torch.matmul(edges.src['z'].unsqueeze(1), dstfeat.unsqueeze(2)).squeeze(1)
        # z2 = torch.cat([edges.src['z'], dfeat, dstfeat], dim=1)  # [edge_num, 2 * out_dim]
        # wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}


    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}


    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}
    
    def forward(self, g, h):
        enode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        edges = g.filter_edges(lambda edges: (edges.src["dtype"]==1) & (edges.dst["dtype"]==1))

        z = self.fc(h)
        g.nodes[enode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=edges)
        g.pull(enode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('sh')
        return h[enode_id]



