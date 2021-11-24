import time
# import logging
import json
import pickle
import nltk
import numpy as np
from tools.logger import *
import torch
import dgl
import copy
from collections import Counter
import nltk
from nltk.corpus import stopwords

FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
FILTERWORD.extend(punctuations)


class Example(object):
    def __init__(self, vocab, senteces, events, sent_max_len, event_max_len, crop_max_len, crop):
        self.event_max_len = event_max_len
        self.sent_max_len = sent_max_len
        self.crop_max_len = crop_max_len
        self.vocab = vocab

        self.events_id = []
        self.events_len = []

        self.sents_id = []
        self.sents_len = []

        self.crop = crop
        self.crop_id = []

        # process sentences
        for sent in senteces:
            sent_words = sent.split()
            self.sents_len.append(len(sent_words))
            self.sents_id.append([vocab.word2id(w.lower()) for w in sent_words])
        
        # process events
        # for event in events:
        #     event_words = [part.split() for part in event]
        #     self.events_len.append(sum([len(part) for part in event_words]))
        #     self.events_id.append([[vocab.word2id(w.lower()) for w in part]for part in event_words])

        for event in events:
            event_words = []
            for e in event:
                event_words += e.split(' ')
            # event_words = event.split()
            self.events_len.append(len(event_words))
            self.events_id.append([vocab.word2id(w.lower()) for w in event_words])
        
        # process crop
        crop_words = crop.split()
        self.crop_id = [vocab.word2id(w.lower()) for w in crop_words]

        self.crop_id = self._pad_input(vocab.word2id("[PAD]"), self.crop_id, self.crop_max_len)
        self.sents_id = self._pad_input(vocab.word2id("[PAD]"), self.sents_id, self.sent_max_len)
        self.events_id = self._pad_input(vocab.word2id("[PAD]"), self.events_id, self.event_max_len)

    def _pad_input(self, pad_id, input_id, max_len):
        output = []
        if isinstance(input_id, list) and isinstance(input_id[0], list) and isinstance(input_id[0][0], list):
            for part in input_id:
                subj = part[0].copy()
                verb = part[1].copy()
                obj = part[2].copy()
                # subj_w = [1/len(subj) for _ in range(len(subj))]
                # verb_w = [1/len(verb) for _ in range(len(verb))]
                # obj_w = [1/len(obj) for _ in range(len(obj))]
                subj_w = [1] * len(subj)
                verb_w = [1] * len(verb)
                obj_w = [1] * len(obj)
                if len(subj) >= max_len:
                    subj = subj[:max_len]
                    subj_w = [1/max_len for _ in range(max_len)]
                else:
                    subj_w.extend([0] * (max_len - len(subj)))
                    subj.extend([pad_id] * (max_len - len(subj)))

                if len(verb) >= max_len:
                    verb = verb[:max_len]
                    verb_w = [1/max_len for _ in range(max_len)]
                else:
                    verb_w.extend([0] * (max_len - len(verb)))
                    verb.extend([pad_id] * (max_len - len(verb)))

                if len(obj) >= max_len:
                    obj = obj[:max_len]
                    obj_w = [1/max_len for _ in range(max_len)]
                else:
                    obj_w.extend([0] * (max_len-len(obj)))
                    obj.extend([pad_id] * (max_len - len(obj)))
                output.append([subj, verb, obj, subj_w, verb_w, obj_w])
                

        elif isinstance(input_id, list) and isinstance(input_id[0], list):
            for i in range(len(input_id)):
                example = input_id[i].copy()
                if len(example) >= max_len:
                    example = example[:max_len]
                else:
                    example.extend([pad_id] * (max_len - len(example)))
                output.append(example)
        elif isinstance(input_id, list):
            example = input_id.copy()
            if len(example) >= max_len:
                example = example[:max_len]
            else:
                example.extend([pad_id] * (max_len-len(example)))
            output = example
        return output
                

class ExampleSet(torch.utils.data.Dataset):
    def __init__(self, data_path, vocab, sent_max_len, event_max_len, crop_max_len, hps, w2s_path, filter_word_path):
        self.event_max_len = event_max_len
        self.sent_max_len = sent_max_len
        self.crop_max_len = crop_max_len
        self.vocab = vocab
        self.hps = hps

        self.filterwords = FILTERWORD
        self.filterids = [vocab.word2id(w.lower()) for w in FILTERWORD]
        self.filterids.append(vocab.word2id("[PAD]"))

        self.tfidf_w = readText(filter_word_path)

        lowtfidf_num = 0
        for w in self.tfidf_w:
            if vocab.word2id(w) != vocab.word2id('[UNK]'):
                self.filterwords.append(w)
                self.filterids.append(vocab.word2id(w))
                lowtfidf_num += 1
            if lowtfidf_num > 5000:
                break

        self.example_list = readJson(data_path)
        self.w2s_tfidf = readJson(w2s_path)


    def get_example(self, index):
        e = self.example_list[index]
        example = Example(self.vocab, e["sentences"], e["events"], self.sent_max_len, self.event_max_len, self.crop_max_len, e["crop"])

        return example, e["label"], e["e2e_edges"], e["e2s_edges"]


    def AddWordNode(self, G, sent_input, event_input):
        # G = dgl.DGLGraph()
        wid2nid = {}
        nid2wid = {}

        for sent in sent_input:
            for word in sent:
                if word not in wid2nid.keys() and word not in self.filterids:
                    wid2nid[word] = len(wid2nid)
                    nid2wid[len(nid2wid)] = word
                else:
                    continue
        
        for sent in event_input:
            # for part in sent:
            for word in sent:
                if word not in wid2nid.keys() and word not in self.filterids:
                    wid2nid[word] = len(wid2nid)
                    nid2wid[len(nid2wid)] = word
                else:
                    continue

        G.add_nodes(len(wid2nid))
        G.set_n_initializer(dgl.init.zero_initializer)
        
        G.ndata["unit"] = torch.zeros(len(wid2nid))
        G.ndata["dtype"] = torch.zeros(len(wid2nid))
        G.ndata["id"] = torch.LongTensor(list(nid2wid.values()))
        return wid2nid, nid2wid


    def CreateGraph(self, sent_pad, event_pad, crop_pad, label, e2e_edges, e2s_edges, w2s_w):
        G = dgl.DGLGraph()
        wid2nid, nid2wid = self.AddWordNode(G, sent_pad, event_pad)

        # add sent node
        w_nodes = len(wid2nid)
        SN = len(sent_pad)
        G.add_nodes(SN)
        G.ndata["unit"][w_nodes:] = torch.ones(SN)
        G.ndata["dtype"][w_nodes:] = torch.ones(SN)
        G.ndata['id'][w_nodes:] = torch.LongTensor([i for i in range(w_nodes, w_nodes+SN)])
        sid2nid = [i for i in range(w_nodes, w_nodes+SN)]

        # add event node
        ws_nodes = w_nodes + SN
        EN = len(event_pad)
        G.add_nodes(EN)
        G.ndata["unit"][ws_nodes:] = torch.ones(EN)
        G.ndata["dtype"][ws_nodes:] = torch.ones(EN) * 2
        G.ndata['id'][ws_nodes:] = torch.LongTensor([i for i in range(ws_nodes, ws_nodes+EN)])
        eid2nid = [i for i in range(ws_nodes, ws_nodes+EN)]

        G.set_e_initializer(dgl.init.zero_initializer)

        # add sent-word edge
        for i in range(SN):
            word_set = Counter(sent_pad[i])
            sent_tfw = w2s_w[str(i)]
            for wid in word_set.keys():
                if self.vocab.id2word(wid) in sent_tfw.keys() and wid in wid2nid.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)
                    G.add_edge(wid2nid[wid], sid2nid[i], data={"dtype": torch.LongTensor([0]), 'tffrac': torch.LongTensor([tfidf_box])})
                    G.add_edge(sid2nid[i], wid2nid[wid], data={"dtype": torch.LongTensor([0]), 'tffrac': torch.LongTensor([tfidf_box])})
        
        # add sent-sent edge
        # for i in range(SN):
        #     if SN == 1:
        #         break
        #     if 0 < i < SN-1:
        #         G.add_edge(sid2nid[i], sid2nid[i-1], data={"dtype": torch.LongTensor([0]), 'tffrac': torch.LongTensor([tfidf_box])})
        #         G.add_edge(sid2nid[i-1], sid2nid[i], data={"dtype": torch.LongTensor([0]), 'tffrac': torch.LongTensor([tfidf_box])})
        #         G.add_edge(sid2nid[i], sid2nid[i+1], data={"dtype": torch.LongTensor([0]), 'tffrac': torch.LongTensor([tfidf_box])})
        #         G.add_edge(sid2nid[i+1], sid2nid[i], data={"dtype": torch.LongTensor([0]), 'tffrac': torch.LongTensor([tfidf_box])})
        #     elif i == 0:
        #         G.add_edge(sid2nid[i], sid2nid[i+1], data={"dtype": torch.LongTensor([0]), 'tffrac': torch.LongTensor([tfidf_box])})
        #         G.add_edge(sid2nid[i+1], sid2nid[i], data={"dtype": torch.LongTensor([0]), 'tffrac': torch.LongTensor([tfidf_box])})
        #     else:
        #         G.add_edge(sid2nid[i], sid2nid[i-1], data={"dtype": torch.LongTensor([0]), 'tffrac': torch.LongTensor([tfidf_box])})
        #         G.add_edge(sid2nid[i-1], sid2nid[i], data={"dtype": torch.LongTensor([0]), 'tffrac': torch.LongTensor([tfidf_box])})
        
        # add event-word edge
        for i in range(EN):
            # for part in event_pad[i][:3]:
            # for part in event_pad[i]:
            # word_set = Counter(part)
            word_set = Counter(event_pad[i])
            for wid in word_set.keys():
                if wid in wid2nid.keys():
                    G.add_edge(wid2nid[wid], eid2nid[i], data={"dtype": torch.LongTensor([1]), 'tffrac': torch.LongTensor([1])})
                    G.add_edge(eid2nid[i], wid2nid[wid], data={"dtype": torch.LongTensor([1]), 'tffrac': torch.LongTensor([1])})
        
        G.nodes[sid2nid].data["words"] = torch.LongTensor(sent_pad)
        event_pad = torch.FloatTensor(event_pad)
        G.nodes[eid2nid].data['data'] = event_pad
        G.nodes[sid2nid].data["position"] = torch.arange(1, SN + 1).view(-1, 1).long()
        G.nodes[eid2nid].data["position"] = torch.arange(1, EN + 1).view(-1, 1).long()

        G.add_nodes(1)
        G.ndata["dtype"][ws_nodes+EN] = torch.ones(1)*4
        # G.ndata["words"][ws_nodes+EN] = torch.LongTensor(crop_pad)
        G.nodes[ws_nodes+EN].data['crop'] = torch.LongTensor(crop_pad).unsqueeze(0)
        G.nodes[ws_nodes+EN].data["position"] = torch.arange(1, 2).view(-1, 1).long()

        G.nodes[eid2nid].data["label"] = torch.zeros(EN) if label == 0 else torch.ones(EN)
        G.nodes[sid2nid].data["label"] = torch.zeros(SN) if label == 0 else torch.ones(SN)

        if self.hps.event2event:
            for edge in e2e_edges:
                try:
                    G.add_edge(eid2nid[edge[0]], eid2nid[edge[1]], data={"dtype": torch.LongTensor([2]), 'tffrac': torch.LongTensor([1])})
                    G.add_edge(eid2nid[edge[1]], eid2nid[edge[0]], data={"dtype": torch.LongTensor([2]), 'tffrac': torch.LongTensor([1])})
                except:
                    continue
        if self.hps.event2sent:
            # logger.info('Using event2event edges')
            # count = 0
            for edge in e2s_edges:
                try:
                    G.add_edge(eid2nid[edge[0]], sid2nid[edge[1]], data={"dtype": torch.LongTensor([3]), 'tffrac': torch.LongTensor([1])})
                    G.add_edge(sid2nid[edge[1]], eid2nid[edge[0]], data={"dtype": torch.LongTensor([3]), 'tffrac': torch.LongTensor([1])})
                    # count += 1
                except:
                    continue
            # logger.info('{} event edges has been added'.format(count))
        
        # add event self-loop
        # for e_node in eid2nid:
        #     G.add_edge(e_node, e_node, data={"dtype": torch.LongTensor([4]), 'tffrac': torch.LongTensor([1])})


        return G


    def __getitem__(self, index):
        item, label, e2e_edges, e2s_edges = self.get_example(index)
        sent_pad = item.sents_id[:self.hps.max_sent_num]
        event_pad = item.events_id[:self.hps.max_event_num]
        G = self.CreateGraph(sent_pad, event_pad, item.crop_id, label, e2e_edges, e2s_edges, self.w2s_tfidf[index])
        return G, index
    
    def __len__(self):
        return len(self.example_list)
        



def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def graph_collate_fn(samples):
    '''
    :param batch: (G, input_pad)
    :return: 
    '''
    graphs, index = map(list, zip(*samples))
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # sent node of graph
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    return batched_graph, [index[idx] for idx in sorted_index]


def readText(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data
