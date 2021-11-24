import argparse
import datetime
import os
import shutil
import time
import random
import csv

import dgl
import numpy as np
import torch
import torch.nn as nn
from rouge import Rouge

from StockGraph import StockGraph
from Tester import SLTester, SLTesterStock
from module.stock_dataloader import ExampleSet, graph_collate_fn
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools.logger import *
from sklearn import metrics

parser = argparse.ArgumentParser(description='HeterSumGraph Model')

# Where to find data
parser = argparse.ArgumentParser(description='HeterSumGraph Model')

# Where to find data
parser.add_argument('--data_dir', type=str, default='../cache/STOCK/jsonlines/',help='The dataset directory.')
parser.add_argument('--cache_dir', type=str, default='../cache/STOCK/',help='The processed dataset directory')
parser.add_argument('--embedding_path', type=str, default='/users5/kxiong/work/stock_prediction/HeterSumGraph/glove.42B.300d.txt', help='Path expression to external word embedding.')

# Important settings
parser.add_argument('--model', type=str, default='HSG', help='model structure[HSG|HDSG]')
parser.add_argument('--restore_model', type=str, default='None', help='Restore model for further training. [bestmodel/bestFmodel/earlystop/None]')

# Where to save output
parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')

# Hyperparameters
parser.add_argument('--gpu', type=str, default='3,0', help='GPU ID to use. [default: 0]')
parser.add_argument('--cuda', action='store_true', default=True, help='GPU or CPU [default: False]')
parser.add_argument('--vocab_size', type=int, default=50000,help='Size of vocabulary. [default: 50000]')
parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs [default: 20]')
parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
parser.add_argument('--n_iter', type=int, default=2, help='iteration hop [default: 1]')

parser.add_argument('--word_embedding', action='store_true', default=False, help='whether to use Word embedding [default: True]')
parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding size [default: 300]')
parser.add_argument('--embed_train', action='store_true', default=True, help='whether to train Word embedding [default: False]')
parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
parser.add_argument('--num_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state [default: 128]')
parser.add_argument('--lstm_layers', type=int, default=2, help='Number of lstm layers [default: 2]')
parser.add_argument('--bidirectional', action='store_true', default=True, help='whether to use bidirectional LSTM [default: True]')
parser.add_argument('--n_feature_size', type=int, default=128, help='size of node feature [default: 128]')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden size [default: 64]')
parser.add_argument('--ffn_inner_hidden_size', type=int, default=512,help='PositionwiseFeedForward inner hidden size [default: 512]')
parser.add_argument('--n_head', type=int, default=8, help='multihead attention number [default: 8]')
parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1,help='recurrent dropout prob [default: 0.1]')
parser.add_argument('--atten_dropout_prob', type=float, default=0.1, help='attention dropout prob [default: 0.1]')
parser.add_argument('--ffn_dropout_prob', type=float, default=0.1,help='PositionwiseFeedForward dropout prob [default: 0.1]')
parser.add_argument('--use_orthnormal_init', action='store_true', default=True,help='use orthnormal init for lstm [default: True]')
parser.add_argument('--sent_max_len', type=int, default=50,help='max length of sentences (max source text sentence tokens)')
parser.add_argument('--doc_max_timesteps', type=int, default=50,help='max length of documents (max timesteps of documents)')

# Training
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--lr_descent', action='store_true', default=False, help='learning rate descent')
parser.add_argument('--grad_clip', action='store_true', default=False, help='for gradient clipping')
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='for gradient clipping max gradient normalization')

parser.add_argument('-m', type=int, default=3, help='decode summary length')
parser.add_argument('--event_max_len', type=int, default=50)
parser.add_argument('--crop_max_len', type=int, default=5)
parser.add_argument('--max_sent_num', type=int, default=50)
parser.add_argument('--max_event_num', type=int, default=50)
parser.add_argument('--event2sent', type=bool, default=True)
parser.add_argument('--event2event', type=bool, default=False)
parser.add_argument('--seed', type=int, default=3184)
# parser.add_argument('--crop_max_len', type=int, default=20)

args = parser.parse_args()
hps = args

# torch.nn.Module.dump_patches = True

# torch.cuda.manual_seed_all(args.seed)
# torch.manual_seed(args.seed)
# random.seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


model_dir = 'save/train_789'

TEST_FILE = os.path.join(args.data_dir, "test.label.jsonl")
VAL_FILE = os.path.join(args.data_dir, "val.label.jsonl")
VOCAL_FILE = os.path.join(args.cache_dir, "vocab")
FILTER_WORD = os.path.join(args.cache_dir, "filter_word.txt")

vocab = Vocab(VOCAL_FILE, 50000)
# embed = torch.nn.Embedding(vocab.size(), 300, padding_idx=0)

test_w2s_path = os.path.join(args.cache_dir, "test.w2s.tfidf.jsonl")
val_w2s_path = os.path.join(args.cache_dir, "val.w2s.tfidf.jsonl")

test_dataset = ExampleSet(TEST_FILE, vocab, hps.sent_max_len, hps.event_max_len, hps.crop_max_len, hps, test_w2s_path, FILTER_WORD)
val_dataset = ExampleSet(VAL_FILE, vocab, hps.sent_max_len, hps.event_max_len, hps.crop_max_len, hps, val_w2s_path, FILTER_WORD)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=32)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=32)

model_name = "earlystop0.6670137843353743"
logger.info("processing model :{}".format(model_name))

model = torch.load(os.path.join(model_dir, model_name))
model.to(torch.device("cuda:0"))

def evaluation(model, data_loader, dtype):
    model.eval()
    labels = []
    prediction = []
    model.eval()
    positive_scores = []
    softmax = nn.Softmax(1)
    length = len(data_loader)
    logger.info('{} batches in total'.format(length))
    _index = 0
    for i, (G, index) in enumerate(data_loader):
        iter_start_time = time.time()

        # model.eval()

        if _index % 10 == 0:
            logger.info('{} batchs have been done, {} in total'.format(_index, length))

        if hps.cuda:
            G.to(torch.device("cuda"))

        outputs = model.forward(G)  # [batch_size, 2]

        glist = dgl.unbatch(G)
        label_list = []
        for g in glist:
            enode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"]==2)
            label_list.append(g.nodes[enode_id[0]].data["label"])
        tmp_labels = torch.cat(label_list, dim=-1).long()
            
        
        predict_label = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
        prediction += predict_label
        labels += tmp_labels.cpu().numpy().tolist()
        
        outputs = softmax(outputs)
        positive_scores += outputs[:, 1].cpu().detach().numpy().tolist()

        _index += 1

    auc_socre = metrics.roc_auc_score(np.array(labels), np.array(positive_scores))

    logger.info('[{}], model:{}, AUC:{}'.format(dtype, model_name, auc_socre))

evaluation(model, val_loader, 'VAL')
evaluation(model, test_loader, 'TEST')


