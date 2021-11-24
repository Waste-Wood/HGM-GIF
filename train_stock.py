import argparse
import datetime
import os
import shutil
import time

import dgl
import numpy as np
import torch
from rouge import Rouge

from StockGraph import StockGraph
from Tester import SLTester
from module.stock_dataloader import ExampleSet, graph_collate_fn
from module.embedding import Word_Embedding
from module.vocabulary import Vocab
from tools.logger import *
import torch.nn as nn
import random
from sklearn import metrics

_DEBUG_FLAG_ = False


def save_model(model, save_file):
    with open(save_file, 'wb') as f:
        torch.save(model, f)
    logger.info('[INFO] Saving model to %s', save_file)


def evaluation(model, val_loader, hps):
    labels = []
    prediction = []
    model.eval()
    positive_scores = []
    softmax = nn.Softmax(1)
    for i, (G, index) in enumerate(val_loader):
        iter_start_time = time.time()

        # model.eval()

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
    
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(labels)):
        if labels[i] == 1 and prediction[i] == 1:
            TP += 1
        elif labels[i] == 1 and prediction[i] == 0:
            FN += 1
        elif labels[i] == 0 and prediction[i] == 1:
            FP += 1
        else:
            TN += 1
    precision = TP / (FP + TP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)

    assert len(labels) == len(prediction)
    
    auc_socre = metrics.roc_auc_score(np.array(labels), np.array(positive_scores))

    return precision, recall, F1, auc_socre



def setup_training(model, train_loader, valid_loader, valset, hps, test_loader):
    """ Does setup before starting training (run_training)
    
        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :return: 
    """

    train_dir = os.path.join(hps.save_root, "train_11_18")
    if os.path.exists(train_dir) and hps.restore_model != 'None':
        logger.info("[INFO] Restoring %s for training...", hps.restore_model)
        bestmodel_file = os.path.join(train_dir, hps.restore_model)
        model.load_state_dict(torch.load(bestmodel_file))
        hps.save_root = hps.save_root + "_reload"
    else:
        logger.info("[INFO] Create new model for training...")
        if os.path.exists(train_dir): shutil.rmtree(train_dir)
        os.makedirs(train_dir)

    try:
        run_training(model, train_loader, valid_loader, valset, hps, train_dir, test_loader)
    except KeyboardInterrupt:
        logger.error("[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
        save_model(model, os.path.join(train_dir, "earlystop"))


def run_training(model, train_loader, valid_loader, valset, hps, train_dir, test_loader):

    logger.info("[INFO] Starting run_training")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    best_train_loss = None
    best_loss = None
    best_F = None
    non_descent_cnt = 0
    saveNo = 0
    patient = 0
    best_auc_score = 0
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(1, hps.n_epochs + 1):
        epoch_loss = 0.0
        train_loss = 0.0
        epoch_start_time = time.time()
        for i, (G, index) in enumerate(train_loader):
            # print(index)
            iter_start_time = time.time()
            model.train()
            if hps.cuda:
                G.to(torch.device("cuda"))

            outputs = model.forward(G)  # [batch_size, 2]
            glist = dgl.unbatch(G)
            label_list = []
            for g in glist:
                enode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"]==2)
                label_list.append(g.nodes[enode_id[0]].data["label"])
            labels = torch.cat(label_list, dim=-1).long()
            
            loss = criterion(outputs, labels).mean()


            if not (np.isfinite(loss.data.cpu())).numpy():
                logger.error("train Loss is not finite. Stopping.")
                logger.info(loss)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        logger.info(name)
                        # logger.info(param.grad.data.sum())
                raise Exception("train Loss is not finite. Stopping.")

            optimizer.zero_grad()
            loss.backward()
            if hps.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hps.max_grad_norm)

            optimizer.step()

            train_loss += float(loss.data)
            epoch_loss += float(loss.data)

            if i % 100 == 0:
                if _DEBUG_FLAG_:
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            logger.debug(name)
                            logger.debug(param.grad.data.sum())
                logger.info('       | end of iter {:3d} | time: {:5.2f}s | train loss {:5.4f} | '
                                .format(i, (time.time() - iter_start_time),float(train_loss / 100)))
                train_loss = 0.0

        if hps.lr_descent:
            new_lr = max(5e-6, hps.lr / (epoch + 1))
            for param_group in list(optimizer.param_groups):
                param_group['lr'] = new_lr
            logger.info("[INFO] The learning rate now is %f", new_lr)

        epoch_avg_loss = epoch_loss / len(train_loader)
        logger.info('   | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.4f} | '
                    .format(epoch, (time.time() - epoch_start_time), float(epoch_avg_loss)))

        p, r, f, auc_score = evaluation(model, valid_loader, hps)
        logger.info("[INFO] DEV result, P:{}, R:{}, F:{}, AUC:{}".format(p, r, f, auc_score))
        
        if auc_score >= best_auc_score:
            save_model(model, os.path.join(train_dir, "earlystop"+str(auc_score)))
            test_p, test_r, test_f, test_auc = evaluation(model, test_loader, hps)
            logger.info("[INFO] TEST result, P:{}, R:{}, F:{}, AUC:{}".format(test_p, test_r, test_f, test_auc))
            best_auc_score = auc_score
            patient = 0
        else:
            patient += 1

        if patient >= 10:
            logger.info("[INFO] Best Dev Result: {}".format(best_auc_score))
            logger.error("[Error] val auc does not descent for ten times. Stopping supervisor...")
            return


def main():
    print('start running')
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

    # Hyperparameters::q
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. [default: 0]')
    parser.add_argument('--cuda', action='store_true', default=True, help='GPU or CPU [default: False]')
    parser.add_argument('--vocab_size', type=int, default=50000,help='Size of vocabulary. [default: 50000]')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs [default: 20]')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
    parser.add_argument('--n_iter', type=int, default=1, help='iteration hop [default: 1]')

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
    parser.add_argument('--sent_max_len', type=int, default=100,help='max length of sentences (max source text sentence tokens)')
    parser.add_argument('--doc_max_timesteps', type=int, default=50,help='max length of documents (max timesteps of documents)')

    # Training
    parser.add_argument('--lr', type=float, default=6e-4, help='learning rate')
    parser.add_argument('--lr_descent', action='store_true', default=False, help='learning rate descent')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='for gradient clipping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='for gradient clipping max gradient normalization')

    parser.add_argument('-m', type=int, default=3, help='decode summary length')
    parser.add_argument('--event_max_len', type=int, default=50)
    parser.add_argument('--crop_max_len', type=int, default=5)
    parser.add_argument('--max_sent_num', type=int, default=50)
    parser.add_argument('--max_event_num', type=int, default=50)
    parser.add_argument('--event2sent', type=bool, default=True)
    parser.add_argument('--event2event', type=bool, default=True)
    parser.add_argument('--sent2sent', type=bool, default=False)
    # 1664 maybe better
    parser.add_argument('--seed', type=int, default=3184)
    parser.add_argument('--set_seed', type=bool, default=False)
    # parser.add_argument('--crop_max_len', type=int, default=20)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_printoptions(threshold=50000)

    if args.set_seed:
        torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)


    # File paths
    DATA_FILE = os.path.join(args.data_dir, "train.label.jsonl")
    VALID_FILE = os.path.join(args.data_dir, "val.label.jsonl")
    TEST_FILE = os.path.join(args.data_dir, "test.label.jsonl")
    VOCAL_FILE = os.path.join(args.cache_dir, "vocab")
    FILTER_WORD = os.path.join(args.cache_dir, "filter_word.txt")
    LOG_PATH = args.log_root

    train_w2s_path = os.path.join(args.cache_dir, "train.w2s.tfidf.jsonl")
    val_w2s_path = os.path.join(args.cache_dir, "val.w2s.tfidf.jsonl")
    test_w2s_path = os.path.join(args.cache_dir, "test.w2s.tfidf.jsonl")

    # train_log setting
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, "train_" + nowTime)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Pytorch %s", torch.__version__)
    logger.info("[INFO] Create Vocab, vocab path is %s", VOCAL_FILE)
    vocab = Vocab(VOCAL_FILE, args.vocab_size)
    embed = torch.nn.Embedding(vocab.size(), args.word_emb_dim, padding_idx=0)
    if args.word_embedding:
        embed_loader = Word_Embedding(args.embedding_path, vocab)
        vectors = embed_loader.load_my_vecs(args.word_emb_dim)
        pretrained_weight = embed_loader.add_unknown_words_by_avg(vectors, args.word_emb_dim)
        embed.weight.data.copy_(torch.Tensor(pretrained_weight))
        embed.weight.requires_grad = args.embed_train

    hps = args
    logger.info(hps)

    model = StockGraph(hps, embed)
    logger.info("[MODEL] Hetergeneous Stock Graph ")
    dataset = ExampleSet(DATA_FILE, vocab, hps.sent_max_len, hps.event_max_len, hps.crop_max_len, hps, train_w2s_path, FILTER_WORD)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=True, num_workers=32, collate_fn=graph_collate_fn)
    del dataset
    valid_dataset = ExampleSet(VALID_FILE, vocab, hps.sent_max_len, hps.event_max_len, hps.crop_max_len, hps, val_w2s_path, FILTER_WORD)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=hps.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=32)
    
    test_dataset = ExampleSet(TEST_FILE, vocab, hps.sent_max_len, hps.event_max_len, hps.crop_max_len, hps, test_w2s_path, FILTER_WORD)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hps.batch_size, shuffle=False, collate_fn=graph_collate_fn, num_workers=32)
    
    

    if args.cuda:
        model.to(torch.device("cuda:0"))
        logger.info("[INFO] Use cuda")

    setup_training(model, train_loader, valid_loader, valid_dataset, hps, test_loader)


if __name__ == '__main__':
    print('sb')
    main()

