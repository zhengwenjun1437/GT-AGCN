import os
import sys
import copy
import random
import logging
import argparse
from time import strftime, localtime

import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from models.gt_agcn import GT_AGCN
from models.gt_agcn_bert import GT_AGCN_BERT
from models.gt_agcn_wo_gt import GT_AGCN_WO_GT
from models.gt_agcn_wo_agcn import GT_AGCN_WO_AGCN
from models.gt_agcn_wo_t import GT_AGCN_WO_T
from models.gt_agcn_wo_g import GT_AGCN_WO_G
from models.gt_agcn_wo_m import GT_AGCN_WO_M
from models.gt_agcn_w_ddt import GT_AGCN_W_DDT
from models.gt_agcn_w_sd import GT_AGCN_W_SD
from data_utils import SentenceDataset, build_tokenizer, build_embedding_matrix, Tokenizer4BertGCN, ABSAGCNData
from prepare_vocab import VocabHelp

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Instructor:
    """模型训练和评估"""
    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4BertGCN(opt.max_length,
                                          opt.pretrained_bert_name)  # actually tokenizer.max_seq_len == opt.max_length
            bert = BertModel.from_pretrained('bert-base-uncased')
            # bert = BertModel.from_pretrained("./bert_path/models--bert-base-uncased/snapshots/0a6aa9128b6194f4f3c4db429b6cb4891cdb421b") # locally after downloading
            # for name, param in bert.named_parameters():
            # print(name + ": ", param.requires_grad, sep='')
            # exit()
            dep_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep.vocab')  # deprel
            pos_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pos.vocab')  # 加载 pos_vocab
            opt.deprel_size = len(dep_vocab)
            self.model = opt.model_class(bert, opt).to(opt.device)
            trainset = ABSAGCNData(opt.dataset_file['train'], tokenizer, pos_vocab, dep_vocab, opt=opt)
            testset = ABSAGCNData(opt.dataset_file['test'], tokenizer, pos_vocab, dep_vocab, opt=opt)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_length=opt.max_length,
                data_file='{}/{}_tokenizer.dat'.format(opt.vocab_dir, opt.dataset))
            embedding_matrix = build_embedding_matrix(
                vocab=tokenizer.vocab,
                embed_dim=opt.embed_dim,
                data_file='{}/{}d_{}_embedding_matrix.dat'.format(opt.vocab_dir, str(opt.embed_dim), opt.dataset))

            logger.info("Loading vocab...")
            token_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_tok.vocab')  # token
            post_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_post.vocab')  # position
            pos_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pos.vocab')  # POS
            dep_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep.vocab')  # deprel
            pol_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pol.vocab')  # polarity
            logger.info(
                "token_vocab: {}, post_vocab: {}, pos_vocab: {}, dep_vocab: {}, pol_vocab: {}".format(len(token_vocab),
                                                                                                      len(post_vocab),
                                                                                                      len(pos_vocab),
                                                                                                      len(dep_vocab),
                                                                                                      len(pol_vocab)))

            # opt.tok_size = len(token_vocab)
            opt.post_size = len(post_vocab)
            opt.pos_size = len(pos_vocab)
            opt.deprel_size = len(dep_vocab)

            vocab_help = (post_vocab, pos_vocab, dep_vocab, pol_vocab)
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
            trainset = SentenceDataset(opt.dataset_file['train'], tokenizer, opt=opt, vocab_help=vocab_help)
            testset = SentenceDataset(opt.dataset_file['test'], tokenizer, opt=opt, vocab_help=vocab_help)

        self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')

        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)  # xavier_uniform_
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def get_bert_optimizer(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        diff_part = ["bert.embeddings", "bert.encoder"]

        logger.info("layered learning rate on")
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                "weight_decay": self.opt.finetune_weight_decay,
                "lr": self.opt.bert_lr
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": self.opt.bert_lr
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": self.opt.weight_decay,
                "lr": self.opt.learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": self.opt.learning_rate
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon)
        # optimizer = AdamW(optimizer_grouped_parameters)

        return optimizer

    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = ''
        if self.opt.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, int(self.opt.warmup * len(self.train_dataloader)),
                self.opt.num_epoch * len(self.train_dataloader))
        elif self.opt.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(self.opt.warmup * len(self.train_dataloader)),
                self.opt.num_epoch * len(self.train_dataloader))
        elif self.opt.scheduler == 'none':
            scheduler = None
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1

                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('./state_dict'):
                                os.mkdir('./state_dict')
                            model_path = './state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name,
                                                                                          self.opt.dataset, test_acc,
                                                                                          f1)
                            self.best_model = copy.deepcopy(self.model)
                            logger.info('>> saved: {}'.format(model_path))
                    if f1 > max_f1:
                        max_f1 = f1
                    logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc,
                                                                                                 test_acc, f1))
        return max_test_acc, max_f1, model_path

    def _evaluate(self, show_results=False):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None
        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                # outputs, penal = self.model(inputs)
                outputs = self.model(inputs)
                n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test_total += len(outputs)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, test_acc, f1

        return test_acc, f1

    @torch.no_grad()
    def _show_cases(self):  # For case study
        self.model.eval()
        cases_result = open("target_predict.txt", 'w')
        for sample_batched in self.test_dataloader:
            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            targets = sample_batched['polarity'].to(self.opt.device)
            outputs = self.model(inputs)
            predict = torch.argmax(outputs, -1)
            for i in range(targets.size()[0]):
                cases_result.write(str(targets[i].item()))
                cases_result.write(", ")
                cases_result.write(str(predict[i].item()))
                cases_result.write("\n")
        cases_result.close()

    def _test(self):
        self.model = self.best_model
        self.model.eval()
        test_report, test_confusion, acc, f1 = self._evaluate(show_results=True)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion)

    def run(self):
        label_weights = torch.tensor([1, 1, 1.], device=self.opt.device)

        if self.opt.balance_loss:
            if self.opt.dataset == 'restaurant':
                label_weights = torch.tensor([1 / 2164, 1 / 807, 1 / 637], device=self.opt.device)
            elif self.opt.dataset == 'laptop':
                label_weights = torch.tensor([1 / 976, 1 / 851, 1 / 455], device=self.opt.device)
            elif self.opt.dataset == 'twitter':
                label_weights = torch.tensor([1 / 1507, 1 / 1528, 1 / 3016], device=self.opt.device)
            elif self.opt.dataset == 'rest16':
                label_weights = torch.tensor([1 / 1240, 1 / 439, 1 / 69], device=self.opt.device)

        criterion = nn.CrossEntropyLoss(weight=label_weights)
        if 'bert' not in self.opt.model_name:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.learning_rate
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                    "lr": self.opt.learning_rate
                }
            ]
            optimizer = self.opt.optimizer(optimizer_grouped_parameters)

            # _params = filter(lambda p: p.requires_grad, self.model.parameters())
            # optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        else:
            optimizer = self.get_bert_optimizer(self.model)
        max_test_acc_overall = 0
        max_f1_overall = 0
        # if 'bert' not in self.opt.model_name: # use default initilization of every module
        #     self._reset_params()
        max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall)
        logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)
        torch.save(self.best_model.state_dict(), model_path)
        logger.info('>> saved: {}'.format(model_path))
        logger.info('#' * 60)
        logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall))
        logger.info('max_f1_overall:{}'.format(max_f1_overall))
        self._test()


def main():
    model_classes = {
        'gt-agcn': GT_AGCN,
        'gt-agcn-wo-gt': GT_AGCN_WO_GT,
        'gt-agcn-wo-agcn': GT_AGCN_WO_AGCN,
        'gt-agcn-wo-g': GT_AGCN_WO_G,
        'gt-agcn-wo-t': GT_AGCN_WO_T,
        'gt-agcn-wo-m': GT_AGCN_WO_M,
        'gt-agcn-bert': GT_AGCN_BERT,
        'gt-agcn-w-ddt': GT_AGCN_W_DDT,
        'gt-agcn-w-sd': GT_AGCN_W_SD,
    }  # 模型类别

    vocab_dirs = {
        'restaurant': './dataset/Restaurants_corenlp',
        'laptop': './dataset/Laptops_corenlp',
        'twitter': './dataset/Tweets_corenlp',
    }  # 数据词表位置

    dataset_files = {
        'restaurant': {
            'train': './dataset/Restaurants_corenlp/train.json',
            'test': './dataset/Restaurants_corenlp/test.json',
        },
        'laptop': {
            'train': './dataset/Laptops_corenlp/train.json',
            'test': './dataset/Laptops_corenlp/test.json',
        },
        'twitter': {
            'train': './dataset/Tweets_corenlp/train.json',
            'test': './dataset/Tweets_corenlp/test.json',
        }
    }  # 数据集位置

    input_colses = {
        'non-bert': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'adj', 'adj_2', 'adj_3', 'adj_dis'],
        'bert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'adj_matrix',
                 'src_mask', 'aspect_mask', 'adj_dis'],
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }  # 参数初始化方式

    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }  # 优化器选择

    # 超参数配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gt-agcn', type=str, choices=list(model_classes.keys()))
    parser.add_argument('--dataset', default='restaurant', type=str, choices=list(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, choices=list(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, choices=list(initializers.keys()))
    parser.add_argument('--learning_rate', default='0.001', type=float)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--post_dim', default=60, type=int, help='Position embedding dimension.')
    parser.add_argument('--pos_dim', default=30, type=int, help='Part-of-speech embedding dimension.')
    parser.add_argument('--deprel_dim', default=30, type=int, help='Dependent relation embedding dimension.')
    parser.add_argument('--hidden_dim', default=60, type=int, help='隐藏层维度')
    parser.add_argument('--input_dropout', default=0.7, type=float, help='Input dropout rate.')
    parser.add_argument('--lower', default=True, help='Lowercase all words.')

    # 图神经网络常规设置
    parser.add_argument('--directed', default=False, help='Directed graph or undirected graph.')
    parser.add_argument('--add_self_loop', default=True, help='Graph self loop.')

    # RNN模型常规设置
    parser.add_argument('--use_rnn', action='store_true')   # 如果没有指定--use_rnn，默认为False
    parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
    parser.add_argument('--rnn_hidden', default=60, type=int, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', default=0.1, type=float, help='RNN dropout rate.')

    parser.add_argument('--max_length', default=85, type=int)
    parser.add_argument('--device', default=None, type=str, help='CPU or CUDA.')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='权重衰减')
    parser.add_argument('--vocab_dir', default='./dataset/Restaurants_corenlp', type=str, help='词表的位置')
    parser.add_argument('--pad_id', default=0, type=int, help='填充标记')

    # GT-AGCN模型超参数
    parser.add_argument('--graph_conv_type', default='gin', type=str, choices=['gat-mod', 'gcn', 'gin', 'gan'])
    parser.add_argument('--graph_conv_attention_heads', default=4, type=int)
    parser.add_argument('--graph_conv_attn_dropout', default=0.0, type=float)
    parser.add_argument('--attention_heads', default=4, type=int)
    parser.add_argument('--attn_dropout', default=0.1, type=float)
    parser.add_argument('--ffn_dropout', default=0.3, type=float)
    parser.add_argument('--norm', default='ln', type=str, choices=['ln', 'bn'], help='规范化方式')
    parser.add_argument('--max_position', default=9, type=int)
    parser.add_argument('--alpha', default=0.9, type=float, help='距离权重')
    parser.add_argument('--sd', default=2, type=int, help='sd')
    parser.add_argument('--num_layers', default=8, type=int, help='G-Transformer模块的层数')
    parser.add_argument('--gcn_layers', default=4, type=int, help='A-GCN模块的层数')

    parser.add_argument('--scheduler', default='none', type=str, choices=['linear', 'cosine', 'none'])
    parser.add_argument('--warmup', default=2, type=float)
    parser.add_argument('--balance_loss', action='store_true')
    parser.add_argument('--cuda', default='0', type=str)

    # BERT
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Epsilon for Adam optimizer.')
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--bert_dropout', default=0.5, type=float)
    parser.add_argument('--bert_lr', default=2e-5, type=float)
    parser.add_argument('--finetune_weight_decay', default=0.01, type=float)
    opt = parser.parse_args()

    opt.model_class = model_classes[opt.model_name]         # 选择模型
    opt.dataset_file = dataset_files[opt.dataset]           # 选择数据
    opt.initializer = initializers[opt.initializer]         # 选择参数初始化方式
    opt.optimizer = optimizers[opt.optimizer]               # 选择优化器
    opt.vocab_dir = vocab_dirs[opt.dataset]                 # 选择词表

    if 'bert' in opt.model_name:
        opt.inputs_cols = input_colses['bert']
        opt.hidden_dim = 768
        opt.max_length = 100
        opt.num_epoch = 10
        opt.attention_heads = 6
    else:
        opt.inputs_cols = input_colses['non-bert']
        opt.max_length = 85
        opt.num_epoch = 50

    opt.device = torch.device('cuda:' + opt.cuda)
    setup_seed(opt.seed)

    if not os.path.exists('./logging'):
        os.makedirs('./logging', mode=0o777)
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H-%M-%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./logging', log_file)))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()






