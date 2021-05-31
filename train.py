# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pytorch_pretrained_bert import BertModel
from sklearn import metrics

from data_utils import build_tokenizer, build_embedding_matrix, ABSADataset, Tokenizer4Bert
from models import METNet

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        #bert模型的数据处理器对象
        UNCASED = 'bert-base-uncased'  # your path for model and vocab
        VOCAB = 'bert-base-uncased-vocab.txt'
        #tokenizer = Tokenizer4Bert(opt.max_seq_len, os.path.join(UNCASED, VOCAB))
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            #创建bert模型并加载预训练模型;https://blog.csdn.net/wangpeiyi9979/article/details/89191709
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        bert.is_training = False
        for p in bert.parameters():
            p.requires_grad = False

            #这里的opt.model_class就相当于基于Bert的模型的类名称，比如bert_spc等
            # bert_spc的初始化函数需要bert, opt这两个参数。
            # 这句话就相当于创建了一个模型对象，并将其送入GPU。
        self.model = opt.model_class(bert, opt).to(opt.device)
        
        #对数据集中的数据做处理，准备好之后需要用的数据资源。
        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)#形参：训练集路径；分词器
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)#形参：测试集路径；分词器
        #opt.valset_ratio：验证集的占比
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            #len(self.trainset)：训练集中句子的数量
            #计算验证集的大小
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            #对self.trainset进行分割，得到两个子数据集，长度分别为len(self.trainset)-valset_len和valset_len。
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:#如果该参数为0，则测试集充当验证集
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        #遍历所有的参数
        #parameters()函数得到的每一个参数对象都有data和requires_grad属性，代表了具体数据和是否需要梯度。
        #每一个参数对象的data的类型都是tensor类型的。
        #for循环：统计模型需要训练和不需要训练的参数的数量。
        for p in self.model.parameters():
            #torch.prod：将tensor中的各个元素相乘
            #通过维度p.shape计算一组参数p的参数量，存入n_params。
            n_params = torch.prod(torch.tensor(p.shape))
            #requires_grad：参数是否需要梯度，默认为True。
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        #vars(object)返回由object的属性和属性值组成的字典对象
        #vars(self.opt)：self.opt中的参数名和参数值组成的字典对象。
        #arg取参数名
        for arg in vars(self.opt):
            #getattr(self.opt, arg)：返回self.opt的arg属性的属性值。
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
    #参数初始化
    def _reset_params(self):
        #遍历每个模型的各个子模块
        for child in self.model.children():
            #if type(child) != BertModel:  # skip bert params
            #遍历每个子模块的各个参数
            for p in child.parameters():
                #对于需要梯度的参数，进行初始化操作
                if p.requires_grad:

                    if len(p.shape) > 1:#p.shape代表了参数对象中的数据的维度
                        self.opt.initializer(p)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])#sqrt平方根函数
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)#用均匀分布初始化参数对象的值

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            #enumerate()用于将一个可遍历的数据对象组合为一个索引序列，返回数据和数据下标。
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()
                #因此inputs包含sample_batched对应的三部分内容。
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                #outputs即为y^
                outputs = self.model(inputs)
                #targets即为y
                targets = sample_batched['polarity'].to(self.opt.device)
                #logger.info('outputs: {0}'.format(outputs))
                #logger.info('targets: {0}'.format(targets))

                loss = criterion(outputs, targets)
                #logger.info('梯度: {:.4f}'.format(loss))
                #梯度更新参数
                loss.backward()
                optimizer.step()
                #argmax()：返回指定维度最大值下标
                #此处的下标0、1、2分别代表了情感的负向、中性和正向。
                #将预测值和标记值进行比较，统计预测正确的样例数量。
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                #统计进行预测的样例的总数量
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                #监测准确率的变化
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
            #每一轮训练后模型在验证集上的准确率和F1分数
            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                #判断文件是否存在
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                #round()：返回浮点数的四舍五入值。
                path = 'state_dict/{0}_{1}_val_acc{2}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                #state_dict()：返回一个包含了模块当前所有状态的字典。
                #save()：把模型的状态字典保存到磁盘文件path中。
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
        #此时path对应的文件中存储的是，模型在得到最好结果时的状态字典
        return path
    #参数：验证集
    #计算验证集上的准确率和f1分数
    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return acc, f1

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        #filter()：https://www.runoob.com/python3/python3-func-filter.html
        #filter()：两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数，返回 True 或 False，所有返回 True 的元素组成最终的结果。
        #lambda是定义匿名函数的关键字，冒号前面是参数，后面是表达式，表达式计算的结果即为该匿名函数的结果。
        #_params用于存储需要梯度的参数。
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        #？？？L2范数？？？
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        #DataLoader：根据参数要求将数据集封装成batch_size大小的tensor，用于之后的训练
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
        #参数初始化
        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        #将模型的参数状态设置为best_model_path文件中保存的。
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        #计算模型在测试集上的准确率和F1分数
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))


def main():
    # Hyper Parameters参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='metnet', type=str)
    parser.add_argument('--dataset', default='laptop', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adadelta', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=1, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=0.0001, type=float)
    parser.add_argument('--num_epoch', default=80, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=32, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=20, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)#词嵌入的维度
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    #valset_ratio：验证集占初始训练集的比率
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    # semantic-relative-distance, see the paper of LCF-BERT model
    parser.add_argument('--SRD', default=3, type=int, help='set SRD')
    #opt相当于一个参数集
    opt = parser.parse_args()

    if opt.seed is not None:
        #seed()用于改变随机数生成器的种子，相同的种子生成相同的随机数，不同的种子生成不同的随机数
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        #设置生成随机数的种子
        torch.manual_seed(opt.seed)
        #设置opt.seed为当前GPU生成随机数的种子
        torch.cuda.manual_seed(opt.seed)
        #用以保证实验的可重复性代码
        torch.backends.cudnn.deterministic = True
        #https://www.cnblogs.com/hizhaolei/p/10177439.html
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'metnet': METNet,
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }
    #每个模型的输入组成部分
    input_colses = {
        'metnet': ['text_raw_bert_indices', 'aspect_bert_indices', 'aspect_in_text'],
    }
    initializers = {
        #设置均匀分布初始化
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        #设置正态分布初始化
        'xavier_normal_': torch.nn.init.xavier_normal,
        #设置（半）正定矩阵初始化
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        #torch.optim,这是一个实现各种优化算法的包
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    #新创建了三个参数model_class,dataset_file,inputs_cols
    opt.model_class = model_classes[opt.model_name]#根据模型的名字，找到模型函数
    opt.dataset_file = dataset_files[opt.dataset]#存储了dataset领域的一个训练集和一个数据集的路径
    opt.inputs_cols = input_colses[opt.model_name]#输入数据组成部分
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    #？？？
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    #创建日志
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))
    #加载模型，处理数据集，统计参数量
    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
