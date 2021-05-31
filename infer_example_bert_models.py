# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer
import torch.nn.functional as F

from models.metnet import METNet
from pytorch_pretrained_bert import BertModel
from torch.utils.data import DataLoader, random_split
from data_utils import Tokenizer4Bert
from data_utils import build_tokenizer, build_embedding_matrix, ABSADataset, Tokenizer4Bert
import logging
from sklearn import metrics
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def _evaluate_acc_f1(self,model, data_loader):
    n_correct, n_total = 0, 0
    t_targets_all, t_outputs_all = None, None
    # switch model to evaluation mode
    model.eval()
    with torch.no_grad():
        for t_batch, t_sample_batched in enumerate(data_loader):
            t_inputs = [t_sample_batched[col].to(self.opt.device) for col in ['text_raw_bert_indices', 'aspect_bert_indices', 'aspect_in_text']]
            t_targets = t_sample_batched['polarity'].to(self.opt.device)
            t_outputs = model(t_inputs)

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

if __name__ == '__main__':

    model_classes = {
        'metnet': METNet,
    }
    # set your trained models here
    model_state_dict_paths = {
        'metnet': 'state_dict/metnet_laptop_val_acc0.7837'
    }
    
    class Option(object): pass
    opt = Option()
    opt.model_name = 'metnet'
    opt.model_class = model_classes[opt.model_name]
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.max_seq_len = 80
    opt.pretrained_bert_name='bert-base-uncased'
    opt.polarities_dim = 3
    opt.dropout = 0.5
    opt.bert_dim = 768
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
    
    bert = BertModel.from_pretrained(opt.pretrained_bert_name)
    bert.is_training = False
    for p in bert.parameters():
        p.requires_grad = False
    model = opt.model_class(bert, opt).to(opt.device)
    
    print('loading model {0} ...'.format(opt.model_name))
    model.load_state_dict(torch.load(opt.state_dict_path))
    model.eval()
    torch.autograd.set_grad_enabled(False)



    testset = ABSADataset('./datasets/semeval14/Laptops_Test_Gold.xml.seg', tokenizer)  # 形参：测试集路径；分词器
    test_data_loader = DataLoader(dataset=testset, batch_size=16, shuffle=False)

    n_correct, n_total = 0, 0
    t_targets_all, t_outputs_all = None, None
    # switch model to evaluation mode
    model.eval()
    with torch.no_grad():
        for t_batch, t_sample_batched in enumerate(test_data_loader):
            t_inputs = [t_sample_batched[col].to(opt.device) for col in
                        ['text_raw_bert_indices', 'aspect_bert_indices', 'aspect_in_text']]
            t_targets = t_sample_batched['polarity'].to(opt.device)
            t_outputs = model(t_inputs)
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
    logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(acc, f1))


