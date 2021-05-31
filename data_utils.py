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

#fnames存储了训练和测试数据集的路径
#函数的作用：返回分词器对象
#如果存储该对象的文件存在，即分词器对象已经创建，则读取并返回
#否则分词器对象还未创建，则进行创建并返回
def build_tokenizer(fnames, max_seq_len, dat_fname):
    #判断dat_fname是否存在，dat_fname可以是文件路径；https://blog.csdn.net/u012424313/article/details/82216092
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        #读取文件内容重构为python对象，文件必须以二进制只读形式打开；
        ## https://blog.csdn.net/weixin_38278334/article/details/82967813；https://www.cnblogs.com/lincappu/p/8296078.html
        #load序列化对象
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        #将训练集和数据集中的所有句子存储在变量text中
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            #从fin文件读取所有的内容
            lines = fin.readlines()
            fin.close()
            #range(start,stop,step)，到stop但是不包含stop
            for i in range(0, len(lines), 3):
                #s.lower().strip()，去除字符串头和尾的空格
                #对完整的句子以属性标志为分割符进行分割，三个变量分别存储了属性标志左边的句子、属性标志本身和其右边的句子
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                #text_raw存储了完整的句子
                text_raw = text_left + " " + aspect + " " + text_right
                #数据集中的所有的句子以空格分隔
                text += text_raw + " "
        #创建一个分词器对象
        tokenizer = Tokenizer(max_seq_len)
        #调用函数分词，得到两个词典
        tokenizer.fit_on_text(text)
        #dump反序列化对象，将tokenizer保存至dat_fname
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    #返回分词器对象，其中有两个词典
    return tokenizer

#加载词向量
def _load_word_vec(path, word2idx=None):
    #newline='\n'区分换行符,保留原有的换行
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        #rstrip，删除字符串末尾的指定字符，默认为空格
        tokens = line.rstrip().split()
        #依靠词典或者path指定的文本文件加载词向量
        if word2idx is None or tokens[0] in word2idx.keys():
            #np.asarray和np.array的区别：https://www.jianshu.com/p/a050fecd5a29
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    #返回加载的词向量（字典对象）
    return word_vec
#word2idx是数据集构成的词典
#embed_dim是嵌入矩阵的维度
#dat_fname代表存储词嵌入的文件的名称
def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    #判断存储词嵌入的文件是否存在，若存在则直接读取其中内容并返回；
    #否则，创建词嵌入数组，并将其存储在该文件中，并将创建的词嵌入数组返回
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        #生成维度是len(word2idx) + 2行embed_dim列的数组embedding_matrix。
        #？？为什么要+2？
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        #如果embed_dim不是300，fname即为if语句前面的字符串，否则为后面的字符串
        #？？fname代表了什么？是词向量存储文件？
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        #_load_word_vec的作用是加载词向量，它的形参一个是.txt文件的路径，一个是字典对象
        #{词：词向量}
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            #获取词word的词向量vec，然后将其存储到嵌入矩阵对应的数组embedding_matrix的位置i处。
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        #for循环结束即得到一个词嵌入数组
        #把词嵌入数组存储到dat_fname文件中
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    #返回词嵌入数组
    return embedding_matrix

#sequence的长度不一定就是maxlen
#？？？为什么这样截取和填充？？？
def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    #x初始化为长度是maxlen，值为value，值得类型为dtype的数组
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        #-代表从右侧开始读取
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}#字典对象，键值对<词，编号>
        self.idx2word = {}#字典对象，键值对<编号，词>
        self.idx = 1
    #text是某一数据集中训练集和测试集中所有句子的集合
    #根据数据集创建两个字典self.word2idx和self.idx2word
    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        #句子词汇列表
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
    #text：一段文本
    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        #对于文本当中的每一个词，如果其在词典word2idx中，则该位置为相应的id值，否则为值unknownidx
        #sequence即为text对应的id序列
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            #翻转序列
            sequence = sequence[::-1]
        #返回对sequence进行截取和填充（用0）后的结果
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        #从预训练模型中获得数据处理器；https://blog.csdn.net/weixin_41519463/article/details/100863313
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        #tokenize()函数实现分词
        #？？？先分词然后将其转换为id序列？？？
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            #分别抽取句子的左半句、右半句（以属性标志为分界点）、属性词和情感极性
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            #text_to_sequence()，先将给定的文本转换为id序列，再进行截取和填充。
            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            text_left_indices = tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            #text_left_indices是numpy数组类型
            #所以，np.sum(text_left_indices != 0)的含义为，计算text_left_indices其中非0项的个数
            left_context_len = np.sum(text_left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            #属性在整个句子中的起始下标
            aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
            #原本数据集中-1、0、1表示情感的负向、中性、正向，现将该值+1，作为模型中表示情感极性的值。
            #0、1、2：负向、中性、正向
            polarity = int(polarity) + 1
            #？？？为什么要加上“+ aspect + " [SEP]"”？？？
            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            #+2代表text_bert_indices中两个标志符的位置，aspect_len + 1代表了text_bert_indices中最后一个属性词和标识符的位置
            #两个数组的+代表连接。
            #？？？为什么用0和1就确定了id序列？？？
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

            text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_bert_indices': text_raw_bert_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_in_text': aspect_in_text,
                'polarity': polarity,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
