import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy
from layers.dynamic_rnn import DynamicLSTM
from layers.squeeze_embedding import SqueezeEmbedding

import datetime
'''
import logging
import sys
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
'''
class Absolute_Position_Embedding(nn.Module):
    def __init__(self, opt, size=None, mode='sum'):
        self.opt = opt
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Absolute_Position_Embedding, self).__init__()

    def forward(self, x, pos_inx):
        #x的维度应该是三维的（batch_size，seq_len，embedding_size）
        if (self.size is None) or (self.mode == 'sum'):
            #self.size有什么用处？？
            self.size = int(x.size(-1))
        batch_size, seq_len = x.size()[0], x.size()[1]
        #三个参数的含义分别为属性在整个句子中的起始下标，batch的大小，句子的长度
        weight = self.weight_matrix(pos_inx, batch_size, seq_len).to(self.opt.device)
        #相对位置值和隐状态向量的乘积
        #unsqueeze：weight原本是二维的，在最后添加一个维度，变成三维，因为x是三维的。
        #学习网址：https://blog.csdn.net/flysky_jay/article/details/81607289
        x = weight.unsqueeze(2) * x
        return x



    def weight_matrix(self, pos_inx, batch_size, seq_len):
        pos_inx = pos_inx.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        #一次训练数据的大小是batch，所以用batch_size控制，以遍历当前的一组句子
        for i in range(batch_size):
            for j in range(0,seq_len):
                if j < pos_inx[i][1]:
                    relative_pos = pos_inx[i][1] - j
                    weight[i].append(1 - relative_pos / 40)
                else:
                    relative_pos = j - pos_inx[i][0]
                    weight[i].append(1 - relative_pos / 40)
            #句子中从第一个词到属性的最后一个词，计算相对位置值
            '''for j in range(pos_inx[i][1]):
                relative_pos = pos_inx[i][1] - j
                weight[i].append(1 - relative_pos / 40)
            #计算属性词之后的词的相对位置值
            for j in range(pos_inx[i][1], seq_len):
                relative_pos = j - pos_inx[i][0]
                weight[i].append(1 - relative_pos / 40)'''
        weight = torch.tensor(weight)
        return weight

class METNet(nn.Module):
    #def __init__(self, embedding_matrix, opt):
    def __init__(self, bert, opt):
        #super()用于解决继承的问题。
        super(METNet, self).__init__()
        print("this is METNet model")
        #加载预训练的词向量
        #self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.bert = bert
        self.position = Absolute_Position_Embedding(opt)
        self.opt = opt
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)
        #D = opt.embed_dim  # 模型词向量维度
        C = opt.polarities_dim  # 分类数目，正向、负向、中性
        #L = opt.max_seq_len
        #HD = opt.hidden_dim
        #定义LSTM单元
        #batch_first：输入输出tensor的第一维是否为 batch_size，默认值 False(seq,batch,feature)。
        self.lstm1 = DynamicLSTM(opt.bert_dim, 384, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = DynamicLSTM(opt.bert_dim, 384, num_layers=1, batch_first=True, bidirectional=True)
        #nn.Conv1d：https://blog.csdn.net/qq_36323559/article/details/102937606
        #nn.Conv1d(词嵌入向量的维度，卷积核的数量，卷积核的大小)
        self.convs1 = nn.Conv1d(opt.bert_dim, 50, 3, padding=1)
        self.convs2 = nn.Conv1d(opt.bert_dim, 50, 3, padding=1)
        #self.convs3 = nn.Conv1d(opt.bert_dim, 50, 3, padding=1)
        self.bert_linear1 = nn.Linear(4 * opt.bert_dim, opt.bert_dim)
        self.bert_linear2 = nn.Linear(4 * opt.bert_dim, opt.bert_dim)
        self.fc1 = nn.Linear(2 * opt.bert_dim, opt.bert_dim)
        self.fc2 = nn.Linear(opt.bert_dim, 50)
        self.fc = nn.Linear(50, C)

        self.gru = DynamicLSTM(opt.bert_dim, 384, num_layers=1, batch_first=True, bidirectional=True, rnn_type = 'GRU')

    def forward(self, inputs):
        text_raw_indices, aspect_indices, aspect_in_text = inputs[0], inputs[1], inputs[2]
        #logger.info('>>> text_raw_indices: {0}'.format(text_raw_indices.shape))
        #torch.sum(text_raw_indices != 0, dim=-1)结果是一个tensor，内容是text_raw_indices每一行非0元素的长度。
        feature_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        #得到text_raw_indices的词嵌入表示
        #feature = self.embed(text_raw_indices)
        #aspect = self.embed(aspect_indices)
        #返回lstm的输出，h，c状态向量？？
        #v：句子经过LSTM的结果
        #v, (_, _) = self.lstm1(feature, feature_len)
        #e：属性经过LSTM的结果
        #e, (_, _) = self.lstm2(aspect, aspect_len)


        v = self.squeeze_embedding(text_raw_indices, feature_len)
        v, _ = self.bert(v, output_all_encoded_layers=False)
        #print("bert:")
        #print(v.shape)
        #print(feature_len)
        #v = torch.cat((v[-1],v[-2],v[-3],v[-4]),dim=2)
        #v = self.bert_linear1(v)
        v = self.dropout(v)

        e = self.squeeze_embedding(aspect_indices, aspect_len)
        e, _ = self.bert(e, output_all_encoded_layers=False)
        #e = torch.cat((e[-1],e[-2],e[-3],e[-4]),dim=2)
        #e = self.bert_linear2(e)
        e = self.dropout(e)

        #logger.info('>>> v: {0}'.format(v))
        #logger.info('>>> e: {0}'.format(e))

        v, (_, _) = self.lstm1(v, feature_len)
        #print("lstm")
        #print(v.shape)

        e, (_, _) = self.lstm2(e, aspect_len)
        #print(e.shape)

        #对属性隐状态向量e进行平均池化
        e_pool = torch.sum(e, dim=1)
        e_len = aspect_len.float()
        e_pool = torch.div(e_pool.float(), e_len.view(e_len.size(0), 1))
        e = e_pool
        #print(e.shape)
        #logger.info('>>> e: {0}'.format(e_pool))
        v = v.transpose(1, 2)

        # range(2)：代表了2层CPT层
        for i in range(4):
            if i is not 0:
                v, (_, _) = self.lstm1(v.transpose(1, 2), feature_len)
                #v, (_, _) = self.gru(v.transpose(1, 2), feature_len)
                # 对经过GRU的上下文隐状态向量v进行平均池化
                v_pool = torch.sum(v, dim=1)
                v_len = feature_len.float()
                v_pool = torch.div(v_pool.float(), v_len.view(v_len.size(0), 1))
                # 用上下文向量更新属性向量
                e = e + v_pool
                v = v.transpose(1, 2)
            else:
                # 对经过双向LSTM的上下文隐状态向量v进行平均池化
                v_pool = torch.sum(v.transpose(1, 2), dim=1)
                v_len = feature_len.float()
                v_pool = torch.div(v_pool.float(), v_len.view(v_len.size(0), 1))
                # 用上下文向量更新属性向量
                #e = e + v_pool.transpose(1,2)
                e = e + v_pool

            '''
            #a：计算了属性向量和隐状态的相关性，即分配给属性中每个词的权重。
            #torch.bmm矩阵乘法；bmm：三维张量乘；mm二维张量乘
            a = torch.bmm(e.transpose(1, 2), v)
            #softmax(,1)按行softmax，行和为1
            a = F.softmax(a, 1)  # (aspect_len,context_len)
            #aspect_mid：属性的权重和向量，即论文中的r。
            aspect_mid = torch.bmm(e, a)
            '''
            # 此时属性向量e的维度是[64,2*HD]，要把维度变成[64,v.size(2),2*HD]，以便CPT模块中同上下文向量做连接
            '''
            e_enlarge = []
            for i in e:
                e_temp = [list(i) for j in range(v.size(2))]
                e_enlarge.append(e_temp)
            e_enlarge = torch.tensor(e_enlarge)
            '''
            #print("e:")
            #print(e.shape)
            e_enlarge = e.unsqueeze(1)
            e_enlarge = e_enlarge.repeat(1, v.size(2), 1)
            #print("e_enlarge")
            #print(e_enlarge.shape)
            #print("v")
            #print(v.shape)
            # cat(,dim=1)按列拼接
            aspect_mid = torch.cat((e_enlarge.transpose(1, 2), v), dim=1)
            #print("aspect_mid")
            #print(aspect_mid.shape)
            # 该行代码对应论文中的公式（5）
            aspect_mid = F.relu(self.fc1(aspect_mid.transpose(1, 2)).transpose(1, 2))
            #print("aspect_mid2")
            #print(aspect_mid.shape)
            # 该行代码对应CPT模块中的LF机制
            v = aspect_mid + v
            #print("v2")
            #print(v.shape)
            # 实现AS机制
            # as_score = F.sigmoid(v.float())
            # v = as_score.mul(aspect_mid) + (1-as_score).mul(v)
            # 此时的v是融合了词的相对位置值的v

        v = self.position(v.transpose(1, 2), aspect_in_text).transpose(1, 2)
        v = v.type(torch.cuda.FloatTensor)
        e_enlarge = self.fc2(e_enlarge)
        v1 = self.convs1(v)
        v2 = self.convs2(v)
        v2 = v2 + e_enlarge.transpose(1, 2)
        # z = F.relu(self.convs3(v))  # [(N,Co,L), ...]*len(Ks)
        z1 = F.tanh(v1)#si

        z2 = F.relu(v2)#ai

        z = z1.mul(z2)

        # ？？？z.size(2)？？？squeeze(2)？？？，z：2维
        z = F.max_pool1d(z, z.size(2)).squeeze(2)

        out = self.fc(z)
        # 在论文中模型的最后一层是softmax()，这体现在计算loss的过程中。
        return out
