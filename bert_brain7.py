# encoding=utf8
'''
bert模型超参数设置
'''
maxlen = 286
predictsize = 100 #预测集抽取的样本数
epochsize = 400
batch_size = 100
max_pred = 30  # max tokens of prediction
n_layers = 3
n_heads = 6
d_model = 284
d_ff = d_model * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2

'''
查看和显示nii文件
'''

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
img = nib.load('data/tfMRI_MOTOR_LR.nii.gz')
#OrthoSlicer3D(img.dataobj).show()

#这是把四维nii数据转成二维数据的代码
import nilearn
from nilearn.input_data import NiftiMasker
import numpy as np
masker = NiftiMasker()
inputdata2d = masker.fit_transform(img)  #要转了(316,34059)
print(inputdata2d.shape)
#img3 = np.multiply(inputdata2d,img2)
#components_img = masker.inverse_transform(inputdata2d)

'''
将input2d数据进行预处理，归一化之后保留3位有效数字，变量为normalinput2d
'''
import nibabel as nib
import numpy as np
import pandas as pd
data = inputdata2d.T
def normalization(data):
    range = np.max(data) - np.min(data)
    return (data - np.min(data)) / range
data1=normalization(data)
normalinput2d = np.round(data1, 3)
np.save("data/normalinput2d.npy", normalinput2d)
#第二种保留三位有效数字的方法
# data3 = np.zeros((33714,316))
# import numpy as np
# for i in range(data1.shape[0]):
#     for j in range(data2.shape[1]):
#         data3[i,j] =('{:.3f}'.format(data1[i,j]))
# print(data3)
'''
将归一化后的数据定义进行放入bert模型前的预处理
'''

import re
import math

import torch
import datetime
import numpy as np
from random import *
from random import shuffle
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
start = datetime.datetime.now()
#normalinput2d = np.load('brain_motor/data/normalinput2d.npy')
three_list = []
list_test1 = normalinput2d.tolist()
sentences = list_test1
print(len(list_test1))
list_test2 = normalinput2d.reshape(284*34059).tolist()
word_list = list(set(list_test2))

word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
    word2idx[w] = i + 4

idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx)

token_list = list()
for sentence in range(normalinput2d.shape[0]):
    arr = [word2idx[s] for s in normalinput2d[sentence, :]]
    token_list.append(arr)


'''
训练集数据制作
'''
def make_data(start):
    batch = []
    #tokens_a_index = randrange(len(sentences)) # sample random index in sentences
    tokens_a_index = start # sample random index in sentences
    for j in range(10):
        tokens_a = token_list[tokens_a_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1)

        # MASK LM
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  ## 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]  # candidate masked position
        cand_maked_pos.remove(1)
        cand_maked_pos.remove(284)
        shuffle(cand_maked_pos)#随机打算mask的程序

        #mask_start = randint(21, 610)

        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word2idx['[MASK]']  # make mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1)  # random index in vocabulary
                while index < 4:  # can't involve 'CLS', 'SEP', 'PAD'
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index  # replace

        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        batch.append([input_ids, segment_ids, masked_tokens, masked_pos])

    return batch
# Proprecessing Finished
batch = []
for m in range(3000):
    batch.extend(make_data(m*10))

input_ids, segment_ids, masked_tokens, masked_pos = zip(*batch)

input_ids, segment_ids, masked_tokens, masked_pos = torch.LongTensor(input_ids), torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens), torch.LongTensor(masked_pos)


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx]


loader = Data.DataLoader(MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos), batch_size)

'''
预测集数据制作
'''
def make_predict(start):
    batch_predict = []
    positive = negative = 0
    tokens_a_index = start  # sample random index in sentences

    tokens_a = token_list[tokens_a_index]
    input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']]
    segment_ids = [0] * (1 + len(tokens_a) + 1)

    # MASK LM
    n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  ## 15 % of tokens in one sentence
    cand_maked_pos = [i for i, token in enumerate(input_ids)
                      if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]  # candidate masked position
    #print(cand_maked_pos)
    mask_start = 40
    del cand_maked_pos[39:59]
    remove_mask1 = []
    remove_mask2 = []
    for re in range(20):
        remove_mask1.append(mask_start+re)
        remove_mask2.append(mask_start + 20 + re * 8)
        cand_maked_pos.remove(mask_start + 20 + re * 8)

    shuffle(remove_mask2)
    cand_maked_pos.remove(1)
    cand_maked_pos.remove(2)
    cand_maked_pos.remove(3)
    cand_maked_pos.remove(284)
    #shuffle(cand_maked_pos)#随机打算mask的程序
    cand_maked_pos = remove_mask1+remove_mask2+cand_maked_pos

    #mask_start = randint(21, 260)

    masked_tokens, masked_pos = [], []
    for pos in cand_maked_pos[:n_pred]:
        masked_pos.append(pos)
        masked_tokens.append(input_ids[pos])
        if random() < 0.8:  # 80%
            input_ids[pos] = word2idx['[MASK]']  # make mask
        elif random() > 0.9:  # 10%
            index = randint(0, vocab_size - 1)  # random index in vocabulary
            while index < 4:  # can't involve 'CLS', 'SEP', 'PAD'
                index = randint(0, vocab_size - 1)
            input_ids[pos] = index  # replace

    # Zero Paddings
    n_pad = maxlen - len(input_ids)
    input_ids.extend([0] * n_pad)
    segment_ids.extend([0] * n_pad)

    # Zero Padding (100% - 15%) tokens
    if max_pred > n_pred:
        n_pad = max_pred - n_pred
        masked_tokens.extend([0] * n_pad)
        masked_pos.extend([0] * n_pad)

    batch_predict.append([input_ids, segment_ids, masked_tokens, masked_pos])

    return batch_predict

batch_predict = []
for n in range(300):
    batch_predict.extend(make_predict(n*100+5))

'''
Bert模型搭建

'''

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]


def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        pos  = pos.to(device)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        embedding =embedding.to(device)

        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        return context,attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.LayerNorm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context,attn= ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_heads * d_v)  # context: [batch_size, seq_len, n_heads, d_v]
        output = self.linear(context)
        out = self.LayerNorm(output + residual)

        #return nn.LayerNorm(d_model)(output + residual)  # output: [batch_size, seq_len, d_model]
        return out,attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        a = self.fc1(x)
        b = gelu(a)
        c = self.fc2(b)
        # return self.fc2(gelu(self.fc1(x)))
        return c



class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs,attn= self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                         enc_self_attn_mask)  # enc_inputs to same Q,K,V
        attention_outputs = enc_outputs
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]

        return enc_outputs, attention_outputs,attn


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, attention_output,enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        # it will be decided by first token(CLS)
        h_pooled = self.activ1(self.fc(output[:, 0]))  # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2]

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))  # [batch_size, max_pred, d_model]
        # get masked position from final output of transformer.
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]\

        h_masked = self.norm(self.activ2(self.linear(h_masked)))

        logits_lm = self.decoder(h_masked) + self.decoder_bias  # [batch_size, max_pred, n_vocab]
        return logits_lm, logits_clsf, h_pooled, output, attention_output,enc_self_attn


model = BERT().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adamax(model.parameters(), lr=0.0002)
'''
 开始训练数据
'''
from tqdm import tqdm
for epoch in range(epochsize):
    for input_ids, segment_ids, masked_tokens, masked_pos in loader:
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        masked_tokens = masked_tokens.to(device)
        masked_pos = masked_pos.to(device)

        logits_lm, logits_clsf, h_pooled, output, attention_output,enc_self_attn = model(input_ids, segment_ids, masked_pos)

        loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1))  # for masked LM
        loss_lm = (loss_lm.float()).mean()
        loss = loss_lm
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


end = datetime.datetime.now()
print("训练时间为")
print(end-start)
# Predict mask tokens
'''
 测试集预测
'''
#自定义测试数据集
import matplotlib.pyplot as plt
import numpy as np
loss_plot = []
value_plot = []

for r in range(predictsize):
    input_ids1, segment_ids1, masked_tokens1, masked_pos1= batch_predict[r]

    masked_word2 = []
    token_list_use = [word2idx['[CLS]']] + token_list[5+r*100] + [word2idx['[SEP]']]
    print('==============进行预测的原始段落为==================')
    print('================================')
    print([idx2word[w] for w in input_ids1 if idx2word[w] != '[PAD]'])
    print(input_ids1)

    logits_lm1, logits_clsf1, h_pooled1, output1, attention_output1,enc_self_attn1 = model(torch.LongTensor([input_ids1]).to(device),
                                                                       torch.LongTensor([segment_ids1]).to(device),
                                                                       torch.LongTensor([masked_pos1]).to(device))

    logits_lm1 = logits_lm1.cpu().data.max(2)[1][0].data.numpy()
    print('masked tokens list : ', [pos for pos in masked_tokens1 if pos != 0])
    print('predict masked tokens list : ', [pos for pos in logits_lm1 if pos != 0])
    predict_tokens1 = logits_lm1.tolist()
    print('================================')

    x = np.linspace(0, 286,286)  # 设置横轴的取值点
    y1= input_ids1
    word1 = []
    masked_word1 = []
    predict_maskword1 = []
    i = 0
    j = 0
    k = 0
    for i in range(286):
        data1 = idx2word[input_ids1[i]]
        word1.append(data1)
    for j in range(max_pred):
        word1[masked_pos1[j]] = idx2word[predict_tokens1[j]]
    for k in range(max_pred):
        masked_word1.append(idx2word[masked_tokens1[k]])
        predict_maskword1.append(idx2word[predict_tokens1[k]])

    print('================================')
    # print(masked_word1)
    # print(predict_maskword1)
    loss_p = 0.0
    for l in range(max_pred):
        loss_p = abs(masked_word1[l]-predict_maskword1[l])+loss_p
    loss_p = loss_p/max_pred
    print(loss_p)
    loss_plot.append(loss_p)

    plt.plot(masked_word1)
    plt.plot(predict_maskword1, color='red', linewidth=1, linestyle='--')
    plt.savefig('result/' + 'plot' + str(r) + '.jpg')
    plt.cla()
    #plt.show()
loss_average = sum(loss_plot)/len(loss_plot)
print('所有预测集平均误差为')
print(loss_average)
plt.plot(loss_plot)
plt.savefig('result/' + 'loss_plot' + '.jpg')
#plt.show()
'''
 预测完成后对前6000个点输出attentionmap值
'''
input_ids1, segment_ids1, masked_tokens1, masked_pos1= make_predict(0)[0]

logits_lm1, logits_clsf1, h_pooled1, output1, attention_output1,enc_self_attn1= model(torch.LongTensor([input_ids1]).to(device),
                                                                       torch.LongTensor([segment_ids1]).to(device),
                                                                       torch.LongTensor([masked_pos1]).to(device))
print(h_pooled1.shape)
print(output1.shape)
print(attention_output1.shape)
h_pooledmap11 = h_pooled1.cpu().data.numpy()
output1map11 = output1.cpu().data.numpy()
attention_outputmap11 = attention_output1.cpu().data.numpy()
enc_self_attn11 = enc_self_attn1.cpu().data.numpy()
from tqdm import tqdm
for j in range(68):
    del input_ids1, segment_ids1, masked_tokens1, masked_pos1
    del logits_lm1, logits_clsf1, h_pooled1, output1, attention_output1, enc_self_attn1
    input_ids1, segment_ids1, masked_tokens1, masked_pos1 = make_predict(j*500)[0]

    logits_lm1, logits_clsf1, h_pooled1, output1, attention_output1, enc_self_attn1 = model(
        torch.LongTensor([input_ids1]).to(device),
        torch.LongTensor([segment_ids1]).to(device),
        torch.LongTensor([masked_pos1]).to(device))
    #print(h_pooled1.shape)
    #print(output1.shape)
    #print(attention_output1.shape)
    h_pooledmap11 = h_pooled1.cpu().data.numpy()
    output1map11 = output1.cpu().data.numpy()
    attention_outputmap11 = attention_output1.cpu().data.numpy()
    enc_self_attn11 = enc_self_attn1.cpu().data.numpy()
    for a_map01 in tqdm(range(499)):
        input_ids1, segment_ids1, masked_tokens1, masked_pos1= make_predict(a_map01+j*500+1)[0]

        logits_lm1, logits_clsf1, h_pooled1, output1, attention_output1,enc_self_attn1= model(torch.LongTensor([input_ids1]).to(device),
                                                                           torch.LongTensor([segment_ids1]).to(device),
                                                                           torch.LongTensor([masked_pos1]).to(device))
        h_pooled01 = h_pooled1.cpu().data.numpy()
        output101 = output1.cpu().data.numpy()
        attention_output01 = attention_output1.cpu().data.numpy()
        enc_self_attn01 = enc_self_attn1.cpu().data.numpy()


        h_pooledmap11= np.concatenate((h_pooledmap11,h_pooled01),axis = 0)
        output1map11= np.concatenate((output1map11,output101),axis = 0)
        attention_outputmap11= np.concatenate((attention_outputmap11,attention_output01),axis = 0)
        enc_self_attn11= np.concatenate((enc_self_attn11,enc_self_attn01),axis = 0)
    np.save('result/h_pooledmap' + str(j) +'.npy',h_pooledmap11)
    np.save('result/output1map' + str(j) +'.npy',output1map11)
    np.save('result/attention_outputmap' + str(j) +'.npy',attention_outputmap11)
    np.save('result/enc_self_attn' + str(j) +'.npy',enc_self_attn11)

    print(h_pooledmap11.shape)
    print(output1map11.shape)
    print(attention_outputmap11.shape)
    print(enc_self_attn11.shape)

del input_ids1, segment_ids1, masked_tokens1, masked_pos1
del logits_lm1, logits_clsf1, h_pooled1, output1, attention_output1, enc_self_attn1
input_ids1, segment_ids1, masked_tokens1, masked_pos1= make_predict(34000)[0]

logits_lm1, logits_clsf1, h_pooled1, output1, attention_output1,enc_self_attn1= model(torch.LongTensor([input_ids1]).to(device),
                                                                       torch.LongTensor([segment_ids1]).to(device),
                                                                       torch.LongTensor([masked_pos1]).to(device))
print(h_pooled1.shape)
print(output1.shape)
print(attention_output1.shape)
h_pooledmap11 = h_pooled1.cpu().data.numpy()
output1map11 = output1.cpu().data.numpy()
attention_outputmap11 = attention_output1.cpu().data.numpy()
enc_self_attn11 = enc_self_attn1.cpu().data.numpy()
for a_map01 in tqdm(range(58)):
    input_ids1, segment_ids1, masked_tokens1, masked_pos1= make_predict(a_map01+34001)[0]

    logits_lm1, logits_clsf1, h_pooled1, output1, attention_output1,enc_self_attn1= model(torch.LongTensor([input_ids1]).to(device),
                                                                       torch.LongTensor([segment_ids1]).to(device),
                                                                       torch.LongTensor([masked_pos1]).to(device))
    h_pooled01 = h_pooled1.cpu().data.numpy()
    output101 = output1.cpu().data.numpy()
    attention_output01 = attention_output1.cpu().data.numpy()
    enc_self_attn01 = enc_self_attn1.cpu().data.numpy()


    h_pooledmap11= np.concatenate((h_pooledmap11,h_pooled01),axis = 0)
    output1map11= np.concatenate((output1map11,output101),axis = 0)
    attention_outputmap11= np.concatenate((attention_outputmap11,attention_output01),axis = 0)
    enc_self_attn11= np.concatenate((enc_self_attn11,enc_self_attn01),axis = 0)
np.save('result/h_pooledmap68.npy',h_pooledmap11)
np.save('result/output1map68.npy',output1map11)
np.save('result/attention_outputmap68.npy',attention_outputmap11)
np.save('result/enc_self_attn68.npy',enc_self_attn11)

print(h_pooledmap11.shape)
print(output1map11.shape)
print(attention_outputmap11.shape)
print(enc_self_attn11.shape)



