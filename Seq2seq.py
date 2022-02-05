import torch.nn as nn
from Attn import Attn
import torch.nn.functional as F
import torch
class EncoderRNN(nn.Module):
    def __init__(self,hidden_size,embedding,n_layers=1,dropout=0):
        super(EncoderRNN,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size,hidden_size,n_layers,dropout = (0 if n_layers == 1 else dropout),bidirectional=True)
    def forward(self,input_seq,input_lengths):
        embedded = self.embedding(input_seq)
        # 把padding pack掉,输入三维矩阵和长度
        packed = nn.utils.rnn.pack_padded_sequence(embedded,input_lengths)
        outputs,hidden = self.gru(packed)
        # 只需要处理output,因为output包含时序
        outputs,_ = nn.utils.rnn.pad_packed_sequence(outputs)
        # 将双向相加,内置的处理是dim=2拼接
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        return outputs,hidden

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self,attn_model,embedding,hidden_size,output_size,n_layers=1,dropout=0.1):
        super(LuongAttnDecoderRNN,self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size,hidden_size,n_layers,dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2,hidden_size)
        self.out = nn.Linear(hidden_size,output_size)
        self.attn = Attn(attn_model,hidden_size)
    def forward(self,input_step,last_hidden,encoder_outputs):
        # 先过word embedding和dropout
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # gru过一下
        rnn_output,hidden = self.gru(embedded,last_hidden)
        # 根据输出和encoder算一下attention
        attn_weights = self.attn(rnn_output,encoder_outputs)
        # bmm: batch matmul,批量的矩阵乘法,计算一批context矢量
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))
        # 一个时间过gru,可以挤压
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        # 拼接后进行Linear再tanh,是luong给出的方案(见计算公式)
        concat_input = torch.cat((rnn_output,context),1)
        concat_output = torch.tanh(self.concat(concat_input))
        # 过一下linear
        output = self.out(concat_output)
        # 过一下softmax
        output = F.softmax(output,dim=1)
        return output,hidden