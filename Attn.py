import torch.nn as nn
import torch
import torch.nn.functional as F
# luong
class Attn(nn.Module):
    # method: dot,general,concat
    def __init__(self,method,hidden_size):
        super(Attn,self).__init__()
        self.method = method
        if self.method not in ['dot','general','concat']:
            raise ValueError(self.method," is not method!")
        self.hidden_size = hidden_size
        if self.method == 'general':
            # 内含训练参数W
            self.attn = nn.Linear(self.hidden_size,hidden_size)
        elif self.method == 'concat':
            # 内含训练参数W
            self.attn = nn.Linear(self.hidden_size * 2,hidden_size)
            # 将v加入训练参数
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
    def dot_score(self,hidden,encoder_output):
        # 广播乘法再相加,形成内积
        return torch.sum(hidden * encoder_output,dim=2)
    def general_score(self,hidden,encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy,dim=2)
    def concat_score(self,hidden,encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0),-1,-1),encoder_output,2))).tanh()
        return torch.sum(self.v * energy,dim=2)
    def forward(self,hidden,encoder_output):
        if self.method == 'general':
            attn_energies = self.general_score(hidden,encoder_output)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden,encoder_output)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden,encoder_output)
        attn_energies = attn_energies.t()
        return F.softmax(attn_energies,dim=1).unsqueeze(1)
