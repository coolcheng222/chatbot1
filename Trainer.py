import torch
import random
from Voc import SOS_token,PAD_token
class SeqTrainer:
    def __init__(self,model,encoder_optimizer,decoder_optimizer,device):
        
        # self.embedding = embedding
        # self.encoder = encoder
        # self.decoder = decoder
        self.model = model
        self.device = device
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
    def train(self,input_variable,lengths,target_variable,mask,max_target_len,clip=50):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_variable = input_variable.to(self.device)
        target_variable = target_variable.to(self.device)
        mask = mask.to(self.device)
        # packing的length参数放cpu
        lengths = lengths.to("cpu")

        loss = 0
        print_losses = []
        # n_totals = 0

        loss,print_losses,n_totals = self.model(input_variable,lengths,target_variable,mask,max_target_len)
        loss.backward()
        self.model.clip(clip)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        # print(list(encoder.parameters()))
        return sum(print_losses) / n_totals