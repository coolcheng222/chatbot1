import torch 
import torch.nn as nn 
import random
from Voc import SOS_token,PAD_token
class Seq2seq(nn.Module):
    def __init__(self,encoder,decoder,device,losser):
        super(Seq2seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.generator = generator
        self.device = device
        self.losser = losser
    def forward(self,input_variable,lengths,target_variable,mask,max_target_len,teacher_forcing_ratio=1.0):
        print_losses = []
        loss = 0
        batch_size = input_variable.shape[1]
        n_totals = 0
        
        encoder_outputs,encoder_hidden = self.encoder(input_variable,lengths)
        

        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
        decoder_input = decoder_input.to(self.device)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
        for t in range(max_target_len):
            # decoder登场
            decoder_output,decoder_hidden = self.decoder(
                decoder_input,decoder_hidden,encoder_outputs
            )
            # 新的input
            if use_teacher_forcing:
                decoder_input = target_variable[t].view(1,-1)
            else:
                # 取出最大概率的一组
                _,topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                decoder_input = self.decoder.to(self.device)
            # print(decoder_output)
            mask_loss,nTotal = self.losser(decoder_output,target_variable[t],mask[t],self.device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
        # loss.backward()
        self.avg = sum(print_losses) / n_totals
        return loss
    def clip(self,clip):
        _ = nn.utils.clip_grad_norm_(self.encoder.parameters(),clip)
        _ = nn.utils.clip_grad_norm_(self.decoder.parameters(),clip)
    def generate(self,input_seq,input_length,max_length):
        encoder_outputs,encoder_hidden = self.encoder(input_seq,input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = torch.ones(1,1,device=self.device,dtype=torch.long) * SOS_token
        all_tokens = torch.zeros([0],device=self.device,dtype=torch.long)
        all_scores = torch.zeros([0],device=self.device)
        for _ in range(max_length):
            decoder_output,decoder_hidden = self.decoder(decoder_input,decoder_hidden,encoder_outputs)
            decoder_scores,decoder_input = torch.max(decoder_output,dim=1)
            all_tokens = torch.cat((all_tokens,decoder_input),dim=0)
            all_scores = torch.cat((all_scores,decoder_scores))
            decoder_input = torch.unsqueeze(decoder_input,0)
        return all_tokens,all_scores
    