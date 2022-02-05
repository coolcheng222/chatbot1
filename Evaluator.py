import torch

class InputBotEvaluator:
    def __init__(self,model,voc,device,max_length):
        self.model = model
        self.device = device
        self.voc = voc
        self.max_length = max_length
    def evaluate(self,sentence):
        self.model.eval()
        indexes_batch = [self.voc.indexesFromSentenceLeft(sentence)]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        input_batch = torch.LongTensor(indexes_batch).transpose(0,1)
        input_batch = input_batch.to(self.device)
        lengths = lengths.to("cpu")
        tokens,scores = self.model.generate(input_batch,lengths,self.max_length)
        # print(tokens)
        decoded_words = [self.voc.index2wordF(token.item()) for token in tokens]
        return decoded_words
