PAD_token = 0
SOS_token = 1 # start of sentence
EOS_token = 2
import itertools
import torch
class BaseVoc:
    def addSentence(self,sentence,loc):pass
    def addWord(self,word,loc):pass 
    def trim(self, min_count):pass
    def batch2TrainData(self,pair_batch):pass
    def indexesFromSentenceLeft(self,sentence):pass
    def indexesFromSentenceRight(self,sentence):pass
    def word2indexLeft(self,word):pass
    def word2indexRight(self,word):pass
    def index2wordF(self,index,loc='l'):pass
class Voc(BaseVoc):
    def __init__(self,name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.num_words = 3
    def addSentence(self,sentence,loc='l'):
        for word in sentence.split(' '):
            self.addWord(word)
    def addWord(self,word,loc='l'):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))
        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)
    # 整合
    def batch2TrainData(self,pair_batch):
        pair_batch.sort(key = lambda x:len(x[0].split(" ")),reverse=True)# 按单词数降序
        input_batch,output_batch = [],[]
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp,lengths = self._inputVar(input_batch)
        output,mask,max_target_len = self._outputVar(output_batch)
        return inp,lengths,output,mask,max_target_len
    def _inputVar(self,l):
        indexes_batch = [self.indexesFromSentence(sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = self._zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        return padVar,lengths
    def _outputVar(self,l):
        indexes_batch = [self.indexesFromSentence(sentence) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        padList = self._zeroPadding(indexes_batch)
        mask = self._binaryMatrix(padList)
        mask = torch.BoolTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, max_target_len
    
    def indexesFromSentence(self,sentence):
        
        return [self.word2index[word] for word in sentence.split(' ')] + [EOS_token]
    def indexesFromSentenceLeft(self,sentence):
        return self.indexesFromSentence(sentence)
    def indexesFromSentenceRight(self,sentence):
        return self.indexesFromSentence(sentence)    
    def _zeroPadding(self,l,fillvalue=PAD_token):
        return list(itertools.zip_longest(*l,fillvalue=fillvalue))
    def _binaryMatrix(self,l,value=PAD_token):
        m = []
        for i,seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == PAD_token:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m
    def word2indexLeft(self,word):
        return self.word2index[word]
    def word2indexRight(self,word):
        # print(word)
        return self.word2index[word]
    def index2wordF(self,index,loc='l'):
        return self.index2word[index]

class SplitVoc(Voc):
    def __init__(self,name):
        self.name = name
        self.trimmed = False
        self.word2index = {'l':{},'r':{}}
        self.word2count = {'l':{},'r':{}}
        baseWord = {PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.index2word = {'l':{**baseWord},'r':{**baseWord}}
        self.num_words = {'l':3,'r':3}
    def addSentence(self,sentence,loc='l'):
        for word in sentence:
            self.addWord(word,loc)
    def addWord(self,word,loc='l'):
        if word not in self.word2index[loc]:
            self.word2index[loc][word] = self.num_words[loc]
            self.word2count[loc][word] = 1
            self.index2word[loc][self.num_words[loc]] = word
            self.num_words[loc] += 1
        else:
            self.word2count[loc][word] += 1
        
    def batch2TrainData(self,pair_batch):
        pair_batch.sort(key = lambda x:len(x[0].split(" ")),reverse=True)# 按单词数降序
        input_batch,output_batch = [],[]
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp,lengths = self._inputVar(input_batch)
        output,mask,max_target_len = self._outputVar(output_batch)
        return inp,lengths,output,mask,max_target_len
    def indexesFromSentenceLeft(self,sentence):
        loc = 'l'
        return [self.word2index[loc][word] for word in sentence] + [EOS_token]
    def indexesFromSentenceRight(self,sentence):
        loc = 'r'
        return [self.word2index[loc][word] for word in sentence] + [EOS_token]
    def word2indexLeft(self,word):
        loc = 'l'
        return self.word2index[loc][word]
    def word2indexRight(self,word):
        loc = 'r'
        return self.word2index[loc][word]
    def index2wordF(self,index,loc='l'):
        return self.index2word[loc][index]
    def _inputVar(self,l):
        indexes_batch = [self.indexesFromSentenceLeft(sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = self._zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        return padVar,lengths
    def _outputVar(self,l):
        indexes_batch = [self.indexesFromSentenceRight(sentence) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        padList = self._zeroPadding(indexes_batch)
        mask = self._binaryMatrix(padList)
        mask = torch.BoolTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, max_target_len
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))
        # Reinitialize dictionaries
        self.word2index = {'l':{},'r':{}}
        self.word2count = {'l':{},'r':{}}
        baseWord = {PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.index2word = {'l':{**baseWord},'r':{**baseWord}}
        self.num_words = {'l':3,'r':3}

        for word in keep_words:
            self.addWord(word)

# voc = SplitVoc("haha")
# voc.addSentence("吃饭",'l')
# voc.addSentence("稀饭",'r')
# voc.addSentence("池",'l')
# voc.addSentence("饭",'r')
# res = voc.batch2TrainData([["吃饭","稀饭"],["池","饭"]])
# print(voc.word2index)
# print(res)