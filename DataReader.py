
import torch
from functions import normalizeString
class BaseReader:
    def loadPrepareData(self,corpus_name, datafile):pass
class DataReader(BaseReader):
    def __init__(self,VocClass,isTrim=True,MAX_LENGTH=10,MIN_COUNT=3):
        self.Voc = VocClass
        self.max = MAX_LENGTH
        self.min = MIN_COUNT
        self.isTrim = isTrim
    def readVocs(self,datafile,corpus_name):
        print("Reading lines...")
        lines = open(datafile,encoding='utf-8').read().strip().split('\n')
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
        voc = self.Voc(corpus_name)
        # 创建一个新的voc和一组对话pair
        return voc,pairs
    def _fileterPair(self,p):
        # 根据单词长度滤
        return len(p[0].split(' ')) < self.max and len(p[1].split(' ')) < self.max
    def filterPairs(self,pairs):
        return [pair for pair in pairs if self._fileterPair(pair)]
    def trimRareWords(self,voc,pairs):
        voc.trim(self.min)
        keep_pairs = []
        for pair in pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_outpus = True
            for word in input_sentence.split(' '):
                if word not in voc.word2index:
                    keep_input = False
                    break
            for word in output_sentence.split(' '):
                if word not in voc.word2index:
                    keep_outpus = False
                    break
            if keep_input and keep_outpus:
                keep_pairs.append(pair)
        print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
        return keep_pairs
    def loadPrepareData(self,corpus_name, datafile):
        print("Start preparing training data ...")
        voc, pairs = self.readVocs(datafile, corpus_name)
        print("Read {!s} sentence pairs".format(len(pairs)))
        pairs = self.filterPairs(pairs)
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))
        print("Counting words...")
        for pair in pairs:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
        print("Counted words:", voc.num_words)
        if self.isTrim:
            pairs = self.trimRareWords(voc,pairs)
        return voc, pairs
