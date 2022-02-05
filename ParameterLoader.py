import torch
import os
class ParameterLoader:
    def __init__(self,save_dir, model_name, corpus_name,prefix="checkpoint"):
        self.directory = os.path.join(save_dir, model_name, corpus_name)
        self.prefix = prefix
    def save(self,iteration,model,encoder_optimizer,decoder_optimizer,loss,voc):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        torch.save({
            'iteration': iteration,
            'model': model.state_dict(),
            'en_opt': encoder_optimizer.state_dict(),
            'de_opt': decoder_optimizer.state_dict(),
            'loss': loss,
            'voc_dict': voc.__dict__,
        }, os.path.join(self.directory, '{}_{}.tar'.format(iteration, self.prefix)))
    def load(self,loadFilename,checkpoint_iter):
        filename = os.path.join(self.directory,
                        '{}_{}.tar'.format(checkpoint_iter,self.prefix))
        checkpoint = torch.load(filename)
        model_sd = checkpoint['model']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        voc_dict = checkpoint['voc_dict']
        iteration = checkpoint['iteration']
        return model_sd,encoder_optimizer_sd,decoder_optimizer_sd,voc_dict,iteration