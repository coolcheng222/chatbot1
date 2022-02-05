import torch
import os
import matplotlib.pyplot as plt
import time
import random
 
class NLPOperator:
    def __init__(self,trainer, voc,loader, print_every, save_every,clip, checkpoint=None):
        self.print_every = print_every
        self.save_every = save_every
        self.trainer = trainer
        self.loader = loader
        self.checkpoint = checkpoint
        self.voc = voc
        self.clip = clip
    def train(self,pairs, n_iteration, batch_size):
        mat_loss = []
        # Load batches for each iteration
        self.trainer.model.train()
        training_batches = [self.voc.batch2TrainData([random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

        # Initializations
        print('Initializing ...')
        start_iteration = 1
        print_loss = 0
        if self.checkpoint:
            start_iteration = self.checkpoint + 1

        # Training loop
        print("Training...")
        for iteration in range(start_iteration, n_iteration + 1):
            training_batch = training_batches[iteration - 1]
            # Extract fields from batch
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            # Run a training iteration with batch
            loss = self.trainer.train(input_variable,lengths,target_variable,mask,max_target_len,self.clip)
            print_loss += loss


            # Print progress
            if iteration % self.print_every == 0:
                print_loss_avg = print_loss / self.print_every
                # print(print_every)
                # print(print_loss)
                mat_loss.append(print_loss_avg)
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
                print_loss = 0

            # Save checkpoint
            if (iteration % self.save_every == 0):
                self.loader.save(iteration,self.trainer.model,self.trainer.encoder_optimizer,self.tariner.decoder_optimizer,loss,self.voc)        
        plt.plot(list(range(len(mat_loss))),mat_loss)
        plt.show()
