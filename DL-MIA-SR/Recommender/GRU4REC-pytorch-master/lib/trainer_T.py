import os
import lib
import time
import torch
import numpy as np
from tqdm import tqdm


class Trainer_T(object):
    def __init__(self, model, train_data, eval_data, optim, use_cuda, loss_func, batch_size, args):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.optim = optim
        self.loss_func = loss_func
        self.evaluation = lib.Evaluation(self.model, self.loss_func, use_cuda, k=args.k_eval)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.batch_size = batch_size
        self.args = args

    def train(self, start_epoch, end_epoch, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            print('Start Epoch #', epoch)
            train_loss = self.train_epoch(epoch)
            loss, recall, mrr = self.evaluation.eval(self.eval_data, self.batch_size)

            print("Epoch: {}, train loss: {:.4f}, loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch,
                                                                                                              train_loss,
                                                                                                              loss,
                                                                                                              recall,
                                                                                                              mrr,
                                                                                                              time.time() - st))
            checkpoint = {
                'model': self.model,
                'args': self.args,
                'epoch': epoch,
                'optim': self.optim,
                'loss': loss,
                'recall': recall,
                'mrr': mrr
            }
            model_name = os.path.join(self.args.checkpoint_dir, "model_{0:05d}.pt".format(epoch))
            torch.save(checkpoint, model_name)
            print("Save model as %s" % model_name)

        print('generate recommendations')
        new_itemmap = {}
        # fw = open("data/ml-1m_Tmember_recommendations", 'w')
        # with open("data/ml-1m_Titemmap.txt", 'r') as fr:
        fw = open("data/beauty_Tmember_recommendations", 'w')
        with open("data/beauty_Titemmap.txt", 'r') as fr:
            for line in fr.readlines():
                line = line.strip().split(',')
                if line[0] != 'ItemID':
                    new_itemmap[int(line[1])] = int(line[0])
        dataloader = lib.DataLoader(self.train_data, self.batch_size)
        last_logit = torch.FloatTensor(self.batch_size, len(self.train_data.items))
        last_mask = []
        with torch.no_grad():
            hidden = self.model.init_hidden()
            for ii, (input, target, mask, iters, iters_old) in \
                    tqdm(enumerate(dataloader),
                         total=len(dataloader.dataset.df) // dataloader.batch_size,
                         miniters=1000):
                # for input, target, mask in dataloader:
                input = input.to(self.device)
                if len(mask) != 0:
                    if len(mask) != len(last_mask) or (len(mask) == len(last_mask) and any(mask != last_mask)):
                        for i in range(len(mask)):
                            SessionID = iters_old[mask[i]]
                            _, indices = torch.topk(last_logit[mask[i]], 100, -1)
                            indices = indices.cpu().detach().tolist()
                            item_ids = [new_itemmap[indices[x]] for x in range(len(indices))]
                            for j in range(len(item_ids)):
                                fw.write(str(SessionID) + '\t' + str(item_ids[j]) + '\t' + '1' + '\n')
                logit, hidden = self.model(input, hidden)
                last_logit = logit
                last_mask = mask.copy()

    def train_epoch(self, epoch):
        self.model.train()
        losses = []

        def reset_hidden(hidden, mask):
            """Helper function that resets hidden state when some sessions terminate"""
            if len(mask) != 0:
                hidden[:, mask, :] = 0
            return hidden

        hidden = self.model.init_hidden()

        dataloader = lib.DataLoader(self.train_data, self.batch_size)
        # for ii,(data,label) in tqdm(enumerate(train_dataloader),total=len(train_data)):
        for ii, (input, target, mask, iters, iters_old) in tqdm(enumerate(dataloader),
                                                                total=len(
                                                                    dataloader.dataset.df) // dataloader.batch_size,
                                                                miniters=1000):
            input = input.to(self.device)
            target = target.to(self.device)
            self.optim.zero_grad()
            hidden = reset_hidden(hidden, mask).detach()
            logit, hidden = self.model(input, hidden)
            logit_sampled = logit[:, target.view(-1)]
            loss = self.loss_func(logit_sampled)
            losses.append(loss.item())
            loss.backward()
            self.optim.step()

        mean_losses = np.mean(losses)
        return mean_losses

