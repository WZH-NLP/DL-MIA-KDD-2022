"""
@author: Javier Rodriguez (jrzaurin@gmail.com)
"""

import numpy as np
import pandas as pd
import math
import os
import argparse
import heapq
import torch

from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader, Dataset
from GMF_pytorch import train, evaluate, checkpoint

from Dataset import Dataset as ml1mDataset
from time import time
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datadir", type=str, default="Data_Movielens/",
        help="data directory.")
    parser.add_argument("--modeldir", type=str, default="models",
        help="models directory")
    parser.add_argument("--dataname", type=str, default="ml-1m",
        help="chose a dataset.")
    parser.add_argument("--epochs", type=int, default=20,
        help="number of epochs.")
    parser.add_argument("--batch_size", type=int, default=256,
        help="batch size.")
    parser.add_argument("--layers", type=str, default="[64,32,16,8]",
        help="layer architecture. The first elements is used for the embedding \
        layers and equals n_emb*2")
    parser.add_argument("--dropouts", type=str, default="[0,0,0]",
        help="dropout per dense layer. len(dropouts) = len(layers)-1")
    parser.add_argument("--l2reg", type=float, default=0.,
        help="l2 regularization")
    parser.add_argument("--lr", type=float, default=0.01,
        help="learning rate.")
    parser.add_argument("--learner", type=str, default="adam",
        help="Specify an optimizer: adagrad, adam, rmsprop, sgd")
    parser.add_argument("--validate_every", type=int, default=1,
        help="validate every n epochs")
    parser.add_argument("--save_model", type=int , default=1)
    parser.add_argument("--n_neg", type=int, default=4,
        help="number of negative instances to consider per positive instance.")
    parser.add_argument("--topK", type=int, default=100,
        help="number of items to retrieve for recommendation.")

    return parser.parse_args()


class MLP(nn.Module):
    """
    Concatenate Embeddings that are then passed through a series of Dense
    layers
    """
    def __init__(self, n_user, n_item, layers, dropouts):
        super(MLP, self).__init__()

        self.layers = layers
        self.n_layers = len(layers)
        self.dropouts = dropouts
        self.n_user = n_user
        self.n_item = n_item

        self.embeddings_user = nn.Embedding(n_user, int(layers[0]/2))
        self.embeddings_item = nn.Embedding(n_item, int(layers[0]/2))

        self.mlp = nn.Sequential()
        for i in range(1,self.n_layers):
            self.mlp.add_module("linear%d" %i, nn.Linear(layers[i-1],layers[i]))
            self.mlp.add_module("relu%d" %i, torch.nn.ReLU())
            self.mlp.add_module("dropout%d" %i , torch.nn.Dropout(p=dropouts[i-1]))

        self.out = nn.Linear(in_features=layers[-1], out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight)

    def forward(self, users, items):

        user_emb = self.embeddings_user(users)
        item_emb = self.embeddings_item(items)
        emb_vector = torch.cat([user_emb,item_emb], dim=1)
        emb_vector = self.mlp(emb_vector)
        # preds = torch.sigmoid(self.out(emb_vector))
        preds = self.out(emb_vector)
        return preds


if __name__ == '__main__':
    args = parse_args()

    datadir = args.datadir
    dataname = args.dataname
    modeldir = args.modeldir
    layers = eval(args.layers)
    ll = str(layers[-1]) #last layer
    dropouts = eval(args.dropouts)
    dp = "wdp" if dropouts[0]!=0 else "wodp"
    l2reg = args.l2reg
    n_emb = int(layers[0]/2)
    batch_size = args.batch_size
    epochs = args.epochs
    learner = args.learner
    lr = args.lr
    validate_every = args.validate_every
    save_model = args.save_model
    topK = args.topK
    n_neg = args.n_neg

    modelfname = "pytorch_MLP" + \
        "_".join(["_bs", str(batch_size)]) + \
        "_".join(["_reg", str(l2reg).replace(".", "")]) + \
        "_".join(["_lr", str(lr).replace(".", "")]) + \
        "_".join(["_n_emb", str(n_emb)]) + \
        "_".join(["_ll", ll]) + \
        "_".join(["_dp", dp]) + \
        ".pt"
    modelpath = os.path.join(modeldir, modelfname)
    resultsdfpath = os.path.join(modeldir, 'results_df.p')

    dataset = ml1mDataset(os.path.join(datadir, dataname))
    trainRatings, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    n_users, n_items = trainRatings.shape

    test_dataset = get_test_instances(testRatings, testNegatives)
    test_loader = DataLoader(dataset=test_dataset,
        batch_size=2000,
        shuffle=False)

    model = MLP(n_users, n_items, layers, dropouts)
    if learner.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=l2reg)
    elif learner.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2reg)
    elif learner.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2reg)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2reg)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    best_hr, best_ndcgm, best_iter=0,0,0
    for epoch in range(1,epochs+1):
        t1 = time()
        loss = train(model, criterion, optimizer, epoch, batch_size, use_cuda,
            trainRatings,n_items,n_neg,testNegatives)
        t2 = time()
        if epoch % validate_every == 0:
            (hr, ndcg) = evaluate(model, test_loader, use_cuda, topK)
            print("Iteration {}: {:.2f}s, HR = {:.4f}, NDCG = {:.4f}, loss = {:.4f}, validated in {:.2f}s"
                .format(epoch, t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter, train_time = hr, ndcg, epoch, t2-t1
                if save_model:
                    checkpoint(model, modelpath)

    print("End. Best Iteration {}:  HR = {:.4f}, NDCG = {:.4f}. ".format(best_iter, best_hr, best_ndcg))
    if save_model:
        print("The best MLP model is saved to {}".format(modelpath))

    if save_model:
        if not os.path.isfile(resultsdfpath):
            results_df = pd.DataFrame(columns = ["modelname", "best_hr", "best_ndcg", "best_iter",
                "train_time"])
            experiment_df = pd.DataFrame([[modelfname, best_hr, best_ndcg, best_iter, train_time]],
                columns = ["modelname", "best_hr", "best_ndcg", "best_iter","train_time"])
            results_df = results_df.append(experiment_df, ignore_index=True)
            results_df.to_pickle(resultsdfpath)
        else:
            results_df = pd.read_pickle(resultsdfpath)
            experiment_df = pd.DataFrame([[modelfname, best_hr, best_ndcg, best_iter, train_time]],
                columns = ["modelname", "best_hr", "best_ndcg", "best_iter","train_time"])
            results_df = results_df.append(experiment_df, ignore_index=True)
            results_df.to_pickle(resultsdfpath)
