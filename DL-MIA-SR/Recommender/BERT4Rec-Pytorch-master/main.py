import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *

# # debug
# import pydevd_pycharm
# pydevd_pycharm.settrace('172.25.155.3', port=3931, stdoutToServer=True, stderrToServer=True)

def train():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader, usermap, itemmap = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, usermap, itemmap, export_root)
    trainer.train()

    test_model = 1
    if test_model:
        trainer.test()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
