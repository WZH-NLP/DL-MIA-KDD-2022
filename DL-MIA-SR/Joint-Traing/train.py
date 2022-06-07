import datetime
import os
import sys

import numpy as np
import torch
import config
import train_helper
import data_utils
import argparse
import ast
import json
import random

import pandas as pd
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from models import jointModel
from classifier import MLP
from tensorboardX import SummaryWriter
from time import time


# debug
# import pydevd_pycharm
# pydevd_pycharm.settrace('172.25.243.68', port=3931, stdoutToServer=True, stderrToServer=True)

def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def freeze_params(model, freeze=None):
    for name, param in model.named_parameters():
        if freeze is not None and freeze in name:
            param.requires_grad = False
            # print('freeze_params', name, param.size())


def unfreeze_params(model, unfreeze=None):
    for name, param in model.named_parameters():
        if unfreeze is not None and unfreeze in name:
            param.requires_grad = True
            # print('unfreeze_params', name, param.size())


def run(e):
    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    dp = data_utils.data_processor(
        train_path=e.config.train_file,
        eval_path=None,
        experiment=e)
    data, W = dp.process()

    set_seed(2021)
    model = jointModel(
        vocab_size=len(data.vocab),
        embed_dim=e.config.edim if W is None else W.shape[1],
        embed_init=W,
        experiment=e)

    true_it = 0
    if e.config.continue_train:
        start_epoch, _ = model.load(name=e.config.continue_from)
        continue_epoch = start_epoch + 1
        if e.config.use_cuda:
            model.cuda()
            e.log.info("transferred model to gpu")
        e.log.info(
            "resumed from previous checkpoint: start epoch: {}, "
            "iteration: {}".format(start_epoch, true_it))

    if e.config.use_pretrainD:
        _, _ = model.load_portion(name=e.config.pretrain_modelD)
        e.log.info("use pretrained model:{}".format(e.config.pretrain_modelD))
        if e.config.use_cuda:
            model.cuda()
            e.log.info("transferred model to gpu")

    if e.config.use_pretrainC:
        _, _ = model.load_portion(name=e.config.pretrain_modelC)
        e.log.info("use pretrained model:{}".format(e.config.pretrain_modelC))
        if e.config.use_cuda:
            model.cuda()
            e.log.info("transferred model to gpu")

    if e.config.summarize:
        writer = SummaryWriter(e.experiment_dir)

    if e.config.decoder_type.startswith("bag"):
        minibatcher = data_utils.bow_minibatcher
        e.log.info("using BOW batcher")
    else:
        minibatcher = data_utils.minibatcher
        e.log.info("using sequential batcher")
    train_stats = train_helper.tracker(["loss", "vgvae_loss", "vmf_kl", "gauss_kl", "rec_logloss",
                                        "classifier_loss", "reweight_loss"])

    if not e.config.use_pretrainD and not e.config.use_pretrainC:
        e.log.info("Vgvae Training start ...")
        trainD_batch = minibatcher(
            data1=data.train_data[0],
            data2=data.train_data[1],
            vocab_size=len(data.vocab),
            batch_size=e.config.batch_size,
            score_func=model.vgvae.score,
            shuffle=True,
            mega_batch=0 if not e.config.continue_train else e.config.mb,
            p_scramble=e.config.ps)

        startD_epoch = true_it = 0
        if e.config.continue_train:
            startD_epoch = continue_epoch
        freeze_params(model, "mlp1")
        freeze_params(model, "mlp2")
        freeze_params(model, "mlp3")
        freeze_params(model, "reweight")
        for epoch in range(startD_epoch, e.config.D_epoch):
            epoch_loss = 0.0
            epoch_vgloss = 0.0
            epoch_kl = 0.0
            epoch_recloss = 0.0
            epoch_closs = 0.0
            accs = 0
            numData = 0
            epoch_rloss = 0.0

            if epoch > 1 and trainD_batch.mega_batch != e.config.mb:  # mega_batch
                trainD_batch.mega_batch = e.config.mb  # mb:size of mega batching mega-batch大批量
                trainD_batch._reset()  # generate mega-batch data for sampling
            for it, (s1, m1, s2, m2, t1, tm1, t2, tm2,
                     n1, nm1, nt1, ntm1, n2, nm2, nt2, ntm2, _) in \
                    enumerate(trainD_batch):
                true_it = it + 1 + epoch * len(trainD_batch)
                loss, vgvae_loss, vkl, gkl, rec_logloss, classifier_loss, acc, reweight_loss = \
                    model(s1, m1, s2, m2, t1, tm1, t2, tm2,
                          n1, nm1, nt1, ntm1, n2, nm2, nt2, ntm2,
                          e.config.vmkl, e.config.gmkl,
                          epoch > 1 and e.config.dratio and e.config.mb > 1, "vgvae")
                epoch_loss += loss.item()
                epoch_vgloss += vgvae_loss.item()
                epoch_kl += (vkl.item() + gkl.item())
                epoch_recloss += rec_logloss.item()
                epoch_closs += classifier_loss.item()
                accs += acc.item()
                numData += len(s1)
                epoch_rloss += reweight_loss.item()
                model.optimize(loss)

                train_stats.update(
                    {"loss": loss, "vgvae_loss": vgvae_loss, "vmf_kl": vkl, "gauss_kl": gkl, "rec_logloss": rec_logloss,
                     "classifier_loss": classifier_loss, "reweight_loss": reweight_loss}, len(s1))
                if (true_it + 1) % e.config.print_every == 0 or \
                        (true_it + 1) % len(trainD_batch) == 0:
                    summarization = train_stats.summarize(
                        "epoch: {}, it: {} (max: {})".format(epoch, it, len(trainD_batch)))
                    # e.log.info(summarization)
                    if e.config.summarize:
                        for name, value in train_stats.stats.items():
                            writer.add_scalar("train/" + name, value, true_it)
                    train_stats.reset()

            epoch_loss /= it + 1
            epoch_vgloss /= it + 1
            epoch_kl /= it + 1
            epoch_recloss /= it + 1
            epoch_closs /= it + 1
            accs /= numData
            epoch_rloss /= it + 1

            # output_str = "Epoch:{}\tepoch_loss:{:.4f}\tepoch_vgloss:{:.4f}\tepoch_kl:{:.4f}\t" \
            #              "epoch_rec_logloss:{:.4f}\t" \
            #              "epoch_closs:{:.4f}\ttrain_acc:{:.4f}\tepoch_rloss:{:.4f}\n" \
            #     .format(epoch,
            #             epoch_loss,
            #             epoch_vgloss,
            #             epoch_kl,
            #             epoch_recloss,
            #             epoch_closs,
            #             accs,
            #             epoch_rloss)
            # e.log.info(output_str)

            output_str = "Epoch:{}\tloss:{:.4f}" .format(epoch, epoch_loss)
            e.log.info(output_str)

            # if (epoch + 1) % 10 == 0:
            #     checkpoint = {
            #         "epoch": epoch,
            #         "iteration": true_it,
            #         "state_dict": model.state_dict(),
            #         "opt_state_dict": model.opt.state_dict(),
            #         "config": e.config
            #     }
            #     save_path = os.path.join(e.config.checkpointD_dir, "model_{0:05d}.pt".format(epoch))
            #     torch.save(checkpoint, save_path)
            #     e.log.info("model saved to {}".format(save_path))

        e.log.info("Vgvae Training end!")
        unfreeze_params(model, "mlp1")
        unfreeze_params(model, "mlp2")
        unfreeze_params(model, "mlp3")

    if not e.config.use_pretrainC:
        e.log.info("Classifier Training start ...")
        trainC_batch = minibatcher(
            data1=data.train_data[0][:e.config.shadow_size],
            data2=data.train_data[1][:e.config.shadow_size],
            vocab_size=len(data.vocab),
            batch_size=1,
            score_func=model.vgvae.score,
            shuffle=True,
            mega_batch=0 if not e.config.continue_train else e.config.mb,
            p_scramble=e.config.ps)
        Dict_shadow = {}
        for it, (s1, m1, s2, m2, t1, tm1, t2, tm2,
                 n1, nm1, nt1, ntm1, n2, nm2, nt2, ntm2, _) in enumerate(trainC_batch):
            sent1_semantic, sent1_syntax, sent2_semantic, sent2_syntax = model.vgvae.getvector(s1, m1, s2, m2)
            sent1_semantic = np.squeeze(sent1_semantic.detach().cpu().numpy()).tolist()
            sent1_syntax = np.squeeze(sent1_syntax.detach().cpu().numpy()).tolist()
            sent1, sent1_label = model.dp_s.postprocess2(s1, model.s_label_dict)
            Dict_shadow.setdefault(it, []).append([sent1, sent1_semantic, sent1_syntax, sent1_label])

        startC_epoch = 0
        freeze_params(model, "vgvae")
        for epoch in range(startC_epoch, e.config.C_epoch):
            epoch_loss = 0.0
            epoch_vgloss = 0.0
            epoch_kl = 0.0
            epoch_recloss = 0.0
            epoch_closs = 0.0
            accs = 0
            numData = 0
            epoch_rloss = 0.0

            for it, (s1, m1, s2, m2, t1, tm1, t2, tm2,
                     n1, nm1, nt1, ntm1, n2, nm2, nt2, ntm2, _) in \
                    enumerate(trainC_batch):
                vector_shadow = Dict_shadow[it][0][0]
                vector1_shadow = Dict_shadow[it][0][1]
                vector2_shadow = Dict_shadow[it][0][2]
                vector_labels = Dict_shadow[it][0][3]
                # vector_target, vector1_target, vector2_target, label_target
                loss, vgvae_loss, vkl, gkl, rec_logloss, classifier_loss, acc, reweight_loss = \
                    model(s1, m1, s2, m2, t1, tm1, t2, tm2,
                          vector_shadow, vector1_shadow, vector2_shadow, vector_labels,
                          n2, nm2, nt2, ntm2,
                          e.config.vmkl, e.config.gmkl,
                          epoch > 1 and e.config.dratio and e.config.mb > 1, "classifier")
                epoch_loss += loss.item()
                epoch_vgloss += vgvae_loss.item()
                epoch_kl += (vkl.item() + gkl.item())
                epoch_recloss += rec_logloss.item()
                epoch_closs += classifier_loss.item()
                accs += acc.item()
                numData += len(s1)
                epoch_rloss += reweight_loss.item()
                model.optimize_classifier(loss)

                train_stats.update(
                    {"loss": loss, "vgvae_loss": vgvae_loss, "vmf_kl": vkl, "gauss_kl": gkl, "rec_logloss": rec_logloss,
                     "classifier_loss": classifier_loss, "reweight_loss": reweight_loss}, len(s1))

                if (true_it + 1) % e.config.print_every == 0 or \
                        (true_it + 1) % len(trainC_batch) == 0:
                    summarization = train_stats.summarize(
                        "epoch: {}, it: {} (max: {})".format(epoch, it, len(trainC_batch)))
                    # e.log.info(summarization)
                    if e.config.summarize:
                        for name, value in train_stats.stats.items():
                            writer.add_scalar("train/" + name, value, true_it)
                    train_stats.reset()

            epoch_loss /= it + 1
            epoch_kl /= it + 1
            epoch_recloss /= it + 1
            epoch_closs /= it + 1
            accs /= numData
            epoch_rloss /= it + 1

            # output_str = "Epoch:{}\tepoch_loss:{:.4f}\tepoch_vgloss:{:.4f}\t" \
            #              "epoch_gkl:{:.4f}\tepoch_rec_logloss:{:.4f}\t" \
            #              "epoch_closs:{:.4f}\ttrain_acc:{:.4f}\tepoch_rloss:{:.4f}\n" \
            #     .format(epoch,
            #             epoch_loss,
            #             epoch_vgloss,
            #             epoch_kl,
            #             epoch_recloss,
            #             epoch_closs,
            #             accs,
            #             epoch_rloss)
            # e.log.info(output_str)
            output_str = "Epoch:{}\tloss:{:.4f}".format(epoch, epoch_loss)
            e.log.info(output_str)

            # if (epoch + 1) % 20 == 0:
            #     checkpoint = {
            #         "epoch": epoch,
            #         "iteration": true_it,
            #         "state_dict": model.state_dict(),
            #         "opt_state_dict": model.opt.state_dict(),
            #         "config": e.config
            #     }
            #     save_path = os.path.join(e.config.checkpointC_dir, "model_{0:05d}.pt".format(epoch))
            #     torch.save(checkpoint, save_path)
            #     e.log.info("model saved to {}".format(save_path))

        e.log.info("Classifier Training end!")
        unfreeze_params(model, "vgvae")
        unfreeze_params(model, "reweight")

    train_batch = minibatcher(
        data1=data.train_data[0],
        data2=data.train_data[1],
        vocab_size=len(data.vocab),
        batch_size=e.config.batch_size,
        score_func=model.vgvae.score,
        shuffle=True,
        mega_batch=0 if not e.config.continue_train else e.config.mb,
        p_scramble=e.config.ps)

    Bias_shadow = []
    Bias_shadow1 = []
    Bias_shadow2 = []
    Label_shadow = []
    Bias_target = []
    Bias_target1 = []
    Bias_target2 = []
    # get bias feature vector
    for it, (s1, m1, s2, m2, t1, tm1, t2, tm2,
             n1, nm1, nt1, ntm1, n2, nm2, nt2, ntm2, _) in \
            enumerate(train_batch):
        bias_shadow_, bias_shadow1_, bias_shadow2_, label_shadow_, \
        bias_target_, bias_target1_, bias_target2_, label_target_ = \
            model.midtest(s1, m1, s2, m2, t1, tm1, t2, tm2,
                          n1, nm1, nt1, ntm1, n2, nm2, nt2, ntm2,
                          e.config.vmkl, e.config.gmkl, _)
        Bias_shadow.append(bias_shadow_)
        Bias_shadow1.append(bias_shadow1_.tolist())
        Bias_shadow2.append(bias_shadow2_.tolist())
        Label_shadow.append(label_shadow_)

        Bias_target.append(bias_target_)
        Bias_target1.append(bias_target1_.tolist())
        Bias_target2.append(bias_target2_.tolist())

    e.log.info("Joint Training start ...")
    start_epoch = true_it = 0
    train_weight = True

    for epoch in range(start_epoch, e.config.n_epoch):
        if epoch % 10 == 0:
            train_weight = bool(1 - train_weight)
        epoch_loss = 0.0
        epoch_vgloss = 0.0
        epoch_kl = 0.0
        epoch_recloss = 0.0
        epoch_closs = 0.0
        epoch_rloss = 0.0
        accs = 0
        numData = 0
        if epoch > 1 and train_batch.mega_batch != e.config.mb:  # mega_batch
            train_batch.mega_batch = e.config.mb  # mb:size of mega batching mega-batch大批量
            train_batch._reset()  # generate mega-batch data for sampling
        model.index = -e.config.batch_size
        model.index1 = -e.config.batch_size
        time1 = time()
        if train_weight:
            for it, (s1, m1, s2, m2, t1, tm1, t2, tm2,
                     n1, nm1, nt1, ntm1, n2, nm2, nt2, ntm2, _) in \
                    enumerate(train_batch):
                bias_shadow = torch.autograd.Variable(torch.Tensor(Bias_shadow[it]))
                bias_shadow1 = torch.autograd.Variable(torch.Tensor(Bias_shadow1[it]))
                bias_shadow2 = torch.autograd.Variable(torch.Tensor(Bias_shadow2[it]))
                label_shadow = torch.autograd.Variable(torch.Tensor(Label_shadow[it]).long())

                bias_target = torch.autograd.Variable(torch.Tensor(Bias_target[it]))
                bias_target1 = torch.autograd.Variable(torch.Tensor(Bias_target1[it]))
                bias_target2 = torch.autograd.Variable(torch.Tensor(Bias_target2[it]))

                true_it = it + 1 + epoch * len(train_batch)
                loss, vgvae_loss, vkl, gkl, rec_logloss, classifier_loss, acc, reweight_loss = \
                    model(s1, m1, s2, m2, t1, tm1, t2, tm2,
                          bias_shadow, bias_shadow1, bias_shadow2, label_shadow,
                          bias_target, bias_target1, bias_target2, _,
                          e.config.vmkl, e.config.gmkl,
                          epoch > 1 and e.config.dratio and e.config.mb > 1, "joint_weight")

                epoch_loss += loss.item()
                epoch_vgloss += vgvae_loss.item()
                epoch_kl += (vkl.item() + gkl.item())
                epoch_recloss += rec_logloss.item()
                epoch_closs += classifier_loss.item()
                accs += acc.item()
                numData += (len(s1) + len(s2))
                epoch_rloss += reweight_loss.item()

                model.optimize(loss)

                train_stats.update(
                    {"loss": loss, "vgvae_loss": vgvae_loss, "vmf_kl": vkl, "gauss_kl": gkl, "rec_logloss": rec_logloss,
                     "classifier_loss": classifier_loss, "reweight_loss": reweight_loss}, len(s1))

                if (true_it + 1) % e.config.print_every == 0 or \
                        (true_it + 1) % len(train_batch) == 0:
                    summarization = train_stats.summarize(
                        "epoch: {}, it: {} (max: {})".format(epoch, it, len(train_batch)))
                    # e.log.info(summarization)
                    if e.config.summarize:
                        for name, value in train_stats.stats.items():
                            writer.add_scalar("train/" + name, value, true_it)
                    train_stats.reset()
        else:
            for it, (s1, m1, s2, m2, t1, tm1, t2, tm2,
                     n1, nm1, nt1, ntm1, n2, nm2, nt2, ntm2, _) in \
                    enumerate(train_batch):
                loss, vgvae_loss, vkl, gkl, rec_logloss, classifier_loss, acc, reweight_loss = \
                    model(s1, m1, s2, m2, t1, tm1, t2, tm2,
                          n1, nm1, nt1, ntm1, n2, nm2, nt2, ntm2,
                          e.config.vmkl, e.config.gmkl,
                          epoch > 1 and e.config.dratio and e.config.mb > 1, "joint")
                epoch_loss += loss.item()
                epoch_vgloss += vgvae_loss.item()
                epoch_kl += (vkl.item() + gkl.item())
                epoch_recloss += rec_logloss.item()
                epoch_closs += classifier_loss.item()
                accs += acc.item()
                numData += (len(s1) + len(s2))
                epoch_rloss += reweight_loss.item()

                freeze_params(model, "ps")
                freeze_params(model, "pt")

                model.optimize(loss)

                unfreeze_params(model, "ps")
                unfreeze_params(model, "pt")

                train_stats.update(
                    {"loss": loss, "vgvae_loss": vgvae_loss, "vmf_kl": vkl, "gauss_kl": gkl, "rec_logloss": rec_logloss,
                     "classifier_loss": classifier_loss, "reweight_loss": reweight_loss}, len(s1))

                if (true_it + 1) % e.config.print_every == 0 or \
                        (true_it + 1) % len(train_batch) == 0:
                    summarization = train_stats.summarize(
                        "epoch: {}, it: {} (max: {})".format(epoch, it, len(train_batch)))
                    # e.log.info(summarization)
                    if e.config.summarize:
                        for name, value in train_stats.stats.items():
                            writer.add_scalar("train/" + name, value, true_it)
                    train_stats.reset()

        time2 = time()
        epoch_loss /= it + 1
        epoch_vgloss /= it + 1
        epoch_kl /= it + 1
        epoch_recloss /= it + 1
        epoch_closs /= it + 1
        accs /= numData
        epoch_rloss /= it + 1

        # output_str = "Epoch:{}\tepoch_loss:{:.4f}\tepoch_vgloss:{:.4f}\t" \
        #              "epoch_kl:{:.4f}\tepoch_rec_logloss:{:.4f}\t" \
        #              "epoch_closs:{:.4f}\ttrain_acc:{:.4f}\tepoch_rloss:{:.4f}\tTime:{:.4f}\n" \
        #     .format(epoch,
        #             epoch_loss,
        #             epoch_vgloss,
        #             epoch_kl,
        #             epoch_recloss,
        #             epoch_closs,
        #             accs,
        #             epoch_rloss,
        #             time2 - time1)
        # e.log.info(output_str)
        output_str = "Epoch:{}\tloss:{:.4f}\tTime:{:.4f}".format(epoch,  epoch_loss, time2 - time1)
        e.log.info(output_str)

        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "epoch": epoch,
                "iteration": true_it,
                "state_dict": model.state_dict(),
                "opt_state_dict": model.opt.state_dict(),
                "config": e.config
            }
            save_path = os.path.join(e.config.checkpoint_dir, "model_{0:05d}.pt".format(epoch))
            torch.save(checkpoint, save_path)
            e.log.info("model saved to {}".format(save_path))

    e.log.info("Joint Training end!")


def eval(e, save_dir):
    e.log.info("Test start ...")
    dp = data_utils.data_processor(
        train_path=e.config.train_file,
        eval_path=None,
        experiment=e)
    data, W = dp.process()

    model = jointModel(
        vocab_size=len(data.vocab),
        embed_dim=e.config.edim if W is None else W.shape[1],
        embed_init=W,
        experiment=e)

    dir_model = save_dir
    print("loading trained model from {}".format(dir_model))
    _, _ = model.load(name=dir_model)
    if e.config.use_cuda:
        model.cuda()
        e.log.info("transferred model to gpu")

    if e.config.decoder_type.startswith("bag"):
        minibatcher = data_utils.bow_minibatcher
        e.log.info("using BOW batcher")
    else:
        minibatcher = data_utils.minibatcher
        e.log.info("using sequential batcher")

    test_batch = minibatcher(
        data1=data.train_data[0][:e.config.target_size],
        data2=data.train_data[1][:e.config.target_size],
        vocab_size=len(data.vocab),
        batch_size=1,
        score_func=model.vgvae.score,
        shuffle=False,
        mega_batch=0 if not e.config.continue_train else e.config.mb,
        p_scramble=e.config.ps)

    acc_ans = 0
    TruePositive = 0
    FalsePositive = 0
    TrueNegative = 0
    FalseNegative = 0
    for it, (s1, m1, s2, m2, t1, tm1, t2, tm2,
             n1, nm1, nt1, ntm1, n2, nm2, nt2, ntm2, _) in enumerate(test_batch):

        outputs, labels = model.test(s1, m1, s2, m2, t1, tm1, t2, tm2,
                                     n1, nm1, nt1, ntm1, n2, nm2, nt2, ntm2,
                                     e.config.vmkl, e.config.gmkl, False)
        _, pred = torch.max(outputs, 1)
        pred = pred.cpu()
        if int(pred) == labels.cpu().numpy()[0]:
            acc_ans += 1
            if int(pred) == 1:
                TruePositive = TruePositive + 1
            else:
                TrueNegative = TrueNegative + 1
        else:
            if int(pred) == 1:
                FalsePositive = FalsePositive + 1
            else:
                FalseNegative = FalseNegative + 1
    num_sum = it + 1
    # print('TruePositive:', TruePositive)
    # print('FalsePositive:', FalsePositive)
    # print('TrueNegative:', TrueNegative)
    # print('FalseNegative', FalseNegative)
    print("accuarcy: ")
    print((acc_ans / num_sum))
    _A = TruePositive + FalsePositive
    _B = TruePositive + FalseNegative
    _C = FalsePositive + TrueNegative
    if _A != 0 and _B != 0 and _C != 0:
        print("precsion: ")
        print((TruePositive / (TruePositive + FalsePositive)))
        print("recall: ")
        print((TruePositive / (TruePositive + FalseNegative)))

        TPRate = TruePositive / (TruePositive + FalseNegative)
        FPRate = FalsePositive / (FalsePositive + TrueNegative)
        area = 0.5 * TPRate * FPRate + 0.5 * (TPRate + 1) * (1 - FPRate)
        print("AUC: ")
        print(area)
    else:
        print("ZeroDivisionError")

    e.log.info("Test end ...")


def make_checkpoint_dir(checkpoint_dir):
    # print("PARAMETER" + "-" * 10)
    now = datetime.datetime.now()
    S = '{:02d}{:02d}{:02d}{:02d}'.format(now.month, now.day, now.hour, now.minute)
    save_dir = os.path.join(checkpoint_dir, S)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_dir = save_dir
    with open(os.path.join(checkpoint_dir, 'parameter.txt'), 'w') as f:
        for attr, value in sorted(e.config.__dict__.items()):
            # print("{}={}".format(attr.upper(), value))
            f.write("{}={}\n".format(attr.upper(), value))
    # print("---------" + "-" * 10)
    return save_dir


if __name__ == '__main__':
    args = config.get_base_parser().parse_args()
    args.use_cuda = torch.cuda.is_available()


    def exit_handler(*args):
        print(args)
        exit()


    train_helper.register_exit_handler(exit_handler)
    with train_helper.experiment(args, args.save_prefix) as e:
        if e.config.is_eval:
            for ind in range(9, e.config.n_epoch, 10):
                save_path_ind = os.path.join(e.config.load_model_from, "model_{0:05d}.pt".format(ind))
                eval(e, save_path_ind)  # eval joint model

        else:
            save_dirD = make_checkpoint_dir(e.config.checkpointD_dir)
            save_dirC = make_checkpoint_dir(e.config.checkpointC_dir)
            save_dir = make_checkpoint_dir(e.config.checkpoint_dir)
            e.config.checkpointD_dir = save_dirD
            e.config.checkpointC_dir = save_dirC
            e.config.checkpoint_dir = save_dir
            run(e)
            for ind in range(9, e.config.n_epoch, 10):
                save_path_ind = os.path.join(e.config.checkpoint_dir, "model_{0:05d}.pt".format(ind))
                eval(e, save_path_ind)  # eval joint model
