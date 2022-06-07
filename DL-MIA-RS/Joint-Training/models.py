import ast
import math
import os
import pickle

import pandas as pd
import torch
import data_utils

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter

from von_mises_fisher import VonMisesFisher
from decorators import auto_init_args, auto_init_pytorch
from torch.autograd import Variable
from classifier import MLP, reweight, reweightB
from vgvae import vgvae

MAX_LEN = 101


class base(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_init, experiment):
        super(base, self).__init__()
        self.expe = experiment
        self.eps = self.expe.config.eps
        self.margin = self.expe.config.m
        self.use_cuda = self.expe.config.use_cuda
        self.fn = "vocab_" + str(self.expe.config.vocab_size)
        self.vocab_file = os.path.join(self.expe.config.vocab_file, self.fn)
        self.raw_feature1 = self.expe.config.raw_feature1
        self.raw_feature2 = self.expe.config.raw_feature2
        self.alpha = self.expe.config.alpha
        self.beta = self.expe.config.beta
        self.vgvae = vgvae(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            embed_init=embed_init,
            experiment=experiment)

        self.mlp1 = MLP(self.expe.config.x_dim1, experiment)
        self.mlp2 = MLP(self.expe.config.x_dim2, experiment)
        self.mlp3 = MLP(self.expe.config.x_dim3, experiment)

        self.reweight = reweight(experiment)

        self.lambda_s = 1.0
        self.lambda_t = 1.0
        self.ps = Parameter(torch.rand(1, self.expe.config.pair_size), requires_grad=True)
        self.pt = Parameter(torch.rand(1, self.expe.config.pair_size), requires_grad=True)

        self.index = -self.expe.config.batch_size
        self.index1 = -self.expe.config.batch_size

        self.dp_s = data_utils.data_postprocess(vocabfile=self.vocab_file, filetohelps=self.expe.config.raw_feature1)
        self.dp_t = data_utils.data_postprocess(vocabfile=self.vocab_file, filetohelps=self.expe.config.raw_feature2)
        self.s_label_dict = self.dp_s.postprocess1()
        self.t_label_dict = self.dp_t.postprocess1()

    def optimize(self, loss):
        self.opt.zero_grad()
        loss.backward()
        if self.expe.config.gclip is not None:
            torch.nn.utils.clip_grad_norm(
                self.parameters(), self.expe.config.gclip)
        self.opt.step()

    def init_optimizer(self, opt_type, learning_rate, weight_decay):
        if opt_type.lower() == "adam":
            optimizer = torch.optim.Adam
        elif opt_type.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop
        elif opt_type.lower() == "sgd":
            optimizer = torch.optim.SGD
        else:
            raise NotImplementedError("invalid optimizer: {}".format(opt_type))

        opt = optimizer(
            params=filter(
                lambda p: p.requires_grad, self.parameters()
            ),
            weight_decay=weight_decay,
            lr=learning_rate)

        return opt

    def optimize_classifier(self, loss):
        self.opt1.zero_grad()
        loss.backward()
        if self.expe.config.gclip is not None:
            torch.nn.utils.clip_grad_norm(
                self.parameters(), self.expe.config.gclip)
        self.opt1.step()

    def init_optimizer_classifier(self, learning_rate, weight_decay, momentum):
        optimizer = torch.optim.SGD
        opt1 = optimizer(
            params=filter(
                lambda p: p.requires_grad, self.parameters()
            ),
            weight_decay=weight_decay,
            lr=learning_rate, momentum=momentum)

        return opt1

    # def save(self, epoch, iteration=None, name="best"):
    #     save_path = os.path.join(self.expe.experiment_dir, name + ".ckpt")
    #     checkpoint = {
    #         "epoch": epoch,
    #         "iteration": iteration,
    #         "state_dict": self.state_dict(),
    #         "opt_state_dict": self.opt.state_dict(),
    #         "config": self.expe.config
    #     }
    #     torch.save(checkpoint, save_path)
    #     self.expe.log.info("model saved to {}".format(save_path))
    #
    def load(self, checkpointed_state_dict=None, name="best"):
        if checkpointed_state_dict is None:
            save_path = name
            checkpoint = torch.load(
                save_path, map_location=lambda storage, loc: storage)
            self.load_state_dict(checkpoint['state_dict'])
            if checkpoint.get("opt_state_dict"):
                self.opt.load_state_dict(checkpoint.get("opt_state_dict"))

                if self.use_cuda:
                    for state in self.opt.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
            self.expe.log.info("model loaded from {}".format(save_path))
            return checkpoint.get('epoch', 0), checkpoint.get('iteration', 0)
        else:
            self.load_state_dict(checkpointed_state_dict)
            self.expe.log.info("model loaded!")

    def load_portion(self, checkpointed_state_dict=None, name="best"):
        if checkpointed_state_dict is None:
            save_path = name
            checkpoint = torch.load(
                save_path, map_location=lambda storage, loc: storage)
            state_dict_ = checkpoint['state_dict']
            new_state_dict = {}
            for key, value in state_dict_.items():
                if "reweight" not in key:
                    new_state_dict[key] = value
            model_dict = self.state_dict()
            model_dict.update(new_state_dict)
            self.load_state_dict(model_dict)

            self.expe.log.info("model loaded from {}".format(save_path))
            return checkpoint.get('epoch', 0), checkpoint.get('iteration', 0)
        else:
            self.load_state_dict(checkpointed_state_dict)
            self.expe.log.info("model loaded!")

    def to_var(self, inputs):
        if self.use_cuda:
            if isinstance(inputs, Variable):
                inputs = inputs.cuda()
                inputs.volatile = self.volatile
                return inputs
            else:
                if not torch.is_tensor(inputs):
                    inputs = torch.from_numpy(inputs)
                return Variable(inputs.cuda(), volatile=self.volatile)
        else:
            if isinstance(inputs, Variable):
                inputs = inputs.cpu()
                inputs.volatile = self.volatile
                return inputs
            else:
                if not torch.is_tensor(inputs):
                    inputs = torch.from_numpy(inputs)
                return Variable(inputs, volatile=self.volatile)

    def to_vars(self, *inputs):
        return [self.to_var(inputs_) if inputs_ is not None and
                                        inputs_.size else None for inputs_ in inputs]

    @property
    def volatile(self):
        return not self.training


class jointModel(base):
    @auto_init_pytorch
    @auto_init_args
    def __init__(self, vocab_size, embed_dim, embed_init, experiment):
        super(jointModel, self).__init__(
            vocab_size, embed_dim, embed_init, experiment)

    def forward(self, sent1, mask1, sent2, mask2, tgt1,
                tgt_mask1, tgt2, tgt_mask2,
                neg_sent1, neg_mask1, ntgt1, ntgt_mask1,
                neg_sent2, neg_mask2, ntgt2, ntgt_mask2, vtemp,
                gtemp, use_margin, submodel):
        self.train()
        loss_vgvae_mean = torch.tensor(0.0)
        vkl = torch.tensor(0.0)
        gkl = torch.tensor(0.0)
        rec_logloss = torch.tensor(0.0)
        classifier_loss = torch.tensor(0.0)
        acc = torch.tensor(0.0)
        loss_reweight = torch.tensor(0.0)
        loss = torch.tensor(0.0)
        if submodel == "vgvae":
            loss_vgvae, loss_vgvae_mean, vkl, gkl, rec_logloss, \
            sent1_semantic, sent1_syntax, sent2_semantic, sent2_syntax = \
                self.vgvae(sent1, mask1, sent2, mask2, tgt1,
                           tgt_mask1, tgt2, tgt_mask2,
                           neg_sent1, neg_mask1, ntgt1, ntgt_mask1,
                           neg_sent2, neg_mask2, ntgt2, ntgt_mask2, vtemp,
                           gtemp, use_margin)
            loss = loss_vgvae_mean
        elif submodel == "classifier":
            vector_shadow = torch.autograd.Variable(torch.Tensor(neg_sent1))
            vector1_shadow = torch.autograd.Variable(torch.Tensor(neg_mask1))
            vector2_shadow = torch.autograd.Variable(torch.Tensor(ntgt1))
            vector_label = torch.autograd.Variable(torch.Tensor(ntgt_mask1).long())

            vector_shadow, vector1_shadow, vector2_shadow, vector_label = \
                self.to_vars(vector_shadow, vector1_shadow, vector2_shadow, vector_label)

            outputs1 = self.mlp1(vector_shadow)
            classifier_loss1 = self.mlp1.criterion(outputs1, vector_label)
            outputs2 = self.mlp2(vector1_shadow)
            classifier_loss2 = self.mlp2.criterion(outputs2, vector_label)
            outputs3 = self.mlp3(vector2_shadow)
            classifier_loss3 = self.mlp3.criterion(outputs3, vector_label)
            classifier_loss = self.alpha * classifier_loss2 + \
                              self.beta * classifier_loss3 + (1 - self.alpha - self.beta) * classifier_loss1
            outputs = self.alpha * outputs2 + self.beta * outputs3 + \
                      (1 - self.alpha - self.beta) * outputs1
            _, pred = torch.max(outputs, 1)
            results = pred.eq_(vector_label)
            acc = torch.sum(results)
            loss = classifier_loss

        elif submodel == "joint_weight":
            loss_vgvae, loss_vgvae_mean, vkl, gkl, rec_logloss, \
            sent1_semantic, sent1_syntax, sent2_semantic, sent2_syntax = \
                self.vgvae(sent1, mask1, sent2, mask2, tgt1,
                           tgt_mask1, tgt2, tgt_mask2,
                           neg_sent1, neg_mask1, ntgt1, ntgt_mask1,
                           neg_sent2, neg_mask2, ntgt2, ntgt_mask2, vtemp,
                           gtemp, use_margin)
            # bias_shadow, bias_shadow1, bias_shadow2, label_shadow, --> neg_sent1, neg_mask1, ntgt1, ntgt_mask1,
            bias_shadow, bias_shadow1, bias_shadow2, label_shadow = \
                self.to_vars(neg_sent1, neg_mask1, ntgt1, ntgt_mask1)
            # shadow
            # biased feature vector
            bias_outputs1 = self.mlp1(bias_shadow)
            bias_outputs2 = self.mlp2(bias_shadow1)
            bias_outputs3 = self.mlp3(bias_shadow2)
            bias_closs1 = self.mlp1.criterion(bias_outputs1, label_shadow)
            bias_closs2 = self.mlp2.criterion(bias_outputs2, label_shadow)
            bias_closs3 = self.mlp3.criterion(bias_outputs3, label_shadow)
            bias_closs = self.alpha * bias_closs2 + self.beta * bias_closs3 + \
                         (1 - self.alpha - self.beta) * bias_closs1
            fd = self.alpha * bias_outputs2 + self.beta * bias_outputs3 + \
                 (1 - self.alpha - self.beta) * bias_outputs1
            value_fdi, pred_fdi = torch.max(fd, 1)

            # debiased feature vector
            vector_shadow_, vector_label_ = self.dp_s.postprocess2(sent1, self.s_label_dict)
            vector_shadow = torch.autograd.Variable(torch.Tensor(vector_shadow_))
            vector_label = torch.autograd.Variable(torch.Tensor(vector_label_).long())
            vector_shadow, vector_label = self.to_vars(vector_shadow, vector_label)
            outputs1 = self.mlp1(vector_shadow)
            outputs2 = self.mlp2(sent1_semantic)
            outputs3 = self.mlp3(sent1_syntax)
            classifier_loss1 = self.mlp1.criterion(outputs1, vector_label)
            classifier_loss2 = self.mlp2.criterion(outputs2, vector_label)
            classifier_loss3 = self.mlp3.criterion(outputs3, vector_label)
            sclassifier_loss = self.alpha * classifier_loss2 + self.beta * classifier_loss3 + \
                              (1 - self.alpha - self.beta) * classifier_loss1
            fd_ = self.alpha * outputs2 + self.beta * outputs3 + \
                  (1 - self.alpha - self.beta) * outputs1
            value_fdi_, pred_fdi_ = torch.max(fd_, 1)
            fd, pred_fdi, fd_, pred_fdi_ = \
                self.to_vars(fd, pred_fdi, fd_, pred_fdi_)

            self.index += self.expe.config.batch_size
            ps = self.ps[0][self.index:self.index+self.expe.config.batch_size]
            pt = self.pt[0][self.index:self.index+self.expe.config.batch_size]
            ps, pt = self.to_vars(ps, pt)

            dist1 = self.reweight.dist(fd, label_shadow)
            dist2 = self.reweight.dist(fd_, vector_label)
            loss_reweight_s = self.lambda_s * torch.pow(((1.0/ps * dist1) - dist2), 2)

            # target
            # debiased feature vector
            vector_target_, _ = self.dp_t.postprocess2(sent2, self.t_label_dict)
            vector_target = torch.autograd.Variable(torch.Tensor(vector_target_))
            # biased feature vector
            bias_target, bias_target1, bias_target2, vector_target = \
                self.to_vars(neg_sent2, neg_mask2, ntgt2, vector_target)
            tbias_outputs1 = self.mlp1(bias_target)
            tbias_outputs2 = self.mlp2(bias_target1)
            tbias_outputs3 = self.mlp3(bias_target2)
            fdt = tbias_outputs = self.alpha * tbias_outputs2 + self.beta * tbias_outputs3 + \
                                 (1 - self.alpha - self.beta) * tbias_outputs1
            predicts, pseudo_label = torch.max(tbias_outputs, 1)

            # debiased feature vector
            tdebias_outputs1 = self.mlp1(vector_target)
            tdebias_outputs2 = self.mlp2(sent2_semantic)
            tdebias_outputs3 = self.mlp3(sent2_syntax)
            fdt_ = tdebias_outputs = self.alpha * tdebias_outputs2 + self.beta * tdebias_outputs3 + \
                                    (1 - self.alpha - self.beta) * tdebias_outputs1
            predicts_, pseudo_label_ = torch.max(tdebias_outputs, 1)


            fdt, pseudo_label, fdt_, pseudo_label_ = \
                self.to_vars(fdt, pseudo_label, fdt_, pseudo_label_)

            tbias_closs1 = self.mlp1.criterion(tbias_outputs1, pseudo_label)
            tbias_closs2 = self.mlp2.criterion(tbias_outputs2, pseudo_label)
            tbias_closs3 = self.mlp3.criterion(tbias_outputs3, pseudo_label)
            tbias_closs = self.alpha * tbias_closs2 + self.beta * tbias_closs3 + \
                          (1 - self.alpha - self.beta) * tbias_closs1

            tclassifier_loss1 = self.mlp1.criterion(tdebias_outputs1, pseudo_label_)
            tclassifier_loss2 = self.mlp2.criterion(tdebias_outputs2, pseudo_label_)
            tclassifier_loss3 = self.mlp3.criterion(tdebias_outputs3, pseudo_label_)
            tclassifier_loss = self.alpha * tclassifier_loss2 + self.beta * tclassifier_loss3 + \
                               (1 - self.alpha - self.beta) * tclassifier_loss1

            dist3 = self.reweight.dist(fdt, pseudo_label)
            dist4 = self.reweight.dist(fdt_, pseudo_label_)
            loss_reweight_t = self.lambda_t * torch.pow(((1.0 / pt * dist3) - dist4), 2)

            loss_reweight = torch.sum(loss_reweight_s + loss_reweight_t)/2

            classifier_loss = (bias_closs+sclassifier_loss+tbias_closs+tclassifier_loss)/4

            loss = loss_reweight  #1/2
            loss.requires_grad_(True)

        elif submodel == "joint":
            loss_vgvae, loss_vgvae_mean, vkl, gkl, rec_logloss, \
            sent1_semantic, sent1_syntax, sent2_semantic, sent2_syntax = \
                self.vgvae(sent1, mask1, sent2, mask2, tgt1,
                           tgt_mask1, tgt2, tgt_mask2,
                           neg_sent1, neg_mask1, ntgt1, ntgt_mask1,
                           neg_sent2, neg_mask2, ntgt2, ntgt_mask2, vtemp,
                           gtemp, use_margin)
            vector_shadow_, vector_label_ = self.dp_s.postprocess2(sent1, self.s_label_dict)
            vector_shadow = torch.autograd.Variable(torch.Tensor(vector_shadow_))
            vector_label = torch.autograd.Variable(torch.Tensor(vector_label_).long())
            vector_shadow, vector_label = self.to_vars(vector_shadow, vector_label)

            s_outputs1 = self.mlp1(vector_shadow)
            s_outputs2 = self.mlp2(sent1_semantic)
            s_outputs3 = self.mlp3(sent1_syntax)
            s_closs1 = self.mlp1.criterion(s_outputs1, vector_label)
            s_closs2 = self.mlp2.criterion(s_outputs2, vector_label)
            s_closs3 = self.mlp3.criterion(s_outputs3, vector_label)
            s_classifier_loss = self.alpha * s_closs2 + self.beta * s_closs3 + \
                                (1 - self.alpha - self.beta) * s_closs1
            outputs = self.alpha * s_outputs2 + self.beta * s_outputs3 + \
                      (1 - self.alpha - self.beta) * s_outputs1
            _, predict = torch.max(outputs, 1)
            s_results = predict.eq_(vector_label)
            s_acc = torch.sum(s_results)

            # target
            vector_target_, target_label_ = self.dp_t.postprocess2(sent2, self.t_label_dict)
            vector_target1 = torch.autograd.Variable(torch.Tensor(vector_target_))
            target_label1 = torch.autograd.Variable(torch.Tensor(target_label_).long())
            vector_target, target_label = self.to_vars(vector_target1, target_label1)

            t_outputs1 = self.mlp1(vector_target)
            t_outputs2 = self.mlp2(sent2_semantic)
            t_outputs3 = self.mlp3(sent2_syntax)
            t_outputs = self.alpha * t_outputs2 + self.beta * t_outputs3 + \
                        (1 - self.alpha - self.beta) * t_outputs1
            _, pseudo_label_ = torch.max(t_outputs, 1)
            t_results = pseudo_label_.eq_(target_label)
            t_acc = torch.sum(t_results)


            # pi = torch.autograd.Variable(torch.squeeze(torch.cat([self.ps, self.pt], 1)))
            pseudo_target_label_ = torch.autograd.Variable(pseudo_label_.long())
            ps = torch.autograd.Variable(self.ps)
            pt = torch.autograd.Variable(self.pt)
            ps, pt, pseudo_target_label = self.to_vars(ps, pt, pseudo_target_label_)

            t_closs1 = self.mlp1.criterion(t_outputs1, pseudo_target_label)
            t_closs2 = self.mlp2.criterion(t_outputs2, pseudo_target_label)
            t_closs3 = self.mlp3.criterion(t_outputs3, pseudo_target_label)
            t_classifier_loss = self.alpha * t_closs2 + self.beta * t_closs3 + \
                                (1 - self.alpha - self.beta) * t_closs1

            classifier_loss = (s_classifier_loss + t_classifier_loss)/2
            acc = s_acc + t_acc

            alpha_ps = self.reweight(ps)
            alpha_pt = self.reweight(pt)
            self.index1 += self.expe.config.batch_size
            ps = alpha_ps[0][self.index1:self.index1 + self.expe.config.batch_size]
            pt = alpha_pt[0][self.index1:self.index1 + self.expe.config.batch_size]
            alpha_p = torch.sum(pt + ps).item() / (pt.shape[0]*2)

            loss = alpha_p * (loss_vgvae_mean + classifier_loss)

        return loss, loss_vgvae_mean, vkl, gkl, rec_logloss, classifier_loss, acc, loss_reweight

    def test(self, sent1, mask1, sent2, mask2, tgt1,
             tgt_mask1, tgt2, tgt_mask2,
             neg_sent1, neg_mask1, ntgt1, ntgt_mask1,
             neg_sent2, neg_mask2, ntgt2, ntgt_mask2, vtemp,
             gtemp, use_margin):
        loss_vgvae, loss_vgvae_mean,vkl, gkl, rec_logloss, \
        sent1_semantic, sent1_syntax, sent2_semantic, sent2_syntax = \
            self.vgvae(sent1, mask1, sent2, mask2, tgt1,
                       tgt_mask1, tgt2, tgt_mask2,
                       neg_sent1, neg_mask1, ntgt1, ntgt_mask1,
                       neg_sent2, neg_mask2, ntgt2, ntgt_mask2, vtemp,
                       gtemp, use_margin)

        vector_target_, vector_label_ = self.dp_t.postprocess2(sent2, self.t_label_dict)
        vector_target = torch.autograd.Variable(torch.Tensor(vector_target_))
        vector_label = torch.autograd.Variable(torch.Tensor(vector_label_).long())
        vector_target, vector_label = self.to_vars(vector_target, vector_label)
        outputs1 = self.mlp1(vector_target)
        outputs2 = self.mlp2(sent2_semantic)
        outputs3 = self.mlp3(sent2_syntax)

        outputs = self.alpha * outputs2 + self.beta * outputs3 + \
                  (1 - self.alpha - self.beta) * outputs1
        return outputs, vector_label

    def midtest(self, sent1, mask1, sent2, mask2, tgt1,
                tgt_mask1, tgt2, tgt_mask2,
                neg_sent1, neg_mask1, ntgt1, ntgt_mask1,
                neg_sent2, neg_mask2, ntgt2, ntgt_mask2, vtemp,
                gtemp, use_margin):
        loss_vgvae, loss_vgvar_mean, vkl, gkl, rec_logloss, \
        sent1_semantic, sent1_syntax, sent2_semantic, sent2_syntax = \
            self.vgvae(sent1, mask1, sent2, mask2, tgt1,
                       tgt_mask1, tgt2, tgt_mask2,
                       neg_sent1, neg_mask1, ntgt1, ntgt_mask1,
                       neg_sent2, neg_mask2, ntgt2, ntgt_mask2, vtemp,
                       gtemp, use_margin)
        vector_shadow, label_shadow = self.dp_s.postprocess2(sent1, self.s_label_dict)
        vector_target, _ = self.dp_t.postprocess2(sent2, self.t_label_dict)

        return vector_shadow, sent1_semantic, sent1_syntax, label_shadow, \
               vector_target, sent2_semantic, sent2_syntax, _
