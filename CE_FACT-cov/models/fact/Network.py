import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
import numpy as np
from torch.distributions import Normal
from utils import *

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100', 'manyshotcifar']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet', 'manyshotmini', 'imagenet100', 'imagenet1000']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True,
                                    args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.pre_allocate = self.args.num_classes
        self.fc = nn.Linear(self.num_features, self.pre_allocate, bias=False)

        nn.init.orthogonal_(self.fc.weight)
        self.dummy_orthogonal_classifier = nn.Linear(self.num_features, self.pre_allocate - self.args.base_class,
                                                     bias=False)
        self.dummy_orthogonal_classifier.weight.requires_grad = False


        if self.args.cov_restriction:
            self.distribution_estimator = nn.Linear(self.num_features, 2 * self.num_features)
        self.dummy_orthogonal_classifier.weight.data = self.fc.weight.data[self.args.base_class:, :]
        print(self.dummy_orthogonal_classifier.weight.data.size())

        print('self.dummy_orthogonal_classifier.weight initialized over.')

    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:

            x1 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x2 = F.linear(F.normalize(x, p=2, dim=-1),
                          F.normalize(self.dummy_orthogonal_classifier.weight, p=2, dim=-1))

            x = torch.cat([x1[:, :self.args.base_class], x2], dim=1)

            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x

    def forpass_fc(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:

            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def pre_encode(self, x):

        if self.args.dataset in ['cifar100', 'manyshotcifar']:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)

        elif self.args.dataset in ['mini_imagenet', 'manyshotmini', 'cub200']:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)

        return x

    def post_encode(self, x):
        if self.args.dataset in ['cifar100', 'manyshotcifar']:

            x = self.encoder.layer3(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)

        elif self.args.dataset in ['mini_imagenet', 'manyshotmini', 'cub200']:

            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)

        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x

        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')


    def update_fc(self, dataloader, class_list, session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data = self.encode(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune

            self.update_fc_ft_training_vae(new_fc, dataloader, session)



    def update_fc_avg(self, data, label, class_list):
        new_fc = []
        for class_index in class_list:
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index] = proto
        new_fc = torch.stack(new_fc, dim=0)
        return new_fc

    def get_logits(self, x, fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x, fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self, new_fc, data, label, session):
        new_fc = new_fc.clone().detach()
        new_fc.requires_grad = True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_new, momentum=0.9, dampening=0.9,
                                    weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data, fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[
        self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(
            new_fc.data)



    def update_fc_ft_training_vae(self, new_fc, dataloader, session):

        new_fc = new_fc.clone().detach()
        new_fc.requires_grad = True

        self.linear_transformation = nn.Linear(self.num_features, 2 * self.num_features).cuda()
        self.linear_transformation.requires_grad = True

        if self.args.dataset == 'mini_imagenet':

            if self.args.cov_restriction:


                optimizer = torch.optim.Adam([{'params': new_fc, 'lr': 0.001 * self.args.lr_new},
                                              {'params': self.encoder.layer4.parameters(),
                                               'lr': 0.0002 * self.args.lr_new},
                                              {'params': self.linear_transformation.parameters(),
                                               'lr':self.args.lr_new}], weight_decay=0)

            else:

                optimizer = torch.optim.SGD([{'params': new_fc, 'lr': self.args.lr_new},
                                             {'params': self.encoder.layer4.parameters(),
                                              'lr': 0.001 * self.args.lr_new},
                                             {'params': self.linear_transformation.parameters(),
                                              'lr': self.args.lr_new}],
                                            momentum=0.9, dampening=0.9, weight_decay=0)


        elif self.args.dataset == 'cifar100':

            if self.args.cov_restriction:

                optimizer = torch.optim.SGD([
                                            # {'params': new_fc, 'lr': 0.0001 * self.args.lr_new},
                    {'params': new_fc, 'lr': 0.0001 * self.args.lr_new},
                                             {'params': self.encoder.layer3.parameters(),
                                              'lr': 0.001 * self.args.lr_new},
                                             {'params': self.linear_transformation.parameters(),
                                              'lr': self.args.lr_new}],
                                            momentum=0.9, dampening=0.9, weight_decay=0)
            else:

                optimizer = torch.optim.SGD([{'params': new_fc, 'lr': 0.01 * self.args.lr_new},
                                             {'params': self.encoder.layer3.parameters(),
                                              'lr': 0.001 * self.args.lr_new},
                                             {'params': self.linear_transformation.parameters(),
                                              'lr': self.args.lr_new}],
                                            momentum=0.9, dampening=0.9, weight_decay=0)


        else:
            if self.args.cov_restriction:

                optimizer = torch.optim.SGD([{'params': new_fc, 'lr': self.args.lr_new},
                                             {'params': self.encoder.layer4.parameters(),
                                              'lr': 0.004 * self.args.lr_new},
                                             {'params': self.linear_transformation.parameters(),
                                              'lr': self.args.lr_new}],
                                            momentum=0.9, dampening=0.9, weight_decay=0)
            else:

                optimizer = torch.optim.SGD([{'params': new_fc, 'lr': self.args.lr_new},
                                             {'params': self.encoder.layer4.parameters(),
                                              'lr': 0.01 * self.args.lr_new},
                                             {'params': self.linear_transformation.parameters(),
                                              'lr': self.args.lr_new}],
                                            momentum=0.9, dampening=0.9, weight_decay=0)


        old_class = self.args.base_class + self.args.way * (session - 1)
        all_class = self.args.base_class + self.args.way * session
        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                for batch in dataloader:
                    all_data, label = [_ for _ in batch]
                    data = all_data.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)

                    feature = self.encode(data)
                    logits = self.get_logits(feature, fc)
                    loss_clean_ce = F.cross_entropy(logits, label)

                    latent_distribution = self.linear_transformation(feature)
                    mu, logsigma = latent_distribution.chunk(2, dim=-1)

                    selected_vector = torch.zeros_like(feature).cuda()
                    for i in range(label.size(0)):
                        mask = torch.ones(all_class, dtype=torch.bool).cuda()
                        mask[label[i]] = False
                        selected_fc = fc[mask]
                        dist = self.get_logits(feature[i], selected_fc)
                        score = F.softmax(dist, dim=-1)
                        selected_vector[i] += torch.matmul(score, selected_fc)

                    loss_kl = -0.5 * torch.sum(1 + logsigma - (1 / self.args.a) * (logsigma.exp()) - torch.log(torch.tensor(self.args.a)) - (1 / self.args.a) * ((mu - selected_vector).pow(2)))
                    loss_kl =loss_kl/logsigma.size(0)

                    feature_trans = self.reparameterize(feature, mu, logsigma)
                    logits_trans = self.get_logits(feature_trans, fc)
                    loss_trans_ce = F.cross_entropy(logits_trans, label)

                    loss = loss_clean_ce + loss_trans_ce + loss_kl* self.args.incremental_cov_balance
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    pass

        self.fc.weight.data[
        self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(
            new_fc.data)



    def reparameterize(self, feature, mu, logvar):

        std = torch.exp(0.5 * logvar).cuda()

        return  mu + std * feature

