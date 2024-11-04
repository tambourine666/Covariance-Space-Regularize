# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F

from losses import SupContrastive

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def base_train(model, trainloader, criterion, optimizer, scheduler, epoch, transform, args):
    tl = Averager()
    tl_joint = Averager()
    tl_moco = Averager()
    tl_moco_global = Averager()
    tl_moco_small = Averager()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, single_labels = [_ for _ in batch]
        b, c, h, w = data[1].shape
        original = data[0].cuda(non_blocking=True)
        data[1] = data[1].cuda(non_blocking=True)
        data[2] = data[2].cuda(non_blocking=True)
        single_labels = single_labels.cuda(non_blocking=True)
        if len(args.num_crops) > 1:
            data_small = data[args.num_crops[0] + 1].unsqueeze(1)
            for j in range(1, args.num_crops[1]):
                data_small = torch.cat((data_small, data[j + args.num_crops[0] + 1].unsqueeze(1)), dim=1)
            data_small = data_small.view(-1, c, args.size_crops[1], args.size_crops[1]).cuda(non_blocking=True)
        else:
            data_small = None

        data_classify = transform(original)
        data_query = transform(data[1])
        data_key = transform(data[2])
        data_small = transform(data_small)
        m = data_query.size()[0] // b
        joint_labels = torch.stack([single_labels * m + ii for ii in range(m)], 1).view(-1)

        joint_preds, output_global, output_small, target_global, target_small = model(im_cla=data_classify,
                                                                                      im_q=data_query, im_k=data_key,
                                                                                      labels=joint_labels,
                                                                                      im_q_small=data_small)

        loss_moco_global = criterion(output_global, target_global)
        loss_moco_small = criterion(output_small, target_small)
        loss_moco = args.alpha * loss_moco_global + args.beta * loss_moco_small

        joint_preds = joint_preds[:, :args.base_class * m]
        joint_loss = F.cross_entropy(joint_preds, joint_labels)

        agg_preds = 0
        for i in range(m):
            agg_preds = agg_preds + joint_preds[i::m, i::m] / m

        if args.cov_restriction:
            embeddings, _ = model.encode_q(data_classify)
            latent_distribution = model.distribution_estimator(embeddings)
            mu, sigma = latent_distribution.chunk(2, dim=-1)
            cov_loss = -0.5 * torch.sum(1 + sigma - (1 / args.a) * (sigma.exp()) - torch.log(torch.tensor(args.a)))
            cov_loss = cov_loss / sigma.size(0)
        else:
            cov_loss = 0

        total_loss = joint_loss + loss_moco + args.cov_balance * cov_loss


        acc = count_acc(agg_preds, single_labels)
        # print(cov_loss.item())
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        tl_joint.add(joint_loss.item())
        tl_moco_global.add(loss_moco_global.item())
        tl_moco_small.add(loss_moco_small.item())
        tl_moco.add(loss_moco.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        #
        if args.dataset == 'cifar100':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    tl_joint = tl_joint.item()
    tl_moco = tl_moco.item()
    tl_moco_global = tl_moco_global.item()
    tl_moco_small = tl_moco_small.item()
    return tl, tl_joint, tl_moco, tl_moco_global, tl_moco_small, ta




def replace_base_fc(trainset, test_transform, data_transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = test_transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = data_transform(data)
            m = data.size()[0] // b
            labels = torch.stack([label * m + ii for ii in range(m)], 1).view(-1)
            model.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(labels.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class * m):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.fc.weight.data[:args.base_class * m] = proto_list

    return model



def update_fc_ft_vae(trainloader, data_transform, model, m, session, args):
    # incremental finetuning
    old_class = args.base_class + args.way * (session - 1)
    new_class = args.base_class + args.way * session
    new_fc = nn.Parameter(torch.rand(args.way * m, model.num_features, device="cuda"), requires_grad=True)
    new_fc.data.copy_(model.fc.weight[old_class * m: new_class * m, :].data)

    linear_transformation = nn.Linear(model.num_features, 2 * model.num_features).cuda()
    # linear_transformation = nn.Sequential(nn.Linear(model.num_features, 4 * model.num_features),nn.ReLU(), nn.Linear(model.num_features*4, 2 * model.num_features)).cuda()
    linear_transformation.requires_grad = True

    if args.dataset == 'mini_imagenet':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
                                     {'params': model.encoder_q.fc.parameters(), 'lr': 0.05 * args.lr_new},
                                     {'params': model.encoder_q.layer4.parameters(), 'lr': 0.002 * args.lr_new},
                                     {'params': linear_transformation.parameters(), 'lr': 0.1 * args.lr_new}],
                                    momentum=0.9, dampening=0.9, weight_decay=0)

    if args.dataset == 'cub200':
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
                                     {'params': model.encoder_q.fc.parameters(), 'lr': 10 * args.lr_new},
                                     {'params': model.encoder_q.layer4.parameters(), 'lr': 2 * args.lr_new},
                                     {'params': linear_transformation.parameters(), 'lr': 100 * args.lr_new}],
                                    momentum=0.9, dampening=0.9, weight_decay=0)




    elif args.dataset == 'cifar100':
        optimizer = torch.optim.Adam([{'params': new_fc, 'lr': args.lr_new},
                                      {'params': model.encoder_q.fc.parameters(), 'lr': 0.05 * args.lr_new},
                                      {'params': model.encoder_q.layer3.parameters(), 'lr': 0.005 * args.lr_new},
                                      {'params': linear_transformation.parameters(), 'lr': args.lr_new}
                                      ],
                                     weight_decay=0)



    criterion = SupContrastive().cuda()
    with torch.enable_grad():
        for epoch in range(args.epochs_new):
            for batch in trainloader:
                data, single_labels = [_ for _ in batch]
                b, c, h, w = data[1].shape
                origin = data[0].cuda(non_blocking=True)
                data[1] = data[1].cuda(non_blocking=True)
                data[2] = data[2].cuda(non_blocking=True)
                single_labels = single_labels.cuda(non_blocking=True)
                if len(args.num_crops) > 1:
                    data_small = data[args.num_crops[0] + 1].unsqueeze(1)
                    for j in range(1, args.num_crops[1]):
                        data_small = torch.cat((data_small, data[j + args.num_crops[0] + 1].unsqueeze(1)), dim=1)
                    data_small = data_small.view(-1, c, args.size_crops[1], args.size_crops[1]).cuda(non_blocking=True)
                else:
                    data_small = None
            data_classify = data_transform(origin)
            data_query = data_transform(data[1])
            data_key = data_transform(data[2])
            data_small = data_transform(data_small)
            joint_labels = torch.stack([single_labels * m + ii for ii in range(m)], 1).view(-1)

            old_fc = model.fc.weight[:old_class * m, :].clone().detach()
            fc = torch.cat([old_fc, new_fc], dim=0)

            features, _ = model.encode_q(data_classify)
            features.detach()
            logits = model.get_logits(features, fc)
            joint_loss = F.cross_entropy(logits, joint_labels)

            latent_distribution = linear_transformation(features)
            mu, logsigma = latent_distribution.chunk(2, dim=-1)

            pd_logits = logits
            label_one_hot = 1 - F.one_hot(joint_labels, num_classes=fc.size(0))
            pd_logits = pd_logits * label_one_hot
            prediction = F.softmax(pd_logits, dim=-1)
            selected_vector = torch.matmul(prediction, fc)

            loss_kl = -0.5 * torch.sum(
                1 + logsigma - (1 / args.a) * (logsigma.exp()) - torch.log(torch.tensor(args.a)) - (1 / args.a) * (
                    (mu - selected_vector).pow(2)))

            loss_kl = loss_kl / logsigma.size(0)
            feature_trans = reparameterize(features, mu, logsigma)
            logits_trans = model.get_logits(feature_trans, fc)
            loss_trans = F.cross_entropy(logits_trans, joint_labels)

            _, output_global, output_small, target_global, target_small = model(im_cla=data_classify, im_q=data_query,
                                                                                im_k=data_key, labels=joint_labels,
                                                                                im_q_small=data_small, base_sess=False,
                                                                                last_epochs_new=(
                                                                                            epoch == args.epochs_new - 1))
            loss_moco_global = criterion(output_global, target_global)
            loss_moco_small = criterion(output_small, target_small)
            loss_moco = args.alpha * loss_moco_global + args.beta * loss_moco_small

            cls_loss = joint_loss + loss_moco

            if args.dataset == 'cub200':
                loss = cls_loss + loss_trans + loss_kl * args.incremental_cov_balance
            else:
                loss = cls_loss + (loss_trans + loss_kl) * args.incremental_cov_balance


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.fc.weight.data[old_class * m: new_class * m, :].copy_(new_fc.data)


def test(model, testloader, epoch, transform, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = transform(data)
            m = data.size()[0] // b
            joint_preds = model(data)
            joint_preds = joint_preds[:, :test_class * m]

            agg_preds = 0
            for j in range(m):
                agg_preds = agg_preds + joint_preds[j::m, j::m] / m

            loss = F.cross_entropy(agg_preds, test_label)
            acc = count_acc(agg_preds, test_label)

            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va


def reparameterize(feature, mu, logvar):
    # std = torch.sqrt(torch.exp(logvar)).cuda()
    std = torch.exp(0.5 * logvar).cuda()
    # eps = torch.rand_like(std).cuda()
    eps = torch.randn_like(std).cuda()
    return mu + std * feature


def test_incremental(model, testloader, epoch, transform, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()

    acc_novel = []
    class_sample_count = {}
    class_correct_count = {}

    all_pred, all_label = [], []

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = transform(data)
            m = data.size()[0] // b
            joint_preds = model(data)
            joint_preds = joint_preds[:, :test_class * m]

            agg_preds = 0
            for j in range(m):
                agg_preds = agg_preds + joint_preds[j::m, j::m] / m

            prediction = agg_preds.argmax(dim=1)
            correct = prediction.eq(test_label).float()

            for l in test_label:
                if l.item() in class_sample_count:
                    class_sample_count[l.item()] += 1
                else:
                    class_sample_count[l.item()] = 1

            for l, c in zip(test_label, correct):
                if l.item() in class_correct_count:
                    class_correct_count[l.item()] += c.item()
                else:
                    class_correct_count[l.item()] = c.item()

            all_pred.append(prediction.cpu().numpy())
            all_label.append(test_label.cpu().numpy())
            loss = F.cross_entropy(agg_preds, test_label)
            acc = count_acc(agg_preds, test_label)

            vl.add(loss.item())
            va.add(acc)

        acc_up2now = []
        for i in range(session + 1):
            if i == 0:
                classes = np.arange(args.num_classes)[:args.base_class]
            else:
                classes = np.arange(args.num_classes)[
                          (args.base_class + (i - 1) * args.way):(args.base_class + i * args.way)]
            Acc_Each_Session = caculate_session_acc(classes, class_sample_count, class_correct_count)
            acc_up2now.append("{:.2f}%".format(Acc_Each_Session))

        if session > 0:
            novel_classes_so_far = np.arange(args.num_classes)[args.base_class:(args.base_class + session * args.way)]
            Acc_All_Novel = caculate_session_acc(novel_classes_so_far, class_sample_count, class_correct_count)
            Acc_All_Base = caculate_session_acc(np.arange(args.num_classes)[:args.base_class], class_sample_count,
                                                class_correct_count)
            acc_novel.append("{:.2f}%".format(Acc_All_Novel))
        print(f'{acc_up2now} Current Avg Acc:{"{:.2f}%".format(acc)} Novel classes Avg Acc:{acc_novel}\n')

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va, acc_up2now, Acc_All_Novel, Acc_All_Base


def caculate_session_acc(classes, class_sample_count, class_correct_count):
    test_data_num, correct_data_num = 0, 0
    for itm in classes:
        test_data_num += class_sample_count[itm]
        correct_data_num += class_correct_count[itm]

    return (correct_data_num / test_data_num) * 100


