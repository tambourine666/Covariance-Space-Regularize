# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda(non_blocking=True) for _ in batch]

        logits = model(data)
        logits = logits[:, :args.base_class]
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)

        if args.cov_restriction:
            embeddings = model.module.encode(data)
            latent_distribution = model.module.distribution_estimator(embeddings)
            mu, sigma = latent_distribution.chunk(2, dim=-1)
            cov_loss = (-0.5 * torch.sum(1 + sigma - (1 / args.a) * (sigma.exp()) - torch.log(torch.tensor(args.a))))/sigma.size(0)
            total_loss = loss + args.cov_balance * cov_loss
        else:
            total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description('Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model


def test(model, testloader, epoch, args, session, validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            top5acc = count_acc_topk(logits, test_label)

            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)

            lgt = torch.cat([lgt, logits.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])
        vl = vl.item()
        va = va.item()
        va5 = va5.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}'.format(epoch, vl, va, va5))

        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)
        if validation is not True:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm = confmatrix(lgt, lbs, save_model_dir)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            print('Seen Acc:', seenac, 'Unseen ACC:', unseenac)
    return vl, va


def test_withfc(model, testloader, epoch, args, session, validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])

    acc_novel = []
    class_sample_count = {}
    class_correct_count = {}

    all_pred, all_label = [], []

    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]

            logits = model.module.forward_metric(data)

            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)

            prediction = logits.argmax(dim=1)

            correct = prediction.eq(test_label).float()

            accuracy = correct.sum(0).mul_(100.0 / test_label.size(0))

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

            acc = count_acc(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
            lgt = torch.cat([lgt, logits.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])
        vl = vl.item()
        va = va.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

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

        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)
        if validation is not True:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm = confmatrix(lgt, lbs, save_model_dir)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            print('Seen Acc:', seenac, 'Unseen ACC:', unseenac)

        all_pred = np.concatenate(all_pred)
        all_label = np.concatenate(all_label)

        confusion_mat = confusion_matrix(all_label, all_pred, labels=range(len(class_sample_count)))

    return vl, va, acc_up2now, Acc_All_Novel, Acc_All_Base, confusion_mat


def caculate_session_acc(classes, class_sample_count, class_correct_count):
    test_data_num, correct_data_num = 0, 0
    for itm in classes:
        test_data_num += class_sample_count[itm]
        correct_data_num += class_correct_count[itm]

    return (correct_data_num / test_data_num) * 100


