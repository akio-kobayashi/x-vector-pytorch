#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:22:26 2020

@author: krishna
"""



import torch
import numpy as np
from torch.utils.data import DataLoader
from generator import SpeechDataSet
import generator
import torch.nn as nn
import os
import numpy as np
from torch import optim
import argparse
from models.x_vector import X_vector
from sklearn.metrics import accuracy_score
from utils.utils import speech_collate
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
from adacos import AdaCos

########## Argument parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--data', type=str)
parser.add_argument('--train-keys',type=str)
parser.add_argument('--valid-keys',type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--input-dim', type=int, default=60)
parser.add_argument('--classes', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=256)
#parser.add_argument('--use_gpu', action="store_true", default=True)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--crop', type=int, default=0)
parser.add_argument('--mtl', type=int, default=0, help="number of MTL classes")
parser.add_argument('--weight', type=float, default=0.5, help="weight for MTL objective")
parser.add_argument('--log', type=str, default="log.txt")

args = parser.parse_args()

### Data related
train_dataset=SpeechDataset(args.data, keypath=args.train_keys, crop=args.crop)
train_loader =data.DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=lambda x: generator.data_processing(x,'train'),**kwargs)
valid_dataset=SpeechDataset(args.data, keypath=args.valid_keys)
valid_loader=data.DataLoader(dataset=valid_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             collate_fn=lambda x: generator.data_processing(x, 'valid'),**kwargs)
eval_dataset=SpeechDataset(args.data, keypath=args.eval_keys)
eval_loader=data.DataLoader(dataset=eval_dataset,
                            batch_size=1, shuffle=False,
                            collate_fn=lambda x: generator.data_processing(x, 'eval'),**kwargs)

## Model related
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = X_vector(args.input_dim, args.classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
#
#loss_fun = nn.CrossEntropyLoss()
criterion = AdaCos(512, 2)
if args.mtl > 0:
    criterion_mtl = AdaCos(512, args.mtl)


def train(train_loader,epoch):
    train_loss_list=[]
    full_preds=[]
    full_gts=[]
    model.train()
    for i_batch, sample_batched in enumerate(train_loader):
        features, labels, speakers, _, _=sample_batched
        features, labels = features.to(device),labels.to(device)
        features.requires_grad = True
        optimizer.zero_grad()
        if args.mtl == 0:
            pred_logits,x_vec = model(features)
            loss = criterion(pred_logits,labels)
        else:
            pred_logits, pred_logits_mtl, x_vec = model(features)
            loss = (1-args.weight) * criterion(pred_logits, labels)
            + args.weight * criterion_mtl(pred_logits_mtl, speakers)

        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())

        '''
        predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
        for pred in predictions:
            full_preds.append(pred)
        for lab in labels.detach().cpu().numpy():
            full_gts.append(lab)
        '''

    mean_acc = accuracy_score(full_gts,full_preds)
    mean_loss = np.mean(np.asarray(train_loss_list))
    print('Total training loss {} and training Accuracy {} after {} epochs'.format(mean_loss,mean_acc,epoch))

    return mean_acc, mean_loss


def validation(loader,epoch):
    model.eval()
    with torch.no_grad():
        val_loss_list=[]
        full_preds=[]
        full_gts=[]
        for i_batch, sample_batched in enumerate(loader):
            features,labels, speakers, input_lengths, _ = sample_batched
            features, labels = features.to(device),labels.to(device)
            if args.mtl == 0:
                pred_logits,x_vec = model(features)
                loss = criterion(pred_logits,labels)
            else:
                pred_logits, pred_logits_mtl, x_vec = model(features)
                loss = (1-args.weight) * criterion(pred_logits, labels)
                + args.weight * criterion_mtl(pred_logits_mtl, speakers)

            val_loss_list.append(loss.item())
            predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)

        mean_acc = accuracy_score(full_gts,full_preds)
        mean_loss = np.mean(np.asarray(val_loss_list))
        print('Total validation loss {} and Validation accuracy {} after {} epochs'.format(mean_loss,mean_acc,epoch))

        return mean_acc, mean_loss

if __name__ == '__main__':
    max_acc=0.
    with open(args.log, 'w') as wf:
        for epoch in range(args.epochs):
            train_acc, train_loss = train(trai_loadern,epoch)
            valid_acc, valid_loss = validation(valid_loader,epoch)
            if valid_acc > max_acc:
                max_acc = valid_acc
                mess = 'Maximum validation ACC changed at {} : {}\n'.format(epoch, max_acc)
                wf.write(mess)
                print('Maximum validation ACC changed to %.3f' % max_acc)
                print('Saving model to %s' % args.output)
                torch.save(model.to('cpu').state_dict(), args.output)
                model.to(device)
        # final evaluation
        model.load_state_dict(torch.load(args.output, map_location=torch.device('cpu')))
        model = model.to(device)
        eval_acc, eval_loss = validation(eval_loader, epoch)
        mess = 'Evaluation ACC at {} : {}\n'.format(epoch, eval_acc)
        print('Evaluation ACC: %.3f' % eval_acc)
        wf.write(mess)
