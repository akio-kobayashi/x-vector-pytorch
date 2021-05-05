import numpy as np
import sys, os, re, gzip, struct
import random
import h5py
import torch
import torch.nn as nn
#import seq2seq_generator
#from specAugment import spec_augment_pytorch

def speaker_map(h5fd):
    spk2id={}
    id2spk={}
    spk2id['<unk>']=0
    id2spk[0]='<unk>'

    num=0
    keys = h5fd.keys()
    for key in keys:
        spk=h5fd[key+'/speaker'][()]
        if spk not in spk2id:
            spk2id[spk]=num
            id2spk[num]=spk
            num+=1
    return spk2id, id2spk

def compute_norm(h5fd):
    keys=h5fd.keys()
    rows=0
    mean=None
    var=None

    for key in keys:
        if key == 'mean':
            continue
        if key == 'var':
            continue
        mat = h5fd[key+'/data'][()]
        rows += mat.shape[0]
        if mean is None:
            mean=np.sum(mat, axis=0).astype(np.float64)
            var=np.sum(np.square(mat), axis=0).astype(np.float64)
        else:
            mean=np.add(np.sum(mat, axis=0).astype(np.float64), mean)
            var=np.add(np.sum(np.square(mat), axis=0).astype(np.float64), var)

    mean = mean/rows
    var = np.sqrt(var/rows - np.square(mean))

    return mean, var

class SpeechDataset(torch.utils.data.Dataset):

    def __init__(self, path, keypath=None, crop=0, stats=None, speakers=None, train=True):
        super(SpeechDataset, self).__init__()

        #self.augment=augment
        self.train=train
        self.crop=crop

        self.h5fd = h5py.File(path, 'r')
        if stats is None:
            self.mean, self.var = compute_norm(self.h5fd)
        else:
            self.mean, self.var = stats

        self.keys=[]

        if keypath is not None:
            with open(keypath,'r') as f:
                lines=f.readlines()
                for l in lines:
                    self.keys.append(l.strip())
        if speakers is None:
            self.spk2id, self.id2spk = speaker_map(self.h5fd)
        else:
            self.spk2id, self.id2spk = speakers

    def get_stats(self):
        return self.mean, self.var

    def get_speakers(self):
        return self.spk2id, self.id2spk

    def get_keys(self):
        return self.keys

    def __len__(self):
        return len(self.keys)

    def input_size(self):
        mat = self.h5fd[self.keys[0]+'/data'][()]
        return mat.shape[1]

    def get_data(self, keys):
        data=[]
        for key in keys:
            dt = self.__getitem__(self.keys.index(key))
            data.append(dt)
        _data=data_processing(data)
        return _data

    def num_speakers(self):
        return (len(self.spk2id))

    def __getitem__(self, idx):
        # (time, feature)
        input=self.h5fd[self.keys[idx]+'/data'][()]

        # randomly crop
        if self.crop > 0:
            max_len = input.shape[0]
            start=random.randint(0, max_len-self.crop)
            mat = input[start:start+self.crop, :]
            input=mat

        '''
        TODO specaugment
        if self.augment:
            x = torch.from_numpy(np.expand_dims(np.transpose(input),axis=0).astype(np.float32)).clone()
            input=spec_augment_pytorch.spec_augment(x,frequency_masking_para=8)
            input = x.to('cpu').detach().numpy().copy()
            input=np.transpose(np.squeeze(input))
            aug_mask=input==0.0
            input -= self.mean
            input /= self.var
            input[aug_mask]=0.0 # masked feature value = (0,0, 1.0)
        else:
            input -= self.mean
            input /= self.var
        '''

        label=self.h5fd[self.keys[idx]+'/label'][()]
        spk=self.spk2id[self.h5fd[self.keys[idx]+'/speaker'][()]]

        return input, label, spk, self.keys[idx]


def data_processing(data, data_type="train"):
    inputs = []
    labels = []
    speakers = []
    lengths=[]
    keys = []

    for input, label, spk, key in data:
        """ inputs : (batch, time, feature) """
        # w/o channel
        inputs.append(torch.from_numpy(input.astype(np.float32)).clone())
        labels.append(torch.from_numpy(label.astype(np.int)).clone())
        speakers.append(torch.from_numpy(speaker.astype(np.int)).clone())
        lengths.append(input.shape[0])
        keys.append(key)

    inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    #labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    labels=torch.from_numpy(labels.astype(np.int)).clone()
    speakers=torch.from_numpy(speakers.astype(np.float32)).clone()
    #speakers = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return inputs, labels, speakers, lengths, keys
