#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:15:47 2020

@author: krishna
"""

import os
import numpy as np
from utils import utils
import h5py
import argparse

def extract_features(audio_filepath):
    features = utils.feature_extraction(audio_filepath)
    return features
    
    

def FE_pipeline(wav_list,h5fd):
    
    for row in wav_list:
        filepath = row.split(' ')[1]
        tag = row.split(' ')[0]
        if not os.path.exists(create_folders):
            os.makedirs(create_folders)
        extract_feats = extract_features(filepath)
        if '_DT' in tag:
            label=0
        else:
            label=1 # deaf
        h5fd.create_group(tag)
        h5fd.create_dataset(tag+'/data', data=extract_feats,
                            compression='gzip', compression_opts=9)
        h5fd.create_dataset(tag+'/label')
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=str, required=True, help='wav.scp')
    parser.add_argument('--output', type=str, required=Tre, help='output hdf5')
    args = parser.parse_args()

    with h5py.File(args.output, 'w') as h5fd:
        lines = [line.rstrip('\n') for line in open(args.list)]
        FE_pipeline(lines,h5fd)
