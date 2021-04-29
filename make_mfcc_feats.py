import os
import numpy as np
import argparse
import kaldi_io_py
import subprocess
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--feats', type=str, required=True, help='input feature (feats.scp)')
parser.add_argument('-o', '--outdir', type=str, required=True, help='output directory')
args = parser.parse_args()

with h5py.File(args.output, 'w') as w:
    speakers={}
    spk_num=1 # 0 for <unk>

    spk_file=os.path.join(args.outdir,'speakers')
    
    with open(spk_file, 'w') as wf:
        with open(args.feats, 'r') as f:
            lines=f.readlines()
            for line in lines:
                spk = re.sub('\S\S\S$', "", re.sub('_DT',"",line.split()[0]))
                if spk in speakers:
                    speakers[spk]=spk_num
                    out="{0} {1}\n".format(spk, spk_num)
                    wf.write(out)
                    spk_num+=1

    hdf_file=os.path.join(args.outdir,'data.h5')
    generator=kaldi_io_py.read_mat_scp(args.feats)

    with h5py.File(hdf_file, 'w') as hdf:
        for key, mat in generator:
            label = self.textset[key]
            hdf.create_group(key)
            hdf.create_dataset(key+'/data', data=mat, compression='gzip', compression_opts=9)
            if '_DT' in key:
                label=0
            else:
                label=1 # 1 if speaker is deaf
            spk=re.sub('\S\S\S$', "", re.sub('_DT',"", key))
            spk_label=speakers[$spk]
            hdf.create_dataset(key+'/label', data=label)
            hdf.create_dataset(key+'/speaker',data=spk_label)
        
