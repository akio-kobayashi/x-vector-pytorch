#!/bin/sh

minlen=200

. parse_options.sh

# wav.scp
rootdir=/media/akio/hdd1/kaldi/egs/deaf-ivector/data60/
python3 make_mfcc_feats.py --feats $rootdir/all/feats.scp \
	--output $rootdir/all/

feats_len=$rootdir/all/feats.len
for spk in F001 F002 F003 M001 M002 M003 M004 M005 M006 M007 M008 M009;
do
    deaf_root=$rootdir/${spk}/
    tgtdir=$deaf_root/$minlen/
    if [ ! -e $tgtdir ];then
	mkdir -p $tgtdir
    fi
    for cond in train valid eval; do
	cat $deaf_root/$spk/$cond/feats.scp | \
	    perl pickup.pl $feats_len $minlen > $tgtdir/${cond}.keys
    done
done
