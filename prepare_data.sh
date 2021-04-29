#!/bin/sh

. parse_options.sh

# wav.scp
rootdir=/media/akio/hdd1/kaldi/egs/deaf-ivector/data60/
python3 make_mfcc_feats.py --feats $rootdir/all/feats.scp \
	--output $rootdir/all/

feats_len=$rootdir/all/feats.len
for minlen in 100 200 400;do
    for spk in F001 F002 F003 M001 M002 M003 M004 M005 M006 M007 M008 M009;
    do
	datadir=$rootdir/${spk}/${minlen}/
	for cond in train valid eval; do
	    cat $datadir/$cond/feats.scp |
		perl -lane "print @F[0]" > $datadir/$cond/${cond}.keys
	done
    done
done
