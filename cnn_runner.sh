#! /bin/bash

dim="100"
windows="4 12 20"
strategies="no_weight pmi pmi_all"
datasets="polarity webkb r8 20ng"
cuts="5 10 20"
pooling="global_max"

for d in $dim; do
	for w in $windows; do
		for dr in $datasets; do
			for cp in $cuts; do
				for s in $strategies; do
					for p in $pooling; do
						python3 cnn_main.py --dataset $dr --cut_percent $cp --strategy $s --window $w --emb_dim $d --pool_type $p &
					done
				done
			done
		done
	done
	wait
done
