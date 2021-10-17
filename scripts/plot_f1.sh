#! /bin/bash

datasets="polarity webkb r8 20ng"
windows="4 12 20"
cuts="5 10 20"

for dt in $datasets; do
	for w in $windows; do
		for ct in $cuts; do
            python results_main.py --dataset $dt --emb_dim 100 --window $w --cut_percent $ct
		done
	done
done
