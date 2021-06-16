#! /bin/bash

dim="100"
windows="4 12 20"
strategies="no_weight"
datasets="polarity webkb r8"
cuts="0"

for d in $dim; do
	for w in $windows; do
		for dr in $datasets; do
			for cp in $cuts; do
				for s in $strategies; do
					python3 cnn_main.py --dataset $dr --cut_percent $cp --strategy $s --window $w --emb_dim $d &
				done
			done
		done
		wait
	done
done

datasets="20ng"

for d in $dim; do
	for w in $windows; do
		for dr in $datasets; do
			for cp in $cuts; do
				for s in $strategies; do
					python3 cnn_main.py --dataset $dr --cut_percent $cp --strategy $s --window $w --emb_dim $d
				done
			done
		done
	done
done
