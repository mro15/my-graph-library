#! /bin/bash

datasets="polarity webkb r8"
windows="12"

for dt in $datasets; do
	for w in $windows; do
        python f1_graphics_emnlp_2022.py --dataset $dt --emb_dim 100 --window $w
	done
done
