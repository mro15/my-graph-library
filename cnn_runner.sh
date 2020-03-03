#! /bin/bash

dim="100"
#dim="100 300"
windows="4 7 20 12"
#strategies="no_weight pmi_2019 pmi_1990 dice llr chi_square"
strategies="no_weight pmi_1990 pmi_1990_all"
datasets="20ng ohsumed webkb polarity"
methods="node2vec"
pooling="global_max"

for d in $dim; do
	for w in $windows; do
		for dr in $datasets; do
			for m in $methods; do
				for s in $strategies; do
					for p in $pooling; do
						python3 cnn_main.py --dataset $dr --method $m --strategy $s --window $w --emb_dim $d --pool_type $p > logs/$dr/$m/$s.$w.$d.$p.txt
					done
				done
			done
		done
	done
done
