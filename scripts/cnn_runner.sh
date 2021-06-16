#! /bin/bash

dim="100"
windows="4 12 20"
strategies="pmi pmi_all"
datasets="20ng"
cuts="5 10 20"

for d in $dim; do
	for w in $windows; do
		for dr in $datasets; do
			for cp in $cuts; do
				for s in $strategies; do
					python3 cnn_main.py --dataset $dr --cut_percent $cp --strategy $s --window $w --emb_dim $d &
				done
				wait
			done
		done
	done
done

dim="100"
windows="4 12 20"
strategies="llr llr_all"
datasets="20ng"
cuts="5 10 20"

for d in $dim; do
	for w in $windows; do
		for dr in $datasets; do
			for cp in $cuts; do
				for s in $strategies; do
					python3 cnn_main.py --dataset $dr --cut_percent $cp --strategy $s --window $w --emb_dim $d &
				done
				wait
			done
		done
	done
done

dim="100"
windows="4 12 20"
strategies="chi_square chi_square_all"
datasets="20ng"
cuts="5 10 20"

for d in $dim; do
	for w in $windows; do
		for dr in $datasets; do
			for cp in $cuts; do
				for s in $strategies; do
					python3 cnn_main.py --dataset $dr --cut_percent $cp --strategy $s --window $w --emb_dim $d &
				done
				wait
			done
		done
	done
done
