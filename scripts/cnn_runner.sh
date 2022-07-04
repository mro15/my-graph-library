#! /bin/bash


dim="100"
windows="12"
strategies="pmi pmi_all"
datasets="r8 webkb polarity"
cuts="5 10 20 30 50 70 80 90"

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
windows="12"
strategies="llr llr_all"
datasets="r8 webkb polarity"
cuts="5 10 20 30 50 70 80 90"

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
windows="12"
strategies="chi_square chi_square_all"
datasets="r8 webkb polarity"
cuts="5 10 20 30 50 70 80 90"

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
