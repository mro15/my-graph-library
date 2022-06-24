#! /bin/bash


dim="100"
windows="12"
strategies="pmi pmi_all"
datasets="r8 webkb polarity"
methods="node2vec"
cuts="5 10 20 30 50 70 80 90"
time_interval="60"

for d in $dim; do
	for w in $windows; do
		for dr in $datasets; do
			for cp in $cuts; do
				for s in $strategies; do
					mprof run --output time_mem/mprofile.$dr.$s.$w.$cp.dat --interval $time_interval feature_generator.py --dataset $dr --strategy $s --window $w --emb_dim $d  --cut_percent $cp &
					echo "python3 feature_generator.py --dataset $dr --strategy $s --window $w --emb_dim $d --cut_percent $cp" >> executed_features_new_measures.txt &
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
methods="node2vec"
cuts="5 10 20 30 50 70 80 90"
time_interval="60"

for d in $dim; do
	for w in $windows; do
		for dr in $datasets; do
			for cp in $cuts; do
				for s in $strategies; do
					mprof run --output time_mem/mprofile.$dr.$s.$w.$cp.dat --interval $time_interval feature_generator.py --dataset $dr --strategy $s --window $w --emb_dim $d  --cut_percent $cp &
					echo "python3 feature_generator.py --dataset $dr --strategy $s --window $w --emb_dim $d --cut_percent $cp" >> executed_features_new_measures.txt &
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
methods="node2vec"
cuts="5 10 20 30 50 70 80 90"
time_interval="60"

for d in $dim; do
	for w in $windows; do
		for dr in $datasets; do
			for cp in $cuts; do
				for s in $strategies; do
					mprof run --output time_mem/mprofile.$dr.$s.$w.$cp.dat --interval $time_interval feature_generator.py --dataset $dr --strategy $s --window $w --emb_dim $d  --cut_percent $cp &
					echo "python3 feature_generator.py --dataset $dr --strategy $s --window $w --emb_dim $d --cut_percent $cp" >> executed_features_new_measures.txt &
				done
				wait
			done
		done
	done
done
