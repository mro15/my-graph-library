#! /bin/bash

datasets="polarity webkb r8"
windows="12"

for dt in $datasets; do
	for w in $windows; do
        python resume_time_men.py --dataset $dt --window $w
	done
done
