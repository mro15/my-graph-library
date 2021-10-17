#! /bin/bash

datasets="r8 webkb polarity"
windows="12"
cuts="5 10 20 30 50 70 80 90"

for dt in $datasets; do
	for w in $windows; do
		for ct in $cuts; do
            output="${dt}_${w}_${ct}.png"
			mprof plot mprofile.$dt.no_weight.$w.0.dat mprofile.$dt.pmi.$w.$ct.dat  mprofile.$dt.pmi_all.$w.$ct.dat mprofile.$dt.llr.$w.$ct.dat mprofile.$dt.llr_all.$w.$ct.dat mprofile.$dt.chi_square.$w.$ct.dat mprofile.$dt.chi_square_all.$w.$ct.dat  --legend "SEM PESO" --legend PMI --legend "PMI GLOBAL" --legend LLR --legend "LLR GLOBAL" --legend "CHI SQUARE" --legend "CHI SQUARE GLOBAL" --title "Dataset: $dt Janela: $w Corte: $ct %" --output $output 
		done
	done
done
