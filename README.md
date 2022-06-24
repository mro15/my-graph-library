# Enhancing text to graph representation by edge weighting and filtering through word association measures


# Datasets
- Polarity
- WEBKB
- R8
- 20 Newsgroups

# Reproducing the experiments
We implemented auxiliar scripts in the `scripts/` directory. To use these scripts, copy them to the project root directory (this directory). Example:

```sh
cp scripts/feature_runner.sh .
```

## Requirements

The implementation was constructed using Python 3.6.13

### Optional step

There is a **optional** step, to use a virtual environment before install the dependencies:

Install the virtualenv package:

```sh
pip install virtualenv
```

Create a new environment:

```sh
virtualenv venv
```

Activate the new environment:

```sh
source venv/bin/activate
```

### Install the dependencies:

The dependencies are listed in the `requirements.txt` file. To install, run:


```sh
pip install -r requirements.txt
```


## First step: Graphs generation and representation learning

### Running all experiments

To run the proposed approach:

```sh
./feature_runner.sh
```
To run the baseline:

```sh
./baseline_feature_runner.sh
```

### Running a single estimation

```sh
mprof run --output <time_mem_output_file.dat> --interval 60 feature_generator.py --dataset <dataset> --strategy <weight_strategy> --window 12 --emb_dim 100  --cut_percent <cut_p>
```

Where:

\<time_mem_output_file> : Output file to register time x memory values.

\<dataset>              : Input dataset: polarity, r8 or webkb

\<weight_strategy>      : Weight strategy: pmi, pmi_all,llr, llr_all, chi_square, chi_square_all (for proprosed weight approaches) or no_weight (for baseline)

\<cut_p>          : Cut p: 5, 10, 20, 30, 50, 70, 90 (for the proposed approach) or 0 (for baseline)


## Second step: Classification

### Running all experiments

To run the proposed approach:


```sh
./cnn_runner.sh
```


To run the baseline:


```sh
./baseline_cnn_runner.sh
```

### Running a single estimation

```sh
python3 cnn_main.py --dataset <dataset> --cut_percent <cut_p> --strategy <weight_strategy> --window 12 --emb_dim 100
```

Where:

\<dataset>              : Input dataset: polarity, r8 or webkb

\<weight_strategy>      : Weight strategy: pmi, pmi_all,llr, llr_all, chi_square, chi_square_all (for proprosed weight approaches) or no_weight (for baseline)

\<cut_p>          : Cut p: 5, 10, 20, 30, 50, 70, 90 (for the proposed approach) or 0 (for baseline)


## Evaluation


The evaluation is implemented on `results_main.py` file.
This script performs for a dataset and cut p, the calculation of the 10-fold mean f1 score for all weight strategies and the baseline, and also, runs the Wilcoxon test comparing each weight strategy with the baseline. This results are printed in the console runing the script (for example the linux terminal) and also the results are writen to output files in the `plots/next_level/<cut p>/<dataset>` directory, in txt files name as `f1_polarity_12.txt`

For example, runnin the `results_main.py` script as:

```sh
python results_main.py --dataset r8 --emb_dim 100 --window 12 --cut_percent 50
```

will generate the file: `plots/next_level/0.05/polarity/f1_polarity_12.txt`, containing:


```txt
no_weight,77.12544679853906,0.050221219618395326
chi_square,78.55941729972031,0.050606510004414684,p=0.275390625
chi_square_all,79.69882465466792,0.028751551738617296,p=0.10546875
llr,79.39449196564627,0.024537426341991925,p=0.193359375
llr_all,79.65264759486207,0.04178546879605646,p=0.16015625
pmi,79.42541259746658,0.03398688281606517,p=0.16015625
pmi_all,78.89614764450108,0.046800758344722936,p=0.275390625
```

Where:

The first line contains the mean f1 score and the standard deviation for the baseline (no_weight).
And the other lines contain for each weight strategy the mean f1 score and the standard deviation and the p value for the Wilcoxon test compared to the baseline.

The `results_main.py` runs for a single estimation on a combination of dataset and cut p. To run the evaluation for all experiments, run:

```sh
    ./plot_f1.sh
```

