#!/bin/bash
mkdir -p ae_generated

for i in `seq 0 12`;
do
	echo $i
	python eval_ae.py --ae_args output/ae/args.json --vocab_file output/ae/vocab.json --ae_model output/ae/autoencoder_model_$i.pt --data_path chunks.json --dict_file vocab.txt --seed 1111 --sample_size 100 --outf ae_generated/$i.txt
done