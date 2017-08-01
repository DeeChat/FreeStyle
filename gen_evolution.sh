#!/bin/bash
mkdir -p generated

for i in `seq 0 19`;
do
	num=`expr $i \* 10000 + 9999`
	echo $num
	python generate.py --ae_args output/ae/args.json --gan_args output/gan/args.json --vocab_file output/ae/vocab.json --ae_model output/ae/autoencoder_model_5.pt --g_model output/gan/gan_gen_model_$num.pt --d_model output/gan/gan_disc_model_$num.pt --data_path chunks.json --dict_file vocab.txt --noprint --seed 1111 --ngenerations 50 --outf generated/$i.txt
done