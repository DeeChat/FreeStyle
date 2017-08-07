#!/bin/bash
mkdir -p generated

for i in `seq 0 19`;
do
	num=`expr $i \* 10000 + 9999`
	echo $num
	python answer_quiz.py --quiz_file quiz.json --ae_model output/ae/autoencoder_model_5.pt --g_model output/gan/gan_gen_model_$num.pt --d_model output/gan/gan_disc_model_$num.pt --outf answer_$num.json
done