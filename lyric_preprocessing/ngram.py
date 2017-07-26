#! usr/bin/python
# coding: utf-8
# Author: Pan Li (pl1748@nyu.edu)

import os
import codecs
import pandas as pd

def reshape_lyric(input_path,output_path,length):
	with codecs.open(output_path+'/'+str(length),'w','utf-8') as fwrite:
		folders = os.listdir(input_path)
		for folder in folders:
			files = os.listdir(input_path+'/'+folder)
			for file in files: 
				with codecs.open(input_path+'/'+folder+'/'+file,'r','utf-8') as fread:
					lyrics = fread.readlines()
					lyrics = [x.strip() for x in lyrics]
					lyric_length = len(lyrics)
					if lyric_length >= length:
						for index in range(lyric_length-length):
							lyric=lyrics[index:index+length]
							fwrite.write('|'.join(lyric)+'\n')
	return True
def filter(sentence,min_sentence_length,max_sentence_length,dictionary):
	sentence = sentence.strip().split('|')
	sentence = [x.split(',') for x in sentence]
	for idx in range(len(sentence)):
		if len(sentence[idx]) > max_sentence_length or len(sentence[idx]) < min_sentence_length:
			return False
		for index in range(len(sentence[idx])):
			if int(sentence[idx][index]) > dictionary_num:
				return False
	return True

def filter_lyric(filter_input_path,filter_output_path,dictionary_num,min_sentence_length,max_sentence_length):
	with codecs.open(filter_output_path,'w','utf-8') as fwrite:
		with codecs.open(filter_input_path,'r','utf-8') as fread:
			lyrics = fread.readlines()
			for index in range(len(lyrics)):
				if filter(lyrics[index],min_sentence_length,max_sentence_length,dictionary_num)==True:
					fwrite.write(lyrics[index])
	return True

if __name__ =='__main__':
	input_path = 'numerical_lyric'
	output_path = 'ngram'
	length = 2
	filter_input_path = output_path+'/'+str(length)
	filter_output_path = filter_input_path+'_filter'
	dictionary_num = 10000
	min_sentence_length = 5
	max_sentence_length = 15
	
	#reshape_lyric(input_path,output_path,length)
	filter_lyric(filter_input_path,filter_output_path,dictionary_num,min_sentence_length,max_sentence_length)