#! usr/bin/python
# coding: utf-8
# Author: Pan Li (pl1748@nyu.edu)

import re
import os
import jieba
from nltk.tokenize import WordPunctTokenizer
import codecs
import time

def dedupulicate(path):
	'''
	Dedupilicate lyric files, reserve the original version
	while discarding other versions (live,粤语,etc..)
	
	path: path of folder to dedupulicate
	'''
	fileList = []
	files = os.listdir(path)
	num_files = len(files)
	dedupulicate_files = []
	for i in range(num_files):
		if files[i].find('(') == -1:
			if files[i].find('+') == -1:
				if files[i].find('live') == -1:
					dedupulicate_files.append(files[i])
	dedupulicate_num_files = len(dedupulicate_files)
	return dedupulicate_files

def clean_lyric(string):
	'''
	Clean certain lyric lines by deleting time, punctuation 
	and other useless symbols

	string: string to be cleaned
	'''
	if re.search(r'找歌词|作曲|作词|编曲|监制|\@|\-|编辑人|《|www|QQ|:|：|歌词|by|演唱|LRC', string) is not None:
		string = ''
	else:
		regx_time = re.compile('\[.*\]',re.X)
		punctuation = '!,;.:?"\'！~～（）()'
		string = re.sub(regx_time,'',string)
		string = re.sub(r'[{}]+'.format(punctuation),'',string)
	return string

def tokenizer(string):
	'''
	Tokenize the referred strings by "jieba" tokenizer

	string: string to be tokenized
	'''
	if len(string) == 0:
		return string
	first_char = string[0]
	if first_char >= u'\u4e00' and first_char <= u'\u9fa5':
		regx_space = re.compile('\s',re.X)
		string = re.sub(regx_space,'',string)
		seg_list = jieba.cut(string, cut_all = False)
		seg_result = ','.join(seg_list)
		return seg_result
	elif (first_char >= u'\u0041' and first_char <= u'\u005a') or (first_char >= u'\u0061' and first_char <= u'\u007a'):
		seg_list = WordPunctTokenizer().tokenize(string)
		seg_result = ','.join(seg_list)
		return seg_result
	else:
		return ''

def main():
	'''
	Main function to exectute the preprocessing techniques
	'''
	file_path = '.../raw_lyric'
	file_output_path = '.../processed_lyric'
	os.mkdir(file_output_path)
	inputs = os.listdir(file_path)
	regx=re.compile('[a-zA-Z]',re.X)
	for input in inputs:
		input_path = file_path + '/' + input
		output_path = file_output_path + '/' + input
		os.mkdir(output_path)
		files = dedupulicate(input_path)
		for file in files:
			with codecs.open(input_path + '/' + file,'r','utf-8') as fopen:
				raw_lyric = fopen.readlines()
				raw_lyric = [ tokenizer(x) for x in raw_lyric ]
				string_lyric = str(raw_lyric)
				if re.search(regx,string_lyric) is None:
					fwrite = codecs.open(output_path + '/' + file,'w','utf-8')
					for index in range(len(raw_lyric)):
						if len(raw_lyric[index]) >=2:
							fwrite.write(raw_lyric[index] + '\n')

if __name__=='__main__':
	jieba.load_userdict('.../dictbig.txt')
	time_start = time.time()
	main()
	time_end = time.time()
	print(time_end-time_start)
	##209.00322484970093s
