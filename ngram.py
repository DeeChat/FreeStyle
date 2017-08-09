import nltk
from nltk.util import ngrams
import pickle

n = 3

###For Generating Language Models With Customized Options
#
#Probdist = nltk.KneserNeyProbDist
#i = 0
#alllist = []
#
#with codecs.open('data.json','r','utf-8') as f:
#	data = json.load(f)['data'][:5000]
#
#for song in data:
#	for line in song['text']:
#		generated_ngrams = ngrams(line, n, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')
#		alllist = alllist + list(generated_ngrams)
#		i += 1
#		if i%100 == 0:
#			print(i)
#
#freq_dist = nltk.FreqDist(alllist)
#Dist_p = Probdist(freq_dist,1)
#pickle.dumps(Dist_p,open('Dist_p','wb'))

#Load Language Model From Default
Dist_p = pickle.loads(open('Dist_p','rb'))

def score(sentence):
	generated_ngrams = ngrams(sentence, n, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')
	score = 0
	for gram in generated_ngrams:
		score += Dist_p.prob(gram)
	return score
