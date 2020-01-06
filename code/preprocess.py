import sys
import re
import string
import time
import logging
import argparse

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import preprocess_string
from gensim.utils import tokenize
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.corpora.textcorpus import lower_to_unicode
from gensim.utils import any2unicode
import gensim.models.keyedvectors as kv
from nltk.corpus import stopwords


def preprocess_text(data, percentage, remove_sw):

	# \n removal
	data = data.strip("\n")

	# convert text to lowercase
	data = data.lower()

	# punctuation removal
	result = data.translate(str.maketrans('', '', string.punctuation))
	data = result

	# stop words removal
	if remove_sw == 1:
		words = data.split(" ")
		stop_words = set(stopwords.words('english'))
		words = [w for w in words if w not in stop_words]
		data = " ".join(words)

	if percentage != 1.0:
		words = data.split(" ")
		size = int(len(words)*percentage)
		words = words[:size]
		data = " ".join(words)

	return data

def parse_args():

	p = argparse.ArgumentParser()
	p.add_argument('-f', '--file', type=str, required=True,
					help = 'file containing the text.')
	p.add_argument('-o', '--output', type=str, required=True,
					help = 'file containing the preprocessed text.')
	p.add_argument('-p', '--percentage', type=float, default=1.0,
					help = 'percentage of the text to consider (number of words).')
	p.add_argument('-r', '--stop_words', type=int, default=0,
					help = 'remove stop words in the text.')

	parsed = p.parse_args()

	return parsed


def main():

	args = parse_args()

	file = args.file
	file2 = args.output
	percentage = args.percentage
	remove_sw = args.stop_words

	# read file
	with open(file, "r") as data:
		data = data.readlines()

	# the file contains just one line of text
	data = data[0]
	data = preprocess_text(data, percentage, remove_sw)

	corpus = open(file2, "w")
	corpus.writelines(data)


if __name__ == "__main__":
	start_time = time.time()
	main()
	print("--- %s minutes ---" % round(((time.time() - start_time)/60), 4))
