import sys
import re
import string
import time
import logging
import argparse

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train_model(file, window_size, iter, size, method):

	if method == "CBOW":
		sg = 0
	else:
		sg = 1

	# load sentences
	sentences = LineSentence(file)

	model = Word2Vec(sentences, size=size, window=window_size, sg=sg, min_count=1,
					workers=8, iter=iter)

	return model


def parse_args():

	p = argparse.ArgumentParser()
	p.add_argument('-f', '--file', type=str, required=True,
					help = 'file containing the text.')
	p.add_argument('-o', '--model', type=str, required=True,
					help = 'trained model.')
	p.add_argument('-w', '--window', type=int, default=5,
					help = 'window size used to train the model.')
	p.add_argument('-i', '--iter', type=int, default=10,
					help = 'number of iterations of word2vec.')
	p.add_argument('-s', '--size', type=int, default=100,
					help = 'size of embedding.')
	p.add_argument('-m', '--method', type=str, default='CBOW',
					help = 'method used to train the model.')
	parsed = p.parse_args()

	return parsed


def main():

	args = parse_args()

	file = args.file
	output_model = args.model
	window_size = args.window
	iter = args.iter
	method = args.method
 	size = args.size

	corpus = open(file, "r")
	model = train_model(corpus, window_size, iter, size, method)
	model.save(output_model)


if __name__ == "__main__":
	start_time = time.time()
	main()
	print("--- %s minutes ---" % round(((time.time() - start_time)/60), 4))
