# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import sys
import re
import string
import time
from numpy import dot
from numpy.linalg import norm
import numpy as np

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import preprocess_string
from gensim.utils import tokenize
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import gensim.models.keyedvectors as kv


def cosine_distance(model, predicted, expected):

	a = model.wv[predicted]
	b = model.wv[expected]
	cos_sim = dot(a, b)/(norm(a)*norm(b))

	return cos_sim


def evaluate_model(model, words):

	original_vocab = model.wv.vocab
	ok_vocab = [(w, model.wv.vocab[w]) for w in model.wv.index2word]
	ok_vocab = dict(ok_vocab)

	count = 0
	errors = 0
	distances = []
	for word in words:
		a, b, c, expected = word
		# print(a, "*", b, "*", c, "*", expected)
		ignore = {a, b, c}
		if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
			errors += 1
			continue

		sims = model.wv.most_similar(positive=[b, c], negative=[a], topn=5)
		for element in sims:
			predicted = element[0]
			if predicted not in ignore:
				break
		distance = cosine_distance(model, predicted, expected)
		distances.append(distance)
		count += 1

	return distances, count, errors


def main():

	file = sys.argv[1]			# questions-words.txt
	model_file = sys.argv[2]	# model
	out_file = sys.argv[3]		# output file

	words = []

	# read file
	with open(file, "r") as data:
		data.readline()
		lines = data.readlines()
		for line in lines:
			word = line.strip("\n").split(" ")
			if len(word) == 4:
				words.append([w.lower() for w in word])

	model = Word2Vec.load(model_file)

	distances, total, errors = evaluate_model(model, words)

	print("Vocabulary size:", len(model.wv.vocab))
	print('Total:', total)
	print('Errors:', errors)
	print("Mean distance:", np.mean(distances))
	print("Std distance:", np.std(distances))

	with open(out_file, "w") as result:
		result.write(str(len(model.wv.vocab)) + "\n")
		result.write(str(total) + "\n")
		result.write(str(errors) + "\n")
		result.write(str(np.mean(distances)) + "\n")
		result.write(str(np.std(distances)) + "\n")


if __name__ == "__main__":
	start_time = time.time()
	main()
	print("--- %s minutes ---" % round(((time.time() - start_time)/60), 4))
