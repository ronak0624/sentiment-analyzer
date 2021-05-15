# STEP 1: rename this file to hw3_sentiment.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
# import numpy as np
import sys
from collections import Counter
import math

"""
Your name and file comment here:
"""


"""
Cite your sources here:
"""

"""
Implement your functions that are not methods of the Sentiment Analysis class here
"""


def generate_tuples_from_file(training_file_path):
	# open file
	# parse lines
	with open(training_file_path) as w:
		lines = w.read().splitlines()
		data = []
		for l in lines:
			info = l.split('\t')

			# Tests if we are splitting up a training set with labels or a test set with no labels
			if len(info) == 3:
				data.append((info[0], info[1], int(info[2])))
			else:
				data.append((info[0], info[1]))

	return data


def precision(gold_labels, classified_labels):
	tp, tn, fp, fn = comparison(gold_labels, classified_labels)
	
	return tp / (tp + fp)


def recall(gold_labels, classified_labels):
	tp, tn, fp, fn = comparison(gold_labels, classified_labels)

	return tp / (tp + fn)


def f1(gold_labels, classified_labels):
	r = recall(gold_labels, classified_labels)
	p = precision(gold_labels, classified_labels)

	return (2 * (r * p)) / (r + p)

def comparison(gold_labels, classified_labels):
	tp = 0
	tn = 0
	fp = 0
	fn = 0

	for i in range(len(gold_labels)):
		gold = int(gold_labels[i])
		classified = int(classified_labels[i])

		if (gold == 1 and classified == 1):
			tp += 1
		elif (gold == 0 and classified == 0):
			tn += 1
		elif (gold == 0 and classified == 1):
			fp += 1
		elif (gold == 1 and classified == 0):
			fn += 1
	return (tp, tn, fp, fn)


"""
Implement any other non-required functions here
"""


"""
implement your SentimentAnalysis class here
"""


class SentimentAnalysis:

	def __init__(self):
		# do whatever you need to do to set up your class here
		self.pos_count = {}
		self.neg_count = {}
		self.pos_prob = {}
		self.neg_prob = {}
		self.pos = 0
		self.neg = 0
		self.pos_statement_p = 0
		self.neg_statement_p = 0

		pass

	def train(self, examples):

		feat = []

		for e in examples:
			feat = self.featurize(e)
			if(e[2] == 1):
				self.pos_statement_p += 1

				for f in feat:
					if(f[0] not in self.pos_count):
						self.pos_count[f[0]] = 0
					self.pos_count[f[0]] += 1
					self.pos += 1

			if(e[2] == 0):
				self.neg_statement_p += 1

				for f in feat:
					if(f[0] not in self.neg_count):
						self.neg_count[f[0]] = 0
					self.neg_count[f[0]] += 1
					self.neg += 1

		for k, v in self.pos_count.items():
			if k not in self.neg_count:
				self.neg_count[k] = 0
		for k, v in self.neg_count.items():
			if k not in self.pos_count:
				self.pos_count[k] = 0
		
		for k, v in self.pos_count.items():
			self.pos_prob[k] = (self.pos_count[k] + 1) / (self.pos + len(self.pos_count))
		for k, v in self.neg_count.items():
			self.neg_prob[k] = (self.neg_count[k] + 1) / (self.neg + len(self.neg_count))

		# Turns pos and neg statement counts into proportions for use in calculating the probability
		sum_statement_counts = self.pos_statement_p + self.neg_statement_p
		self.pos_statement_p = self.pos_statement_p / sum_statement_counts
		self.neg_statement_p = self.neg_statement_p / sum_statement_counts

		return feat

	def score(self, data):
		data_p = (data[0], data[1], 1)
		data_n = (data[0], data[1], 0)

		feat_p = self.featurize(data_p)
		feat_n = self.featurize(data_n)

		pos_p = self.pos_statement_p
		neg_p = self.neg_statement_p

		size_of_pos = len(self.pos_count)
		size_of_neg = len(self.neg_count)
		
		for f in feat_p:
			prob_w = 1 / (self.pos + size_of_pos + 1)

			if f[0] in self.pos_prob:
				prob_w = self.pos_prob[f[0]]
			pos_p = pos_p * prob_w
		for f in feat_n:
			prob_w = 1 / (self.neg + size_of_neg + 1)

			if f[0] in self.neg_prob:
				prob_w = self.neg_prob[f[0]]
			neg_p = neg_p * prob_w
		return (neg_p, pos_p)

	def classify(self, data):
		neg_p, pos_p = self.score(("ID", data))

		if(pos_p > neg_p):
			return 1
		else:
			return 0
		pass

	def featurize(self, data):
		sen = data[1]
		label = data[2]

		tokens = sen.split()

		feat = []
		for t in tokens:
			feat.append((t, label))

		return feat

	def __str__(self):
		return "Naive Bayes - bag-of-words baseline"


class SentimentAnalysisImproved:

	def __init__(self):
		# do whatever you need to do to set up your class here
		self.pos_count = {}
		self.neg_count = {}
		self.pos_prob = {}
		self.neg_prob = {}
		self.pos = 0
		self.neg = 0
		self.pos_statement_p = 0
		self.neg_statement_p = 0

		pass

	def train(self, examples):

		feat = []

		for e in examples:
			feat = self.featurize(e)
			if(e[2] == 1):
				self.pos_statement_p += 1

				for f in feat:
					if(f[0] not in self.pos_count):
						self.pos_count[f[0]] = 0
					self.pos_count[f[0]] += 1
					self.pos += 1

			if(e[2] == 0):
				self.neg_statement_p += 1

				for f in feat:
					if(f[0] not in self.neg_count):
						self.neg_count[f[0]] = 0
					self.neg_count[f[0]] += 1
					self.neg += 1

		for k, v in self.pos_count.items():
			if k not in self.neg_count:
				self.neg_count[k] = 0
		for k, v in self.neg_count.items():
			if k not in self.pos_count:
				self.pos_count[k] = 0
		
		for k, v in self.pos_count.items():
			self.pos_prob[k] = (self.pos_count[k] + 1) / (self.pos + len(self.pos_count))
		for k, v in self.neg_count.items():
			self.neg_prob[k] = (self.neg_count[k] + 1) / (self.neg + len(self.neg_count))

		# Turns pos and neg statement counts into proportions for use in calculating the probability
		sum_statement_counts = self.pos_statement_p + self.neg_statement_p
		self.pos_statement_p = self.pos_statement_p / sum_statement_counts
		self.neg_statement_p = self.neg_statement_p / sum_statement_counts

		return feat

	def score(self, data):
		data_p = (data[0], data[1], 1)
		data_n = (data[0], data[1], 0)

		feat_p = self.featurize(data_p)
		feat_n = self.featurize(data_n)

		pos_p = self.pos_statement_p
		neg_p = self.neg_statement_p

		size_of_pos = len(self.pos_count)
		size_of_neg = len(self.neg_count)
		
		for f in feat_p:
			prob_w = 1

			if f[0] in self.pos_prob:
				prob_w = self.pos_prob[f[0]]
			pos_p = pos_p * prob_w
		for f in feat_n:
			prob_w = 1

			if f[0] in self.neg_prob:
				prob_w = self.neg_prob[f[0]]
			neg_p = neg_p * prob_w
		return (neg_p, pos_p)

	def classify(self, data):
		neg_p, pos_p = self.score(("ID", data))

		if(pos_p > neg_p):
			return 1
		else:
			return 0
		pass

	def featurize(self, data):
		sen = data[1]
		label = data[2]

		tokens = sen.split()

		feat = []
		for t in tokens:
			feat.append((t.lower(), label))

		return feat

	def __str__(self):
		return "The Ronalgorithm"


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage:", "python hw3_sentiment.py training-file.txt testing-file.txt")
		sys.exit(1)

	training = sys.argv[1]
	testing = sys.argv[2]

	sa = SentimentAnalysis()
	print(sa)
	examples = generate_tuples_from_file(training)
	# Trains the Naive Bayes Classifier based on the tuples from the training data

	sa.train(examples)

	test = generate_tuples_from_file(testing)
	test_result = []
	actual_result = []
	# Creates list of gold labels and classified labels for checks
	for t in test:
		current = sa.classify(t[1])
		actual_result.append(t[2])
		test_result.append(current)
	
	print("Recall:", recall(actual_result, test_result))
	print("Precision:", precision(actual_result, test_result))
	print("F1:", f1(actual_result, test_result))
	
	# Writes labels for testset for submission
	# out = open("label_test_data.txt", "w")
	# sa = SentimentAnalysis()
	# examples = generate_tuples_from_file("bigger/train_file.txt")
	
	# sa.train(examples)
	# test = generate_tuples_from_file("HW3-testset.txt")

	# for line in test:
	# 	out.write("" + str(line[0]) + " " + str(sa.classify(line[1])) + "\n")
	
	# out.close()

	improved = SentimentAnalysisImproved()
	print(improved)

	improved.train(examples)

	actual_result = []
	test_result = []
	
	for t in test:
		current = improved.classify(t[1])
		actual_result.append(t[2])
		test_result.append(current)

	print("Recall:", recall(actual_result, test_result))
	print("Precision:", precision(actual_result, test_result))
	print("F1:", f1(actual_result, test_result))
