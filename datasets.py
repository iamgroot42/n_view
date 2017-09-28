import numpy as np
import re
import os
from sets import Set


DATA_DIRECTORY = "datasets"


class Dataset:
	def __init__(self, dataest_path, u_ratio, test_ratio=0.25):
		self.dataest_path = dataest_path
		self.parse_data(dataest_path)
		self.split_data(u_ratio, test_ratio)

	def split_data(self, u_ratio, test_ratio):
		# Shuffle data
		shuffle = np.random.permutation(self.X.shape[0])
		self.X = self.X[shuffle]
		self.Y = self.Y[shuffle]
		Y_flat = np.argmax(self.Y, axis=1)
		classwise_indices = []
		labels = np.arange(self.Y.shape[1])
		# Organize data to comupute subsets that follow the given data distribution
		label_indices = []
		for label in labels:
			label_indices.append(np.nonzero(Y_flat == label)[0])
		self.L_x = []
		self.L_y = []
		self.test_x = []
		self.test_y = []
		self.U = []
		# Split data into labelled, unlabelled and test sets
		for label in label_indices:
			test_indices = label[:int(len(label)*test_ratio)]
			U_indices = label[int(len(label)*test_ratio):int(len(label)*(test_ratio + (1 - test_ratio)*u_ratio))]
			L_indices = label[int(len(label)*(test_ratio + (1 - test_ratio)*u_ratio)):]
			for i in test_indices:
				self.test_x.append(self.X[i])
				self.test_y.append(self.Y[i])
			for i in U_indices:
				self.U.append(self.X[i])
			for i in L_indices:
				self.L_x.append(self.X[i])
				self.L_y.append(self.Y[i])
		self.L_x = np.array(self.L_x)
		self.L_y = np.array(self.L_y)
		self.U = np.array(self.U)
		self.test_x = np.array(self.test_x)
		self.test_y = np.array(self.test_y)

	def get_data(self):
		shuffle = np.random.permutation(self.L_x.shape[0])        
		return (self.L_x[shuffle], self.L_y[shuffle]), self.U, (self.test_x, self.test_y)

	def numerical_y(self, y):
		y_uniq = list(Set(y))
		if '?' in y_uniq:
			y_uniq.remove('?')
		label_names = {x:i for i,x in enumerate(y_uniq)}
		y_ = []
		for i in range(len(y)):
			if y[i] in label_names:
				y_.append(label_names[y[i]])
		y_ = np.array(y_)
		one_hot = np.zeros((len(y_), len(np.unique(y_))))
		one_hot[np.arange(len(y_)), y_] = 1
		return one_hot

	def numerical_x(self, x):
		columns = [Set([]) for _ in range(len(x[0]))]
		for i,row in enumerate(x):
			for j in range(len(row)):
				columns[j].add(row[j])
		columns = [list(c) for c in columns]
		column_mappings = []
		for c in columns:
			c_ = {}
			# Either Column has float/int values
			try:
				for cv in c:
					c_[cv] = float(cv)
			except:
				# Or Column has string values; map to integers
				c_ = {x:i for i,x in enumerate(c)}
			column_mappings.append(c_)
		# Remap data to numerical values
		X_ = []
		for x_ in x:
			row = []
			for j in range(len(x_)):
				row.append(column_mappings[j][x_[j]])
			X_.append(row)
		return np.array(X_)


	def parse_data(self, filename):
		X = []
		Y = []
		# Read data file
		with open(filename, 'r') as file:
			for line in file:
				parsed_line = ' '.join(line.rstrip().split())
				parsed_line = re.split(r'[, |\t]+', parsed_line)
				X.append(parsed_line[:-1])
				Y.append(parsed_line[-1])
		# Transform X and Y into numerical representations
		self.X = self.numerical_x(X)
		self.Y = self.numerical_y(Y)


def get_dataset(dataset, u_ratio, test_ratio):
	try:
		return Dataset(os.path.join(DATA_DIRECTORY, dataset), u_ratio, test_ratio)
	except:
		print "Dataset doesn't exist!"
	return None
