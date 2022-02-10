import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam
import string
from random import randrange
import os

class LSTMLyricGen():
	def __init__(self):
		print("Model Initialized")
		self.songLines = []
		self.vocabulary_size= None
		self.max_seq_length = None
		self.embedding_matrix = None
		self.x_train = None
		self.y_train = None
		self.model = None
		self.tokenizer = None
		self.ave_line_len = None
		self.max_line_len = 0
		self.dataFilePath = './data/baseline/'

	def loadData(self):
		dataFiles = os.listdir(self.dataFilePath)
		#path = './data/baseline/genre_country.txt'
		songLines = []
		songLineLengths = []
		
		for file in dataFiles:
			textLines = open(self.dataFilePath + file).readlines()
			for index in range(0, len(textLines)):
				if "LYRICS:" in textLines[index]:
					l = 1
					while not("LYRICS:" in textLines[index+l] or "/END LYRICS" in textLines[index+l]):
						# make all lower case and remove whitespace and newlines
						songLine = textLines[index+l].lower().strip()

						# find the max line length
						if len(songLine.split(' ')) > self.max_line_len:
							self.max_line_len = len(songLine.split(' '))

						songLineLengths.append(len(songLine.split(' ')))

						self.songLines.append(songLine)
						l += 1
		
		self.ave_line_len = int(sum(songLineLengths) / len(songLineLengths))

		# prints song lines for debugging
		#print(" ".join(self.songLines))
		#for line in songLines:
			#print(line)
	
	def processData(self):
		self.tokenizer = Tokenizer()
		self.tokenizer.fit_on_texts(self.songLines)
		word_index = self.tokenizer.word_index
		self.vocabulary_size = len(self.tokenizer.word_index) + 1
		#print(total_unique_word)
		#print(word_index)
		
		# create n_grams
		input_sequences = []

		for line in self.songLines:
			token_list = self.tokenizer.texts_to_sequences([line])[0]
			#print(token_list)
			for i in range(1, len(token_list)):
				n_gram_seq = token_list[:i+1]
				input_sequences.append(n_gram_seq)

		#print(len(input_sequences))
		#print(input_sequences)

		# pad each n_gram to the length of the longest n_gram
		self.max_seq_length = max([len(x) for x in input_sequences])
		input_seqs = np.array(pad_sequences(input_sequences, 
											maxlen=self.max_seq_length, 
											padding='pre'))

		#print(max_seq_length)
		#print(input_seqs[:5])

		# set the training x and y 
		# use one hot encoding for labels

		self.x_train, labels = input_seqs[:, :-1], input_seqs[:, -1]

		self.y_train = to_categorical(labels, num_classes=self.vocabulary_size)

		# processes a word embedding dictionary to help the model 
		# better understand contexual relationships among the words

		embeddings_index = {}
		with open('./glove.6B.100d.txt') as f:
			for line in f:
				values = line.split()
				word = values[0]
				coeffs = np.array(values[1:], dtype='float32')
				embeddings_index[word] = coeffs

		#print(dict(list(embeddings_index.items())[0:2]))

		# create a matrix which contains words from the embedding dictionary
		# for words only in the data vocabulary

		self.embedding_matrix = np.zeros((self.vocabulary_size, 100))

		for word, i in word_index.items():
			embedding_vector= embeddings_index.get(word)
			if embedding_vector is not None:
				self.embedding_matrix[i] = embedding_vector;


	def buildModel(self):

		# build model
		self.model = Sequential([
		# the embedding layer requires in input_dim of the total number of unique words (size of vocabulary)
		# and an output_dim which specifies the number of word embedding dimensions we want. Since this model 
		# uses the GloVe 100D we wil use 100 as our out_dim
		tf.keras.layers.Embedding(input_dim = self.vocabulary_size, 
								 output_dim = 100, 
								 weights=[self.embedding_matrix],
								 input_length=self.max_seq_length-1,
								 trainable=False),

		tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dense(self.vocabulary_size, activation='softmax')])


		# compile model
		self.model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
		self.model.summary()
		
		# used to create a graph representation of the model
		#tf.keras.utils.plot_model(self.model, show_shapes=True)

	def fitModel(self):
		history = self.model.fit(self.x_train, self.y_train, epochs=20, verbose=1)

	def makePredictions(self, userInput):

		if userInput == "":
			randInt = int(randrange(0, len(list(self.tokenizer.word_index.keys()))))
			print(list(self.tokenizer.word_index.keys())[randInt])
		
		for i in range(0, 30):
			output_line = ""
			for k in range(0, randrange(self.ave_line_len, self.max_line_len)):
				
				encoded_text = self.tokenizer.texts_to_sequences([userInput])[0]

				encoded_text = pad_sequences([encoded_text], maxlen=self.max_seq_length-1, truncating='pre')

				predicted = np.argmax(self.model.predict(encoded_text), axis=-1)

				output_word = ""
				for word, index in self.tokenizer.word_index.items():
					if index == predicted:
						output_word = word
						break
			
				if k > 0:
					output_line += " " + output_word
				else:
					output_line = output_word

				userInput = output_line
		
			print(output_line)
		

if __name__ == "__main__":
	lGen = LSTMLyricGen()
	print('Loading Data')
	lGen.loadData()
	print('Loading Data - DONE')
	print('Processing Data.')
	lGen.processData()
	print('Processing Data - DONE')
	print('Building Model.')
	lGen.buildModel()
	print('Building Model. - DONE')
	print('Training Model.')
	lGen.fitModel()
	print('Training Model. - DONE')
	print("Type some words start lyric generation")
	print("Or press enter to use a random seed word")
	userInput = input().strip().lower()
	lGen.makePredictions(userInput)

