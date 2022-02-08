import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
import string

class LSTMLyricGen():
	def __init__(self):
		print("Model Initialized")
		self.songLines = []
		self.train_inputs = None
		self.train_labels = None
		self.vocabulary_size= None
		self.seq_len = None
		self.model = None
		self.tokenizer = None

	def loadData(self):
		path = './data/baseline/genre_country.txt'
		songLines = []
		textLines = open(path).readlines()
		for index in range(0, len(textLines)):
			if "LYRICS:" in textLines[index]:
				l = 1
				while not("LYRICS:" in textLines[index+l] or "/END LYRICS" in textLines[index+l]):
					# make all lower case and remove whitespace and newlines
					sanitizedLine = textLines[index+l].lower().strip()
					# remove punctuation
					sanitizedLine = sanitizedLine.strip(string.punctuation)
					sanitizedLine = sanitizedLine.replace(',', '')
					sanitizedLine = sanitizedLine.replace("'", "")
					self.songLines.append(sanitizedLine)
					l += 1
		
		# prints song lines for debugging
		#print(" ".join(self.songLines))
		#for line in songLines:
			#print(line)
	
	def processData(self):
		wordTokens = " ".join(self.songLines).split(" ")
		train_len = 4
		text_sequences = []

		for i in range(train_len, len(wordTokens)):
			seq = wordTokens[i-train_len:i]
			text_sequences.append(seq)

		sequences = {}
		count = 1

		for i in range(len(wordTokens)):
			if wordTokens[i] not in sequences:
				sequences[wordTokens[i]] = count
				count += 1

		self.tokenizer = Tokenizer()
		self.tokenizer.fit_on_texts(text_sequences)
		sequences = self.tokenizer.texts_to_sequences(text_sequences)

		# vocabulary size increased by 1 for the cause of padding
		self.vocabulary_size = len(self.tokenizer.word_counts)+1
		n_sequences = np.empty([len(sequences), train_len], dtype='int32')

		for i in range(len(sequences)):
			n_sequences[i] = sequences[i]

		self.train_inputs = n_sequences[:, :-1]
		self.train_targets = n_sequences[:, -1]
		self.train_targets = to_categorical(self.train_targets, num_classes=self.vocabulary_size)
		self.seq_len = self.train_inputs.shape[1]

	def buildModel(self):

		# build model
		self.model = Sequential()
		self.model.add(Embedding(self.vocabulary_size, self.seq_len, input_length=self.seq_len))
		self.model.add(LSTM(50, return_sequences=True))
		self.model.add(LSTM(50))
		self.model.add(Dense(50, activation='relu'))
		self.model.add(Dense(self.vocabulary_size, activation='softmax'))

		# compile model
		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		self.model.summary()

	def fitModel(self):
		self.model.fit(self.train_inputs, self.train_targets, epochs=200, verbose=1)

	def makePredictions(self):
		print("Type some words to make a prediction")
		userInput = input().strip().lower()
		encoded_text = self.tokenizer.texts_to_sequences([userInput])[0]
		pad_encoded = pad_sequences([encoded_text], maxlen=self.seq_len, truncating='pre')
		print(encoded_text, pad_encoded)

		for i in (self.model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
			pred_word = self.tokenizer.index_word[i]
			print("Next word suggestion: ", pred_word)


		

if __name__ == "__main__":
	lGen = LSTMLyricGen()
	lGen.loadData()
	lGen.processData()
	lGen.buildModel()
	lGen.fitModel()
	lGen.makePredictions()

