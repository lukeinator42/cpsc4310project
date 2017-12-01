import csv
import os
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Activation, Dropout, Flatten
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

epochs = 50
batch_size = 16

training_data = []
training_labels = []

validation_data = []
validation_labels = []

label_to_index = {}

pos=0
with open('dcase_data/evaluation_setup/fold1_train.txt') as train_file:
    for row in train_file:
        label = row.split()[1].replace("/", "_")

        if not label in label_to_index.keys():
            label_to_index[label] = pos
            pos += 1



with open('dcase_data/evaluation_setup/fold1_train.txt') as train_file:
    for row in train_file:
        filename = 'dcase_data/embeddings/'+row.split()[0].split("/")[1].split(".")[0]+'.npy'
        label = row.split()[1].replace("/", "_")

        embedding_data = np.load(open(filename))

        training_data.append(embedding_data)
        training_labels.append(label_to_index[label])

with open('dcase_data/evaluation_setup/fold1_evaluate.txt') as test_file:
    for row in test_file:
        filename = 'dcase_data/embeddings/'+row.split()[0].split("/")[1].split(".")[0]+'.npy'
        label = row.split()[1].replace("/", "_")


        embedding_data = np.load(open(filename))

        validation_data.append(embedding_data)
        validation_labels.append(label_to_index[label])

training_data = np.asarray(training_data)
training_labels = to_categorical(training_labels)

validation_data = np.asarray(validation_data)
validation_labels = to_categorical(validation_labels)

#print training_labels[1].shape

tbCallBack = TensorBoard(log_dir='./logs/deep')
checkpoint = ModelCheckpoint('./checkpoints/deep/weights.best.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

print training_data[1,:].shape

model = Sequential()
print training_data[1,:].shape
#Original
model.add(Flatten(input_shape=training_data[1,:].shape))
model.add(BatchNormalization())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='softmax'))

# model.add(Dense(2048, input_shape=training_data[1,:].shape))
# model.add(LSTM(2048, dropout=0.2, return_sequences=True))
# model.add(Dense(2048))
# model.add(LSTM(2048, dropout=0.2))
# model.add(Dense(15, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(training_data, training_labels, batch_size=16, validation_data=(validation_data, validation_labels), epochs=500, callbacks=[tbCallBack, checkpoint])

model.save('model/cnn_embedding_classifier.h5')
