import csv
import os
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Activation, Dropout, Flatten
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping

epochs = 200
batch_size = 16

training_data = []
training_labels = []

validation_data = []
validation_labels = []

label_to_index = {}

foldName = 'fold3'

pos=0
with open('dcase_data/evaluation_setup/'+foldName+'_train.txt') as train_file:
    for row in train_file:
        label = row.split()[1].replace("/", "_")

        if not label in label_to_index.keys():
            label_to_index[label] = pos
            pos += 1



with open('dcase_data/evaluation_setup/'+foldName+'_train.txt') as train_file:
    for row in train_file:
        filename = 'dcase_data/embeddings/'+row.split()[0].split("/")[1].split(".")[0]+'_embeddings.npy'
        label = row.split()[1].replace("/", "_")

        embedding_data = np.load(open(filename))

        training_data.append(embedding_data)
        training_labels.append(label_to_index[label])

with open('dcase_data/evaluation_setup/'+foldName+'_evaluate.txt') as test_file:
    for row in test_file:
        filename = 'dcase_data/embeddings/'+row.split()[0].split("/")[1].split(".")[0]+'_embeddings.npy'
        label = row.split()[1].replace("/", "_")


        embedding_data = np.load(open(filename))

        validation_data.append(embedding_data)
        validation_labels.append(label_to_index[label])

training_data = np.asarray(training_data)
training_labels = to_categorical(training_labels)

validation_data = np.asarray(validation_data)
validation_labels = to_categorical(validation_labels)

#print training_labels[1].shape

tbCallBack = TensorBoard(log_dir='./logs/'+foldName)
#checkpoint = ModelCheckpoint('./checkpoints/deep/weights.best.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')


print training_data[1,:].shape
# print label_to_index.keys()

model = Sequential()
#Original
# model.add(BatchNormalization(input_shape=training_data[1,:].shape))
# model.add(Dense(1024, input_shape=training_data[1,:].shape))
# model.add(LSTM(1024, activation='relu'))
# model.add(Dropout(.2))
#model.add(Dense(15, activation='softmax'))

#other lstm
model.add(Dense(512, input_shape=training_data[1,:].shape))
model.add(Dropout(.2))
model.add(LSTM(512))
model.add(Dense(15, activation='softmax'))

#model.add(BatchNormalization(input_shape=training_data[1,:].shape))
#model.add(LSTM(1024,stateful=False,
                          #batch_input_shape=(None, training_data[1].shape[0], training_data[1].shape[1])))
#model.add(LSTM(64, return_sequences=True, stateful=False))
#model.add(LSTM(64, stateful=False))
# model.add(Bidirectional(LSTM(1024, init='normal', activation='relu', dropout=0.5)))
# model.add(Dense(1024))
# model.add(Dropout(.25))

# model.add(Flatten(input_shape=training_data[1,:].shape))
# model.add(BatchNormalization())
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(training_data, training_labels,batch_size=64, validation_data=(validation_data, validation_labels), epochs=epochs, callbacks=[tbCallBack])

model.save('model/'+foldName+'_cnn_embedding_classifier.h5')
