from keras.models import load_model
import numpy as np


label_to_index = {}

pos=0
with open('../dcase_data/evaluation_setup/fold1_train.txt') as train_file:
    for row in train_file:
        label = row.split()[1].replace("/", "_")
        if not label in label_to_index.keys():
            label_to_index[label] = pos
            pos += 1

model = load_model('cnn_wavenet_classifier.h5')

embedding_data = np.load(open('a086_10_20_embeddings.npy'))
embedding_data = np.expand_dims(embedding_data, 0)

res = model.predict(embedding_data)
ndex_min = np.argmax(res[0])

for key in label_to_index.keys():
    if label_to_index[key] == ndex_min:
        print key
