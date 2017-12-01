## CPSC 4310 Project
### Classification of Auditory Scenes using Transfer Learning
#### by Austin Kothig and Lukas Grasse

The goal of our project has been to classify recordings of acoustic scenes into categories. The dataset that was used was from challenge 1 of the DCASE 2017 (Detection and Classification of Acoustic Scenes and Events)[2] conference.  This dataset contains 10 second audio recordings of different scenes. All audio segments used will go through a preprocessing phase to generate log-mel spectrogram to be used as input to make discovery of features easier.

## VGG Embedding Classifier

The cnn_embedding_classifier folder contains the code for our VGG Embedding Classifier experiment. The generate_embeddings.py script was adapted from the vggish scripts and is used to generate the sparse embeddings that our neural networks are trained on. The train_classifier.py, train_simple_classifier.py, and train_lstm_classifier.py scripts were used to train machine learning models using keras. The best performing model was the train_lstm_classifier.py script, which used a layer of bidirectional lstm nodes. The results of this model are shown in the report.

## Wavenet Embedding Classifier

The wavenet_classifier folder contains train_lstm_classifier.py, the script we used to train models on embeddings generated from the [NSynth Wavenet Autoencoder](https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth). The autoencoder is described in the paper "Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders". The nsynth package can be installed using pip, and the command to generate embeddings from audio is similar to:

'''nsynth_save_embeddings \
--checkpoint_path=/<path>/wavenet-ckpt/model.ckpt-200000 \
--source_path=/<path> \
--save_path=/<path> \
--batch_size=4'''

Once the embeddings are generated, the train_lstm_classifier.py script can be used.

## CNN Spectrogram Classifier

The cnn_classifier folder contains the scripts used to train a model directly on the audio spectrograms. The generate_spectrograms.py script can be used to convert the audio files into spectrograms. The keras_train.py script implements a model similar to the one used in the paper "Convolutional Neural Networks with Binaural Representations and Background Subtraction for Acoustic Scene Classification". The resnet_train.py script trains a model based on the [ResNet](https://arxiv.org/abs/1512.03385) architecture.

## Examples

The Wavenet Embedding Classifier and CNN Embedding Classifier folders contain example folders. In each example folder is an audio file, the corresponding embedding as a numpy array, and a script that classifies the embedding using a given model.

## Logs

The Wavenet Embedding Classifier and CNN Embedding Classifier folders have a logs folder that contains the training logs. These logs can be viewed using TensorBoard.

## Report

The final report can be viewed [here](https://github.com/lukeinator42/cpsc4310project/raw/master/final_report.pdf).