from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, ZeroPadding2D, BatchNormalization, GlobalAveragePooling2D, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import Sequential

img_width, img_height = 320, 240
#img_width, img_height = 640, 480
top_model_weights_path = 'model.h5'
train_data_dir = 'old_spectrograms/train'
test_data_dir = 'old_spectrograms/test'
epochs = 300
batch_size = 16
validation_batch_size = 16
# create the base pre-trained model
#base_model = InceptionV3(weights='imagenet', include_top=False)

train_datagen = ImageDataGenerator();
train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size)

test_datagen = ImageDataGenerator();
test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size)

model = Sequential()
model.add(ZeroPadding2D(input_shape=(img_height, img_width, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(ZeroPadding2D())
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
#model.add(Dropout(0.5))

model.add(ZeroPadding2D())
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(ZeroPadding2D())
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(3, 3)))
#model.add(Dropout(0.5))

model.add(ZeroPadding2D())
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(ZeroPadding2D())
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(MaxPooling2D(pool_size=(3, 3)))
#model.add(Dropout(0.5))

model.add(ZeroPadding2D())
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(ZeroPadding2D())
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))

model.add(GlobalAveragePooling2D())

# model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
#model.add(Dense(1024, activation='relu'))

model.add(Dense(15))
model.add(Activation('softmax'))

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

tbCallBack = TensorBoard(log_dir='./cnn_logs')
#checkpointCB = ModelCheckpoint(filepath='./keras_weights.hdf5', verbose=1, save_best_only=True)

# train the model on the new data for a few epochs
model.fit_generator(train_generator,
		          validation_data=test_generator,
                  epochs=epochs,
                  shuffle=True,
                  #validation_steps=validation_batch_size,
                  callbacks=[tbCallBack]);

model.save_weights(top_model_weights_path)
