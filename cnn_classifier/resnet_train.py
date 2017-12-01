from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import Sequential

from keras.optimizers import SGD


img_width, img_height = 320, 240
top_model_weights_path = 'model.h5'
train_data_dir = 'spectrograms/train'
test_data_dir = 'spectrograms/test'
epochs = 500
batch_size = 16
validation_batch_size=16


train_datagen = ImageDataGenerator(rescale=1. / 255);
train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size)

test_datagen = ImageDataGenerator(rescale=1. / 255);
test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size)


# create the base pre-trained model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# and a logistic layer -- let's say we have 15 classes
predictions = Dense(15, activation='softmax')(x)


# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:245]:
    layer.trainable = False
for layer in model.layers[245:]:
    layer.trainable = True


#
# model.add(BatchNormalization())
# model.add(Conv2D(256, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Dropout(0.5))
#
# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
# model.add(Dense(4096, activation='relu'))
#
# model.add(Dense(15))
# model.add(Activation('softmax'))


# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
#from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

tbCallBack = TensorBoard(log_dir='./keras_logs')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(train_generator,
		            validation_data=test_generator,
                    epochs=epochs,
                    shuffle=True,
                    #validation_steps=validation_batch_size,
                    callbacks=[tbCallBack]);



model.save_weights(top_model_weights_path)
