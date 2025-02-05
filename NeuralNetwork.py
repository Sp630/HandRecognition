import tensorflow as tf
from numpy.ma.core import shape
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Input, Dropout, Flatten
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras import models
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

train = ImageDataGenerator(rescale= 1/255)
validation = ImageDataGenerator(rescale= 1/255)

train_dataset = train.flow_from_directory("Data/BaseData", target_size=(300, 300), batch_size= 32, class_mode= "categorical")
test_dataset = validation.flow_from_directory("Data/Test", target_size=(300, 300), batch_size= 32, class_mode= "categorical")

print(train_dataset.class_indices)
print(test_dataset.class_indices)

model = tf.keras.models.Sequential([
    Input(shape=(300, 300, 3)),
    Conv2D(32, 3, activation= relu),
    Conv2D(32, 3, activation= relu),
    MaxPool2D(2),

    Conv2D(64, 3, activation= relu),
    Conv2D(64, 3, activation= relu),
    MaxPool2D(2),

    Flatten(),
    Dense(128, activation=relu),
    Dense(3, activation=softmax)

])

model.compile(loss= CategoricalCrossentropy(), optimizer = Adam(), metrics = ['accuracy'])

history = model.fit(train_dataset, epochs = 10, validation_data = test_dataset)

model.save("model1")
model.save("model1.h5")
print("banana")

