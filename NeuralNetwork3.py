import tensorflow as tf
from numpy.ma.core import shape
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Input, Dropout, Flatten, InputLayer
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras import models
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.preprocessing as prep




train_dataset = prep.image_dataset_from_directory(
    directory="Data/Bulgarian",
    image_size=(300, 300),
    validation_split=0.4,
    subset= "training",
    batch_size= 32,
    label_mode= "categorical",
    seed= 1

)

test_dataset = prep.image_dataset_from_directory(
    directory="Data/Bulgarian",
    image_size=(300, 300),
    validation_split=0.4,
    subset= "validation",
    batch_size = 32,
    label_mode = "categorical",
    seed=1
)


print(train_dataset.class_names)

print(test_dataset.class_names)


model = tf.keras.models.Sequential([
    Input(shape=(300, 300, 3)),
    Conv2D(32, 3, activation= relu),
    Conv2D(32, 3, activation= relu),
    Dropout(0.2),
    MaxPool2D(2),

    Conv2D(64, 3, activation= relu),
    Conv2D(64, 3, activation= relu),
    Dropout(0.2),
    MaxPool2D(2),

    Conv2D(128, 3, activation= relu),
    Conv2D(128, 3, activation= relu),
    Dropout(0.2),
    MaxPool2D(2),

    Conv2D(256, 3, activation=relu),
    Conv2D(256, 3, activation=relu),
    Dropout(0.2),
    MaxPool2D(2),

    Flatten(),
    Dense(512, activation=relu),
    Dropout(0.2),
    Dense(256, activation=relu),
    Dense(20, activation=softmax)

])


model.compile(loss= CategoricalCrossentropy(), optimizer = Adam(), metrics = ['accuracy'])

history = model.fit(train_dataset, epochs = 1, validation_data = test_dataset)

model.save("Models/model7")
print("banana")

