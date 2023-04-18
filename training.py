#Importing the Dependencies

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

#reading data folder and assigning to classes
data_file = r"data"
classes = ["with_mask", "without_mask"]


data = []
labels = []

for c in classes:
    path = os.path.join(data_file, c)
    for i in os.listdir(path):
        image_path = os.path.join(path, i)
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        image_array = preprocess_input(img)

        data.append(image_array)
        labels.append(c)

# binarizing labels  
binarizer = LabelBinarizer() #creating an instance of LabelBinarizer
binary_labels = binarizer.fit_transform(labels) 
labels = to_categorical(binary_labels)

#converting image_array and labels to numpy array
X = np.array(data, dtype="float32")
y = np.array(labels)

#splitting the data and labels
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = 0.2, stratify = labels, random_state = 42)

#generating batches of  image data with data augmentation
aug = ImageDataGenerator(rotation_range = 20,
    zoom_range = 0.15,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.15,
    horizontal_flip = True)

#transfer learning using MobileNet architecture
base_model = MobileNetV2(include_top = False, input_tensor = Input(shape = (224, 224, 3)))

head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

#keeping parameters of base_model layers fixed
for layer in base_model.layers:
    layer.trainable = False

print("Compiling model...")

#compiling the model
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate= 0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

print("Model compiled.")

#training the model
print("Training head...")
hist = model.fit(
    aug.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    validation_data=(X_test, y_test),
    validation_steps=len(X_test) // 32,
    epochs=20)

#evaluating the model
print("Evaluating model...")
pred_prob = model.predict(X_test, batch_size=32)

pred_id = np.argmax(pred_prob, axis=1)

print(classification_report(y_test.argmax(axis=1), pred_id, target_names=binarizer.classes_))

#saving the model
print("Saving mask detector model...")
model.save("mask_detector2.model", save_format="h5")

#plotting the training loss and accuracy against epoch
epochs = 20
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), hist.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), hist.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), hist.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), hist.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")