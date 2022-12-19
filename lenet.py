import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn import metrics
warnings.filterwarnings("ignore")
train_path = 'animals_split/train'
test_path = 'animals_split/val'
e = 300
b = 20
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                target_size = (224, 224),
                                                batch_size = b,
                                                class_mode = 'categorical')

val_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224,224),
                                            batch_size = b,
                                            class_mode = 'categorical')
print(training_set)

model1 = Sequential()
model1.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(224, 224, 3)))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Flatten())
model1.add(Dense(120, activation='relu'))
model1.add(Dense(84, activation='relu'))
model1.add(Dense(4, activation='softmax'))
model1.summary()
model1.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model_1 = model1.fit(training_set, epochs=e, verbose=1, validation_data=val_set)
model1.save('test/lenet.h5')

Number_Of_Epochs = range(0, e)

plt.plot(Number_Of_Epochs, model_1.history['accuracy'], color = 'green', marker = '*', label = 'Training Accuracy')
plt.plot(Number_Of_Epochs, model_1.history['val_accuracy'], color = 'blue', marker = 'o', label = 'Validation Accuracy')
plt.title('Training Accuracy and Validation Accuracy Vs Epochs for lenet')
plt.legend()
plt.show()

plt.plot(Number_Of_Epochs, model_1.history['loss'], color = 'green', marker = '*', label = 'Training Loss')
plt.plot(Number_Of_Epochs, model_1.history['val_loss'], color = 'blue', marker = 'o', label = 'Validation Loss')
plt.title('Training Loss and Validation Loss Vs Epochs for lenet')
plt.legend()
plt.show()

test_steps_per_epoch = np.math.ceil(val_set.samples / val_set.batch_size)
predictions = model1.predict_generator(val_set , steps=test_steps_per_epoch)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_set.classes
class_labels = list(val_set.class_indices.keys())
print("lenet Architecture")  
report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

cm = tf.math.confusion_matrix(labels=true_classes,predictions=predicted_classes)
plt.figure(figsize = (10,7))
ax= plt.subplot()
sn.heatmap(cm, annot=True, fmt='d',ax=ax)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels') 
ax.set_title('Confusion Matrix') 
ax.xaxis.set_ticklabels(['buffalo','elephant','rhino','zebra']); ax.yaxis.set_ticklabels(['buffalo','elephant','rhino','zebra'])
plt.show()

