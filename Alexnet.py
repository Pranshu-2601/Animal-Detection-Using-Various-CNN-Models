import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn import metrics
warnings.filterwarnings("ignore")

train_path = 'animals_split/train'
test_path = 'animals_split/val'

s = 224
e = 200
b = 20

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                target_size = (s,s),
                                                batch_size = b,
                                                class_mode = 'categorical')

val_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (s, s),
                                            batch_size = b,
                                            class_mode = 'categorical')

model2 = Sequential()
model2.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11),strides=(4,4), padding='valid'))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model2.add(BatchNormalization())
model2.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model2.add(BatchNormalization())
model2.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model2.add(BatchNormalization())
model2.add(Flatten())
model2.add(Dense(4096, input_shape=(224*224*3,)))
model2.add(Activation('relu'))
model2.add(Dropout(0.4))
model2.add(BatchNormalization())
model2.add(Dense(4096))
model2.add(Activation('relu'))
model2.add(Dropout(0.4))
model2.add(BatchNormalization())
model2.add(Dense(1000))
model2.add(Activation('relu'))
model2.add(Dropout(0.4))
model2.add(BatchNormalization())
model2.add(Dense(4))
model2.add(Activation('softmax'))
model2.summary()
model2.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(1E-5),metrics=['accuracy'])
 


model_2 = model2.fit(training_set,
                        epochs = e,
                        validation_data = val_set
                           )
model2.save('test/Alexnet1.h5')

Number_Of_Epochs = range(0, e)

plt.plot(Number_Of_Epochs, model_2.history['accuracy'], color = 'green', marker = '*', label = 'Training Accuracy')
plt.plot(Number_Of_Epochs, model_2.history['val_accuracy'], color = 'blue', marker = 'o', label = 'Validation Accuracy')
plt.title('Training Accuracy and Validation Accuracy Vs Epochs for Alexnet')
plt.legend()
plt.show()

plt.plot(Number_Of_Epochs, model_2.history['loss'], color = 'green', marker = '*', label = 'Training Loss')
plt.plot(Number_Of_Epochs, model_2.history['val_loss'], color = 'blue', marker = 'o', label = 'Validation Loss')
plt.title('Training Loss and Validation Loss Vs Epochs for Alexnet')
plt.legend()
plt.show()

test_steps_per_epoch = np.math.ceil(val_set.samples / val_set.batch_size)
predictions = model2.predict_generator(val_set , steps=test_steps_per_epoch)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_set.classes
class_labels = list(val_set.class_indices.keys())
print("Alexnet Architecture")  
report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)




cm = tf.math.confusion_matrix(labels=true_classes,predictions=predicted_classes)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('confusion matrix for Alexnet')
plt.show()

cm = tf.math.confusion_matrix(labels=true_classes,predictions=predicted_classes)
plt.figure(figsize = (10,7))
ax= plt.subplot()
sn.heatmap(cm, annot=True, fmt='d',ax=ax)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels') 
ax.set_title('Confusion Matrix for Alexnet') 
ax.xaxis.set_ticklabels(['buffalo','elephant','rhino','zebra']); ax.yaxis.set_ticklabels(['buffalo','elephant','rhino','zebra'])
plt.show()