from keras.applications.resnet import ResNet50
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from glob import glob
from sklearn import metrics
import seaborn as sn
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
e = 2
b = 40
IMAGE_SIZE = [224, 224]
IMAGE_RESIZE = 224
train_path = 'animals_split/train'
test_path = 'animals_split/val'
folders = glob(train_path + '/*')
print("Number of classes:", len(folders))
num_classes = len(folders)
nb_epochs = e
image_size = IMAGE_RESIZE
NUM_CLASSES = num_classes
CHANNELS = 3

IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'

LOSS_METRICS = ['accuracy']

NUM_EPOCHS = 60
EARLY_STOP_PATIENCE = 3
STEPS_PER_EPOCH_TRAINING = 10
STEPS_PER_EPOCH_VALIDATION = 10
BATCH_SIZE_TRAINING = b
BATCH_SIZE_VALIDATION = b

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
    train_path,
    target_size=(image_size, image_size),
    batch_size=BATCH_SIZE_TRAINING,
    class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
    test_path,
    target_size=(image_size, image_size),
    batch_size=BATCH_SIZE_VALIDATION,
    class_mode='categorical')


resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)





model4 = Sequential()
model4.add(ResNet50(include_top=False, pooling=RESNET50_POOLING_AVERAGE, weights='imagenet'))
model4.add(Dense(num_classes, activation='softmax'))
model4.layers[0].trainable = False
model4.summary()
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)#, nesterov=True
model4.compile(optimizer=sgd, loss=OBJECTIVE_FUNCTION, metrics=LOSS_METRICS)


fit_history = model4.fit(
    train_generator,
    epochs=nb_epochs,
    validation_data=validation_generator

)  

model4.save("test/resnet50.h5")

Number_Of_Epochs = range(0, e)
plt.title('Training Accuracy and Validation Accuracy Vs Epochs for VGG')
plt.plot(Number_Of_Epochs, fit_history.history['accuracy'], color='green', marker='*', label='Training Accuracy')
plt.plot(Number_Of_Epochs, fit_history.history['val_accuracy'], color='blue', marker='o', label='Validation Accuracy')
plt.legend()
plt.show()
plt.title('Training Loss and Validation Loss Vs Epochs for VGG')
plt.plot(Number_Of_Epochs, fit_history.history['loss'], color='green', marker='*', label='Training Los')
plt.plot(Number_Of_Epochs, fit_history.history['val_loss'], color='blue', marker='o', label='Validation Loss')
plt.legend()
plt.show()

test_steps_per_epoch = np.math.ceil(validation_generator.samples / validation_generator.batch_size)
predictions = model4.predict(validation_generator, steps=test_steps_per_epoch)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())
print("ResNet Architecture")
report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)




cm = tf.math.confusion_matrix(labels=true_classes,predictions=predicted_classes)
plt.figure(figsize = (10,7))
ax= plt.subplot()
sn.heatmap(cm, annot=True, fmt='d',ax=ax)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels') 
ax.set_title('Confusion Matrix for Resnet') 
ax.xaxis.set_ticklabels(['buffalo','elephant','rhino','zebra']); ax.yaxis.set_ticklabels(['buffalo','elephant','rhino','zebra'])
plt.show()


