import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
import seaborn as sn


train_path = 'animals_split/train'
test_path = 'animals_split/val'
e = 200
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

pretrained_model3 = tf.keras.applications.VGG16(input_shape=(224,224,3),include_top=False,weights='imagenet',pooling='avg')
pretrained_model3.trainable = False
inputs3 = pretrained_model3.input
x3 = tf.keras.layers.Dense(224, activation='relu')(pretrained_model3.output)
outputs3 = tf.keras.layers.Dense(4, activation='softmax')(x3)
model = tf.keras.Model(inputs=inputs3, outputs=outputs3)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
his=model.fit(training_set, epochs=e, verbose=1, validation_data=val_set)
model.save("test/VGG.h5")
model = keras.models.load_model("VGG.h5")
model.summary()


Number_Of_Epochs = range(0, e)
plt.title('Training Accuracy and Validation Accuracy Vs Epochs for VGG')
plt.plot(Number_Of_Epochs, his.history['accuracy'],color = 'green', marker = '*',label='Training Accuracy')
plt.plot(Number_Of_Epochs, his.history['val_accuracy'],color = 'blue',marker = 'o',label='Validation Accuracy')
plt.legend()
plt.show()
plt.title('Training Loss and Validation Loss Vs Epochs for VGG')
plt.plot(Number_Of_Epochs, his.history['loss'],color = 'green',marker = '*',label='Training Los')
plt.plot(Number_Of_Epochs, his.history['val_loss'],color = 'blue',marker = 'o',label='Validation Loss')
plt.legend()
plt.show()


test_steps_per_epoch = np.math.ceil(val_set.samples / val_set.batch_size)
predictions = model.predict(val_set , steps=test_steps_per_epoch)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_set.classes
class_labels = list(val_set.class_indices.keys())
print("VGG Architecture")
report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

cm = tf.math.confusion_matrix(labels=true_classes,predictions=predicted_classes)
plt.figure(figsize = (10,7))
ax= plt.subplot()
sn.heatmap(cm, annot=True, fmt='d',ax=ax)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels') 
ax.set_title('Confusion Matrix for VGG') 
ax.xaxis.set_ticklabels(['buffalo','elephant','rhino','zebra']); ax.yaxis.set_ticklabels(['buffalo','elephant','rhino','zebra'])
plt.show()



