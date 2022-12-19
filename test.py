import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('alexnet.h5')

image=cv2.imread(r"C:\Users\clins\Desktop\Ai Processor final prj\Animals\rhino\001.jpg")

img=Image.fromarray(image)

img=img.resize((224,224))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result=np.argmax(model.predict(input_img), axis=-1)

print(result)

# import cv2
# import os
# import tensorflow as tf
# from tensorflow import keras
# from PIL import Image
# import numpy as np
# from sklearn.model_selection import train_test_split
# from keras.utils.np_utils import normalize 
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense
# from keras.utils.np_utils import to_categorical

# #Change Directory according to the image directory on your computer
# buffalo_1="Animals/buffalo"
# elephant_1="Animals/elephant"
# rhino_1="Animals/rhino"
# zebra_1="Animals/zebra"

# buffalo=os.listdir("Animals/buffalo")
# elephant=os.listdir("Animals/elephant")
# rhino=os.listdir("Animals/rhino")
# zebra=os.listdir("Animals/zebra")

# dataset=[]
# label=[]

# INPUT_SIZE=64

# for i , image_name in enumerate(buffalo):
#     print(i,image_name)
#     if(image_name.split('.')[1]=='jpg'):
#         image=cv2.imread(buffalo_1+image_name)
#         print(image)
#         image=Image.fromarray(image,'RGB')
#         image=image.resize((INPUT_SIZE,INPUT_SIZE))
#         dataset.append(np.array(image))
#         label.append(0)

# for i , image_name in enumerate(melanoma_images):
#     if(image_name.split('.')[1]=='jpg'):
#         image=cv2.imread(melanoma_dir+image_name)
#         image=Image.fromarray(image, 'RGB')
#         image=image.resize((INPUT_SIZE,INPUT_SIZE))
#         dataset.append(np.array(image))
#         label.append(1)

# dataset=np.array(dataset)
# label=np.array(label)


# x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.20, random_state=0)


# x_train=normalize(x_train, axis=1)
# x_test=normalize(x_test, axis=1)

# y_train=to_categorical(y_train , num_classes=4)
# y_test=to_categorical(y_test , num_classes=4)