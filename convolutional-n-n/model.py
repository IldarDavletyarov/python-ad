import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import  Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keract import get_activations, display_activations
datagen = ImageDataGenerator()

train_data = datagen.flow_from_directory("Covid19-dataset/train/",target_size=(64,64))
test_data = datagen.flow_from_directory("Covid19-dataset/test/",target_size=(64,64))

model = Sequential()
model.add(Conv2D(filters=48,kernel_size=3,activation="relu",input_shape=[64,64,3]))
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Conv2D(filters=48,kernel_size=3,activation="relu"))
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Conv2D(filters=32,kernel_size=3,activation="relu"))
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(3,activation="softmax"))
#adam
#sgd
#rmsprop
#____
#categorical_hinge
#cosine_similarity
#categorical_crossentropy
model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(x=train_data,epochs=1)
sc = model.evaluate(test_data)
print(sc[1]*100)