from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from keras.models import model_from_json
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from resizeImage import fixData, loadAndGetData, getHorVert, convTekst

#Henter inn dataen
trainImages, trainLabels = loadAndGetData('train.npy')
testImages, testLabels = loadAndGetData('test.npy')
img_horizontal, img_vertical = getHorVert()
img_size_flat = img_vertical * img_horizontal
img_shape_full = (img_vertical, img_horizontal, 1)

#1 for gråskala bilder
num_channels = 1

#Antall klasser
num_classes = 2

#Tensorboard
tbCallBack = TensorBoard(log_dir='./tensorboard/test_01',
	histogram_freq=0, write_graph=True, write_images=True)

#Keras sequential model
model = Sequential()
model.add(InputLayer(input_shape=(img_size_flat,)))
model.add(Reshape(img_shape_full))

#Kernel_size er masken som ser på det orginale bildet
#Stride er antall piksler masken blir forskøvet.
#Filter er antall filtere som blir brukt
#Relu activation er for å sette negative vekter til null
model.add(Conv2D(kernel_size=3, strides=1, filters=18, padding='same',
                 activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=3, strides=1, filters=20, padding='same',
                 activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=3, strides=1, filters=24, padding='same',
                 activation='relu', name='layer_conv3'))

model.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same',
                 activation='relu', name='layer_conv4'))

model.add(Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                 activation='relu', name='layer_conv5'))
model.add(MaxPooling2D(pool_size=2, strides=2))

#Bruker flatten for å gjøre klart til denselagene
model.add(Flatten())

#Denselag med relu
model.add(Dense(128, activation='relu'))
model.add(Dense(160, activation='relu'))
model.add(Dense(256 , activation='relu'))

#Siste denselag. Bruker softmax for å summere vektene til 1
model.add(Dense(num_classes, activation='softmax'))

#Læringskurve
optimizer = Adam(lr=1.0e-6)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])	  	  

#Antall ganger den trener(epochs), og hvor mange bilder den tar av gangen(Batch_size)
model.fit(x=trainImages,
		  y=trainLabels,
		  epochs=2, batch_size=16,
		  callbacks=[tbCallBack])

#Lagre modellen
model_json = model.to_json()
with open("model.json", "w") as json_file:
	json_file.write(model_json)
model.save_weights("model.h5")

#Skriver ut loss og nøyaktighet
result = model.evaluate(x=testImages, y=testLabels)
print("\ntest results: ")
for name, value in zip(model.metrics_names, result):
	print(name, value)


