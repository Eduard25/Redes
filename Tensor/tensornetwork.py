#https://www.tutorialspoint.com/keras/keras_installation.htm
#https://docs.python.org/es/3/tutorial/venv.html

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD


#Para escribir logs en wandb
#pip install wandb
#wandb login

learning_rate = 0.4
epochs = 30
batch_size = 50
momentum = 0.5

import wandb
from wandb.keras import WandbCallback, WandbMetricsLogger, WandbModelCheckpoint
wandb.init(project="tensor1")
wandb.config.learning_rate = learning_rate
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size
#wandb.config.patito = "cuacCuac"
###################
#import mlflow
#mlflow.tensorflow.autolog()

#keras.optimizers.SGD(
#    learning_rate=0.01,
#    momentum=0.5,
#    nesterov=False,
#    weight_decay=None,
#    clipnorm=None,
#    clipvalue=None,
#    global_clipnorm=None,
#    use_ema=False,
#    ema_momentum=0.99,
#    ema_overwrite_frequency=None,
#    name="SGD",
    #kwargs
#)

dataset=mnist.load_data()
#print(len(dataset))


(x_train, y_train), (x_test, y_test) = dataset

#print(y_train.shape)
#print(x_train.shape)
#print(x_test.shape)

#x_train=x_train[0:8000]
#x_test=x_train[0:1000]

#y_train=y_train[0:8000]
#y_test=y_train[0:1000]

x_trainv = x_train.reshape(60000, 784)
x_testv = x_test.reshape(10000, 784)

#print(x_trainv[3])

x_trainv = x_trainv.astype('float32')
x_testv = x_testv.astype('float32')

x_trainv /= 255  # x_trainv = x_trainv/255
x_testv /= 255

#print("linea 40--------")
#print(x_trainv[3])

#print(x_train.shape)
#print(x_trainv.shape)

num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)

#print(y_trainc[6:15])

model = Sequential()
model.add(Dense(256, activation='softplus', input_shape=(784,)))
#model.add(Dropout(0.2))
model.add(Dense(256, activation='softmax'))
#model.add(Dense(num_classes, activation='softmax'))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=learning_rate),metrics=['accuracy'])

history = model.fit(x_trainv, y_trainc,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_testv, y_testc),
                    callbacks=[
                        WandbMetricsLogger(log_freq="epochs"),
                        WandbModelCheckpoint("models")
                    ])


score = model.evaluate(x_testv, y_testc, verbose=1)
print(score)
a=model.predict(x_testv)
#b=model.predict_proba(x_testv)
print(a.shape)
print(a[1])
print("resultado correcto:")
print(y_testc[1])

#Para guardar el modelo en disco
model.save("red.h5")
exit()
#para cargar la red:
modelo_cargado = tf.keras.models.load_model('red.h5')

#Softmax, relu, softmax, 64, 0.05 Está como en 78 acurracy
#Softmax, relu, softmax, 64, 0.001 Está como en 15 acurracy
#Softmax, relu, softmax, 64, 0.3 Está como en 90 acurracy
#Softplus, softmax, sigmoide 256, 0.3 Está como en 95 acurracy :0
#Softplus, softmax, tanh 256, 0.5 Está como en 08.92 acurracy y ya no sube ni baja :0