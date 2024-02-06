#### Libraries
# Standard library
import random
import matplotlib.pyplot as plt #Agregamos la librería de Matplot para poder graficar

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)   
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.velocity_w = [np.zeros(w.shape) for w in self.weights]
        self.velocity_b = [np.zeros(b.shape) for b in self.biases]
        #COMENTARIOS DE LA PRIMERA SECCIÓN
        """En esta primera sección definimos el número de capas
        que hay en la red neuronal, a través del tamaño de SIZES
        Inicializamos los valores de pesos y biases como números randoms
        que toma valores entre 0 y 1.
        Así como el valor de los pesos, que se enlazan entre las capas de
        las neuronas"""
        
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights): #Aqui ya relacionamos los biases y los pesos
            #de las neuronas y las capas en un mismo conjunto
            a = sigmoid(np.dot(w, a)+b) #Y definimos a, utilizando los valores de w y b, ya
            #previamente encontrados
        return a #Regresamos el valor de a


    def SGD_Momentum(self, training_data, epochs, mini_batch_size, eta,
            test_data=None): #Definimos SGD junto con sus variables
    
        cost = [] #Inicializamos la lista de costo con cero valores para que esta pueda ser llenada
        if test_data:
            test_data = list(test_data) #Se crea una lista con los valores de prueba
            n_test = len(test_data) #Y definimos el tamaño de la lista

        training_data = list(training_data) #Se crea una lista con los valores
        #de entrenamiento
        n = len(training_data) #Y definimos el tamaño de esta lista
        
        for j in range(epochs):#Cada época se va a componer de la siguiente manera
            random.shuffle(training_data) #Revolvemos aleatoriamente los valores
            #de entrenamiento
            mini_batches = [ 
                training_data[k:k+mini_batch_size] #Los valores de entrenamiento
                 # se acoplan en subconjuntos Para dar un paso de entrenamiento
                for k in range(0, n, mini_batch_size)] 
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, momento=0.5) #Para cada mini_batch se le 
                #aplica el SGD, con el valor de eta
            if test_data:
                print("Epoch {0}: {1} / {2} y tiene costo: {3}".format( #Cuando se han usado todos
                    # los datos se les conoce como época
                    j, self.evaluate(test_data), n_test, self.funcion_costo_cross_entropy(test_data))) #Si diste datos de prueba te regresa el procentaje de aciertos
                cost.append(self.funcion_costo_cross_entropy(test_data)) #Imprimos los valores de eficiencia de encontrar el mínimo
            else:
                print("Epoch {0} complete".format(j)) #Cuando acaba, simplemente
                #marca completado
        
        #Generamos la función para poder visualizar las gráfica, definimos los datos de nuestros ejes que van a ser x, el número de épocas
        #Que se define como la magnitud de la lista costo
        numero_epocas = list(range(len(cost)))

        fig, ax = plt.subplots()

        ax.plot( numero_epocas, cost) #Y el eje y, va a estar determinado por el costo promedio de cada época

        ax.set(xlim=(0, len(cost)),
            ylim=(0, max(cost)*1.5)) #Y aumentamos los límites de nuestra gráfica
        plt.show() #El comando para lograr visualizar la gráfica

    def update_mini_batch(self, mini_batch, eta, momento=0.5):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #Definimos nabla b, como
        #un array de valores de los biases, donde primero los inicializamos
        #con el valor cero
        nabla_w = [np.zeros(w.shape) for w in self.weights] #Definimos nabla w, como
        #un array de valores de los pesos, donde primero los inicializamos
        #con el valor cero
        for x, y in mini_batch: #Aqui se asignan los valores para nabla de b y w
            #de cada neurona de cada capa
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) #Utilizamos el
            #función de backprop (que se explica posteriormente) para dar valores
            #a las variables que determinan el error en estos datos
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #A
            #nabla b se le zipea, se le junta con el valor de delta nabla b
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] #A
            #nabla w se le zipea, se le junta con el valor de delta nabla w
        self.velocity_w = [momento * v_m - (eta / len(mini_batch))*nw #Le damos los valores del SGD momento a las variables de velocidad
                           #Tenemos que el momento se multiplica por el gradiente anterior y este resta a lo que ya conociamos por SGD
                           for v_m, nw in zip(self.velocity_w, nabla_w)] #Esto se repite para cada peso
        self.velocity_b = [momento * v_b - (eta / len(mini_batch))*nb
                           for v_b, nb in zip(self.velocity_b, nabla_b)] # Lo mismo con los biases
        self.weights =[w + v_m for w, v_m in zip(self.weights, self.velocity_w) ] #Aqui actualizamos los valores de los pesos dandole el peso
                        #y las velocidades, así estos pesos serán los que evaluaremos en la red
        self.biases =[b + v_b for b, v_b in zip(self.biases, self.velocity_b) ]

    """Aquí definimos update_mini_batch que nos va a permitir realizar el SGD
    para cada época"""
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases] #Definimos nabla b, como
        #un array de valores de los biases, donde primero los inicializamos
        #con el valor cero
        nabla_w = [np.zeros(w.shape) for w in self.weights] #Definimos nabla w, como
        #un array de valores de los pesos, donde primero los inicializamos
        #con el valor cero
        # feedforward
        activation = x #Aquí es donde a la variable x, le damos el valor de
        #las activaciones
        activations = [x] #lista para guardar todas las activaciones
        zs = [] #lista para guardar todos los vectores z, capa por capa
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b #Determinamos que z es igual al producto de
            #los pesos por el valor de activación más los biases
            zs.append(z) #agregamos al final de la lista de vectores, el valor
            #de z
            activation = sigmoid(z) #Le damos a activación el valor de la función
            #sigmoide que depende de z
            activations.append(activation) #Agregamos al final de la lista de
            #activacions el valor de activation
        # backward pass
        delta = self.delta_new_cost_function(activations[-1], y) #Aquí la variable delta, nos va a permitir
        #calcular el error, entonces en esta primera línea calculamos
        #la derivada de la función de costo respecto a la última capa por
        #la derivada de la función sigmoide, también evaluada en la última capa
        nabla_b[-1] = delta #Le damos el valor de delta a la parcial de la
        #función de costo respecto a los biases en la última capa
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #Ahora
        #nabla w va a tener el valor del producto de delta por la activación
        #de la penúltima capa. Esto nos permite conocer la delta de error,
        #en las capas escondidas
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers): #Aqui tenemos ecuaciones muy
            #importantes del algoritmo backprop, estas ecuaciones principalmente
            #nos ayudan a determinar el "error" en nuestra red neuronal
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp #Esta
            #ecuación nos da el error en una capa, respecto al error de 
            #la capa siguiente
            nabla_b[-l] = delta #Delta se evalua en la misma neurona que bias
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose()) #Es la
            #ecuación por la taza de cambio del costo respecto a cualquier peso
            #de la red
        return (nabla_b, nabla_w) #Nos regresa los valores de las nablas
    

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data] #Buscamos evaluar nuestros
        #valores de prueba, entonces agarramos el máximo de estos para poder
        #determinar un resultado final
        return sum(int(x == y) for (x, y) in test_results) #Sumamos los valores
    #que si fueron "verdaderos", esto determinado por las neuronas para poder
    #determinar el resultado final

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    
    def funcion_costo_cuadratica(self, test_data): #Definimos la función de costo para que se le aplique a cada grupo de datos y poder graficar
        cost_x = [0.5*(np.square(np.argmax(self.feedforward(x)) - y))
                        for (x, y) in test_data] 
        cost_epoch = np.average(cost_x) #El costo de cada epoca, sera el promedio del costo de cada elemento

        return(cost_epoch)
    
    
    def delta_new_cost_function(self, output_activations, y): #Como fue con cost_derivative, veamos que las deltas de la función
        #de costo de cross entropy también se cumple que los bias = delta y los pesos = delta x las activaciones de la capa anterior
        return (output_activations-y)
    
    def vectorizando(self, j): #Veamos que los valores de "y" son vectores para poder operarlos, entonces esto es lo que hacemos
        #en esta función, vectorizar
        v = np.zeros((10, 1))
        v[j] = 1.0
        return v
    
    def funcion_costo_cross_entropy(self, test_data): #Definimos la funcion de costo cross entropy binary, para poder graficar
        cost_x = [] #Inicializamos con valores cero la lista de costo_x
        for (x, y) in test_data:
             y = self.vectorizando(y) #Vectorizamos y
             cost_x.append(np.nan_to_num(-y*np.log(self.feedforward(x)) - (1-y)*np.log(1-self.feedforward(x))))
             #Escribimos el valor de la función y agregamos el append, porque queremos que cada valor del costo_x se agregue como
             #ultimo elemento a la lista costo_x
        cost_epoch = np.average(cost_x) #Sacamos el promedio de todos los elementos de cost_x

        return(cost_epoch)
    

    
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z)) #Aquí simplemente definimos la función sigmoide

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z)) #Definimos la derivada de la función sigmoide