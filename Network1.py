"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

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

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        training_data = list(training_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
#### Libraries
# Standard library
import random

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

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None): #Definimos SGD junto con sus variables
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
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
                self.update_mini_batch(mini_batch, eta) #Para cada mini_batch se le 
                #aplica el SGD, con el valor de eta
            if test_data:
                print("Epoch {0}: {1} / {2}".format( #Cuando se han usado todos
                    # los datos se les conoce como época
                    j, self.evaluate(test_data), n_test)) #Imprimos los valores
                #de eficiencia de encontrar el mínimo
            else:
                print("Epoch {0} complete".format(j)) #Cuando acaba, simplemente
                #marca completado
    """En esta parte, desarrollamos Stochastic Gradient Descent, lo que quiere
    decir que, nuestra red neuronal, va a aprender a llegar al mínimo de la
    función a través de la repetición de esta función"""

    def update_mini_batch(self, mini_batch, eta):
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
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)] #Determinamos que el
        #valor para pesos este relacionado con la tasa de aprendizaje y el tamaño de
        #el mini_batch
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)] #Determinamos que el
        #valor para biases este relacionado con la tasa de aprendizaje y el tamaño de
        #el mini_batch
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
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1]) #Aquí la variable delta, nos va a permitir
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

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z)) #Aquí simplemente definimos la función sigmoide

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z)) #Definimos la derivada de la función sigmoide
