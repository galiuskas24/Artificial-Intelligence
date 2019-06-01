from networkLayers import *
from transferFunctions import *
from neuralNet import *
from geneticAlgorithm import *
import plotter
import matplotlib.pyplot as plt
import numpy as np
import dataLoader
import os, sys
import json


'''
# --------------------CONFIG FILE FORMAT------------------------
config['hidden_layers']       -> Hidden layers architecture
config['elitism']             -> Keep this many of top units in each iteration
config['populationSize']      -> The number of chromosomes
config['mutationProbability'] -> Probability of mutation
config['mutationScale']       -> Standard deviation of the gaussian noise
config['numIterations']       -> Number of iterations to run the genetic algorithm for
config['errorThreshold']      -> Lower threshold for the error while optimizing
config['train_data_path']     -> Path of train data file
config['test_data_path']      -> Path of test data file
'''


def load_config(path):
    with open(path) as f:
        data = json.load(f)

    return data


def parse_function(function):
    function = function.lower()

    if function == "relu": real_function = reLU
    elif function == "sigmoid": real_function = sigmoid
    elif function == "leakyrelu": real_function = leakyReLU
    elif function == "tanh": real_function = tanh
    else: real_function = None

    return real_function


def create_model(NN, architecture):

    print "ARCHITECTURE:", architecture
    last_layer = None

    for layer in architecture.split('-'):
        if layer.isdigit():
            if not last_layer: last_layer = int(layer)
            else:
                this_layer = int(layer)
                NN.addLayer(LinearLayer(last_layer, this_layer))
                last_layer = this_layer
        else:
            NN.addLayer(FunctionLayer(parse_function(layer)))


if __name__ == '__main__':

    # --------------LOAD CONFIGURATION-------------------
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/sin_config.json"
    config = load_config(config_path)
    if len(sys.argv) > 2: config['hidden_layers'] = sys.argv[2] # set hidden layers with arg

    TRAIN = os.path.join('data', config['train_data_path'])
    TEST = os.path.join('data', config['test_data_path'])

    # Setting the random seed forces the same results of randoming each time you start the program
    np.random.seed(11071998)

    X_train, y_train = dataLoader.loadFrom(TRAIN)
    X_test, y_test = dataLoader.loadFrom(TEST)

    print "Train data shapes: ", X_train.shape, y_train.shape
    print "Test data shapes: ", X_test.shape, y_test.shape

    # The dimensionality of the input layer of the network is the second dimension of the shape
    input_size = X_train.shape[1] if len(X_train.shape) > 1 else 1

    # The size of the output layer
    output_size = 1

    # -------------DEFINE NEURAL NETWORK---------------------
    NN = NeuralNetwork()
    layers = str(input_size) + '-' + config['hidden_layers'] + '-' + str(output_size)
    create_model(NN, layers)


    def errorClosure(w):
        """
            A closure is a variable that stores a function along with the environment.
            The environment, in this case are the variables x, y as well as the NN
            object representing a neural net. We store them by defining a method inside
            a method where those values have been initialized. This is a "hacky" way of
            enforcing the genetic algorithm to work in a generalized manner. This way,
            the genetic algorithm can be applied to any problem that optimizes an error
            (in this case, this function) by updating a vector of values (in this case,
            defined only by the initial size of the vector).

            In plain - the genetic algorithm doesn't know that the neural network exists,
            and the neural network doesn't know that the genetic algorithm exists.
        """
        # Set the weights to the pre-defined network
        NN.setWeights(w)
        # Do a forward pass of the network and evaluate the error according to the oracle (y)
        return NN.forwardStep(X_train, y_train)

    # --------------GA SIMULATION-------------------
    print_every = config['print_every']
    plot_every = config['plot_every']

    GA = GeneticAlgorithm(NN.size(), errorClosure,
                          elitism=config['elitism'],
                          populationSize=config['populationSize'],
                          mutationProbability=config['mutationProbability'],
                          mutationScale=config['mutationScale'],
                          numIterations=config['numIterations'],
                          errorThreshold=config['errorThreshold'])

    # emulated do-while loop
    done = False
    while not done:
        done, iteration, best = GA.step()

        if iteration % print_every == 0:
            print "Error at iteration %d = %f" % (iteration, errorClosure(best))

        if iteration % plot_every == 0:
            NN.setWeights(best)
            plotter.plot(X_train, y_train, NN.output(X_train))
            plotter.plot_surface(X_train, y_train, NN)

    print "Training done, running on test set"
    NN.setWeights(best)

    print "Error on test set: ", NN.forwardStep(X_test, y_test)
    plotter.plot(X_test, y_test, NN.output(X_test))
    plotter.plot_surface(X_test, y_test, NN)
