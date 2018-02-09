
# coding: utf-8

# In[1]:


# configuration parameters
class Config:
    def __init__(self):
        self.POPULATION_SIZE = 1 # 150
        self.INPUTS = 4
        self.OUTPUTS = 2

CONFIG = Config()


# In[2]:


# TODO: incoming_edge can be a list of innovation numbers

NEURON_TYPES = ["input", "output", "hidden"]

class Neuron:
    NEURON_ID = 0

    def __init__(self, neuron_type = "hidden", incoming_edges = [], value = 0.0):
        self.id = type(self).NEURON_ID
        self.type = neuron_type
        self.incoming_edges = incoming_edges
        self.value = value
        type(self).NEURON_ID += 1

    def reset(self):
        self.value = 0.0

    def __repr__(self):
        return str(self.__dict__)


# In[3]:


# Edge between 2 neurons

class Gene:
    INNOVATION_NUMBER = 0

    def __init__(self, input = 0, output = 0, weight = 0.0, enabled = True):
        self.input = input
        self.output = output
        self.weight = weight
        self.enabled = enabled
        self.innovation = type(self).INNOVATION_NUMBER
        type(self).INNOVATION_NUMBER += 1

    def __repr__(self):
        return str(self.__dict__)


# In[4]:


from math import exp
import numpy as np

# from config.py import *

# The neural network itself, consisting of neurons and edges (a.k.a genes)
class Genome:
    def __init__(self, neurons = [], genes = []):
        self.neurons = neurons
        self.genes = genes # list of edges
        self.fitness = 0

    def __repr__(self):
        return str(self.__dict__)

    @staticmethod
    def generate():
        input_neurons = [Neuron('input') for _ in range(CONFIG.INPUTS)]
        output_neurons = [Neuron('output') for _ in range(CONFIG.OUTPUTS)]

        neurons = input_neurons + output_neurons

        genes = [Gene(input=input_neuron_index, output=output_neuron_index, weight=np.random.randn())
                 for input_neuron_index, _ in enumerate(input_neurons)
                 for output_neuron_index, _ in enumerate(output_neurons)]

        for output_neuron_index, output_neuron in enumerate(output_neurons):
            output_neuron.incoming_edges = [gene_index for gene_index, gene in enumerate(genes)
                                            if gene.output == output_neuron_index]

        # TODO: add random weights

        return Genome(neurons=neurons, genes=genes)

    @staticmethod
    def sigmoid(x):
#         return 2 / (1 + exp(-4.9 * x)) - 1
        return 1 / (1 + exp(-4.9 * x))

    def evaluate_neuron(self, neuron):
        if neuron.value != 0.0:
            return neuron.value

        neuron.value = self.sigmoid(sum(
            [self.genes[edge_index].weight * self.evaluate_neuron(self.neurons[self.genes[edge_index].input])
             for edge_index in neuron.incoming_edges]
        ))
        return neuron.value

    def activate(self, input_values):
        if len(input_values) != CONFIG.INPUTS:
            raise Error("invalid inputs length of {}".format(len(input_values)))

        input_index = 0
        for neuron in self.neurons:
            if neuron.type == 'input':
                neuron.value = input_values[input_index]
                input_index += 1
            else:
                neuron.reset()

        output_activations = [self.evaluate_neuron(neuron)
                              for neuron in self.neurons
                              if neuron.type == 'output']
        return output_activations


# In[5]:


class Population:
    def __init__(self):
        self.genomes = [Genome.generate() for _ in range(CONFIG.POPULATION_SIZE)]

    def __repr__(self):
        return str(self.__dict__)


# In[6]:


population = Population()

genome = population.genomes[0]
genome.activate([np.random.randn() for _ in range(CONFIG.INPUTS)])


# In[7]:


import numpy as np
n1 = Neuron('hidden', [], 1)
n1.value

np.random.randn()


# In[7]:


import gym
import time
env = gym.make('CartPole-v0')

# print(gym.envs.registry.all())

for episode in range(2000):
    observation = env.reset()
    done = False
    timestamp = 0

    while not done:
        timestamp = timestamp + 1
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        #print("Reward: {}".format(reward))
        if done:
            print("Episode finished after {} timestamps".format(timestamp))

env.close()
