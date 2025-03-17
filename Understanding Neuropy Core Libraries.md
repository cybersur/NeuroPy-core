# Neurons.py - Core Neuron Models

## Overview

This module defines the **Neuron** class, a fundamental building block for artificial neural networks. The `Neuron` class simulates biological neurons by accumulating input signals and activating when a threshold is reached. This basic neuron model can be used to build more complex artificial intelligence or computational neuroscience applications.

## Features

- **Neuron Representation:** Models a simple artificial neuron with an activation threshold.
- **Excitatory and Inhibitory Neurons:** Neurons can be classified as excitatory (amplifying signals) or inhibitory (reducing signals).
- **Threshold-Based Activation:** The neuron fires (activates) when the accumulated potential reaches or exceeds a defined threshold.
- **Signal Accumulation:** Inputs increase the neuron’s potential, determining if it should activate.

---

## Run the Code

## NeuroPy Core Libraries - Initial Structure

Class Definition: Neuron

The Neuron class is a foundational unit in NeuroPy Core, representing a single neuron in Neural Cognitive Networks (NCNs).

Constructor: __init__

# neurons.py - Core Neuron Models

class Neuron:
    def __init__(self, neuron_type='excitatory', threshold=1.0):
        self.neuron_type = neuron_type
        self.threshold = threshold
        self.potential = 0.0

## NeuroPy Context:

    Defines a neuron with neuron_type (excitatory or inhibitory).
    Implements threshold, determining when the neuron activates.
    potential stores the accumulated signal strength.

## Parameters:

    neuron_type (str): Defines neuron behavior ('excitatory' amplifies signals, 'inhibitory' suppresses signals).
    threshold (float): Determines activation level (default: 1.0).

## Attributes:

    self.neuron_type: Stores the neuron classification.
    self.threshold: Defines activation threshold for the neuron.
    self.potential: Accumulates input signals.

# Activation Function: activate
    
    def activate(self, input_signal):
        self.potential += input_signal
        return self.potential >= self.threshold

## NeuroPy Cognitive Processing:

    Accumulates incoming signals to the neuron’s potential.
    If potential reaches threshold, the neuron activates and returns True.
    Used as a fundamental mechanism in NeuroPy learning algorithms.

## Parameters:

    input_signal (float): Incoming signal affecting neuron state.

## Returns:

    True if the neuron reaches threshold (fires).
    False otherwise.

Example Usage in NeuroPy
Creating a Neuron Instance in NeuroPy

	from neuropy.core.neurons import Neuron

	# Create an excitatory neuron with a threshold of 1.5
	neuron = Neuron(neuron_type='excitatory', threshold=1.5)

Activating a Neuron in NeuroPy Cognitive Processing

	print(neuron.activate(0.5))  # Output: False (Potential = 0.5)
	print(neuron.activate(1.0))  # Output: True  (Potential = 1.5, neuron fires)


Integration with NeuroPy Learning Algorithms

The Neuron class is designed to be compatible with NeuroPy learning algorithms, allowing for seamless integration with:

    NeuroPy.Learning – Supports training and adaptation of neural networks.
    NeuroPy.Synapse – Handles neuron interconnections for signal transmission.
    NeuroPy.Cognition – Implements higher-order cognitive functions.

Example of Neuron Integration with Learning Algorithms:

	from neuropy.core.neurons import Neuron
	from neuropy.learning import LearningAlgorithm

	# Create an excitatory neuron
	neuron = Neuron(neuron_type='excitatory', threshold=1.0)

	# Apply a learning algorithm from NeuroPy
	learning_model = LearningAlgorithm()
	learning_model.train(neuron, data=[0.3, 0.6, 0.9])

Future Enhancements in NeuroPy

    NeuroPy Cognitive Leak Model – Implementing decay factors for neuron potential to model signal leakage over time.
    Advanced Activation Functions – Support for Sigmoid, ReLU, Tanh, and custom NeuroPy activation mechanisms.
    Neuron Network Expansion – Integration with NeuroPy Graph Networks for multi-layered connectivity.


# synapses.py - Synaptic Connection Management
class Synapse:
    def __init__(self, weight=0.5):
        self.weight = weight
    
    def transmit(self, signal):
        return signal * self.weight

# 3. learning.py - Learning Algorithms (Hebbian, BP, Reinforcement)
def hebbian_learning(pre_neuron, post_neuron, learning_rate=0.01):
    if pre_neuron.activate(1) and post_neuron.activate(1):
        return learning_rate  # Strengthening the connection
    return 0

# 4. activation.py - Activation Functions (ReLU, Sigmoid, Custom)
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return max(0, x)

# 5. network.py - Neural Network Construction
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, inputs):
        for layer in self.layers:
            inputs = [neuron.activate(inp) for neuron, inp in zip(layer, inputs)]
        return inputs

# 6. cognition.py - Cognitive Models (Predictive Coding, Temporal Processing)
def predictive_coding(input_signal, prediction):
    error_signal = input_signal - prediction
    return error_signal

# 7. neurogen.py - Neurogenesis & Adaptive Growth
class Neurogenesis:
    def __init__(self):
        self.neurons = []
    
    def generate_neuron(self, neuron_type='excitatory'):
        neuron = Neuron(neuron_type)
        self.neurons.append(neuron)
        return neuron

# 8. utils.py - Utility Functions for Neural Computation
def normalize(data):
    min_val, max_val = min(data), max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

Author

Developed by Dr. Rigoberto Garcia – Contributions to the NeuroPy project are welcome!
