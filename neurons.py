# neurons.py - Core Neuron Models
class Neuron:
    def __init__(self, neuron_type='excitatory', threshold=1.0):
        self.neuron_type = neuron_type
        self.threshold = threshold
        self.potential = 0.0
    
    def activate(self, input_signal):
        self.potential += input_signal
        return self.potential >= self.threshold