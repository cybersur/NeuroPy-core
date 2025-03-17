# neurogen.py - Neurogenesis & Adaptive Growth
class Neurogenesis:
    def __init__(self):
        self.neurons = []
    
    def generate_neuron(self, neuron_type='excitatory'):
        neuron = Neuron(neuron_type)
        self.neurons.append(neuron)
        return neuron