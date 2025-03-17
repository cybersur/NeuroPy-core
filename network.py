# network.py - Neural Network Construction
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, inputs):
        for layer in self.layers:
            inputs = [neuron.activate(inp) for neuron, inp in zip(layer, inputs)]
        return inputs