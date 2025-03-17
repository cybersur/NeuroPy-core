#synapses.py - Synaptic Connection Management
class Synapse:
    def __init__(self, weight=0.5):
        self.weight = weight
    
    def transmit(self, signal):
        return signal * self.weight