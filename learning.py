# learning.py - Learning Algorithms (Hebbian, BP, Reinforcement)
def hebbian_learning(pre_neuron, post_neuron, learning_rate=0.01):
    if pre_neuron.activate(1) and post_neuron.activate(1):
        return learning_rate  # Strengthening the connection
    return 0