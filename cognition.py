# cognition.py - Cognitive Models (Predictive Coding, Temporal Processing)
def predictive_coding(input_signal, prediction):
    error_signal = input_signal - prediction
    return error_signal