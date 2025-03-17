# utils.py - Utility Functions for Neural Computation
def normalize(data):
    min_val, max_val = min(data), max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]