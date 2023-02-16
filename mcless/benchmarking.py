import numpy as np
from time import perf_counter

class ModelTest(object):
    def __init__(self, mcless, classifier):
        self.mcless = mcless
        self.model = classifier