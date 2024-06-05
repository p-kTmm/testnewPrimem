import random
import numpy as np

class Memory:
    def __init__(self, size_max, size_min, priority=True, alpha=0.8):
        self._samples = []
        self._priorities = []
        self._size_max = size_max
        self._size_min = size_min
        self._priority = priority
        self._alpha = alpha  # How much prioritization is used (0 - no prioritization, 1 - full prioritization)

    def add_sample(self, sample, priority=1.0):
        """
        Add a sample into the memory with a given priority
        """
        self._samples.append(sample)
        if self._priority:
            self._priorities.append(priority if priority > 0 else 1e-6)  # Avoid zero priority
        if self._size_now() > self._size_max:
            self._samples.pop(0)  # Remove the oldest element
            if self._priority:
                self._priorities.pop(0)  # Remove the oldest priority

    def get_samples(self, n):
        """
        Get n samples randomly from the memory based on their priority if priority is enabled
        """
        if self._size_now() < self._size_min:
            return [], []

        if n > self._size_now():
            n = self._size_now()

        if self._priority:
            priorities = np.array(self._priorities) ** self._alpha
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(len(self._samples), n, p=probabilities)
        else:
            indices = np.random.choice(len(self._samples), n)
        
        samples = [self._samples[i] for i in indices]
        return samples, indices  # Return indices to update priorities later if needed

    def update_priorities(self, indices, td_errors):
        """
        Update the priorities of the samples at the given indices based on TD errors
        """
        if self._priority:
            for idx, td_error in zip(indices, td_errors):
                self._priorities[idx] = abs(td_error) if abs(td_error) > 0 else 1e-6  # Avoid zero priority

    def _size_now(self):
        """
        Check how full the memory is
        """
        return len(self._samples)
