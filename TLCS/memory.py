import random
import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])
        
    def pop(self):
        oldest_index = self.write  # Point to the oldest entry
        idx = oldest_index + self.capacity - 1
        self.update(idx, 0)  # Set the priority to zero to effectively remove it
        # Optionally, reset the data point if necessary
        self.data[oldest_index] = None

class Memory:
    e = 0.01
    a = 0.8
    beta = 0.4
    beta_increment_per_sampling = 0.001
    absolute_error_upper = 1.0  # Giới hạn lỗi tuyệt đối tối đa

    def __init__(self, size_max, size_min):
        self.tree = SumTree(size_max)
        self.size_min = size_min
        self.size_max = size_max 
        
    def _get_priority(self, error):
        # clipped_error = np.minimum(np.abs(error) + self.e, self.absolute_error_upper)  # Giới hạn lỗi
        # return clipped_error ** self.a
        return (np.abs(error) + self.e) ** self.a


    def add_sample(self, sample, error):
        """
        Add a sample into the memory with a given priority
        """
        p = self._get_priority(error)
        if self.tree.n_entries >= self.size_max:
            self.tree.pop()  # Loại bỏ mẫu cũ nhất nếu cây đầy
        self.tree.add(p, sample)


    def get_samples(self, n):
        """
        Get n samples randomly from the memory based on their priority
        """
        if self.tree.n_entries < self.size_min:
            n = self.tree.n_entries  # Return available samples if less than minimum size

        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = self._get_priority(error)
            self.tree.update(idx, p)
