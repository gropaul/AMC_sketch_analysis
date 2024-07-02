import math


class AMSSketchSimple:
    def __init__(self, array_size):
        # Initialize the sketch array
        self.array_size = array_size
        self.array = [0] * array_size

    def _get_bit_at_int(self, hash, bit):
        return (hash >> bit) & 1

    def update(self, hash, w):
        for i in range(self.array_size):
            bit = self._get_bit_at_int(hash, i)
            if bit == 0:
                self.array[i] -= w
            else:
                self.array[i] += w
