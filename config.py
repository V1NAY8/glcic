# Basic Configurations
# Image size taken is 256 * 256
# Mask Size taken is 128 * 128

class Config:
    batch_size = 16
    gpu_num = 0
    d_loss_alpha = 0.0004
    learning_rate = 0.001

    @property
    def input_size(self):
        return self._input_size

    @input_size.setter
    def input_size(self, value):
        self._input_size = value
        self.input_shape = [value, value, 3]

    @property
    def mask_size(self):
        return self._mask_size

    @mask_size.setter
    def mask_size(self, value):
        self._mask_size = value
        self.mask_shape = [value, value, 3]
        self.hole_min = value // 2
        self.hole_max = value // 4 * 3

    def __init__(self):
        self.input_size = 256
        self.mask_size = 128


