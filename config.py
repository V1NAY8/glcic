# Basic Configurations
# Image size taken is 256 * 256
# Mask Size taken is 128 * 128

class Config:
    def __init__(self,total = 256,mask = 128):
        self._total = total
        self._mask = mask
    
    def configuration(self):
        batch_size = 16
        learning_rate = 0.001
        d_loss_alpha = 0.0004
        input_size = self._total
        mask_size = self._mask
        input_shape = [input_size, input_size, 3]
        mask_shape = [mask_size, mask_size, 3]
        hole_min = mask_size // 2
        hole_max = mask_size // 4 * 3

