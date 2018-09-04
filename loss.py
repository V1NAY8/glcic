# Loss Functions
# Reference from https://github.com/tadax/glcic
from keras import backend as K
import tensorflow as tf
def discriminator(alpha):
    def _f(true, pred):
        real = pred[:, 0]
        fake = pred[:, 1]
        loss_real = K.mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real))) * alpha
        loss_fake = K.mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake))) * alpha
        loss = loss_real + loss_fake
        # loss = util.tfprint(loss, "discriminator_loss")
        return loss
    return _f

