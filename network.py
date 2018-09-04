# Generator Network
# Global Discriminator
# Local Discriminator
from logging import getLogger
import keras.layers as KL
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
import loss

logger = getLogger(__name__)
logger.info("---------Network---------")

class GLCIC:
    def __init__(self,batch_size,input_shape = [256,256,3],mask_shape = [128,128,3]):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.mask_shape = mask_shape

    def activation(self,input,trainable = True):
        out = KL.Activation('relu',trainable=trainable)(input)
        return out

    def conv2d(self,input,filters,kernel_size,strides = 1,trainable = True):
        out = KL.Conv2D(filters = filters,kernel_size = kernel_size,strides = strides,padding = 'same',trainable=trainable)(input)
        out = KL.BatchNormalization(trainable = trainable)(out)
        out = self.activation(out,trainable=trainable)
        return out

    def dilated_conv2d(self,input,filters,kernel_size,dilation_rate,strides = 1, trainable = True):
        out = KL.Conv2D(filters = filters,kernel_size = kernel_size,strides = strides,dilation_rate = dilation_rate,padding = 'same')(input)
        out = KL.BatchNormalization(trainable=trainable)(out)
        out = self.activation(out,trainable=trainable)
        return out

    def deconv2d(self,input,filters,kernel_size,strides = 1 ,padding = 'same', trainable = True):
        out = KL.Conv2DTranspose(filters = filters,kernel_size = kernel_size, strides = strides,padding='same',trainable = trainable)(input)
        out - KL.BatchNormalization(trainable = trainable)(out)
        out = self.activation(out,trainable = trainable)
        return out

    def api_model(self, inputs,outputs,trainable = True):
        logger.info("----Creating Model---")
        model = Model(inputs,outputs)
        model.trainable = trainable
        return model
    
    def generator(self,input_image,input_mask,trainable = True):

        logger.info("----Generator----")
        out = self.conv2d(input_image,filters = 64,kernel_size = 5,strides = 1,trainable = trainable)
        out = self.conv2d(out,filters = 128,kernel_size = 3,strides = 2,trainable = trainable)
        out = self.conv2d(out,filters = 128,kernel_size = 3,strides = 1,trainable = trainable)
        out = self.conv2d(out,filters = 256,kernel_size = 3,strides = 2,trainable = trainable)
        out = self.conv2d(out,filters = 256,kernel_size = 3,strides = 1,trainable = trainable)
        out = self.conv2d(out,filters = 256,kernel_size = 3,strides = 1,trainable = trainable)
        # Dilation Layers
        out = self.dilated_conv2d(out,filters = 256,kernel_size = 3,dilation_rate = 2,trainable = trainable)
        out = self.dilated_conv2d(out,filters = 256,kernel_size = 3,dilation_rate = 4,trainable = trainable)
        out = self.dilated_conv2d(out,filters = 256,kernel_size = 3,dilation_rate = 8,trainable = trainable)
        out = self.dilated_conv2d(out,filters = 256,kernel_size = 3,dilation_rate = 16,trainable = trainable)
        # Convolution
        out = self.conv2d(out,filters = 256,kernel_size = 3,strides = 1,trainable = trainable)
        out = self.conv2d(out,filters = 256,kernel_size = 3, strides = 1,trainable = trainable)
        # Deconvolution
        out = self.deconv2d(out,filters = 128,kernel_size = 4,strides = 2,trainable = trainable)
        out = self.conv2d(out,filters = 128, kernel_size = 3, strides = 1,trainable = trainable)
        out = self.deconv2d(out,filters = 64,kernel_size = 4,strides = 2,trainable = trainable)
        out = self.conv2d(out,filters = 32, kernel_size = 3, strides = 1,trainable = trainable)
        # I am using tanh here
        out = KL.Conv2D(filters=3, kernel_size=3,strides=1,padding='same',trainable=trainable)(out)
        out = KL.Activation('tanh', trainable=trainable)(out)
        # Cut out with mask area and merge with correct answer data.
        # [N, 256, 256] → [N, 256, 256, 1]
#         mask = KL.Reshape((self.input_shape[0], self.input_shape[1],1), trainable=False)(input_mask)
        
        # This is taken from tadax/glcic self.imitation and self.completion
        # x [0] * x [2]: Cut out the region where the mask bit is set from out (make the region other than mask 0)
        # x [1] * (1 - x [2]): Cut out the region where the bit of mask is not set from input_image
        # Merge (add) the above two to make the image replaced only with the output of NN for the mask part.
#         out = KL.Lambda(lambda x: x[0] * x[2] + x[1] * (1 - x[2]),trainable=False)([out, input_image, mask])
        
        model = self.api_model([input_image,input_mask],out,trainable = trainable)
        
        return model 
    
    def global_discriminator(self, input):
        logger.info("---Global Discriminator---")
        out = self.conv2d(input, filters=64,kernel_size=5, strides=2)
        out = self.conv2d( out, filters=128,kernel_size=5, strides=2)
        out = self.conv2d( out, filters=256,kernel_size=5, strides=2)
        out = self.conv2d( out, filters=512,kernel_size=5, strides=2)
        out = self.conv2d( out, filters=512,kernel_size=5, strides=2)
        out = self.conv2d( out, filters=512,kernel_size=5, strides=2)
        # out = KL.Flatten (name = 'gd_flatten 7') (out)
        # Flatten (), input_shape is said to be unknown and it was not possible to make it flat so cut out the area where the mask bit is set from reshape # x [0] * x [2]: out from the shape and cut out it Set the area to 0)
        # x [1] * (1 - x [2]): Cut out the region where the bit of mask is not set from input_image
        # Merge (add) the above two to make the image replaced only with the output of NN for the mask part
        out = KL.Reshape((4 * 4 * 512,))(out)
        # We don apply activation function to output layer
        out = KL.Dense(1024)(out)
        return out

    def local_discriminator(self, input):

        logger.info("---Local Discriminator---")
        
        out = self.conv2d(input, filters=64,kernel_size=5, strides=2)
        out = self.conv2d( out, filters=128,kernel_size=5, strides=2)
        out = self.conv2d( out, filters=256,kernel_size=5, strides=2)
        out = self.conv2d( out, filters=512,kernel_size=5, strides=2)
        out = self.conv2d( out, filters=512,kernel_size=5, strides=2)
        # out = KL.Flatten (name = 'ld_flatten6') (out)
        # Flatten () Because input_shape was said to be unknown and it could not be flat, we reshaped shape
        out = KL.Reshape((4 * 4 * 512,), name='ld_flatten6')(out)
        out = KL.Dense(1024)(out)
        
        return out
    
    def discriminator(self, input_global, input_local):
        logger.info("---Concatenating Global and Local Discriminator---")
        g_output = self.global_discriminator(input_global)
        l_output = self.local_discriminator(input_local)
        out = KL.Lambda(lambda x: K.concatenate(x))([g_output, l_output])
        out = KL.Dense(1)(out)        
        # In the paper sigmoid -> cross_entropy, but at the time of sigmoid it is biased to 0or1, the loss is fixed.
        # It seems that gradient has disappeared quickly. . .
        # Therefore, do not sigmoid here and do sigmoid with loss function
        # out = KL.Activation ('sigmoid', name = 'd_sigmoid 3') (out)
        model = self.api_model([input_global,input_local],out)
        print(model.summary())
        return model
    
    def compile_model(self, model, losses, learning_rate):
        logger.info("---Compiling Model---")
        model.compile(loss = losses,optimizer = Adam(lr = learning_rate))
        return model
    
    # Create and compile only with generator
    def compile_generator(self, learning_rate=0.001):
        logger.info("---Compiling Generator---")
        # An image in which the mask portion is filled with a certain color (simply black).
        input_masked_image = KL.Input(shape=self.input_shape, dtype='float32')
        input_bin_mask = KL.Input(shape=self.input_shape[:2], dtype='float32')

        model = self.generator(input_masked_image, input_bin_mask)
        wrapped_model = self.compile_model(model,'mean_squared_error',learning_rate)
        return wrapped_model, model
    
    def _crop_local(self, reals, fakes, mask_areas):
        """Get reals, fakes clipped in the area of mask_areas
        """
        # print("reals, fakes, masks:", reals, fakes, masks)
        # バッチ毎に分割して処理する
        fakes = tf.split(fakes, self.batch_size)
        reals = tf.split(reals, self.batch_size)
        mask_areas = tf.split(mask_areas, self.batch_size)
        real_locals = []
        fake_locals = []
        for real, fake, mask_area in zip(reals, fakes, mask_areas):
            # １次元目のバッチを示す次元を削除
            fake = K.squeeze(fake, 0)
            real = K.squeeze(real, 0)
            mask_area = K.cast(K.squeeze(mask_area, 0), tf.int32)
            top = mask_area[0]
            left = mask_area[1]
            h = mask_area[2] - top
            w = mask_area[3] - left

            fake_local = tf.image.crop_to_bounding_box(
                fake, top, left, h, w)
            fake_locals.append(fake_local)

            real_local = tf.image.crop_to_bounding_box(
                real, top, left, h, w)
            real_locals.append(real_local)

        fake_locals = K.stack(fake_locals)
        real_locals = K.stack(real_locals)
        # print("real_locals, fake_locals", real_locals, fake_locals)
        return [real_locals,  fake_locals]
    
    def compile_all(self,fix_generator_weight = False,learning_rate=0.001, d_loss_alpha=0.0004):
        logger.info("---Compiling All---")
        # # Image filled with fixed color (simply black) mask part
        input_masked_image = KL.Input(shape=self.input_shape, name='input_masked_image', dtype='float32')
        # Binary Mask
        input_bin_mask = KL.Input(shape=self.input_shape[:2], name='input_bin_mask', dtype='float32')
        # Mask rea
        # [y1,x1,y2,x2]
        input_mask_area = KL.Input(shape=[4], name='input_mask_area', dtype='int32')
        # Input Image intack
        input_real_global = KL.Input(shape=self.input_shape, name='input_real_global', dtype='float32')
        input_real_local = KL.Input(shape=self.mask_shape, name='input_real_local', dtype='float32')

        model_gen = self.generator(input_masked_image, input_bin_mask,trainable=not fix_generator_weight)
        model_dis = self.discriminator(input_real_global, input_real_local)

        # fake_global = model_gen([input_masked_image, input_bin_mask])
        fake_global = model_gen.layers[-1].output
        # print("fake_global: ", fake_global)
        
    
        if fix_generator_weight:
            # Weight of Generator is fixed.
            outputs = []
            losses = []
        else:
            outputs = [fake_global]
            losses = ['mean_squared_error']

        # Genuine, let the discriminator evaluate the fake produced by the generator
        # local image is cut out from the mask area image and evaluated
        real_local, fake_local = KL.Lambda(lambda x: self._crop_local(*x),name='crop_local')([input_real_global, fake_global, input_mask_area])
        prob_real = model_dis([input_real_global, real_local])
        prob_fake = model_dis([fake_global, fake_local])
        # print("prob_real: ", prob_real)
        # print("prob_fake: ", prob_fake)

        # 判定結果をバッチ毎にまとめる。
        # [N, 2]の形式にする。
        def _stack(p_real, p_fake):
            # print("p_real: ", p_real)
            # print("p_fake: ", p_fake)
            # [[prob_real, prob_fake], ...] の形状にする
            prob = K.squeeze(K.stack([p_real, p_fake], -1), 1)
            return prob
        prob = KL.Lambda(lambda x: _stack(*x), name='stack_prob')(
            [prob_real, prob_fake])
        # print("prob: ", prob)
        outputs.append(prob)
        losses.append(loss.discriminator(d_loss_alpha))

        model_all = self.api_model([input_masked_image, input_bin_mask,input_mask_area, input_real_global],outputs)

        wrapped_model = self.compile_model(model_all, losses, learning_rate)
        return wrapped_model, model_all