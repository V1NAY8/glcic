# Only for Training
import argparse
import logging
import os
import cv2
import numpy as np
import tensorflow as tf
import keras.callbacks
from keras import backend as K
from keras import utils
from network import GLCIC
from config import Config
import dataset

class PrintAccuracy(keras.callbacks.Callback):
    def __init__(self, logger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

    def on_batch_end(self, batch, logs={}):
        self.logger.info("accuracy: %s", logs.get('acc'))


class SaveGeneratorOutput(keras.callbacks.Callback):
    def __init__(self, data_generator, batch_size, tests, **kwargs):
        super().__init__(**kwargs)
        self.data_generator = data_generator
        self.batch_size = batch_size
        self.tests = tests

    def on_epoch_end(self, epoch, logs={}):
        outputs = self.model.predict(self.tests, batch_size=self.batch_size,verbose=1)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for output in outputs:
            if len(output.shape) == 4 and output.shape[3] == 3:
                # おそらく画像 
                # Probably image
                output = np.split(output, output.shape[0], axis=0)
                for i, image in enumerate(output):
                    image = np.squeeze(image, 0)
                    image = self.data_generator.denormalize_image(image)
                    cv2.imwrite('./out/epoch{}_{}.png'.format(epoch, i), image)

FORMAT = '%(asctime)-15s %(levelname)s #[%(thread)d] %(message)s'
logger = logging.getLogger(__name__)
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger.info("---Starting Training---")

config = Config()

argparser = argparse.ArgumentParser(
    description="Globally and Locally Consistent Image Completion(GLCIC)"
    + " - train model.")
argparser.add_argument('--data_dir', type=str,
                       required=True, help="Put a dataset and divide it into" +
                       "data_dir/train, data_dir/val.")
argparser.add_argument('--stage', type=int,
                       required=True,
                       help="Oh you need help! 1:generator only, " +
                       "2:discriminator only, 3:all",
                       choices=[1, 2, 3])
argparser.add_argument('--weights_path', type=str,
                       required=False, help="If there are weights Please specify")
argparser.add_argument('--testimage_path', type=str,
                       required=False, help="Predict each picture from stored directory.")
args = argparser.parse_args()
logger.info("args: %s", args)

config.batch_size = 16

network = GLCIC(batch_size=config.batch_size, input_shape=config.input_shape,mask_shape=config.mask_shape)

train_generator = True
train_discriminator = True

if args.stage == 1:
    # Only Generator Training
    model, base_model = network.compile_generator(learning_rate=config.learning_rate)
    train_discriminator = False
    steps_per_epoch = 100
    epochs = 100
elif args.stage == 2:
    # discriminator
    model, base_model = network.compile_all(fix_generator_weight=True,learning_rate=config.learning_rate)
    train_generator = False
    steps_per_epoch = 100
    epochs = 100
elif args.stage == 3:
    model, base_model = network.compile_all(fix_generator_weight=False,learning_rate=config.learning_rate,d_loss_alpha=config.d_loss_alpha)
    steps_per_epoch = 100
    epochs = 100

logger.info(model.summary())

if args.weights_path:
    logger.info("load weight:%s", args.weights_path)
    model.load_weights(args.weights_path, by_name=True)

gen = dataset.DataGenerator(config)
train_data_generator = gen.generate(os.path.join(args.data_dir, "train"), train_generator, train_discriminator)
val_data_generator = gen.generate(os.path.join(args.data_dir, "val"), train_generator, train_discriminator)

# Call back preparation.
model_file_path = './nnmodel/glcic-stage{}-{}'.format(
    args.stage, '{epoch:02d}-{val_loss:.2f}.h5')
callbacks = [keras.callbacks.TerminateOnNaN(),
            keras.callbacks.TensorBoard(log_dir='./tb_log',histogram_freq=0,write_graph=True,write_images=False),
            keras.callbacks.ModelCheckpoint(filepath=model_file_path,verbose=1,save_weights_only=True,save_best_only=False,period=20)]

if args.testimage_path and not args.stage == 2:
    # epoch毎にgeneratorの出力を保存
    test_data_generator = gen.generate(args.testimage_path,
                                       train_generator, train_discriminator)
    inputs, _ = next(test_data_generator)
    callbacks.append(SaveGeneratorOutput(gen, config.batch_size, inputs))

model.fit_generator(train_data_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    verbose=1,max_queue_size=10,
                    callbacks=callbacks,
                    validation_data=val_data_generator,
                    validation_steps=5)

model_file_path = './nnmodel/glcic-latest-stage{}.h5'.format(args.stage)
model.save_weights(model_file_path)