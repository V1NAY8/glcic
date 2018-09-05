# Prediction
import matplotlib
matplotlib.use('Agg')  # noqa
import argparse
import glob
import logging
import os
import re
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from network import GLCIC
from config import Config
from dataset import DataGenerator

"""
An image obtained by randomly cutting out a part of the designated image is taken as input.
An input image and a supplement result image are obtained.
"""
FORMAT = '%(asctime)-15s %(levelname)s #[%(thread)d] %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

logger = logging.getLogger(__name__)
logger.info("---start---")


config = Config()

argparser = argparse.ArgumentParser(
    description="Globally and Locally Consistent Image Completion(GLCIC)"
    + " - generae image.")
argparser.add_argument('--input_path', type=str,required=True, help="data_dir/train, data_dir/val ")
argparser.add_argument('--weights_path', type=str,required=True, help="Model Weight file path")
argparser.add_argument('--random_hole', type=int, default=1,required=False, help="Do You want to randomize the position of the hole?")
argparser.add_argument('--dest', type=str, default='./out/', required=False)
args = argparser.parse_args()
logger.info("args: %s", args)

config.batch_size = 1

# Read the input, Clip the image and resize.
gen = DataGenerator(config, random_hole=args.random_hole)

# Model
network = GLCIC(batch_size=config.batch_size, input_shape=config.input_shape,mask_shape=config.mask_shape)
# Compile the model only with the generator
model, _ = network.compile_generator(learning_rate=config.learning_rate)

# Load Weights.
model.load_weights(args.weights_path, by_name=True)


def save_prediction(path):
    # Destination path
    template = os.path.join(args.dest, re.split('/|\.', path)[-2] + '_{}.png')

    # Input Image
    resized_image, bin_mask, masked_image, mask_window = gen.load_image(path)
    if resized_image is None:
        logger.warn("Specified image %s does not exist", args.input_path)
        sys.exit()

    # cv2.imwrite(template.format('masked_image'), masked_image)
    # Input Image Normalization (0 to 255 to -1 to 1)

    in_masked_image = gen.normalize_image(masked_image)

    # Add Batch Dimension
    in_masked_image = np.expand_dims(in_masked_image, 0)
    in_bin_mask = np.expand_dims(bin_mask, 0)

    # Prediction
    out_completion_image =model.predict([in_masked_image, in_bin_mask], verbose=1,batch_size=config.batch_size)
    # Delete Batch Dimension
    completion_image = np.squeeze(out_completion_image, 0)
    # Denormalization (returning from -1 to 1 to 0 to 255)
    completion_image = gen.denormalize_image(completion_image)

    # cv2.imwrite(template.format('_in_res'), resized_image)
    # bin_mask = np.expand_dims(bin_mask, -1)
    # cv2.imwrite(template.format('_in_bin'), bin_mask * 255)
    # cv2.imwrite(template.format('_in_msk'), masked_image)
    # cv2.imwrite(template.format('_out_raw'), completion_image)
    # マスク部分のみ
    # cropped = completion_image * bin_mask
    # cv2.imwrite(template.format('_out_crp'), cropped)

    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    completion_image = cv2.cvtColor(completion_image, cv2.COLOR_BGR2RGB)

    def plot_image(_img, _label, _num):
        plt.subplot(1, 3, _num)
        plt.imshow(_img)
        # plt.axis('off')
        plt.gca().get_xaxis().set_ticks_position('none')
        plt.gca().get_yaxis().set_ticks_position('none')
        plt.tick_params(labelbottom='off')
        plt.tick_params(labelleft='off')
        plt.xlabel(_label)

    plt.figure(figsize=(6, 3))
    plot_image(masked_image, 'Input', 1)
    plot_image(completion_image, 'Output', 2)
    plot_image(resized_image, 'Ground Truth', 3)
    plt.savefig(template.format(''))


if os.path.isdir(args.input_path):
    paths = glob.glob(os.path.join(args.input_path, '*.jpg'))
else:
    paths = args.input_path.split(',')

for path in paths:
    save_prediction(path)

# inputs = generator.generate(args.input_path, False, False)
# # inputs, _ = next(inputs)
# outputs = model.predict_generator(inputs, steps=2, verbose=1)
# outputs = np.split(outputs, outputs.shape[0], axis=0)
# for i, output in enumerate(outputs):
#     output = np.squeeze(output, 0)
#     output = generator.denormalize_image(output)
#     cv2.imwrite('./out/{}.png'.format(i), output)
