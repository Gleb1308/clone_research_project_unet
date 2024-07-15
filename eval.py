import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import os
from Model_data.model import unet_model
from Model_data.data_generator import CustomDataGen
from Loss_metric.metric import Dice_score
import argparse
from copy import copy


if __name__=="__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument('--img_height', type=int, default=128, help='images will be rescaled to such height, should be a power of two')
  parser.add_argument('--img_width', type=int, default=128, help='images will be rescaled to such width, should be a power of two')
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--model_paths', type=str, default="./checkpoints/my_checkpoint",
                                                help='from here will be downloaded weights of pretrained models')
  parser.add_argument('--model_names', type=str, default=None, help='names of the downloaded models')
  parser.add_argument('--path_img_test', type=str, default="small-subset-of-airbus-ship-segmentation-dataset/test_v2/",
                                                    help='from here will be generated batches of images for testing')
  parser.add_argument('--path_y_train', type=str, default="./small-subset-of-airbus-ship-segmentation-dataset/train_ship_segmentations_v2.csv",
                                                  help='path to the masks encoding (run-length encoding format)')

  args = parser.parse_args()
  model_paths = args.model_paths.split(',')
  if args.model_names is None:
    model_names = copy(model_paths)
  else:
    model_names = args.model_names.split(',')
  d = {'model':[], 'test_loss':[], 'test_metric':[]}
  # unify all encoded pixels that belong to the same image
  y_train = pd.read_csv(args.path_y_train)
  y_train['EncodedPixels'] += ' '
  y_group = y_train.groupby(by='ImageId')['EncodedPixels'].sum()
  y_group = y_group.str.split(' ')

  tf.keras.backend.clear_session()
  unet = unet_model((args.img_height, args.img_width, 3))
  var_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  var_optimizer = tf.keras.optimizers.experimental.RMSprop()
  unet.compile(optimizer=var_optimizer, loss=var_loss, metrics=[Dice_score()])
  testgen = CustomDataGen(y_group, args.path_img_test, args.batch_size, use_bool=False, resize=True, height=args.img_height, width=args.img_width)

  for model_path, model_name in zip(model_paths, model_names):
    try:
      unet.load_weights(model_path).expect_partial()
    except tf.errors.NotFoundError:
      continue
    print('model = {}'.format(model_name))
    loss, dice_score = unet.evaluate(testgen)
    d['model'].append(model_name)
    d['test_loss'].append(loss)
    d['test_metric'].append(dice_score)

  df = pd.DataFrame(d)
  df_sort = df.sort_values(by='test_metric', ascending=False, ignore_index=True)
  os.makedirs('./Test_results', exist_ok=True)
  df_sort.to_csv('./Test_results/models_eval.csv')
