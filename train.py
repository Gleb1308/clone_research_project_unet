import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import os
import argparse
from Model_data.model import unet_model
from Model_data.data_generator import CustomDataGen
from Loss_metric.metric import Dice_score

def train(y_group, epochs, img_height, img_width, batch_size, checkpoint_path_load, path_img_train, path_img_test, best_path_save, final_path_save, 
          plot_path_save, save_weights, use_pretrained, save_plot, eval_only, **model_params):

  num_channels = 3
  # creating a model
  tf.keras.backend.clear_session()
  unet = unet_model((img_height, img_width, num_channels), **model_params)
  # defining loss function and optimizer
  var_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  var_optimizer = tf.keras.optimizers.experimental.RMSprop()
  unet.compile(optimizer=var_optimizer, loss=var_loss, metrics=[Dice_score()])

  EPOCHS = epochs
  #checkpoint_path_load = "./checkpoints/my_checkpoint"
  if use_pretrained:
    unet.load_weights(f"{checkpoint_path_load}")

  if not eval_only:
    # training the model using data generator
    traingen = CustomDataGen(y_group, path_img_train, batch_size, use_bool=False, resize=True, height=img_height, width=img_width)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=best_path_save, verbose=1, save_weights_only=True, save_best_only=True, monitor='Dice_score', 
                                                     mode='max')
    model_history = unet.fit(traingen, epochs=EPOCHS, callbacks=[cp_callback])
    if save_plot:
      fig, axs = plt.subplots(1, 2, figsize=(12, 6))
      train_loss = model_history.history['loss']
      train_metric = model_history.history['Dice_score']
      axs[0].plot(train_loss)
      axs[0].set_title('Train loss')
      axs[1].plot(train_metric)
      axs[1].set_title('Train metric')
      plt.show()
      dir_plot = os.path.dirname(plot_path_save)
      os.makedirs(dir_plot, exist_ok=True)
      fig.savefig(plot_path_save)

    #checkpoint_path_save = "./checkpoints/my_checkpoint"
    if save_weights:
      unet.save_weights(f"{final_path_save}")

    print('evaluation\n')
    testgen = CustomDataGen(y_group, path_img_test, batch_size, use_bool=False, resize=True, height=img_height, width=img_width)
    loss, dice_score = unet.evaluate(testgen)
    print('loss on test images = {}\n'.format(loss))
    print('dice score on test images = {}\n'.format(dice_score))
    model_history = model_history.history
  else:     # you can only evaluate already loaded model
    print('evaluation\n')
    testgen = CustomDataGen(y_group, path_img_test, batch_size, use_bool=False, resize=True, height=img_height, width=img_width)
    loss, dice_score = unet.evaluate(testgen)
    model_history = None
    print('loss on test images = {}\n'.format(loss))
    print('dice score on test images = {}\n'.format(dice_score))
  
  return unet, model_history, loss, dice_score    # loss ans dice_score are evaluated on the test data

if __name__=="__main__":
  # reading arguments from the command line
  parser = argparse.ArgumentParser()

  parser.add_argument('--epochs', type=int, default=40, help='quantity of epochs for training')
  parser.add_argument('--img_height', type=int, default=128, help='images will be rescaled to such height, should be a power of two')
  parser.add_argument('--img_width', type=int, default=128, help='images will be rescaled to such width, should be a power of two')
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--checkpoint_path_load', type=str, default="./checkpoints/my_checkpoint",
                                              help='from here will be downloaded weights of pretrained model')
  parser.add_argument('--path_img_train', type=str, default="small-subset-of-airbus-ship-segmentation-dataset/train_v2/",
                                                   help='from here will be generated batches of images for training')
  parser.add_argument('--path_img_test', type=str, default="small-subset-of-airbus-ship-segmentation-dataset/test_v2/",
                                                   help='from here will be generated batches of images for testing')
  parser.add_argument('--best_path_save', type=str, default="./checkpoints/my_checkpoint",
                                                    help='here will be saved best weights of the trained model')
  parser.add_argument('--final_path_save', type=str, default="./checkpoints/my_checkpoint",
                                                    help='here will be saved final weights of the trained model')
  parser.add_argument('--plot_path_save', type=str, default="./checkpoints/my_checkpoint",
                                                    help='here will be saved final plots of the trained model')
  parser.add_argument('--save_weights', action='store_true', help='whether to save weights after training')
  parser.add_argument('--use_pretrained', action='store_true', help='whether to use pretrained model')
  parser.add_argument('--save_plot', action='store_true', help='whether to save graphs of the training loss and metric')
  parser.add_argument('--eval_only', action='store_true', help='whether to evaluate model without training')
  parser.add_argument('--path_y_train', type=str, default="./small-subset-of-airbus-ship-segmentation-dataset/train_ship_segmentations_v2.csv",
                                                help='path to the masks encoding (run-length encoding format)')
  parser.add_argument('--drop_prob', type=float, default=0.3, help='dropout rate')
        
  args = parser.parse_args()
  # unify all encoded pixels that belong to one image
  y_train = pd.read_csv(args.path_y_train)
  y_train['EncodedPixels'] += ' '
  y_group = y_train.groupby(by='ImageId')['EncodedPixels'].sum()
  y_group = y_group.str.split(' ') 
  # call train() with all arguments; it will return the model, history of training loss and metric, loss and dice score on the test data
  unet, hist, loss, dice_score = train(y_group, args.epochs, args.img_height, args.img_width, args.batch_size, args.checkpoint_path_load, args.path_img_train, 
                                       args.path_img_test, args.best_path_save, args.final_path_save, args.plot_path_save, args.save_weights, 
                                       args.use_pretrained, args.save_plot, args.eval_only, drop_prob=args.drop_prob)
