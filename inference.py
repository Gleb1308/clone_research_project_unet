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

def plot_preds(images, list_masks, model, start_sample, num_pict, save_fig, save_preds, name_pict):
  fig, axs = plt.subplots(num_pict, 3, figsize=(18, 6*num_pict))
  preds = model(images, training=False).numpy()>0.0
  imgs = images.numpy()
  masks = list_masks.numpy()
  for i in range(num_pict):
    img = imgs[i]
    h,w,_ = img.shape
    mask = masks[i]
    pred = preds[i]
    ind = i
    axs[ind,0].imshow(img)
    axs[ind,0].set_title('Image {}'.format(start_sample+i))
    axs[ind,1].imshow(mask)
    axs[ind,1].set_title('True mask {}'.format(start_sample+i))
    axs[ind,2].imshow(pred)
    axs[ind,2].set_title('Predicted mask {}'.format(start_sample+i))
  plt.show()
  if save_fig:
    os.makedirs('./Saved preds', exist_ok=True)
    fig.savefig(name_pict+'.png')
  if save_preds:
    return preds
  else:
    return None

if __name__=="__main__":
  # reading arguments from the command line
  parser = argparse.ArgumentParser()

  parser.add_argument('--num_pict', type=int, default=10, help='quantity of images to predict masks')
  parser.add_argument('--start_sample', type=int, default=0)
  parser.add_argument('--path_infer', type=str, default="./small-subset-of-airbus-ship-segmentation-dataset/test_v2/",
                                                help='from here will be generated images for inferencing')
  parser.add_argument('--path_y_train', type=str, default="./small-subset-of-airbus-ship-segmentation-dataset/train_ship_segmentations_v2.csv",
                                                help='path to the masks encoding (run-length encoding format)')
  parser.add_argument('--save_fig', action='store_true', help='whether to save the results of inferencing')
  parser.add_argument('--save_preds', action='store_true', help='whether to save predictions in run-length encoding format')
  parser.add_argument('--checkpoint_path_load', type=str, default="./checkpoints/my_checkpoint",
                                              help='from here will be downloaded weights of pretrained model')
  parser.add_argument('--img_height', type=int, default=128, help='images will be rescaled to such height, should be a power of two')
  parser.add_argument('--img_width', type=int, default=128, help='images will be rescaled to such width, should be a power of two')
  parser.add_argument('--name_pict', type=str, default='True vs predicted masks', help='file name where to save the results of inferencing')
  
  args = parser.parse_args()
  # unify all encoded pixels that belong to the same image
  y_train = pd.read_csv(args.path_y_train)
  y_train['EncodedPixels'] += ' '
  y_group = y_train.groupby(by='ImageId')['EncodedPixels'].sum()
  y_group = y_group.str.split(' ')

  # create the model
  tf.keras.backend.clear_session()
  unet = unet_model((args.img_height, args.img_width, 3))
  # load weights for the model and generate images for inferencing
  unet.load_weights(args.checkpoint_path_load).expect_partial()
  gen = CustomDataGen(y_group, args.path_infer, args.num_pict, use_bool=False, resize=True, height=args.img_height, width=args.img_width)
  gen.im_list = gen.im_list[args.start_sample:]

  # comparing true masks vs predicted masks
  images, masks = gen[0]
  preds = plot_preds(images, masks, unet, start_sample=args.start_sample, num_pict=args.num_pict, save_fig=args.save_fig,
                  save_preds=args.save_preds, name_pict='./Saved preds/'+args.name_pict)

  if args.save_preds:
    # saving predicted masks in run-length encoding format
    name_imgs = gen.im_list[:args.num_pict]
    encodings = []

    for i in range(args.num_pict):
      pred_mask = preds[i].reshape(args.img_height, args.img_width)
      pred_mask = pred_mask.T.flatten()
      if np.any(pred_mask):
        inds = np.arange(pred_mask.size)[pred_mask]
        d_inds = np.append([2],np.diff(inds))
        sub_inds = np.arange(inds.size)[d_inds>1]
        inds_start = inds[sub_inds][...,np.newaxis]
        if sub_inds.size>1:
          inds_fin = inds[np.append((sub_inds-1)[1:], [inds.size-1])][...,np.newaxis]
        else:
          inds_fin = inds[np.array([inds.size-1])][...,np.newaxis]
        encod = np.append(inds_start, inds_fin, axis=1)
        encod[:,1] = encod[:,1]-encod[:,0]+1
        encod = encod.flatten()
        encoding = ' '.join([str(el) for el in encod])
        encodings.append(encoding)
      else:
        encodings.append(np.nan)

    d = {}
    d['ImageId'] = name_imgs
    d['EncodedPixels'] = encodings
    df = pd.DataFrame(d)
    df.to_csv('./Saved preds/Prediction_encodings.csv')
