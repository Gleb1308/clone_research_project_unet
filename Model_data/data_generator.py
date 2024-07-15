import tensorflow as tf
import numpy as np
import pandas as pd
import os

Image_size = (768,768)

def get_masks_batch(list_inds, group_data, use_bool=True):
  batch_size = len(list_inds)
  if use_bool:
    masks = np.zeros((Image_size[0]*Image_size[0]*batch_size),dtype=bool)
  else:
    masks = np.zeros((Image_size[0]*Image_size[0]*batch_size),dtype=np.uint8)
  df = group_data[list_inds]

  for i in range(df.shape[0]):
    arr = np.array(df.iloc[i])
    if arr.size>=2:
      arr = arr[:-1]
      arr = arr.reshape(arr.size//2,2)
      for inds in arr:
        masks[int(inds[0])+Image_size[0]*Image_size[0]*i:int(inds[0])+Image_size[0]*Image_size[0]*i+int(inds[1])] = 1

  masks = masks.reshape(batch_size,Image_size[0],Image_size[0])
  for j in range(batch_size):
    masks[j] = masks[j].T
  return masks

class CustomDataGen(tf.keras.utils.Sequence):

  def __init__(self, df, path_data, batch_size, shuffle=True, use_bool=True, resize=False, height=128, width=128):
    self.df = df.copy()
    if path_data[-1]=='/':
      self.path_data = path_data
    else:
      self.path_data = path_data+'/'
    self.im_list = np.array(os.listdir(path_data))
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.use_bool = use_bool
    self.resize = resize
    self.height = height
    self.width = width
    self.n = self.im_list.size

  def on_epoch_end(self):    # shuffle training dataset at the end of the epoch
    if self.shuffle:
      rng = np.random.default_rng()
      rng.shuffle(self.im_list)

  def __getitem__(self, index):    # generate index-th batch
    batch = self.im_list[index*self.batch_size:(index+1)*self.batch_size]
    l_masks = tf.constant(get_masks_batch(batch, self.df, self.use_bool))[..., tf.newaxis]
    l_imgs = []
    for img_name in batch:
      img = tf.io.read_file(self.path_data+img_name)
      img = tf.image.decode_png(img, channels=3)
      img = tf.image.convert_image_dtype(img, tf.float32)
      l_imgs.append(img)
    l_imgs = tf.stack(l_imgs)
    if self.resize:       # resize images and masks to make computations less expensive
      l_imgs = tf.image.resize(l_imgs, (self.height, self.width), method='nearest')
      l_masks = tf.image.resize(l_masks, (self.height, self.width), method='nearest')
    return l_imgs,l_masks

  def __len__(self):
    return self.n // self.batch_size
