# Project description

In this project Unet model was implemented and tested on the subset of kaggle airbus-ship-segmentation dataset.
Size of training set = 10k, test set = 1,5k as it was too computationally expensive to train on the full dataset.
Despite these small datasets, such model gave us pretty promising results.

# Dataset installation

You can download the whole dataset or using smaller version - glebsukhomlyn/small-subset-of-airbus-ship-segmentation-dataset.
All these datasets can be downloaded from kaggle. (See more kaggle api if you want to
know how to install datasets from command line)

# EDA

<p>After EDA analysis it was found that approximately 78% of the images do not contain any ships.
This is not too large imbalance in data so ordinary bce (binary cross entropy) as a loss function can be used.
Histograms of the distribution of the number of ships and the total area of ​​ships per image were also constructed.
From these histograms it follows that the smaller the number of ships or the smaller the total area, the higher the probability of meeting the corresponding ships.</p>
<p>Thus, if you apply a boxplot, it will treat images with a large number of ships or a large area as outliers, but they are actually quite useful images.
In addition, this may indicate some imbalance in the data and it is possible that together with the usual bce bfce (binary focal cross entropy) should also be tested.</p>
<p>The pie chart of pixels shows that only 0.1% of all pixels belong to some ship but this is unlikely to indicate a large imbalance in the data
because many pixels are from the same image and thus there will be some relationship between them and therefore each pixel cannot be treated as an independent instance of data.</p>
<p>Also further studies during training will show that weighted bce does not provide any improvements in model training.
All information about EDA analysis can be found in the EDA.ipynb file.</p>

# Architecture of the model

In this project Unet architecture was used. It consists of two parts: encoder and decoder (and bottleneck
that can be treated as part of the encoder). Encoder use convolutional blocks max_pooling to get feature maps which
will be concatenated with different inputs of decoder's blocks via skip connections. Such trick help model not to lose
useful information during forward propagation and effectively learning during backpropagation. Also transpose convolution is used
to upsample image_size in the decoder and at the two last blocks of encoder Droppout is used (rate=0.3) to prevent overfitting.
At the end of the model sigmoid is not used, so all losses should have such option:from_logits=True.
You can find implementation of the model in the Model_data/model.py file.

# Training and testing

<p>For training custom data generator was used as it was unreal (at least too ineffective) to download all 10k in RAM. Realization of
this generator you can check in Model_data/data_generator.py file. During training were tested loss functions - bce, bfce and weighted_bce(didn't give any results).</p>
<p>Also were tested optimizers - Adam and RmsProp, metric - dice_score (see its implementation in Loss_metric/metric.py file).
Each model trained 40 epochs with batch_size=32. Adam has nearly the same results with bfce and bce (68-69% dice_score). RmsProp gave the best result -
nearly 70% dice_score. The best results gave us bce with adam and rmsprop (they had the most stable training and one of the best dice_scores).
All information about testing results with different loss fucntions and optimizers you can find in the "Test_results/test_metric.csv".</p>
<p>Rmsprop omtimizer is a better choice also because it gave us smaller (in Mb) model (rmsprop - 60 Mb, adam > 77Mb, however there is no much decrease in dice score)- such effect can be caused by more stable training process compared to Adam optimizer. So after that unet model (rmsprop optimizer) was tested with other values of dropping rate ([0.08, 0.17, 0.5, 0.75, 0.9]). Model was initial value drop_prob=0.3 has dice_score=0.7. Best model according to dice score became model with drop_prob=0.17 - dice score = 0.71 . The results of such tests you can find in “Test_results/models_eval.csv”.</p>

# Pretrained weights

In the folder "Pretrained_models" example of saved checkpoint is folder 'adam_bce_ep40' - checkpoint for unet that was traing 40 epochs using
Adam optimizer and BinaryCrossentropy loss function on images of size (128,128). Another typical example is 'adam_bce_ep40+40_image=256' - checkpoint for unet that was firstly trained on the images of size = (128,128) (40 epochs), then - on images of size = (256,256) (40 epochs) (dice_score of this model = 75% on the images of size 256). There were other models with different dropout rate. Weights for all these models can be found in another github repo - https://github.com/Gleb1308/Weights-for-unet.git . In this repo you will find such folders that contains weghts for various dropout rates - 
"Best_saved_weights" (weights that had the best dice score on training data duiring training), "Final_saved_weights" (weights at the of the training, there aren't such weights that are copies to some other weights in "Best_saved_weights" folder).

# Scripts for training and inferencing

If you want to train model from scratch or use pretrained model you can run file train.py:
```bash
python train.py --epochs 20 --path_img_train '<path_to_train_data>' --checkpoint_path_save '<path to save weights>' --save_weights --save_plot
```
Detailed information you can get by such command - python train.py --help

If you want use already trained model in inference mode you can run file inference.py:
```bash
python inference.py --checkpoint_path_load ./rmsprop_bce_ep40/my_checkpoint --save_fig --save_preds
```
Command (inference.py --save_preds) saves predicted masks in run-length encoding format.
Detailed information you can get by such command - python inference.py --help

If you want use evaluate different models you can run file eval.py:
```bash
python Research_project_unet/eval.py --model_paths $model_paths --model_names $model_names --path_img_test $path_img_test
```
model_paths is a string that contains paths to the model weights split by comma.
Detailed information you can get by such command - python eval.py --help

You can also check examples how to use all these commands in the file - "train/test_notebook.ipynb" .
