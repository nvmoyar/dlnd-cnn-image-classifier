# Deep Learning Nanodegree

## Image Classification

[image1]: ./cifar-10-image-classifier_screenshot.png "Image Classifier"

### Project Overview

In this project, a Convolutional Neural Network is built to classify a subset of the CIFAR-10 dataset. This is a classic multiclass classification problem that illustrates Supervised Learning. This dataset consists of airplanes, dogs, cats, and other objects that will be labeled from 0 to 9 by using the LabelBinarizer function of Sckikit-Learn in a pre-process function, therefore we get a One-Hot encoded Numpy array with targets/labels for each input image. The images need only to be rescaled, getting a 3D input tensor or cube of numbers between [0, 1]. 

This CNN model takes a batch of images and labels to output the logits, and it consists of a stack of: 

* Some Convolution layers and Max Pooling layers to reduce dimensionality preserving the spatial integrity, getting a 4D tensor.

* A Flatten layer that will process the 4D tensor output of the previous stack of layers, to output a 2D tensor.

* Some Fully Connected connected layers. Since we are using the tf.nn module -low abstraction level API- instead of tf.layers -high abstraction level API-, the operations performed at this layer are manually defined. This way we can expressly set weights initialized to random normal rather than random distribution, etc. For this reason, a different Fully Connected layer called Output layer is needed to reduce dimensionality to the number of classes needed for this problem. Please notice that Dropout layers are used to prevent overfitting, and the keep probability has been defined as a 0D Tensor or Scalar. 

* Output layer, the last used Fully Connected layer needed to reduce the dimensionality of the convoluted data to the classes that need to be classified. 

After tuning this model, and training it on this subset of CIFAR-10, we get a ~70% accuracy, which makes that more the predictions are likely, still, it is not a great classifier depending on the final use, but it supposes a great introduction to hands-on on Supervised Learning. 

![Image Classifier][image1]

### Test and Demo

* [Test](http://localhost:8888/notebooks/dlnd_image_classification.ipynb)
* [Demo](https://www.floydhub.com/nvmoyar/projects/image-classificator/)

#### Requirements

FloydHub is a platform for training and deploying deep learning models in the cloud. It removes the hassle of launching your own cloud instances and configuring the environment. For example, FloydHub will automatically set up an AWS instance with TensorFlow, the entire Python data science toolkit, and a GPU. Then you can run your scripts or Jupyter notebooks on the instance. 
For this project: 

> floyd run --mode jupyter --gpu --env tensorflow-1.0

You can see your instance on the list is running and has ID XXXXXXXXXXXXXXXXXXXXXX. So you can stop this instance with Floyd stop XXXXXXXXXXXXXXXXXXXXXX. Also, if you want more information about that instance, use Floyd info XXXXXXXXXXXXXXXXXXXXXX

#### Environments

FloydHub comes with a bunch of popular deep learning frameworks such as TensorFlow, Keras, Caffe, Torch, etc. You can specify which framework you want to use by setting the environment. Here's the list of environments FloydHub has available, with more to come!

#### Datasets 

With FloydHub, you are uploading data from your machine to their remote instance. It's a really bad idea to upload large datasets like CIFAR along with your scripts. Instead, you should always download the data on the FloydHub instance instead of uploading it from your machine.

Further Reading: [How and Why mount data to your job](https://docs.floydhub.com/guides/data/mounting_data/)

### Usage 

floyd run --gpu --env tensorflow-1.2 --message 'Update README' --data udacity/datasets/cifar-10/1:cifar --mode jupyter

[**You only need to mount the data to your job, since the dataset has been already been uploaded for you**]

#### Output

Often you'll be writing data out, things like TensorFlow checkpoints, updated notebooks, trained models and HDF5 files. You will find all these files, you can get links to the data with:

> floyd output run_ID


