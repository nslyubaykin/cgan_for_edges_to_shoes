# Edges to Shoes with cGAN (Pix2Pix)

This [notebook](https://github.com/nslyubaykin/cgan_for_edges_to_shoes/blob/master/GAN_eth.ipynb) contains a TensorFlow 2.2.0 implementation (with some comments and explanations) of conditional GAN  (cGAN, Pix2Pix) for transforming edges to shoes images from scratch as shown below:

![Main Examples](https://github.com/nslyubaykin/cgan_for_edges_to_shoes/blob/master/images/main.png)

# Training Data:

Models are trained on [UT Zappos50K](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/) shoe image dataset. In this notebook I used a square [version](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images-square.zip) of the data.

Images' edges were retreived with scikit-image package, and then used as source images for generation process.

Overall architecture follows the one described in original [pix2pix paper](https://arxiv.org/abs/1611.07004):

- U-net as a generator
- PatchGAN as a discriminator

Please check out this [article](https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/) for more detailed implementation guide. 

# Model Outputs vs Ground Truth:

![result 1](https://github.com/nslyubaykin/cgan_for_edges_to_shoes/blob/master/images/result1.png)

![result 2](https://github.com/nslyubaykin/cgan_for_edges_to_shoes/blob/master/images/result2.png)

![result 3](https://github.com/nslyubaykin/cgan_for_edges_to_shoes/blob/master/images/result3.png)

![result 4](https://github.com/nslyubaykin/cgan_for_edges_to_shoes/blob/master/images/result4.png)

![result 5](https://github.com/nslyubaykin/cgan_for_edges_to_shoes/blob/master/images/result5.png)

# More Generated Shoes:

![fakes 10x10](https://github.com/nslyubaykin/cgan_for_edges_to_shoes/blob/master/images/fakes10x10.png)

# Handwritten Sketches to Shoes:

Examples below are much less realistic probably due to my poor painting skills.

![sketch 1](https://github.com/nslyubaykin/cgan_for_edges_to_shoes/blob/master/images/sketch1.png)

![sketch 2](https://github.com/nslyubaykin/cgan_for_edges_to_shoes/blob/master/images/sketch2.png)
