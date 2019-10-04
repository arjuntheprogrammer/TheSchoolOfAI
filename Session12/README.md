# TSAI_Session12

Project for The School of AI
---------------------------------------------------------------------------------------

## Tasks

1. Read https://mc.ai/tutorial-2-94-accuracy-on-cifar10-in-2-minutes/ and https://myrtle.ai/how-to-train-your-resnet/
2. Write a report on both. Focus on the Fenwicks library defined in the first blog, and focus on all the steps take by David in 8 posts from the second link.
3. Submit it as a readme file. 
---------------------------------------------------------------------------------------

## **Report 1 - 94% accuracy on Cifar10 in 2 minutes**
https://mc.ai/tutorial-2-94-accuracy-on-cifar10-in-2-minutes/

Dataset Used - Cifar10
OBJECTIVE - Reach 94% accuracy on Cifar10 in 2 minutes

### Cifar10
* 32×32 images
* 10 different classes
* getting 90% on Cifar10 requires serious work

### Model
* DavidNet [David C. Page, who built a custom 9-layer Residual ConvNet, or ResNet]

### Fenwicks
* Fenwicks library build by David Yang was used in thie tutorial.
* Github Link - https://github.com/fenwickslab/fenwicks.git

### Steps Followed
1. Make necessary imports, download Fenwicks, and set up Google Cloud Storage (GCS).
2. Define some tunable **hyperparameters**.
3. **Preparing data** - putting the entire data in memory would lead to an out of memory error, we need to store the dataset on GCS[Google Cloud Storage]

    ``` 
    # using Fenwicks we can make connection to GCS
    data_dir, work_dir = fw.io.get_gcs_dirs(BUCKET, PROJECT)
    ```
4. **TFRecord** - In Tensorflow, the preferred file format is TFRecord, which is compact and efficient since it is based on Google’s ubiquitous ProtoBuf serialization library. Fenwicks provides a one-liner for this:
    ```
    train_fn = os.path.join(data_dir, "train.tfrec")
    test_fn = os.path.join(data_dir, "test.tfrec")
    fw.io.numpy_tfrecord(X_train, y_train, train_fn)
    fw.io.numpy_tfrecord(X_test, y_test, test_fn)
    ```
5. **Data augmentation and input pipeline** - Fenwicks provide one liners for data augmentation techniques such as pad 4 pixels to 40×40, crop back to 32×32, and randomly flip left and right. It can also apply Cutout augmentation as a regularization measure, which alleviates overfitting.
    ```
    x = fw.transform.ramdom_pad_crop(x, 4)
    x = fw.transform.random_flip(x)
    x = fw.transform.cutout(x, 8, 8)
    ```
6. **Building ConvNet** - For DavidNet original implementation is in PyTorch, and PyTorch’s default way to set the initial, random weights of layers does not have a counterpart in Tensorflow. Fenwicks takes care of that.
    ```
    model = fw.Sequential()
    model.add(fw.layers.ConvBN(c, **fw.layers.PYTORCH_CONV_PARAMS))
    model.add(fw.layers.ConvResBlk(c*2, res_convs=2,
    **fw.layers.PYTORCH_CONV_PARAMS))
    model.add(fw.layers.ConvBlk(c*4, **fw.layers.PYTORCH_CONV_PARAMS))
    model.add(fw.layers.ConvResBlk(c*8, res_convs=2,
    **fw.layers.PYTORCH_CONV_PARAMS))
    model.add(tf.keras.layers.GlobalMaxPool2D())
    model.add(fw.layers.Classifier(n_classes, 
    kernel_initializer=fw.layers.init_pytorch, weight=0.125))
    ```
7. **Model training** - DavidNet trains the model with Stochastic Gradient Descent with Nesterov momentum, with a slanted triangular learning rate schedule
    ```
    lr_func = fw.train.triangular_lr(LEARNING_RATE/BATCH_SIZE, total_steps, warmup_steps=WARMUP*steps_per_epoch)
    ```

    * SGD Optimizer and model function for TPUEstimator using fenwicks
    ```
    opt_func = fw.train.sgd_optimizer(lr_func, mom=MOMENTUM,wd=WEIGHT_DECAY*BATCH_SIZE)
    model_func = fw.tpuest.get_clf_model_func(build_nn, opt_func,reduction=tf.losses.Reduction.SUM)
    est = fw.tpuest.get_tpu_estimator(n_train, n_test, model_func, work_dir, trn_bs=BATCH_SIZE)
    est.train(train_input_func, steps=total_steps)
    ```
8. Cleaning the GCS using Fenwicks
    ```
    fw.io.create_clean_dir(work_dir)
    ``` 
9. After the slowish initialization and first epoch, each epoch takes around 2.5 seconds. Since there are 24 epochs in total, the total amount of time spent on training is roughly a minute.Most of the time, the evaluation result is over 94% accuracy.

--------------------------------------------------------------------------------------

## **Report 2 - How to Train Your ResNet**
https://myrtle.ai/how-to-train-your-resnet/

Dataset Used - Cifar10
**OBJECTIVE** - How to train Residual networks on the CIFAR10 image classification dataset and how to do so efficiently on a single GPU.

This report is divided into **8 parts** based on the gradual improvements in traing time and keeping the validation accuracy as close as 94%: 

1. **Baseline**: We analyse a baseline and remove a bottleneck in the data loading. **(training time: 297s)**
    * From fast.ai student Ben Johnson, reached 94% accuracy in under 6 minutes (341s). 
        ```
        The network used in the fastest submission was an 18-layer Residual network
        ```
    * Author built a version of the network in PyTorch and replicated the learning rate schedule and hyperparameters from the DAWNBench submission. Training on an AWS p3.2×large instance with a single V100 GPU, 3/5 runs reach a final test accuracy of 94% in 356s.
    
    **Improvements:**
    1. `Removing the duplication`
        * the network starts with two consecutive (yellow-red) batch norm-ReLU groups after the first (purple) convolution.This was presumably not an intentional design and so let’s remove the duplication.
        *  The strange `kink in the learning rate` at epoch 15 has to been removed.
        * With these changed - new results - **94% final test accuracy in 323s**.
    2. `Removing the repeat work and reducing the number of dataloader processes`
        * By doing the common work once before training [like image preprocessing], removing pressure from the preprocessing jobs, we can reduce the number of processes needed to keep up with the GPU down to one.
        * Saved further 15s training time and a new **training time of 308s.**
    3. `Reducing random number generator time`
        * Most preprocessing time is spent calling out to random number generators to select data augmentations.
        * We make several million individual calls to random number generators and by combining these into a small number of bulk calls at the start of each epoch we can shave a further 7s of training time.
        * Overhead of launching even a single process to perform the data augmentation outweighs the benefit - and by doing the work on the main theread we can further save 4s.
        * **Final training time - 297s**


2. **Mini-batches**: We increase the size of mini-batches. Things go faster and don’t break. We investigate how this can be. **(training time: 256s)**
    * We increase batch size to 512, with one minor adjustment to the learning rate: increasing it by 10% - **Training completes in 256s**. The noisier validation results during training at batch size 512 are expected because of batch norm effect.
    * There are two regimes to be consider if one wishes to train a neural network at high learning rates:
        1. For the current model and dataset, at batch size 128 - forgetfulness dominates; we should either focus on methods to reduce this or we should push batch sizes higher.
        2. At batch size 512 we enter the regime where curvature effects dominate and the focus should shift to mitigating these.
    * For a larger dataset such as ImageNet-1K,the effects of forgetfulness are likely to be much more severe - attempts to speed up training at small batch sizes with very high learning rates have failed on this dataset whilst training with batches of size 8000 or more across multiple machines has been successful.


3. **Regularisation**: We remove a speed bump in the code and add some regularisation. Our single GPU is faster than an eight GPU competition winner. **(training time: 154s)**
    * After getting rough timing profile of our current setup - Found the problem with batch norms – the default method of converting a model to half precision in PyTorch (as of version 0.4) triggers a slow code path which doesn’t use the optimized CuDNN routine. If we convert batch norm weights back to single precision then the fast code is triggered.With this improvement new training time- 186s.
    * Regularisation scheme that has been shown to be effective on CIFAR10 is so-called Cutout regularisation which consists of zeroing out a random subset of each training image- median run reaching 94.3%, a small improvement over the baseline.
    * If we accelerate the learning rate schedule to 30 epochs, we can push the batch size higher to 768 - getting **training time of 154s** with 94% test accuracy (momentum=0.9, weight decay=5e-4).


5. **Architecture**: We search for more efficient network architectures and find a 9 layer network that trains well. **(training time: 79s)**
    * We studied how the shortest path through the network trains in isolation and to take steps to improve this before adding back the longer branches.
    * Original backbone: Training the shortest path network for 20 epochs yields an unimpressive test accuracy of 55.9% in 36 seconds
    * No repeat BN-ReLU: Removing repeated batch norm-ReLU groups, reduces training time to 32s but leaves test accuracy approximately unchanged.
    * 3×3 convolutions: Shortcoming of this backbone network is that the downsampling convolutions have 1×1 kernels and a stride of two - they are simply discarding information; we replace these with 3×3 convolutions, things improve considerably and test accuracy after 20 epochs is 85.6% in a time of 36s.
    * Max pool downsample: Improved the downsampling stages by applying 3×3 convolutions of stride one followed by a pooling layer; we choose max pooling with a 2×2 window size leading to a final test accuracy of 89.7% after 43s.
    * Global max pool: The final pooling layer before the classifier is a concatenation of global average pooling and max pooling layers; we replace this with a more standard global max pooling layer and double the output dimension of the final convolution to compensate for the reduction in input dimension to the classifier, leading to a final test accuracy of 90.7% in 47s.
    * Better BN scale init: Initial batch norm scales replaced with a constant initialisation at 1, with this change in place, 20 epoch training reaches a test accuracy of 91.1% in 47s.
    * We shall consider two classes of networks. The first is constructed by optionally adding a convolutional layer (with batch norm-ReLU) after each max pooling layer. The second class is constructed by optionally adding a residual block consisting of two serial 3×3 convolutions with an identity shortcut, after the same max pooling layers.We train each of the 15 networks (improved backbone + 7 variations in each class) for 20 epochs
    * Residual:L1+L3 network achieves 93.8% test accuracy in 66s for a 20 epoch run. If we extend training to 24 epochs, 7 out of 10 runs reach 94% with a mean accuracy of 94.08% and training time of 79s!
    *  We have found a 9 layer deep residual network which trains to **94% accuracy in 79s**, cutting training time almost in half.

5. **Hyperparameters**: We develop some heuristics to aid with hyperparameter tuning.
    * We fix the choice of network, set batch size to 512 and assume a learning rate schedule that increases linearly from zero for the first 5 epochs and decays linearly for the remainder. The hyperparameters that we aim to recover are the maximal learning rate λ, Nesterov momentum ρ, and weight decay α. We assume that we know nothing about reasonable values for these hyperparameters and start with arbitrary choices λ=0.001, ρ=0.5, α=0.01 which achieve a test accuracy of 30.6% after 24 epochs.
    * To optimise these hyperparameters we will follow a type of cyclical coordinate descent in which they tune one parameter at a time with a very crude line search (doubling or halving and retraining until things get worse).They optimise the learning rate λ first, then momentum ρ, then weight decay α before cycling through again. They halt the whole process when things stop improving.The final test accuracy achieved is 94.2%.
    * Weight decay in the presence of batch normalisation acts as a stable control mechanism on the effective step size. If gradient updates get too small, weight decay shrinks the weights and boosts gradient step sizes until equilibrium is restored. The reverse happens when gradient updates grow too large.    
    
    
6. **Weight decay**: We investigate how weight decay controls the learning rate dynamics.
    * LARS[Layer-wise Adaptive Rate Scaling] is employed in most of the recent successful attempts to scale ImageNet training to very large batch sizes. Given the close relation between LARS and ordinary SGD with weight decay, it would be very interesting to understand what leads to the superior performance of LARS.
    * If the early part of training is problematic in SGD, one could adjust weight initialisation scales to remove the initial out-of-equilibrium stage. If gradient noise is the issue, then employing lower learning rates and/or higher momenta for selected layers might improve things, or simply smoothing gradient norms between batches.
    * The stability in optimal learning rates across architectures is somewhat surprising given that we are using ordinary SGD and the same learning rate for all layers.
    * The LARS-like dynamics of SGD with weight decay, provides a useful type of adaptive scaling for the different layers so that each receive the same step size in scale invariant units and that this renders manual tuning of learning rates per layer unnecessary. 



7. **Batch norm**: We learn that batch normalisation protects against covariate shift after all.
    * Batch norm enables the high learning rates crucial for rapid training. Batch norm works by reparameterising the function space such that these constraints are easier to enforce, curvature of the loss landscape is diminished and training can proceed at a high rate.
    * Batch Norm Advantages:
        1. it stabilises optimisation allowing much higher learning rates and faster training
        2. it injects noise (through the batch statistics) improving generalisation
        3. it reduces sensitivity to weight initialisation
        4. it interacts with weight decay to control the learning rate dynamics
    * Batch Norm Disadvantages:
        1. it’s slow (although node fusion can help)
        2. it’s different at training and test time and therefore fragile
        3. it’s ineffective for small batches and various layer types
        4. it has multiple interacting effects which are hard to separate.
    * The ability to use high learning rates allows training to proceed much more rapidly for the model with batch norm. If batch norm is not used, network becomes unstable.
    * In the absence of batch norm, the standard initialisation scheme for deep networks leads to ‘bad’ configurations in which the network effectively computes a constant function. By design, batch norm goes a long way towards fixing this.
    * For the networks without active batch norm, changes to the distribution of outputs in earlier layers, can propagate to changes in distribution at later layers. In other words, internal covariate shift is able to propagate to external shift at the output layer.
    * For the ‘batch norm’ network, changes to the mean and variance at earlier layers get removed by subsequent batch normalisation and so there is little opportunity for early layers to affect the output distribution.
    * The internal layer distributions are also important in order to maintain the expressivity of the function computed by the network. For example, if at some intermediate layer, a small number of channels become dominant, this would introduce a bottleneck, greatly reducing the set of functions that the network is capable of representing. We would expect that directly tackling this kind of internal distributional shift is a further role of batch norm.


8. **Bag of tricks**: We uncover many ways to speed things up further when we find ourselves displaced from the top of the leaderboard. **(final training time: 26s)**
    * Preprocessing on the GPU:  transferring the data to the GPU, preprocessing there and then transferring back to the CPU for random data augmentation and batching - **training time drops just under 70s**.
    * Moving max-pooling layers: Max-pooling commutes with a monotonic-increasing activation function such as ReLU. It should be more efficient to apply pooling first- saves us 3s. Moving max-pooling before batch norm further saved 5s.The net effect brings our **time to 64s**.
    * Label smoothing: It involves blending the one-hot target probabilities with a uniform distribution over class labels inside the cross entropy loss. Accuracy for 23 epochs of training is 94.1% and **training time dips under a minute (59s)!***
    * CELU activations: 20 epoch time of 52s for 94.1% accuracy.
    * Ghost batch norm: We can apply batch norm separately to subsets of a training batch. This technique, known as ‘ghost’ batch norm, is usually used in a distributed setting but is just as useful when using large batches on a single node.We can achieve 94.1% accuracy in 18 epochs and a **training time of 46s**.
    * Frozen batch norm scales: The new test accuracy is 94.1% with a **time of 43s.**
    * Input patch whitening: Batch norm does a good job at controlling distributions of individual channels but doesn’t tackle covariance between channels and pixels. They applied PCA whitening to 3×3 patches of inputs as an initial 3×3 convolution with fixed (non-learnable) weights. 94.1% test accuracy in **36s**.
    * Exponential moving averages: We update the moving average every 5 batches. 13 epoch training reaches a test accuracy of 94.1%, achieving a **training time below 34s**.
    * Test-time augmentation: If they remove the remaining cutout data augmentation – which is getting in the way on such a short training schedule – we can reduce training to 10 epochs and achieve a TTA test accuracy of 94.1% in **26s**!



--------------------------------------------------------------------------------------

