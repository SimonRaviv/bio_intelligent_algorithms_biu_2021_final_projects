import os
import copy
import argparse
import math
import time
import pickle


from abc import ABC, abstractmethod


try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy
    import cv2

    from skimage.util import random_noise
    from skimage.transform import rotate
    from skimage.filters import gaussian
except ImportError:
    print("Failed to import one of Python external modules. \n"
          "Make sure you have the following libraries installed:\n"
          "pandas, matplotlib, numpy, opencv-python and scikit-image installed")
    exit(1)


"""
    Initialize the acceleration library to use for mathematical operations:
 
    The serialized module dump used with: '-l','--load-path' options
    should be compitable with the runtime this application run on.
 
    On CuPy platform:
        CuPy module dump file should be used, unless changed statically, see below.
 
    On NumPy only platform:
        NumPy module dump should be used.
 
    The acceleration module used is CuPy if installed, NumPy otherwise.
    This can be changed statically, by changing the 'lib'.
"""
lib = None
try:
    import cupy
    lib = cupy
except ImportError:
    lib = numpy


########################################################################################################################
# Globgal variables and structures
########################################################################################################################

random_generator_seed = 2021
rand_gen = lib.random.RandomState(seed=random_generator_seed)
np_rand_gen = numpy.random.RandomState(seed=random_generator_seed)

ACTIVATION_FUNCTIONS = ['sigmoid', 'relu', 'leaky_relu']
EARLY_STOP_CRITERIONS = ['validation_loss', 'validation_accuracy']
OPTIMIZERS = ['sgd', 'momentum', 'adagrad', 'rmsprop', 'adam']

DEBUG = False

if lib.__name__ == 'cupy':
    G_MEMPOOL = cupy.get_default_memory_pool()
    G_PINNED_MEMPOOL = cupy.get_default_pinned_memory_pool()
    G_MEMPOOL_FREE_INTERVAL = 100

########################################################################################################################
# Globgal functions
########################################################################################################################


def debug_print(string, arg=''):
    """
    Debug log print.

    In order to activate this log, set DEBUG variable to True.

    Input:
        string: The string to log.
        arg: Argument to the print function.

    Returns:
        None.
    """
    if DEBUG is True:
        print(string, arg)


def format_time(start_time, end_time):
    """
    Format the time passed between the supplied timestamps.

    Input:
        start_time: Start timesamp from time.time() call.
        end_time: End timesamp from time.time() call.

    Returns:
        Formated time string of the form HH:MM:SS.
    """
    hours, reminder = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(reminder, 60)
    return "{:0>2}:{:0>2}:{:02}".format(int(hours), int(minutes), int(seconds))

########################################################################################################################
# Data augmentation implementation
########################################################################################################################


class BaseAugmentation(ABC):
    """
    Base class for Data Augmentors.

    This class exposes data augmentation interfaces for different type of augmentations.
    """

    def __init__(self):
        pass

    def _preprocess(self, image):
        """
        Do preprocessing on the image before the augmentation.

        Input:
            Input image, image.shape == (hight, width, depth)

        Returns:
            Preprocessed image, output image shape == image.shape.
        """
        if lib.__name__ == 'cupy':
            image = lib.asnumpy(image)

        return image

    def _postprocess(self, image):
        """
        Do postprocessing on the image after the augmentation.

        Input:
            Input image, image.shape == (hight, width, depth)

        Returns:
            Postprocessed image, output image shape == image.shape.
        """
        if lib.__name__ == 'cupy':
            image = lib.array(image)

        return image

    @abstractmethod
    def do_augmentation(self, image):
        """
        Do augmentation on the image.

        Input:
            Input image, image.shape == (hight, width, depth)

        Returns:
            Augmented image, output image shape == image.shape.
        """
        pass

    def augment(self, image):
        """
        Helper method that uses the derived class augmentation.

        This method does the augmentation and the pre/post processing needed on the image.

        Input:
            Input image, image.shape == (hight, width, depth)

        Returns:
            Augmented image, output image shape == image.shape.
        """
        image = self._preprocess(image)
        image = self.do_augmentation(image)
        image = self._postprocess(image)

        return image


class AntiClockwiseRotationAugmentation(BaseAugmentation):
    """
    Anti clockwise random angle rotation augmentation.
    """

    def do_augmentation(self, image):
        angle = rand_gen.uniform(low=0, high=180)
        return rotate(image, angle)


class ClockwiseRotationAugmentation(BaseAugmentation):
    """
    Clockwise random angle rotation augmentation.
    """

    def do_augmentation(self, image):
        angle = rand_gen.uniform(low=0, high=180)
        return rotate(image, -angle)


class HorizontalFlipAugmentation(BaseAugmentation):
    """
    Horizontal Flip Augmentation.
    """

    def do_augmentation(self, image):
        return lib.fliplr(image)


class VerticalFlipAugmentation(BaseAugmentation):
    """
    Vertical Flip Augmentation.
    """

    def do_augmentation(self, image):
        return lib.flipud(image)


class RandomNoiseAugmentation(BaseAugmentation):
    """
    Random Noise Augmentation.
    """

    def do_augmentation(self, image):
        return random_noise(image, seed=random_generator_seed)


class BlurAugmentation(BaseAugmentation):
    """
    Blur augmentation
    """

    def do_augmentation(self, image):
        return gaussian(image, sigma=(1, 1), truncate=3.5, multichannel=True)


class NoAugmentation(BaseAugmentation):
    """
    No augmentation.
    """

    def do_augmentation(self, image):
        return image


DATA_AUGMENTORS = [
    AntiClockwiseRotationAugmentation(),
    ClockwiseRotationAugmentation(),
    HorizontalFlipAugmentation(),
    VerticalFlipAugmentation(),
    RandomNoiseAugmentation(),
    BlurAugmentation(),
    NoAugmentation()
]

########################################################################################################################
# Convolutional Layer implementation
########################################################################################################################


class Conv2D(object):
    """
    2D Convolution layer.

    This class implements 2D convolution layer forward and backward propagation.
    The implementation is fully vectorized to multiple examples and filters.
    """

    def __init__(self, stride=1):
        self._stride = stride

    def forward_convolve(self, feature_map, filter_, bias, same_convolution=True):
        """
        Forward propagation using convolution of the filter over the feature map.

        Input:
            feature_map: The input feature map,
                         feature_map.shape == (n_examples, feature_map_dim, feature_map_dim, d_feature_map).
            filter_: The filter weights to use,
                    filter_.shape == (filter_dim, filter_dim, d_filter_current, d_filter_next).
            bias: The bias to use, bias.shape == (d_filter_next,)
            same_convolution: Do same or valid convolution, default: True.

        Returns:
            The convolved feature map, convolved.shape == (n_examples, output_dim, output_dim, d_filter_next).
        """
        filter_dim, _, d_filter_current, d_filter_next = filter_.shape
        n_examples, feature_map_dim, _, d_feature_map = feature_map.shape

        if d_feature_map != d_filter_current:
            print("ERROR: The depth of filter must match the depth of input feature map")
            return None

        if same_convolution is True:
            output_dim = feature_map_dim
        else:
            output_dim = int((feature_map_dim - filter_dim) / self._stride) + 1

        convolved = lib.zeros((n_examples, output_dim, output_dim, d_filter_next))

        current_h = output_h = 0
        while current_h + filter_dim <= feature_map_dim:
            current_w = output_w = 0
            while current_w + filter_dim <= feature_map_dim:
                h_s, h_e = current_h, current_h + filter_dim
                w_s, w_e = current_w, current_w + filter_dim
                convolved[:, output_h, output_w, :] = lib.sum(
                    feature_map[:, h_s:h_e, w_s:w_e, :, lib.newaxis] * filter_[lib.newaxis, :, :, :],
                    axis=(1, 2, 3)
                )

                current_w += self._stride
                output_w += 1
            current_h += self._stride
            output_h += 1

        convolved += bias
        return convolved

    def backward_convolve(self, dbackward_pass_input, forward_pass_input, filter_, reg):
        """
        Backward propagation of the convolution operation.

        Input:
            dbackward_pass_input: The input to the backward pass (right side).
            forward_pass_input: The input to the forward pass (left side),
                                forward_pass_input.shape == (n_examples, original_dim, original_dim, depth).
            filter_: The filter weights to use,
                    filter_.shape == (filter_dim, filter_dim, d_filter_current, d_filter_next).
            reg: The L2 regularization factor.

        Returns:
            The product of the backward pass operation: doutput, dfilter, dbias.

            douput.shape == (forward_pass_input.shape).
            dfilter.shape == (filter_.shape).
            dbias.shape == (num_of_filters,).
        """
        filter_dim, _, _, _ = filter_.shape
        n_examples, original_dim, _, _ = forward_pass_input.shape

        doutput = lib.zeros(forward_pass_input.shape)
        dfilter = lib.zeros(filter_.shape)
        dbias = dbackward_pass_input.sum(axis=(0, 1, 2)) / n_examples

        current_h = output_h = 0
        while current_h + filter_dim <= original_dim:
            current_w = output_w = 0
            while current_w + filter_dim <= original_dim:
                h_s, h_e = current_h, current_h + filter_dim
                w_s, w_e = current_w, current_w + filter_dim
                doutput[:, h_s:h_e, w_s:w_e, :] += lib.sum(
                    filter_[lib.newaxis, :, :, :, :] *
                    dbackward_pass_input[:, output_h:output_h + 1, output_w:output_w + 1, lib.newaxis, :],
                    axis=4
                )
                dfilter += lib.sum(
                    forward_pass_input[:, h_s:h_e, w_s:w_e, :, lib.newaxis] *
                    dbackward_pass_input[:, output_h:output_h + 1, output_w:output_w + 1, lib.newaxis, :],
                    axis=0
                )
                current_w += self._stride
                output_w += 1
            current_h += self._stride
            output_h += 1

        dfilter = (1. / n_examples) * (dfilter + reg * filter_)

        return doutput, dfilter, dbias

########################################################################################################################
# Max pooling Layer implementation
########################################################################################################################


class MaxPool2D(object):
    """
    2D Max Pooling.

    This class implements 2D Max Pooling layer forward and backward propagation.
    The implementation is fully vectorized to multiple examples and filters.
    """

    def __init__(self, kernel_dim=2, stride=2):
        self._kernel_dim = kernel_dim
        self._stride = stride
        self._forward_cache = dict()

    def forward_maxpool(self, feature_map):
        """
        Forward propagation pass of 2D max pool operation.

        Input:
            feature_map: The input feature map,
                         feature_map.shape == (n_examples, feature_map_dim, feature_map_dim, d_feature_map).

        Returns:
            The feature map after the pooling operation.
            pooled_output.shape == (n_examples, hight_output, width_output, d_feature_map).
        """
        self._forward_cache.clear()
        self._forward_cache = dict()
        n_examples, feature_map_dim, _, d_feature_map = feature_map.shape
        hight_output = int((feature_map_dim - self._kernel_dim) / self._stride) + 1
        width_output = int((feature_map_dim - self._kernel_dim) / self._stride) + 1
        pooled_output = lib.zeros((n_examples, hight_output, width_output, d_feature_map))

        current_h = output_h = 0
        while current_h + self._kernel_dim <= feature_map_dim:
            current_w = output_w = 0
            while current_w + self._kernel_dim <= feature_map_dim:
                h_s, h_e = current_h, current_h + self._kernel_dim
                w_s, w_e = current_w, current_w + self._kernel_dim
                window_4d = feature_map[:, h_s:h_e, w_s:w_e, :]
                pooled_output[:, output_h, output_w, :] = lib.max(window_4d, axis=(1, 2))

                mask = lib.zeros_like(window_4d)
                cache = window_4d.reshape(n_examples, window_4d.shape[1] * window_4d.shape[2], d_feature_map)
                index = lib.argmax(cache, axis=1)
                n_example_index, d_window_4d_index = lib.indices((n_examples, d_feature_map))
                mask.reshape(n_examples,
                             window_4d.shape[1] * window_4d.shape[2],
                             d_feature_map)[n_example_index, index, d_window_4d_index] = 1

                self._forward_cache[(output_h, output_w)] = mask

                current_w += self._stride
                output_w += 1
            current_h += self._stride
            output_h += 1

        return pooled_output

    def backward_maxpool(self, dbackward_pass_input, forward_pass_input):
        """
        Backward propagation pass through a max pooling layer.

        Input:
            dbackward_pass_input: The input to the backward pass (right side).
            forward_pass_input: The input to the forward pass (left side),
                                forward_pass_input.shape == (n_examples, original_dim, original_dim, depth).

        Returns:
            The product of the backward pass operation: doutput.

            douput.shape == (forward_pass_input.shape).
        """
        _, original_dim, _, _ = forward_pass_input.shape
        doutput = lib.zeros(forward_pass_input.shape)

        current_h = output_h = 0
        while current_h + self._kernel_dim <= original_dim:
            current_w = output_w = 0
            while current_w + self._kernel_dim <= original_dim:
                h_s, h_e = current_h, current_h + self._kernel_dim
                w_s, w_e = current_w, current_w + self._kernel_dim
                doutput[:, h_s:h_e, w_s:w_e, :] += \
                    dbackward_pass_input[:, output_h:output_h + 1, output_w:output_w + 1, :] * \
                    self._forward_cache[(output_h, output_w)]

                current_w += self._stride
                output_w += 1
            current_h += self._stride
            output_h += 1

        return doutput


########################################################################################################################
# Multi class convolutional neural network model implementation
########################################################################################################################


class MultiClassConvolutionalNeuralNetwork(object):
    """
    Multi Class Convolutional Neural Network implementation.

    The class offers generic implementation for multi class categorial datasets.
    User can modify the parameters of the constructor to use its various features.

    This class is implemented using a GPU accelerated library following the same API as numpy.
    The library is CuPy, accelerates numpy API on a NVIDIA GPU.

    The model offers the following architecture:
        (1) Input layer: controllable by the data_dim parameter for the number of features.
        (2) Convolutional layer 1: Controllable by the f1_num_filter, conv2d_1_filter_dim, conv2d_1_stride
        (3)                        parameteres for the number of filters, filter dimension and filter stride.
        (4) Max pool layer 1: Uses kernel dimension of 2x2 and stride 2.
        (5) Convolutional layer 2: Controllable by the f2_num_filter, conv2d_2_filter_dim, conv2d_2_stride
        (6)                        parameteres for the number of filters, filter dimension and filter stride.
        (7) Max pool layer 2: Uses kernel dimension of 2x2 and stride 2.
        (8) Fully connected layer 1: Controllable by the l1_fc_hidden_size parameter for the number of neurons in this layer.
        (9) Fully connected layer 2: Controllable by the l2_fc_hidden_size parameter for the number of neurons in this layer.
        (10) Output layer: Controllable by the num_of_classes parameter for the number of classes in the dataset.

    Optimizers:
        The class offers various optimizers: ['sgd', 'momentum', 'adagrad', 'rmsprop', 'adam'],
        user can choose by optimizer parameter.

    Regularizes:
        The class offers 3 type of regularizes:
            * Input noise: The user can control the probability of non-active input features by input_noise_p parameter.
            * Dropout: The user can control the probability of the number of non-active neurons by the dropout_p parameter.
            * L2 regularization: The user can control the regularization factor by reg parameter.

    Features:
        * The class offers various activation functions: ['sigmoid', 'relu', 'leaky_relu'],
            user can choose by activation_func parameter.
        * The user can control static learning rate by lr parameter.
        * The class implements learning rate decay algorithm,
            user can control the algorithm using the following parameters: initial_lr,
            lr_decay_drop_ratio and lr_decay_epochs_drop.
        * The class implements early stop algorithm:
            * User can choose the max epochs by epochs parameter.
            * User can control the max number of epochs without improvement in accuracy before
                stopping by changing early_stop_max_epochs parameter.
            * User can choose the creteria for stopping by using early_stop_criteria parameter,
                available options: ['validation_loss', 'validation_accuracy']
        * The class implements Mini-Batch optimizers, user can control the batch size by batch_size parameter.
        * The class implements initial weights by the Gaussian distribution,
            the user can control the expectation and standard deviation by init_weights_mu and init_weights_sigma parameters.
        * The class offers KFold cross validation, the user can run it by
            cross_validate function and control the number of folds by k_fold parameter of the function.
        * The class offers Z-Score Normalization on the input dataset.
        * The class supports data augmentation of the input picutures, the class does the following augmentations:
                Anti Clockwise Rotation, Clockwise Rotation, Horizontal Flip, Vertical Flip, Random Noise and Blur.
                User can control the percentage of the data to do the augmentation on during the training, by using
                the data_augmentation_ratio parameter.
        * The class supports transfer learning from another another trained model, see fit interface.
    """

    def __init__(self,
                 data_dim=3072,
                 num_of_input_chanels=3,
                 input_chanel_dim=32,
                 activation_func='leaky_relu',
                 optimizer='momentum',
                 conv2d_1_filter_dim=5,
                 conv2d_2_filter_dim=3,
                 conv2d_1_stride=1,
                 conv2d_2_stride=1,
                 f1_num_filter=32,
                 f2_num_filter=64,
                 l1_fc_hidden_size=512,
                 l2_fc_hidden_size=128,
                 num_of_classes=10,
                 lr=0.001,
                 initial_lr=0,
                 lr_decay_drop_ratio=0.15,
                 lr_decay_epochs_drop=15,
                 epochs=1000,
                 batch_size=50,
                 reg=0.15,
                 input_noise_p=0,
                 dropout_p=0.2,
                 early_stop_max_epochs=25,
                 early_stop_criteria='validation_loss',
                 init_weights_mu=0,
                 init_weights_sigma=0.1,
                 input_z_score_normalization=False,
                 data_augmentation_ratio=0.3):
        """
        Model constructor.

        Input:
            data_dim: Number of features in the dataset.
            num_of_input_chanels: Number of chanels in the input, defult: 3.
            input_chanel_dim: Input chanel dimension, default: 32 (32x32).
            activation_func: Activation function name as a string, default: 'leaky_relu', supported functions: ['sigmoid', 'relu', 'leaky_relu']
            optimizer: Optimizer name as a string, default: 'momentum', supported optimizers: ['sgd', 'momentum', 'adagrad', 'rmsprop', 'adam']
            conv2d_1_filter_dim: Convolutional layer 1 filter dimension, default:5.
            conv2d_2_filter_dim: Convolutional layer 2 filter dimension, default:3.
            conv2d_1_stride: Convolutional layer 1 filter stride, default: 1.
            conv2d_2_stride: Convolutional layer 2 filter stride, default: 1.
            f1_num_filter: Number of filters in convolutional layer 1, default: 32.
            f2_num_filter: Number of filters in convolutional layer 2, default: 64.
            l1_fc_hidden_size: Number of neurons in the first hidden layer, default: 512.
            l2_fc_hidden_size: Number of neurons in the second hidden layer, default: 128.
            num_of_classes: Number of categorial class of the data, default: 10.
            lr: Learning rate for the optimization algorithm, default: 0.001.
            initial_lr: Initial learning rate for the optimization algorithm, default: 0.
                        This feature has higher priority than the static lr feature.
            lr_decay_drop_ratio: The ratio by which to drop the learning rate in the decay algorithm, default: 0.15.
            lr_decay_epochs_drop: Number of epochs interval to drop the learning rate in decay algorithm, default: 15.
            epochs: Max number of epochs to do, the algorithm might stop before, due to early stop, default: 1000.
            batch_size: Batch size to use for the Mini-Batch Gradient Descent, default: 50.
            reg: Regularization factor, default: 0.15.
            input_noise_p: The probability of non-active input features, default: 0.
            dropout_p: Dropout probability of non-active neurons, default: 0.2.
            early_stop_max_epochs: Maximum number of epochs without substantial improvement in model
                                   accuracy before stopping the training, default: 25.
            early_stop_criteria: Early stop algorithm criteria, default: 'validation_loss',
                                 supported criterions: ['validation_loss', 'validation_accuracy'].
            init_weights_mu: Expectation of the normal distribution for model weights initialization, default: 0.
            init_weights_sigma: Standard deviation of the normal distribution for model weights initialization, default: 0.1.
            input_z_score_normalization: Controls whether Z-Score Normalization on the input dataset on or off, default: Off.
            data_augmentation_ratio: Data augmentation ratio, default: 0.3.

        Returns:
            None
        """
        self._data_dim = data_dim
        self._params = None
        self._activation_cb_map = {
            'sigmoid': (self._sigmoid, self._sigmoid_derivative),
            'relu': (self._relu, self._relu_derivative),
            'leaky_relu': (self._leaky_relu, self._leaky_relu_derivative)
        }
        self._optimizers_cb_map = {
            'sgd': self._sgd,
            'momentum': self._momentum,
            'adagrad': self._adagrad,
            'rmsprop': self._rmsprop,
            'adam': self._adam
        }
        self._activation_func = activation_func
        activation_cb, activation_derivative_cb = self._activation_cb_map.get(
            activation_func, 'relu')
        self._activation_cb = activation_cb
        self._activation_derivative_cb = activation_derivative_cb

        self._optimizer = optimizer
        optimizer_cb = self._optimizers_cb_map.get(
            optimizer, 'sgd')
        self._optimizer_cb = optimizer_cb

        self._input_dim = input_chanel_dim
        self._num_of_input_chanels = num_of_input_chanels
        self._chanel_len = int(self._data_dim / self._num_of_input_chanels)
        self._maxpool_1_kernel_dim = 2
        self._maxpool_1_stride = 2
        self._maxpool_2_kernel_dim = 2
        self._maxpool_2_stride = 2
        self._conv2d_1_filter_dim = conv2d_1_filter_dim
        self._conv2d_2_filter_dim = conv2d_2_filter_dim
        self._conv2d_1_stride = conv2d_1_stride
        self._conv2d_2_stride = conv2d_2_stride
        self._f1_num_filter = f1_num_filter
        self._f2_num_filter = f2_num_filter
        self._maxpool_1_hight = int(((self._input_dim - self._maxpool_1_kernel_dim) / self._maxpool_1_stride) + 1)
        self._maxpool_1_width = int(((self._input_dim - self._maxpool_1_kernel_dim) / self._maxpool_1_stride) + 1)
        self._maxpool_1_depth = self._f1_num_filter
        self._maxpool_2_hight = int(((self._maxpool_1_hight - self._maxpool_2_kernel_dim) / self._maxpool_2_stride) + 1)
        self._maxpool_2_width = int(((self._maxpool_1_width - self._maxpool_2_kernel_dim) / self._maxpool_2_stride) + 1)
        self._maxpool_2_depth = self._f2_num_filter
        self._conv2d_1 = Conv2D(stride=conv2d_1_stride)
        self._conv2d_2 = Conv2D(stride=conv2d_2_stride)
        self._maxpool_1 = MaxPool2D(kernel_dim=self._maxpool_1_kernel_dim, stride=self._maxpool_1_stride)
        self._maxpool_2 = MaxPool2D(kernel_dim=self._maxpool_2_kernel_dim, stride=self._maxpool_2_stride)
        self._l1_fc_hidden_size = l1_fc_hidden_size
        self._l2_fc_hidden_size = l2_fc_hidden_size
        self._num_of_classes = num_of_classes
        self._initial_lr = initial_lr
        self._lr = initial_lr if initial_lr != 0 else lr
        self._epochs = epochs
        self._batch_size = batch_size
        self._reg = reg
        self._input_noise_p = input_noise_p
        self._dropout_p = dropout_p
        self._early_stop_max_epochs = early_stop_max_epochs
        self._initialize_model_architecture()
        self._performance = {
            "train_accuracy": [], "validation_accuracy": [],
            "train_loss": [], "validation_loss": []
        }
        self._orig_performance = copy.deepcopy(self._performance)
        self._init_weights_mu = init_weights_mu
        self._init_weights_sigma = init_weights_sigma
        self._lr_decay_drop_ratio = lr_decay_drop_ratio
        self._lr_decay_epochs_drop = lr_decay_epochs_drop
        self._input_z_score_normalization = input_z_score_normalization
        self._early_stop_criteria_cb_map = {
            'validation_loss': self._early_stop_criteria_validation_loss_cb,
            'validation_accuracy': self._early_stop_criteria_validation_accuracy_cb
        }
        self._early_stop_criteria = early_stop_criteria
        self._early_stop_cb = self._early_stop_criteria_cb_map.get(
            self._early_stop_criteria, 'validation_loss')
        self._validation_criteria_prime = 2**32 if self._early_stop_criteria == 'validation_loss' else -1
        self._early_stop_steps = 0
        self._prime_params = None
        self._prime_epoch = -1
        self._data_augmentors = DATA_AUGMENTORS
        self._data_augmentation_ratio = data_augmentation_ratio
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._eps = 1e-8
        self._optimizer_cache = {}
        self._num_of_updates = 0
        self._transfer_learning_done = False

    @staticmethod
    def one_hot_encoding(Y, num_of_classes=10):
        """
        One hot encoding of categorial classes.

        Input:
            Y: Vector of the data labels, Y.shape == (n_examples, 1).
            num_of_classes: Number of the categorail classes.

        Returns:
            One hot encoded transformation of the Y input vector.
            one_hot_encoded.shape == (num_of_classes, n_examples).
        """
        Y = Y.reshape(Y.shape[0],)
        one_hot_encoded = lib.zeros((Y.size, num_of_classes))
        one_hot_encoded[lib.arange(Y.size), Y.astype(int)] = 1

        return one_hot_encoded.astype('int8')

    @staticmethod
    def load_dataset(dataset_path, without_labels=False):
        """
        Loads dataset.

        The file should contain lines in the following format:
        Label, feature_1, feature_2, ... , feature_n

        Input:
            dataset_path: Full path to the dataset CSV file.
            without_labels: Ignore labels (first column) in the dataset and don't return Y.

        Returns:
            On success:
                A tuple (X, Y) of dataset and labels.
                X.shape == (data_dimension, n_examples).
                Y.shape == (num_of_classes, n_examples).
            On failure:
                A tuple of (None, None)
        """
        first_column = 1
        try:
            if without_labels is True:
                index_col = 0
                first_column = 0
            else:
                index_col = None

            dataset = pd.read_csv(
                dataset_path, header=None, index_col=index_col)
        except FileNotFoundError:
            print("ERROR: Failed to find file: ", dataset_path)
            return None, None

        X = lib.array(dataset.to_numpy(dtype='float64')[:, first_column:])
        X = X.T

        Y = None
        if without_labels is False:
            Y = lib.array(dataset.to_numpy(dtype='int8')[:, :1] - 1)
            Y = MultiClassConvolutionalNeuralNetwork.one_hot_encoding(Y).T

        return X, Y

    @staticmethod
    def dump_model(model, dump_file="model_dump.bin"):
        """
        Serializes the model using Pickle module.

        Platform specifics:
            * Numpy platform: When running a on Numpy platform, 1 model dump will be available:
                * Numpy model dump with the suffix '_numpy'
            * CuPy platform: When running on a CuPy platform, 2 model dumps will be available:
                * CuPy model dump with the suffix '_cupy'
                * Numpy model dump with the suffix '_numpy'

        Input:
            dump_file: The full path to the dump file, including the name, default: './model_dump.bin'.

        Returns:
            On success: The size in bytes of the dump file.
            On failure: None.
        """
        cupy_dump_size = -1
        numpy_dump_size = -1

        # Dump CuPy model:
        try:
            if lib.__name__ == 'cupy':
                cupy_dump_path = dump_file + '_cupy'
                with open(cupy_dump_path, 'wb') as df:
                    pickle.dump(model._params, df)
                    cupy_dump_size = os.path.getsize(cupy_dump_path)

                for key, parameter in model._params.items():
                    model._params[key] = lib.asnumpy(parameter)
        except EnvironmentError:
            print("ERROR: Failed to write to file: ", cupy_dump_path)
            return None, None

        # Dump Numpy model:
        try:
            numpy_dump_path = dump_file + '_numpy'
            with open(numpy_dump_path, 'wb') as df:
                pickle.dump(model._params, df)
                numpy_dump_size = os.path.getsize(numpy_dump_path)
        except EnvironmentError:
            print("ERROR: Failed to write to file: ", numpy_dump_path)
            return None, None

        return numpy_dump_size, cupy_dump_size

    @staticmethod
    def load_model(dump_file="model_dump.bin"):
        """
        Deserialize the model using Pickle module.

        Note:
            The loaded model should be compatible with the runtime platform library.

        Input:
            dump_file: The full path to the dump file, including the name, default: './model_dump.bin'.

        Returns:
            On success: Instance object of MultiClassNeuralNetwork class.
            On failure: None.
        """
        model_params = None

        try:
            with open(dump_file, 'rb') as df:
                model_params = pickle.load(df)

                test_parameter = model_params['B1']
                if isinstance(test_parameter, numpy.ndarray) is True and lib.__name__ == 'cupy':
                    print("ERROR: The loaded model is incompatible with the current library used\n"
                          "       Current library ({current_lib}) != Loaded model library ({loaded_lib})".format(
                              current_lib=lib.__name__, loaded_lib=numpy.__name__
                          ))
                    return None

        except EnvironmentError:
            print("ERROR: Failed to read from the dump file: ", dump_file)
            return None

        return model_params

    def transform_to_cnn_input(self, X):
        """
            Tranforms the input matrix X to the form needed for convolutional network.

            Input:
                X: The input data, X.shape == (data_dimension, n_examples).

            Returns:
                4 D tensor represents the input as multiple images.
                cnn_X.shape == (n_examples, input_dim, input_dim, 3) 
        """
        n_examples = X.shape[1]
        cnn_X = list()

        for example in range(n_examples):
            example_3d = lib.dstack((
                X[0:self._chanel_len, example].reshape(self._input_dim, self._input_dim),
                X[self._chanel_len:self._chanel_len * 2, example].reshape(self._input_dim, self._input_dim),
                X[self._chanel_len * 2:self._chanel_len * 3, example].reshape(self._input_dim, self._input_dim))
            )
            cnn_X.append(example_3d)

        return lib.array(cnn_X)

    def __str__(self):
        """
        Dumps the model to representative string.
        """
        model_str_format = (
            f"{'':-^135}\n{' Model Parameters ':-^135}\n{'':-^135}\n"
            "* model_architecture: {model_architecture}\n"
            "* activation_func: {activation_func}\n"
            "* optimizer: {optimizer}\n"
            "* num_of_classes: {num_of_classes}\n"
            "* data_dim: {data_dim}\n"
            "* chanel_len: {chanel_len}\n"
            "* num_of_input_chanels: {num_of_input_chanels}\n"
            "* input_dim: {input_dim}\n"
            "* f1_num_filter: {f1_num_filter}\n"
            "* f2_num_filter: {f2_num_filter}\n"
            "* l1_fc_hidden_size: {l1_fc_hidden_size}\n"
            "* l2_fc_hidden_size: {l2_fc_hidden_size}\n"
            "* conv2d_1_filter_dim: {conv2d_1_filter_dim}\n"
            "* conv2d_2_filter_dim: {conv2d_2_filter_dim}\n"
            "* conv2d_1_stride: {conv2d_1_stride}\n"
            "* conv2d_2_stride: {conv2d_2_stride}\n"
            "* maxpool_1_kernel_dim: {maxpool_1_kernel_dim}\n"
            "* maxpool_1_stride: {maxpool_1_stride}\n"
            "* maxpool_2_kernel_dim: {maxpool_2_kernel_dim}\n"
            "* maxpool_2_stride: {maxpool_2_stride}\n"
            "* maxpool_1_hight: {maxpool_1_hight}\n"
            "* maxpool_1_width: {maxpool_1_width}\n"
            "* maxpool_1_depth: {maxpool_1_depth}\n"
            "* maxpool_2_hight: {maxpool_2_hight}\n"
            "* maxpool_2_width: {maxpool_2_width}\n"
            "* maxpool_2_depth: {maxpool_2_depth}\n"
            "* epochs: {epochs}\n"
            "* batch_size: {batch_size}\n"
            "* data_augmentation_ratio: {data_augmentation_ratio}\n"
            "* lr: {lr}\n"
            "* initial_lr: {initial_lr}\n"
            "* lr_decay_drop_ratio: {lr_decay_drop_ratio}\n"
            "* lr_decay_epochs_drop: {lr_decay_epochs_drop}\n"
            "* reg: {reg}\n"
            "* input_noise_p: {input_noise_p}\n"
            "* dropout_p: {dropout_p}\n"
            "* early_stop_criteria: {early_stop_criteria}\n"
            "* early_stop_max_epochs: {early_stop_max_epochs}\n"
            "* init_weights_mu: {init_weights_mu}\n"
            "* init_weights_sigma: {init_weights_sigma}\n"
            "* input_z_score_normalization: {input_z_score_normalization}\n"
        )
        model_str = model_str_format.format(
            model_architecture=self._model_architecture,
            data_dim=self._data_dim,
            activation_func=self._activation_func,
            optimizer=self._optimizer,
            data_augmentation_ratio=self._data_augmentation_ratio,
            l1_fc_hidden_size=self._l1_fc_hidden_size,
            l2_fc_hidden_size=self._l2_fc_hidden_size,
            num_of_classes=self._num_of_classes,
            initial_lr=self._initial_lr,
            lr=self._lr,
            epochs=self._epochs,
            batch_size=self._batch_size,
            reg=self._reg,
            input_noise_p=self._input_noise_p,
            dropout_p=self._dropout_p,
            early_stop_criteria=self._early_stop_criteria,
            early_stop_max_epochs=self._early_stop_max_epochs,
            init_weights_mu=self._init_weights_mu,
            init_weights_sigma=self._init_weights_sigma,
            lr_decay_drop_ratio=self._lr_decay_drop_ratio,
            lr_decay_epochs_drop=self._lr_decay_epochs_drop,
            input_z_score_normalization=self._input_z_score_normalization,
            input_dim=self._input_dim,
            num_of_input_chanels=self._num_of_input_chanels,
            chanel_len=self._chanel_len,
            maxpool_1_kernel_dim=self._maxpool_1_kernel_dim,
            maxpool_1_stride=self._maxpool_1_stride,
            maxpool_2_kernel_dim=self._maxpool_2_kernel_dim,
            maxpool_2_stride=self._maxpool_2_stride,
            conv2d_1_filter_dim=self._conv2d_1_filter_dim,
            conv2d_2_filter_dim=self._conv2d_2_filter_dim,
            conv2d_1_stride=self._conv2d_1_stride,
            conv2d_2_stride=self._conv2d_2_stride,
            f1_num_filter=self._f1_num_filter,
            f2_num_filter=self._f2_num_filter,
            maxpool_1_hight=self._maxpool_1_hight,
            maxpool_1_width=self._maxpool_1_width,
            maxpool_1_depth=self._maxpool_1_depth,
            maxpool_2_hight=self._maxpool_2_hight,
            maxpool_2_width=self._maxpool_2_width,
            maxpool_2_depth=self._maxpool_2_depth
        )

        return model_str

    def _reset_state(self, transfer_learning_model_params=None):
        """
        Reset state variables of the model.

        Input:
            None

        Returns:
            None
        """
        self._params = None
        self._lr = self._initial_lr if self._initial_lr != 0 else self._lr
        self._performance = copy.deepcopy(self._orig_performance)
        self._initialize_weights()
        self._validation_criteria_prime = 2**32 if self._early_stop_criteria == 'validation_loss' else -1
        self._early_stop_steps = 0
        self._prime_params = None
        self._prime_epoch = -1
        if (self._optimizer != 'sgd'):
            self._init_optimizer_cache()
        if transfer_learning_model_params is not None:
            self._do_transfer_learning(transfer_learning_model_params)

    def _initialize_model_architecture(self):
        """
        Initializes model architecture.

        Input:
            None

        Returns:
            None
        """
        flatten_dim = self._maxpool_2_depth * self._maxpool_2_hight * self._maxpool_2_width
        self._model_architecture = {
            "F1": (self._conv2d_1_filter_dim,
                   self._conv2d_1_filter_dim,
                   self._num_of_input_chanels,
                   self._f1_num_filter),
            "B1": (self._f1_num_filter,),
            "F2": (self._conv2d_2_filter_dim,
                   self._conv2d_2_filter_dim,
                   self._f1_num_filter,
                   self._f2_num_filter),
            "B2": (self._f2_num_filter,),
            "W3": (self._l1_fc_hidden_size,
                   flatten_dim),
            "B3": (self._l1_fc_hidden_size, 1),
            "W4": (self._l2_fc_hidden_size,
                   self._l1_fc_hidden_size),
            "B4": (self._l2_fc_hidden_size, 1),
            "W5": (self._num_of_classes,
                   self._l2_fc_hidden_size),
            "B5": (self._num_of_classes, 1)
        }

    def _normalize_z_score(self, X):
        """
        Z-Score data normalization.

        Input:
            X: Input dataset, X.shape == (data_dimension, n_examples).

        Returns:
            Z-Score normalized X.
            Z.shape == (data_dimension, n_examples).
        """
        Z = (X - lib.mean(X, axis=0)) / lib.std(X, axis=0).astype('float64')
        return Z

    def shuffle(self, X, Y):
        """
        Shuffles the dataset X and labels Y correspondingly.

        Input:
            X: Input dataset, X.shape == (n_examples, ...).
            Y: Input dataset, Y.shape == (num_of_classes, n_examples).

        Returns:
            A tuple (X, Y) of shuffled X and Y input.
            X.shape == (n_examples, ...).
            Y.shape == (num_of_classes, n_examples).
        """
        Y = Y.T
        assert X.shape[0] == Y.shape[0]

        p = np_rand_gen.permutation(X.shape[0])
        X, Y = X[p], Y[p].T

        return X, Y

    def _softmax(self, Z):
        """
        Implements Softmax function for the output activation layer.

        Input:
            Z: Output activation layer.

        Returns:
            Softmax function of the input.
        """
        return lib.exp(Z) / sum(lib.exp(Z))

    def _sigmoid(self, Z):
        """
        Implements Sigmoid activation function.

        Input:
            Z: Activation input.

        Returns:
            ReLU function of the input.
        """
        return 1.0 / (1 + lib.exp(-Z))

    def _sigmoid_derivative(self, Z):
        """
        Implements the derivative of Sigmoid activation function.

        Input:
            Z: Activation input.

        Returns:
            Derivative of the ReLU function of the input.
        """
        return self._sigmoid(Z) * (1 - self._sigmoid(Z))

    def _relu(self, Z):
        """
        Implements ReLU (Rectified Linear Unit) activation function.

        Input:
            Z: Activation input.

        Returns:
            ReLU function of the input.
        """
        return lib.maximum(Z, 0)

    def _relu_derivative(self, Z):
        """
        Implements the derivative of ReLU (Rectified Linear Unit) activation function.

        Input:
            Z: Activation input.

        Returns:
            Derivative of the ReLU function of the input.
        """
        return Z > 0

    def _leaky_relu(self, Z, alpha=0.3):
        """
        Implements Leaky ReLU (Leaky Rectified Linear Unit) activation function.

        Input:
            Z: Activation input.
            alpha: Slope of function's negative values domain.

        Returns:
            Leaky ReLU function of the input.
        """
        return lib.where(Z > 0, Z, alpha * Z)

    def _leaky_relu_derivative(self, Z, alpha=0.3):
        """
        Implements the derivative of ReLU (Rectified Linear Unit) activation function.

        Input:
            Z: Activation input.
            alpha: Slope of function's negative values domain.

        Returns:
            Derivative of the Leaky ReLU function of the input.
        """
        return lib.where(Z > 0, 1, alpha)

    def _early_stop_helper(self, early_stop_criteria, epoch, update):
        """
        Implements early stop algorithm.

        Input:
            early_stop_criteria: The value of the early stop criteria.
            epoch: The current epoch.
            update: Decides if do/don't do the early stop algorithm update.

        Returns:
            Boolean indicator whether do/don't stop early.
        """
        do_stop = False

        if update is True:
            self._early_stop_steps = 0
            self._prime_params = copy.deepcopy(self._params)
            self._prime_epoch = epoch
            self._validation_criteria_prime = early_stop_criteria
        else:
            self._early_stop_steps += 1
            if self._early_stop_steps == self._early_stop_max_epochs:
                do_stop = True

        return do_stop

    def _early_stop_criteria_validation_loss_cb(self, epoch):
        """
        Implements early stop algorithm based on validation loss.

        Input:
            epoch: The current epoch.

        Returns:
            Boolean indicator whether do/don't stop early.
        """
        early_stop_criteria = early_stop_criteria = self._performance[self._early_stop_criteria][-1]
        update = early_stop_criteria < self._validation_criteria_prime
        do_stop = self._early_stop_helper(early_stop_criteria, epoch, update)
        return do_stop

    def _early_stop_criteria_validation_accuracy_cb(self, epoch):
        """
        Implements early stop algorithm based on validation accuracy.

        Input:
            epoch: The current epoch.

        Returns:
            Boolean indicator whether do/don't stop early.
        """
        early_stop_criteria = early_stop_criteria = self._performance[self._early_stop_criteria][-1]
        update = early_stop_criteria > self._validation_criteria_prime
        do_stop = self._early_stop_helper(early_stop_criteria, epoch, update)
        return do_stop

    def _initialize_weights(self):
        """
        Initiates weights with random normal distribution for the model.

        Input:
            None

        Returns:
            None
        """
        F1 = rand_gen.randn(*self._model_architecture['F1']) * self._init_weights_sigma + self._init_weights_mu
        B1 = lib.zeros(shape=(self._model_architecture['B1']))
        F2 = rand_gen.randn(*self._model_architecture['F2']) * self._init_weights_sigma + self._init_weights_mu
        B2 = lib.zeros(shape=(self._model_architecture['B2']))
        W3 = rand_gen.randn(*self._model_architecture['W3']) * self._init_weights_sigma + self._init_weights_mu
        B3 = lib.zeros(shape=(self._model_architecture['B3']))
        W4 = rand_gen.randn(*self._model_architecture['W4']) * self._init_weights_sigma + self._init_weights_mu
        B4 = lib.zeros(shape=(self._model_architecture['B4']))
        W5 = rand_gen.randn(*self._model_architecture['W5']) * self._init_weights_sigma + self._init_weights_mu
        B5 = lib.zeros(shape=(self._model_architecture['B5']))

        self._params = {"F1": F1, "B1": B1, "F2": F2, "B2": B2,
                        "W3": W3, "B3": B3, "W4": W4, "B4": B4, "W5": W5, "B5": B5}

    def _forward_propagate(self, X, dropout_p=0):
        """
        Implements forward propagation pass through the network.

        Input:
            X: The dataset input, X.shape == (n_examples, input_dim, input_dim, input_depth).
            dropout_p: Dropout probability, default: No dropout.

        Returns:
            Dictionary of forward pass result.
        """
        F1, B1 = self._params['F1'], self._params['B1']
        F2, B2 = self._params['F2'], self._params['B2']
        W3, B3 = self._params['W3'], self._params['B3']
        W4, B4 = self._params['W4'], self._params['B4']
        W5, B5 = self._params['W5'], self._params['B5']

        noise_p = (rand_gen.rand(*X.shape) < (1 - self._input_noise_p)) / (1 - self._input_noise_p)
        _input = X * noise_p

        conv2d_1 = self._conv2d_1.forward_convolve(_input, F1, B1)
        conv2d_1_A = self._activation_cb(conv2d_1)

        maxpool_1 = self._maxpool_1.forward_maxpool(conv2d_1_A)

        dropout1 = (rand_gen.rand(*maxpool_1.shape) < (1 - dropout_p)) / (1 - dropout_p)
        maxpool_1 *= dropout1

        conv2d_2 = self._conv2d_2.forward_convolve(maxpool_1, F2, B2)
        conv2d_2_A = self._activation_cb(conv2d_2)

        maxpool_2 = self._maxpool_2.forward_maxpool(conv2d_2_A)

        dropout2 = (rand_gen.rand(*maxpool_2.shape) < (1 - dropout_p)) / (1 - dropout_p)
        maxpool_2 *= dropout2

        flatten = lib.ravel(maxpool_2).reshape(maxpool_2.shape[0], -1)
        Z3 = W3.dot(flatten.T) + B3

        A3 = self._activation_cb(Z3)
        dropout3 = (rand_gen.rand(*Z3.shape) < (1 - dropout_p)) / (1 - dropout_p)
        A3 *= dropout3

        Z4 = W4.dot(A3) + B4
        A4 = self._activation_cb(Z4)
        dropout4 = (rand_gen.rand(*Z4.shape) < (1 - dropout_p)) / (1 - dropout_p)
        A4 *= dropout4

        Z5 = W5.dot(A4) + B5
        A5 = self._softmax(Z5)

        result = {'conv2d_1': conv2d_1, 'conv2d_1_A': conv2d_1_A,
                  'maxpool_1': maxpool_1,
                  'conv2d_2': conv2d_2, 'conv2d_2_A': conv2d_2_A,
                  'maxpool_2': maxpool_2,
                  'Z3': Z3, 'A3': A3, 'Z4': Z4, 'A4': A4, 'Z5': Z5, 'A5': A5,
                  'dropout1': dropout1, 'dropout2': dropout2, 'dropout3': dropout3, 'dropout4': dropout4}

        return result

    def _backward_propagate(self, X, Y, forward_result):
        """
        Implements backward propagation pass through the network.

        Input:
            X: The dataset input, X.shape == (n_examples, input_dim, input_dim, input_depth).
            Y: The dataset labels, Y.shape == (num_of_classes, n_examples).
            forward_result: Dictionary of forward pass result.

        Returns:
            Dictionary of the calculated gradients.
        """
        batch_size = X.shape[0]

        F1, B1 = self._params['F1'], self._params['B1']
        F2, B2 = self._params['F2'], self._params['B2']
        W3, B3 = self._params['W3'], self._params['B3']
        W4, B4 = self._params['W4'], self._params['B4']
        W5, B5 = self._params['W5'], self._params['B5']

        conv2d_1, conv2d_1_A = forward_result['conv2d_1'], forward_result['conv2d_1_A']
        conv2d_2, conv2d_2_A = forward_result['conv2d_2'], forward_result['conv2d_2_A']
        maxpool_1, maxpool_2 = forward_result['maxpool_1'], forward_result['maxpool_2']

        Z3, A3 = forward_result['Z3'], forward_result['A3']
        Z4, A4 = forward_result['Z4'], forward_result['A4']
        Z5, A5 = forward_result['Z5'], forward_result['A5']

        dropout1, dropout2 = forward_result['dropout1'], forward_result['dropout2']
        dropout3, dropout4 = forward_result['dropout3'], forward_result['dropout4']

        dZ5 = A5 - Y
        dW5 = (1. / batch_size) * (dZ5.dot(A4.T) + self._reg * W5)
        dB5 = (1. / batch_size) * lib.sum(dZ5, axis=1, keepdims=True)

        dZ4 = W5.T.dot(dZ5) * self._activation_derivative_cb(Z4)
        dZ4 *= dropout4
        dW4 = (1. / batch_size) * (dZ4.dot(self._activation_cb(Z3).T) + self._reg * W4)
        dB4 = (1. / batch_size) * lib.sum(dZ4, axis=1, keepdims=True)

        dZ3 = W4.T.dot(dZ4) * self._activation_derivative_cb(Z3)
        dZ3 *= dropout3
        flatten = lib.ravel(maxpool_2).reshape(maxpool_2.shape[0], -1)
        dW3 = (1. / batch_size) * (dZ3.dot(flatten) + self._reg * W3)
        dB3 = (1. / batch_size) * lib.sum(dZ3, axis=1, keepdims=True)

        dflatten = W3.T.dot(dZ3).T

        dmaxpool_2 = dflatten.reshape(maxpool_2.shape)
        dmaxpool_2 *= dropout2
        dconv2d_2 = self._maxpool_2.backward_maxpool(dmaxpool_2, conv2d_2_A) * self._activation_derivative_cb(conv2d_2)

        dmaxpool_1, dF2, dB2 = self._conv2d_2.backward_convolve(dconv2d_2, maxpool_1, F2, self._reg)
        dmaxpool_1 *= dropout1

        dconv2d_1 = self._maxpool_1.backward_maxpool(dmaxpool_1, conv2d_1_A) * self._activation_derivative_cb(conv2d_1)

        dinput, dF1, dB1 = self._conv2d_1.backward_convolve(dconv2d_1, X, F1, self._reg)

        gradients = {'dW5': dW5, 'dB5': dB5, 'dW4': dW4,
                     'dB4': dB4, 'dW3': dW3, 'dB3': dB3,
                     'dF2': dF2, 'dB2': dB2, 'dF1': dF1, 'dB1': dB1}

        return gradients

    def _init_optimizer_cache(self):
        """
        Initializes the caches used by the model optimizer.

        Input:
            None.

        Returns:
            None.
        """
        self._optimizer_cache.clear()
        for parameter, _ in self._params.items():
            if self._optimizer == 'adam':
                self._optimizer_cache['d' + parameter] = {
                    'm_t': lib.zeros_like(self._params[parameter]),
                    'v_t': lib.zeros_like(self._params[parameter])
                }
            else:
                self._optimizer_cache['d' + parameter] = lib.zeros_like(self._params[parameter])

    def _sgd(self, gradients):
        """
        Implements optimizer: Mini-Batch Gradient Descent.

        Input:
            gradients: Dictionary of last step calculated gradients.

        Returns:
            None
        """
        for parameter, _ in self._params.items():
            self._params[parameter] -= self._lr * gradients['d' + parameter]

    def _momentum(self, gradients):
        """
        Implements optimizer: Mini-Batch Gradient Descent with momentum.

        Input:
            gradients: Dictionary of last step calculated gradients.

        Returns:
            None
        """
        for parameter, _ in self._params.items():
            self._optimizer_cache['d' + parameter] = (
                self._beta1 * self._optimizer_cache['d' + parameter] +
                self._lr * gradients['d' + parameter]
            )
            self._params[parameter] -= self._optimizer_cache['d' + parameter]

    def _adagrad(self, gradients):
        """
        Implements Mini-Batch Adagrad optimizer.

        Input:
            gradients: Dictionary of last step calculated gradients.

        Returns:
            None
        """
        for parameter, _ in self._params.items():
            self._optimizer_cache['d' + parameter] += gradients['d' + parameter]**2
            gradients['d' + parameter] = (
                gradients['d' + parameter] /
                (lib.sqrt(self._optimizer_cache['d' + parameter]) + self._eps)
            )
            self._params[parameter] -= self._lr * gradients['d' + parameter]

    def _rmsprop(self, gradients):
        """
        Implements Mini-Batch RMSProp optimizer.

        Input:
            gradients: Dictionary of last step calculated gradients.

        Returns:
            None
        """

        for parameter, _ in self._params.items():
            self._optimizer_cache['d' + parameter] = (
                self._beta1 * self._optimizer_cache['d' + parameter] +
                (1 - self._beta1) * lib.square(gradients['d' + parameter])
            )
            gradients['d' + parameter] = (
                gradients['d' + parameter] /
                (lib.sqrt(self._optimizer_cache['d' + parameter]) + self._eps)
            )
            self._params[parameter] -= self._lr * gradients['d' + parameter]

    def _adam(self, gradients):
        """
        Implements Mini-Batch Adam optimizer.

        Input:
            gradients: Dictionary of last step calculated gradients.

        Returns:
            None
        """
        for parameter, _ in self._params.items():
            m_t_m_1 = self._optimizer_cache['d' + parameter]['m_t']
            v_t_m_1 = self._optimizer_cache['d' + parameter]['v_t']
            m_t = self._beta1 * m_t_m_1 + (1 - self._beta1) * gradients['d' + parameter]
            v_t = self._beta2 * v_t_m_1 + (1 - self._beta2) * lib.square(gradients['d' + parameter])
            m_t_h = m_t / float(1 - lib.power(self._beta1, self._num_of_updates))
            v_t_h = v_t / float(1 - lib.power(self._beta2, self._num_of_updates))

            gradients['d' + parameter] = m_t_h / (lib.sqrt(v_t_h) + self._eps)
            self._params[parameter] -= self._lr * gradients['d' + parameter]

            self._optimizer_cache['d' + parameter]['v_t'] = v_t
            self._optimizer_cache['d' + parameter]['m_t'] = m_t

    def _lr_step_decay(self, epoch, log_change=False):
        """
        Updates learning rate, using learning rate step decay.

        The minimum learning rate will be bounded by 0.0001.

        Input:
            epoch: Current epoch of the algorithm.
            log_change: Print log in case the learning rate changes..

        Returns:
            None
        """
        previous_lr = self._lr
        self._lr = self._initial_lr * lib.power(1 - self._lr_decay_drop_ratio, lib.floor(
            (1 + epoch) / float(self._lr_decay_epochs_drop)))
        self._lr = lib.maximum(self._lr, 0.0001)
        if previous_lr != self._lr:
            log_str = "Learning rate decay step: {previous_lr} --> {current_lr}".format(
                previous_lr=previous_lr, current_lr=self._lr
            )
            print(log_str)

    def _ce_loss(self, Y_hat, Y):
        """
        Implements multi class cross entropy loss.

        Input:
            Y_hat: The predicated labels of the model, Y_hat.shape == (num_of_classes, batch_size).
            Y: The traning dataset labels, Y.shape == (num_of_classes, batch_size).

        Returns:
            Calculated loss.
        """
        m = Y.shape[1]

        # Compute multiclass cross entropy loss:
        matrix_result = Y * lib.log(Y_hat + lib.finfo(float).eps)
        cross_entropy_loss = float((-1. / m) * lib.sum(matrix_result))

        # Compute L2 regularization loss:
        l2_regularization_loss = (self._reg / (2 * m)) * (
            lib.sum(lib.square(self._params['F1'])) +
            lib.sum(lib.square(self._params['F2'])) +
            lib.sum(lib.square(self._params['W3'])) +
            lib.sum(lib.square(self._params['W4'])) +
            lib.sum(lib.square(self._params['W5'])))

        # Compute total loss:
        loss = float(cross_entropy_loss + l2_regularization_loss)

        return loss

    def plot(self, performance):
        """
        Plot the accuracy and loss for the performance model.

        Input:
            performance: Dictionary of model performance.

        Return:
            None
        """
        # Summarize history for accuracy:
        plt.plot(performance['train_accuracy'])
        plt.plot(performance['validation_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        print("")

        # Summarize history for loss:
        plt.plot(performance['train_loss'])
        plt.plot(performance['validation_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def _do_augmentation(self, X):
        """
        Do data augmentation.

        This method does data augmentation on self._data_augmentation_ratio of the data.
        It will do uniformly random type of augmentation from self._data_augmentors.

        Input:
            X: The dataset input, X.shape == (n_examples, input_dim, input_dim, input_depth).

        Returns:
            The X tensor with part of the data augmented, based on the description.

            X.shape == (n_examples, input_dim, input_dim, input_depth).
        """
        n_examples = X.shape[0]
        num_of_augmentations = int(n_examples * self._data_augmentation_ratio)

        if num_of_augmentations > n_examples:
            print("ERROR: Invalid data augmentation percentage, skipping data augmentation.")
            return

        for example in range(num_of_augmentations):
            random_augmentor = int(rand_gen.uniform(low=0, high=len(self._data_augmentors)))
            X[example] = self._data_augmentors[random_augmentor].augment(X[example])

        return X

    def _evaluate_single_epoch(self, train_bs_evaluation, validation_bs_evaluation,
                               num_of_data_splits, X_train, Y_train, X_validation, Y_validation):
        """
        Evaluates single epoch performance.

        The method evaluates the performance and saves it for later use in self._performance dict.

        Input:
            train_bs_evaluation: The batch size of the train to evaluate in a single iteration.
            validation_bs_evaluation: The batch size of the validation to evaluate in a single iteration.
            num_of_data_splits: Number of evaluation needed over the dataset.
            X_train: Training data.
            Y_train: Training lables.
            X_validation: Validation data.
            Y_validation: Validation lables.

        Returns:
            None
        """
        train_predictions_lst = list()
        train_accuracy_lst = list()
        train_loss_lst = list()
        validation_predictions_lst = list()
        validation_accuracy_lst = list()
        validation_loss_lst = list()
        train_low = 0
        train_high = train_bs_evaluation
        validation_low = 0
        validation_high = validation_bs_evaluation
        for _ in range(1, num_of_data_splits + 1):
            train_predictions_lst.append(self.predict(X_train[train_low:train_high, ...]))
            train_accuracy_lst.append(self.evaluate(train_predictions_lst[-1], Y_train[..., train_low:train_high]))
            train_loss_lst.append(self._ce_loss(train_predictions_lst[-1], Y_train[..., train_low:train_high]))
            validation_predictions_lst.append(self.predict(X_validation[validation_low:validation_high, ...]))
            validation_accuracy_lst.append(self.evaluate(
                validation_predictions_lst[-1], Y_validation[..., validation_low:validation_high]))
            validation_loss_lst.append(self._ce_loss(
                validation_predictions_lst[-1], Y_validation[..., validation_low:validation_high]))
            train_low += train_bs_evaluation
            train_high += train_bs_evaluation
            validation_low += validation_bs_evaluation
            validation_high += validation_bs_evaluation

        # Save performance results:
        self._performance['train_accuracy'].append(float(lib.average(train_accuracy_lst)))
        self._performance['validation_accuracy'].append(float(lib.average(validation_accuracy_lst)))
        self._performance['train_loss'].append(float(lib.average(train_loss_lst)))
        self._performance['validation_loss'].append(float(lib.average(validation_loss_lst)))

    def _log_single_epoch_performance(self, epoch, ts_before_epoch, ts_after_epoch, verbose=False):
        """
        """
        if verbose is False:
            return

        performance_str_format = (
            "Epoch: {epoch} | Train - Accuracy: {train_accuracy: <8} % ; Loss: {train_loss: <8} "
            "| Validation - Accuracy: {validation_accuracy: <8} % ; Loss: {validation_loss: <8} "
            "| Duration: {duration}"
        )
        performance_str = performance_str_format.format(
            epoch=epoch,
            train_accuracy=round(self._performance['train_accuracy'][-1], 6),
            validation_accuracy=round(self._performance['validation_accuracy'][-1], 6),
            train_loss=round(self._performance['train_loss'][-1], 6),
            validation_loss=round(self._performance['validation_loss'][-1], 6),
            duration=format_time(ts_before_epoch, ts_after_epoch)
        )
        print(performance_str)

    def _free_mempool(self):
        """
        Frees memory pools used by CuPy.

        In some scenarios, due to memory limitations of the GPU.
        It is better to free the memory pools, in order not to be
        in a state of out of memory available on the GPU.

        Returns:
            None
        """
        if lib.__name__ == 'cupy' and self._num_of_updates % G_MEMPOOL_FREE_INTERVAL == 0:
            G_MEMPOOL.free_all_blocks()
            G_PINNED_MEMPOOL.free_all_blocks()

    def _do_transfer_learning(self, model_params):
        """
        Does transfer learning using the supplied model parameters.

        The transfer learning takes all layers 1-4, besided the last fully connected
        layer and uses it as starting point parameters for the model.

        Input:
            model_params: The parameters to use for the transfer learninig.

        Return:
            None.
        """
        self._transfer_learning_done = True
        self._params['F1'] = model_params['F1']
        self._params['B1'] = model_params['B1']
        self._params['F2'] = model_params['F2']
        self._params['B2'] = model_params['B2']
        self._params['W3'] = model_params['W3']
        self._params['B3'] = model_params['B3']
        self._params['W4'] = model_params['W4']
        self._params['B4'] = model_params['B4']

    def fit(self, X_train, Y_train, X_validation, Y_validation, num_of_data_splits=1,
            verbose=False, transfer_learning_model_params=None):
        """
        Fit the model using Mini-Batch Gradient Descent algorithm.

        Input:
            X_train: Training dataset, X_train.shape == (n_examples, input_dim, input_dim, input_depth).
            Y_train: Training labels,  Y_train.shape == (num_of_classes, n_examples).
            X_validation: Validation dataset, X_validation.shape == (m_examples, input_dim, input_dim, input_depth).
            Y_validation: Validation labels,  Y_validation.shape == (num_of_classes, m_examples).
            verbose: Prints model performance on each epoch, default: False.
            transfer_learning_model_params: Do transfer learning before model fit and use the supplied parameters, default: None.

        Returns:
            The performance history training dictionary.
        """
        # Initialize variables, reset the state in case fit called multiple times.
        self._reset_state(transfer_learning_model_params)
        n_examples = X_train.shape[0]
        n_validation_examples = X_validation.shape[0]
        batch_iterations = int(lib.ceil(n_examples / float(self._batch_size)))
        train_bs_evaluation = int(lib.ceil(n_examples / num_of_data_splits))
        validation_bs_evaluation = int(lib.ceil(n_validation_examples / num_of_data_splits))
        X_train, Y_train = self.shuffle(X_train, Y_train)
        X_validation, Y_validation = self.shuffle(X_validation, Y_validation)

        if self._input_z_score_normalization is True:
            X_train = self._normalize_z_score(X_train)
            X_validation = self._normalize_z_score(X_validation)

        if verbose is True:
            ts_fit_start = time.time()
            time_now = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime())
            print(f"{'Training started at: {time_now}'}\n{'':-<45}".format(time_now=time_now))

        if self._data_augmentation_ratio != 0:
            X_train_original = X_train.copy()
            Y_train_original = Y_train.copy()

        # Epochs training loop, it will be stopped based on range or early stop algorithm:
        for epoch in range(1, self._epochs + 1):
            ts_before_epoch = time.time()

            if self._data_augmentation_ratio != 0:
                X_train = X_train_original.copy()
                Y_train = Y_train_original.copy()

            X_train, Y_train = self.shuffle(X_train, Y_train)

            # Do data augmentation:
            if self._data_augmentation_ratio != 0:
                X_train = self._do_augmentation(X_train)
                X_train, Y_train = self.shuffle(X_train, Y_train)

            # Mini-Batch Gradient Descent loop:
            low = 0
            high = self._batch_size
            for batch in range(1, batch_iterations + 1):
                self._free_mempool()
                # Get the current mini batch, wrap around on last iteration.
                if batch != batch_iterations:
                    X_batch = X_train[low:high, :, :, :]
                    Y_batch = Y_train[:, low:high]
                else:
                    reminder = high - n_examples
                    X_batch = lib.concatenate((X_train[low:n_examples, :, :, :], X_train[0:reminder, :, :, :]), axis=0)
                    Y_batch = lib.concatenate((Y_train[:, low:n_examples], Y_train[:, 0:reminder]), axis=1)

                low += self._batch_size
                high += self._batch_size

              # Do one step of Gradient Descent:
                self._num_of_updates += 1
                forward_result = self._forward_propagate(X_batch, self._dropout_p)
                gradients = self._backward_propagate(X_batch, Y_batch, forward_result)
                self._optimizer_cb(gradients)

            self._evaluate_single_epoch(train_bs_evaluation, validation_bs_evaluation, num_of_data_splits,
                                        X_train, Y_train, X_validation, Y_validation)
            ts_after_epoch = time.time()
            self._log_single_epoch_performance(epoch, ts_before_epoch, ts_after_epoch, verbose)
            if self._early_stop_cb(epoch) is True:
                break

            if self._initial_lr != 0:
                self._lr_step_decay(epoch, verbose)

        # Calculate total fit duration:
        if verbose is True:
            ts_fit_end = time.time()
            time_now = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime())
            print(f"{'':-<45}\n{'Training finished at: {time_now}'}\n{'':-<45}".format(time_now=time_now))
            print(f"{'Training duration: {duration}'}\n{'':-<45}".format(duration=format_time(ts_fit_start, ts_fit_end)))

        self._params = copy.deepcopy(self._prime_params)
        return self._performance

    def predict(self, X, output_file=None, as_labels=False):
        """
        Predict the input data X.

        Optionally save the predictions to the output_file path.

        Input:
            X: Input dataset, X.shape == (n_examples, input_dim, input_dim, input_depth).
            output_file: Optional full path to an output file where to save the predictions in, default: None.
            as_labels: Return the prediction as class labels array.

        Returns:
            The predictions of the model.
            On as_labels=False: prediction.shape == (num_of_classes, n_examples).
            On as_labels=True: prediction.shape == (n_examples, 1).
        """
        if self._input_z_score_normalization is True:
            X = self._normalize_z_score(X)

        forward_result = self._forward_propagate(X)
        prediction = forward_result['A5']

        if as_labels is True:
            prediction = lib.argmax(prediction, axis=0) + 1

        if output_file is not None:
            if lib.__name__ == 'cupy':
                prediction = lib.asnumpy(prediction)
            prediction = prediction.astype(int)
            numpy.savetxt(fname=output_file, X=prediction, fmt='%d')

        return prediction

    def evaluate(self, predictions, Y):
        """
        Evaluates the predictions accuracy.

        Input:
            predictions: Model predictions, predictions.shape == (num_of_classes, batch_size).
            Y: labels of the corresponding data.

        Returns:
            Accuracy of the prediction.
        """
        Y_hat = lib.argmax(predictions, axis=0) + 1
        labels = lib.argmax(Y, axis=0) + 1
        accuracy_percentage = (lib.sum(lib.array([labels == Y_hat])) / float(labels.size)) * 100
        accuracy = float(accuracy_percentage)

        return accuracy

    @staticmethod
    def evaluate_from_file(predictions_file_path=None, labels_file_path=None):
        """
        Evaluates the predictions accuracy from files.

        Input:
            predictions_file_path: Full path to the predictions file.
            labels_file_path: Full path to the labels file.

        Returns:
            Accuracy of the prediction.
        """
        try:
            predictions = pd.read_csv(predictions_file_path, header=None)
            predictions = lib.array(predictions.to_numpy(dtype='int8'))
        except FileNotFoundError:
            print("ERROR: Failed to find predictions file: ",
                  predictions_file_path)
            return None

        try:
            labels = pd.read_csv(labels_file_path, header=None)
            labels = lib.array(labels.to_numpy(dtype='int8'))
        except FileNotFoundError:
            print("ERROR: Failed to find labels file: ", labels_file_path)
            return None

        if labels.size != predictions.size:
            print("ERROR: The number of predictions and labels doesn't match")
            return None

        accuracy_percentage = (
            lib.sum(lib.array([labels == predictions])) / float(labels.size)) * 100
        accuracy = float(accuracy_percentage)

        return accuracy

    @staticmethod
    def cross_validate(model, X, Y, k_fold=10, verbose=False, plot=False, transfer_learning_model_params=None):
        """
        K Fold cross validation of the model.

        Resets global rand_gen, np_rand_gen Random State generators.

        Input:
            model: The model to validate.
            X: Training input dataset, X.shape == (n_examples, input_dim, input_dim, input_depth).
            Y: Training labels dataset, Y.shape == (num_of_classes, n_examples).
            k_fold: Number of folds in KFold algorithm.
            verbose: Print detailed model performance for each fold.
            plot: Plot model performance for each fold.
            transfer_learning_model_params: Do transfer learning before model fit and use the supplied parameters, default: None.

        Returns:
            Averaged accuracy based on K Fold algorithm.

        """
        global rand_gen, np_rand_gen

        n_examples = X.shape[0]
        n_examples_validation = int(n_examples / k_fold)
        n_examples_train = n_examples - n_examples_validation
        fold_stats = []
        X, Y = model.shuffle(X, Y)
        num_of_data_splits = 1

        for fold in range(k_fold):
            rand_gen = lib.random.RandomState(seed=random_generator_seed)
            np_rand_gen = numpy.random.RandomState(seed=random_generator_seed)

            _model = MultiClassConvolutionalNeuralNetwork(
                data_dim=model._data_dim,
                activation_func=model._activation_func,
                optimizer=model._optimizer,
                f1_num_filter=model._f1_num_filter,
                f2_num_filter=model._f2_num_filter,
                l1_fc_hidden_size=model._l1_fc_hidden_size,
                l2_fc_hidden_size=model._l2_fc_hidden_size,
                num_of_classes=model._num_of_classes,
                lr=model._lr,
                initial_lr=model._initial_lr,
                epochs=model._epochs,
                batch_size=model._batch_size,
                reg=model._reg,
                input_noise_p=model._input_noise_p,
                dropout_p=model._dropout_p,
                early_stop_criteria=model._early_stop_criteria,
                early_stop_max_epochs=model._early_stop_max_epochs,
                input_z_score_normalization=model._input_z_score_normalization,
                init_weights_mu=model._init_weights_mu,
                init_weights_sigma=model._init_weights_sigma,
                data_augmentation_ratio=model._data_augmentation_ratio,
                lr_decay_drop_ratio=model._lr_decay_drop_ratio,
                lr_decay_epochs_drop=model._lr_decay_epochs_drop
            )

            sliding_window_left = int(fold * n_examples_validation)
            sliding_window_right = int((fold + 1) * n_examples_validation)

            X_validation = X[sliding_window_left:sliding_window_right, ...]
            Y_validation = Y[:, sliding_window_left:sliding_window_right]

            X_train_left = X[0:sliding_window_left, ...]
            X_train_right = X[sliding_window_right:n_examples, ...]
            Y_train_left = Y[:, 0:sliding_window_left]
            Y_train_right = Y[:, sliding_window_right:n_examples]

            X_train = lib.concatenate((X_train_left, X_train_right), axis=0)
            Y_train = lib.concatenate((Y_train_left, Y_train_right), axis=1)

            performance = _model.fit(X_train, Y_train, X_validation, Y_validation,
                                     num_of_data_splits, verbose, transfer_learning_model_params)
            predictions = _model.predict(X_validation)
            accuracy = _model.evaluate(predictions, Y_validation)
            fold_stats.append(accuracy)

            if plot is True:
                print("")
                _model.plot(performance)
                print("")

            print("\nFold ({fold}) test accuracy: {accuracy} %\n".format(
                fold=fold + 1, accuracy=round(accuracy, 6)))

        total_avg_accuracy = float(lib.average(fold_stats))

        return total_avg_accuracy


def parse_cli(static_args=None):
    """
    Parse command line options.

    Input:
        static_args: Pass list of args for testing/environments without CLI.

    Returns:
        Parsed arguments.
    """
    description = ("Multi Class Neural Network Classification App. "
                   "NVIDIA GPU acceleration supported when CuPy module installed.")
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-f', '--fit',
                        action='store_true',
                        default=False,
                        help=("Run model training with the supplied parameters, default: False.\n"
                              "This option dumps the model as a Pickle file by default named: 'model_dump.bin'"
                              " to the current working directory. This can be controlled by '-d','--dump-path' option."))
    parser.add_argument('-p', '--predict',
                        action='store_true',
                        default=False,
                        help=("Run prediction on the trained model, use '-x', '--test-file-path'"
                              " to choose the test file, default: False."))
    parser.add_argument('-kf', '--k-fold',
                        dest='k_fold',
                        type=int,
                        help="Run KFold cross validation using the supplied K, default: Not set.")
    parser.add_argument('-d', '--dump-path',
                        dest='dump_path',
                        default="model_dump.bin",
                        help=("Dump the model into the supplied full "
                              "path as Pickle serialized file, default: model_dump.bin."))
    parser.add_argument('-l', '--load-path',
                        dest='load_path',
                        help=("Load the model from the supplied full path Pickle serialized file. "
                              "Note: The module used should be compitable with CuPy/NumPy on the current runtime. "
                              "See the note at the top of application code."))
    parser.add_argument('-ltr', '--load-path-transfer-learning',
                        dest='load_path_transfer_learning',
                        help=("Load a trained model for transfer learning from the supplied full path Pickle serialized file. "
                              "Note: a) The module used should be compitable with CuPy/NumPy on the current runtime. "
                              "See the note at the top of application code. "
                              "b) The model should be with the same amount of categories as the model to train."))
    parser.add_argument('-q', '--quiet',
                        action='store_true',
                        default=False,
                        help="Don't print detailed information during the run, default: False.")
    parser.add_argument('-z', '--plot',
                        action='store_true',
                        default=False,
                        help="Plot accuracy and loss graphs, default: False.")
    parser.add_argument('-k', '--train-file-path',
                        dest='train_file_path',
                        default='train.csv',
                        help=("Full path to a CSV file of the training dataset, default: train.csv.\n"
                              "File format: Label, feature_1, feature_2, ... , feature_n"))
    parser.add_argument('-j', '--validation-file-path',
                        dest='validation_file_path',
                        default='validate.csv',
                        help=("Full path to a CSV file of the validation dataset, default: validate.csv.\n"
                              "File format: Label, feature_1, feature_2, ... , feature_n"))
    parser.add_argument('-x', '--test-file-path',
                        dest='test_file_path',
                        default='test.csv',
                        help=("Full path to a CSV file of the test dataset, default: test.csv.\n"
                              "File format: ?, feature_1, feature_2, ... , feature_n"))
    parser.add_argument('-y', '--prediction-file-path',
                        dest='prediction_file_path',
                        default='prediction.txt',
                        help="Full path to the location where to save the prediction file, default: prediction.txt.\n")
    parser.add_argument('-t', '--labels-file-path',
                        dest='labels_file_path',
                        default='labels.txt',
                        help=(
                            "Full path to the location of the labels file for computing accuracy, default: labels.txt.\n"
                            "File format: Label in each row"))
    parser.add_argument('-i', '--compute-accuracy',
                        action='store_true',
                        default=False,
                        help=(
                            "Compute accuracy of given predictions and labels files, use with "
                            "'-y', '--prediction-file-path' and '-t', '--labels-file-path' options, default: False.\n"))
    parser.add_argument('-a', '--activation-func',
                        dest='activation_func',
                        choices=ACTIVATION_FUNCTIONS,
                        default='leaky_relu',
                        help="Activation function name, default: leaky_relu.")
    parser.add_argument('-op', '--optimizer',
                        dest='optimizer',
                        choices=OPTIMIZERS,
                        default='momentum',
                        help="Optimizer name, default: momentum.")
    parser.add_argument('-f1', '--f1-num-filter',
                        dest='f1_num_filter',
                        type=int, default=32,
                        help="Number of filters in the first convolutional layer, default: 32.")
    parser.add_argument('-f2', '--f2-num-filter',
                        dest='f2_num_filter',
                        type=int, default=64,
                        help="Number of filters in the second convolutional layer, default: 64.")
    parser.add_argument('-l1', '--l1-fc-hidden-size',
                        dest='l1_fc_hidden_size',
                        type=int, default=512,
                        help="Number of neurons in the first hidden layer of the FC part of the network, default: 512.")
    parser.add_argument('-l2', '--l2-fc-hidden-size',
                        dest='l2_fc_hidden_size',
                        type=int, default=128,
                        help="Number of neurons in the second hidden layer of the FC part of the network, default: 128.")
    parser.add_argument('-c', '--num-of-classes',
                        dest='num_of_classes',
                        type=int, default=10,
                        help="Number of categorial class of the data, default: 10.")
    parser.add_argument('-g', '--lr',
                        dest='lr',
                        type=float,
                        default=0.001,
                        help="Learning rate for the optimization algorithm, default: 0.001.")
    parser.add_argument('-il', '--initial-lr',
                        dest='initial_lr',
                        type=float,
                        default=0,
                        help=("Initial learning rate for the learning rate decay algorithm, default: 0."
                              " This feature has higher priority than the static lr feature."))
    parser.add_argument('-ldr', '--lr-decay-ratio-drop',
                        dest='lr_decay_drop_ratio',
                        type=float,
                        default=0.15,
                        help=("The ratio by which the learning "
                              "rate decay algorithm drops in each step, default: 0.15."))
    parser.add_argument('-lde', '--lr-decay-epochs-drop',
                        dest='lr_decay_epochs_drop',
                        type=int,
                        default=15,
                        help=("The amount of epochs between each learning rate decay algorithm step, default: 15."))
    parser.add_argument('-da', '--data-augmentation-ratio',
                        dest='data_augmentation_ratio',
                        type=float,
                        default=0.3,
                        help=("The ratio of the data that will be augmented, default: 0.3."))
    parser.add_argument('-e', '--epochs',
                        dest='epochs',
                        type=int,
                        default=1000,
                        help="Max number of epochs to do, the algorithm might stop before, due to early stop, default: 1000.")
    parser.add_argument('-b', '--batch-size',
                        dest='batch_size',
                        type=int,
                        default=50,
                        help="Batch size to use for the Mini-Batch Gradient Descent, default: 50.")
    parser.add_argument('-r', '--reg',
                        dest='reg',
                        type=float,
                        default=0.15,
                        help="Regularization factor, default: 0.15.")
    parser.add_argument('-n', '--input-noise-p',
                        dest='input_noise_p',
                        type=float,
                        default=0,
                        help="The probability of non-active input features, default: 0.")
    parser.add_argument('-u', '--dropout-p',
                        dest='dropout_p',
                        type=float,
                        default=0.2,
                        help="Dropout probability of non-active neurons, default: 0.2.")
    parser.add_argument('-s', '--early-stop-max-epochs',
                        dest='early_stop_max_epochs',
                        type=int,
                        default=25,
                        help=("Maximum number of epochs without substantial improvement in model "
                              "accuracy before stopping the training, default: 25."))
    parser.add_argument('-v', '--early-stop-criteria',
                        dest='early_stop_criteria',
                        choices=EARLY_STOP_CRITERIONS,
                        default='validation_loss',
                        help=("The criteria for the early stop algorithm"
                              ", default: 'validation_loss'."))
    parser.add_argument('-o', '--input-z-score-normalization',
                        action='store_true',
                        default=False,
                        help="Controls whether Z-Score Normalization on the input dataset on or off, default: Off.")
    parser.add_argument('-m', '--init-weights-mu',
                        dest='init_weights_mu',
                        type=float,
                        default=0,
                        help="Expectation of the normal distribution for model weights initialization, default: 0.")
    parser.add_argument('-w', '--init-weights-sigma',
                        dest='init_weights_sigma',
                        type=float,
                        default=0.1,
                        help="Standard deviation of the normal distribution for model weights initialization, default: 0.1.")

    if static_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(static_args)

    return args


def main(static_args=None):
    """
    Main of the program.

    Input:
        static_args: Pass list of args for testing/environments without CLI.

    Returns:
        Exit code.
    """
    header = "Using {lib} as the acceleration library for mathematical operations".format(
        lib=lib.__name__)
    print(f"{'':-^135}\n{' {header} ':-^78}\n{'':-^135}".format(header=header))

    args = parse_cli(static_args)
    model = None

    if args.fit is True:
        print("Loading the dataset...")

        train_X, train_Y = MultiClassConvolutionalNeuralNetwork.load_dataset(args.train_file_path)
        validation_X, validation_Y = MultiClassConvolutionalNeuralNetwork.load_dataset(args.validation_file_path)
        num_of_data_splits = 1

        if train_X is None or validation_X is None:
            return 1

        model = MultiClassConvolutionalNeuralNetwork(
            data_dim=train_X.shape[0],
            activation_func=args.activation_func,
            optimizer=args.optimizer,
            f1_num_filter=args.f1_num_filter,
            f2_num_filter=args.f2_num_filter,
            l1_fc_hidden_size=args.l1_fc_hidden_size,
            l2_fc_hidden_size=args.l2_fc_hidden_size,
            num_of_classes=args.num_of_classes,
            lr=args.lr,
            initial_lr=args.initial_lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            reg=args.reg,
            input_noise_p=args.input_noise_p,
            dropout_p=args.dropout_p,
            early_stop_criteria=args.early_stop_criteria,
            early_stop_max_epochs=args.early_stop_max_epochs,
            input_z_score_normalization=args.input_z_score_normalization,
            init_weights_mu=args.init_weights_mu,
            init_weights_sigma=args.init_weights_sigma,
            data_augmentation_ratio=args.data_augmentation_ratio,
            lr_decay_drop_ratio=args.lr_decay_drop_ratio,
            lr_decay_epochs_drop=args.lr_decay_epochs_drop
        )

        train_X = model.transform_to_cnn_input(train_X)
        validation_X = model.transform_to_cnn_input(validation_X)

        if args.quiet is False:
            print(f"{'':-^135}\n{' Data shapes ':-^135}\n{'':-^135}")
            print("train_X shape: ", train_X.shape)
            print("train_Y shape: ", train_Y.shape)
            print("validation_X shape: ", validation_X.shape)
            print("validation_Y shape: ", validation_Y.shape)

        if args.quiet is False:
            print(model)
            print(f"{'':-^135}\n{' Model Fitting ':-^135}\n{'':-^135}")
        else:
            print("Fitting the model, please wait...")

        transfer_learning_model_params = None
        if args.load_path_transfer_learning is not None:
            transfer_learning_model_params = MultiClassConvolutionalNeuralNetwork.load_model(
                args.load_path_transfer_learning)
            if transfer_learning_model_params is None:
                return 1
            print("Loaded transfer learning model file: {file}\n".format(file=args.load_path_transfer_learning))

        performance = model.fit(train_X, train_Y, validation_X, validation_Y,
                                num_of_data_splits, not args.quiet, transfer_learning_model_params)
        print(f"{'':-^135}")

        if args.plot is True:
            model.plot(performance)

        predictions = model.predict(validation_X)
        accuracy = model.evaluate(predictions, validation_Y)
        print("Total validation accuracy: {accuracy} [%]".format(accuracy=accuracy))

        numpy_dump_size, cupy_dump_size = MultiClassConvolutionalNeuralNetwork.dump_model(
            model, args.dump_path)
        if numpy_dump_size is None:
            return 1

        numpy_dump_size_MB = numpy_dump_size / 10**6
        numpy_dump_str = "Numpy model dump file: {file}_numpy\nDump model size: {size} [MB]".format(
            file=args.dump_path, size=numpy_dump_size_MB)
        print(numpy_dump_str)

        if lib.__name__ == 'cupy':
            cupy_dump_size_MB = cupy_dump_size / 10**6
            cupy_dump_str = "CuPY model dump file: {file}_cupy\nDump model size: {size} [MB]".format(
                file=args.dump_path, size=cupy_dump_size_MB)
            print(cupy_dump_str)

    elif args.load_path is not None:
        model = MultiClassConvolutionalNeuralNetwork()
        model._params = MultiClassConvolutionalNeuralNetwork.load_model(args.load_path)
        if model is None:
            return 1

        print("Loaded model file: {file}\n".format(file=args.load_path))
    elif args.compute_accuracy is True:
        print("Commputing accuracy...")
        accuracy = MultiClassConvolutionalNeuralNetwork.evaluate_from_file(
            predictions_file_path=args.prediction_file_path,
            labels_file_path=args.labels_file_path)
        if accuracy is None:
            return 1
        print("Computed accuracy: {accuracy} [%]".format(
            accuracy=accuracy))
    else:
        print("Please choose one of the follwing options: '-f','--fit'"
              " | '-l','--load-path' | '-i', '--compute-accuracy'")
        return 1

    if args.k_fold is not None:
        if args.fit is False:
            print("Loading the dataset...")
            train_X, train_Y = MultiClassConvolutionalNeuralNetwork.load_dataset(
                args.train_file_path)
            validation_X, validation_Y = MultiClassConvolutionalNeuralNetwork.load_dataset(
                args.validation_file_path)
            if train_X is None or validation_X is None:
                return 1

            train_X = model.transform_to_cnn_input(train_X)
            validation_X = model.transform_to_cnn_input(validation_X)

        print(f"{'':-^135}\n{' Model KFold Cross Validation ':-^135}\n{'':-^135}")
        X = lib.concatenate((train_X, validation_X), axis=0)
        Y = lib.concatenate((train_Y, validation_Y), axis=1)
        kfold_accuracy = MultiClassConvolutionalNeuralNetwork.cross_validate(
            model, X, Y, k_fold=args.k_fold, verbose=(not args.quiet), plot=args.plot)
        print("Total KFold cross validation accuracy: {accuracy} [%]".format(
            accuracy=kfold_accuracy))

    if args.predict is True:
        print("Predicting...")
        test_X, _ = MultiClassConvolutionalNeuralNetwork.load_dataset(
            dataset_path=args.test_file_path, without_labels=True)
        if test_X is None:
            return 1

        test_X = model.transform_to_cnn_input(test_X)
        predictions = model.predict(
            X=test_X, output_file=args.prediction_file_path, as_labels=True)
        print("Prediction is written to file: ", args.prediction_file_path)

    print("")

    return 0


if __name__ == '__main__':
    #############################
    ### Application CLI usage ###
    #############################
    # When using the application from the CLI, no need for static_args list.
    # In this case the arguments are taken from the CLI.
    static_args = None

    ###################################################
    ### Application from interactive platform usage ###
    ###################################################
    # When using the application from interactive platforms, e.g. Colab,
    # there is a need for static_args list as CLI parameters.

    # static_args = [
    #     '--fit',
    #     '--plot',
    #     '--dump-path',
    #     '/content/model_testing/trained_model_dump.bin',
    #     '--train-file-path',
    #     '/content/model_testing/train.csv',
    #     '--validation-file-path',
    #     '/content/model_testing/validate.csv',
    #     # '--load-path-transfer-learning',
    #     # '/content/GoogleDrive/MyDrive/Exercise_2/transfer_learning_models/cifar10_model_dump.bin_cupy_rmsprop_opt',
    #     # '--load-path',
    #     # '/content/model_testing/trained_model_dump.bin_cupy',
    #     # '--compute-accuracy',
    #     # '--predict',
    #     # '--prediction-file-path',
    #     # '/content/model_testing/accuracy_test/validation_prediction.txt',
    #     # '--labels-file-path',
    #     # '/content/model_testing/accuracy_test/validation_labels.txt',
    #     # '--test-file-path',
    #     # '/content/model_testing/validate.csv',
    # ]

    rc = main(static_args)
    if static_args is None:
        exit(rc)
