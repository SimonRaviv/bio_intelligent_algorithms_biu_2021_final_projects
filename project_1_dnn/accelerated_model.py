import os
import copy
import pickle
import argparse

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy
except ImportError:
    print("Failed to import one of Python external modules."
          "Make sure you have pandas, matplotlib and numpy installed !")
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

# Pseudo random seed:
random_generator_seed = 2021
rand_gen = lib.random.RandomState(seed=random_generator_seed)
np_rand_gen = numpy.random.RandomState(seed=random_generator_seed)


class MultiClassNeuralNetwork(object):
    """
    Multi Class Neural Network implementation.

    The class offers generic implementation for multi class categorial datasets.
    User can modify the parameters of the constructor to use its various features.

    This class is implemented using a GPU accelerated library following the same API as numpy.
    The library is CuPy, accelerates numpy API on a NVIDIA GPU.

    The model offers the following architecture:
        * Input layer, controllable by the data_dim parameter for the number of features.
        * Hidden layer 1, controllable by the l1_hidden_size parameter for the number of neurons in this layer.
        * Hidden layer 2, controllable by the l2_hidden_size parameter for the number of neurons in this layer.
        * Output layer, controllable by the num_of_classes parameter for the number of classes in the dataset.

    Optimizer:
        The class implements Mini-Batch Gradient Descent.

    Regularizes:
        The class offers 2 type of regularizes:
            * Input noise: The user can control the probability of non-active input features by input_noise_p parameter.
            * Dropout: The user can control the probability of the number of non-active neurons by the dropout_p parameter.
            * L2 regularization: The user can control the regularization factor by reg parameter.

    Features:
        * The class offers various activation functions: ['sigmoid', 'relu', 'leaky_relu'],
            user can choose by activation_func parameter.
        * The user can control static learning rate by lr parameter.
        * The class implements learning rate decay algorithm,
            user can choose the initial learning rate by initial_lr parameter.
        * The class implements early stop algorithm:
            * User can choose the max epochs by epochs parameter.
            * User can control the max number of epochs without improvement in accuracy before
                stopping by changing early_stop_max_epochs parameter.
        * The class implements Mini-Batch gradient descent, user can control the batch size by batch_size parameter.
        * The class implements initial weights by the Gaussian distribution,
            the user can control the expectation and standard deviation by init_weights_mu and init_weights_sigma parameters.
        * The class offers KFold cross validation, the user can run it by
            cross_validate function and control the number of folds by k_fold parameter of the function.
        * The class offers Z-Score Normalization on the input dataset.
    """

    def __init__(self,
                 data_dim=3072,
                 activation_func='relu',
                 l1_hidden_size=1500,
                 l2_hidden_size=1500,
                 num_of_classes=10,
                 lr=0,
                 initial_lr=0.1,
                 epochs=300,
                 batch_size=65,
                 reg=0.1,
                 input_noise_p=0,
                 dropout_p=0.2,
                 early_stop_max_epochs=18,
                 init_weights_mu=0,
                 init_weights_sigma=0.01,
                 input_z_score_normalization=False):
        """
        Model constructor.

        Input:
            data_dim: Number of features in the dataset.
            activation_func: Activation function name as a string, default: 'relu', supported functions: ['sigmoid', 'relu', 'leaky_relu']
            l1_hidden_size: Number of neurons in the first hidden layer, default: 1500.
            l2_hidden_size: Number of neurons in the second hidden layer, default: 1500.
            num_of_classes: Number of categorial class of the data, default: 10.
            lr: Learning rate for the optimization algorithm, default: 0.
            initial_lr: Initial learning rate for the optimization algorithm, default: 0.1.
                        This feature has higher priority than the static lr feature.
            epochs: Max number of epochs to do, the algorithm might stop before, due to early stop, default: 300.
            batch_size: Batch size to use for the Mini-Batch Gradient Descent, default: 65.
            reg: Regularization factor, default: 0.1.
            input_noise_p: The probability of non-active input features, default: 0.
            dropout_p: Dropout probability of non-active neurons, default: 0.2.
            early_stop_max_epochs: Maximum number of epochs without substantial improvement in model
                                   accuracy before stopping the training, default: 18.
            init_weights_mu: Expectation of the normal distribution for model weights initialization, default: 0.
            init_weights_sigma: Standard deviation of the normal distribution for model weights initialization, default: 0.01.
            input_z_score_normalization: Controls whether Z-Score Normalization on the input dataset on or off, default: Off.

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
        self._activation_func = activation_func
        activation_cb, activation_derivative_cb = self._activation_cb_map.get(
            activation_func, 'relu')
        self._activation_cb = activation_cb
        self._activation_derivative_cb = activation_derivative_cb
        self._l1_hidden_size = l1_hidden_size
        self._l2_hidden_size = l2_hidden_size
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
        self._early_stop_drop = 0.5
        self._early_stop_epochs_drop = 10
        self._input_z_score_normalization = input_z_score_normalization

    @staticmethod
    def one_hot_encoding(Y, num_of_classes=10):
        """
        One hot encoding of categorial classes.

        Input:
            Y: Vector of the data labels, Y.shape == (n_samples, 1).
            num_of_classes: Number of the categorail classes.

        Returns:
            One hot encoded transformation of the Y input vector.
            one_hot_encoded.shape == (num_of_classes, n_samples).
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
                X.shape == (data_dimension, n_samples).
                Y.shape == (num_of_classes, n_samples).
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
            Y = MultiClassNeuralNetwork.one_hot_encoding(Y).T

        return X, Y

    @staticmethod
    def dump_model(model, dump_file="model_dump.bin"):
        """
        Serializes the model using Pickle module.

        Input:
            dump_file: The full path to the dump file, including the name, default: './model_dump.bin'.

        Returns:
            On success: The size in bytes of the dump file.
            On failure: None.
        """
        dump_size = -1

        try:
            with open(dump_file, 'wb') as df:
                pickle.dump(model, df)
                dump_size = os.path.getsize(dump_file)
        except EnvironmentError:
            print("ERROR: Failed to write to file: ", dump_file)
            return None

        return dump_size

    @staticmethod
    def load_model(dump_file="model_dump.bin"):
        """
        Deserialize the model using Pickle module.

        Input:
            dump_file: The full path to the dump file, including the name, default: './model_dump.bin'.

        Returns:
            On success: Instance object of MultiClassNeuralNetwork class.
            On failure: None.
        """
        model = None

        try:
            with open(dump_file, 'rb') as df:
                model = pickle.load(df)
        except EnvironmentError:
            print("ERROR: Failed to read from the dump file: ", dump_file)
            return None

        return model

    def __str__(self):
        """
        Dumps the model to representative string.
        """
        model_str_format = (
            f"{'':-^135}\n{' Model Parameters ':-^135}\n{'':-^135}\n"
            "* model_architecture: {model_architecture}\n"
            "* data_dim: {data_dim}\n"
            "* activation_func: {activation_func}\n"
            "* l1_hidden_size: {l1_hidden_size}\n"
            "* l2_hidden_size: {l2_hidden_size}\n"
            "* num_of_classes: {num_of_classes}\n"
            "* initial_lr: {initial_lr}\n"
            "* lr: {lr}\n"
            "* epochs: {epochs}\n"
            "* batch_size: {batch_size}\n"
            "* reg: {reg}\n"
            "* input_noise_p: {input_noise_p}\n"
            "* dropout_p: {dropout_p}\n"
            "* early_stop_max_epochs: {early_stop_max_epochs}\n"
            "* init_weights_mu: {init_weights_mu}\n"
            "* init_weights_sigma: {init_weights_sigma}\n"
            "* early_stop_drop: {early_stop_drop}\n"
            "* early_stop_epochs_drop: {early_stop_epochs_drop}\n"
            "* input_z_score_normalization: {input_z_score_normalization}"
        )
        model_str = model_str_format.format(
            model_architecture=self._model_architecture,
            data_dim=self._data_dim,
            activation_func=self._activation_func,
            l1_hidden_size=self._l1_hidden_size,
            l2_hidden_size=self._l2_hidden_size,
            num_of_classes=self._num_of_classes,
            initial_lr=self._initial_lr,
            lr=self._lr,
            epochs=self._epochs,
            batch_size=self._batch_size,
            reg=self._reg,
            input_noise_p=self._input_noise_p,
            dropout_p=self._dropout_p,
            early_stop_max_epochs=self._early_stop_max_epochs,
            init_weights_mu=self._init_weights_mu,
            init_weights_sigma=self._init_weights_sigma,
            early_stop_drop=self._early_stop_drop,
            early_stop_epochs_drop=self._early_stop_epochs_drop,
            input_z_score_normalization=self._input_z_score_normalization
        )

        return model_str

    def _reset_state(self):
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

    def _initialize_model_architecture(self):
        """
        Initializes model architecture.

        Input:
            None

        Returns:
            None
        """
        self._model_architecture = {
            "W1": (self._l1_hidden_size, self._data_dim), "B1": (self._l1_hidden_size, 1),
            "W2": (self._l2_hidden_size, self._l1_hidden_size), "B2": (self._l2_hidden_size, 1),
            "W3": (self._num_of_classes, self._l2_hidden_size), "B3": (self._num_of_classes, 1)
        }

    def _normalize_z_score(self, X):
        """
        Z-Score data normalization.

        Input:
            X: Input dataset, X.shape == (data_dimension, n_samples).

        Returns:
            Z-Score normalized X.
            Z.shape == (data_dimension, n_samples).
        """
        Z = (X - lib.mean(X, axis=0)) / lib.std(X, axis=0).astype('float64')
        return Z

    def shuffle(self, X, Y):
        """
        Shuffles the dataset X and labels Y correspondingly.

        Input:
            X: Input dataset, X.shape == (data_dimension, n_samples).
            Y: Input dataset, Y.shape == (num_of_classes, n_samples).

        Returns:
            A tuple (X, Y) of shuffled X and Y input.
            X.shape == (data_dimension, n_samples).
            Y.shape == (num_of_classes, n_samples).
        """
        X, Y = X.T, Y.T
        assert len(X) == len(Y)
        p = np_rand_gen.permutation(len(X))
        X, Y = X[p], Y[p]

        return X.T, Y.T

    def _softmax(self, Z):
        """
        Implements Softmax function for the output activation layer.

        Input:
            Z: Output activation layer, Z.shape == (num_of_classes, batch_size).

        Returns:
            Softmax function of the input.
            output.shape == (num_of_classes, batch_size).
        """
        return lib.exp(Z) / sum(lib.exp(Z))

    def _sigmoid(self, Z):
        """
        Implements Sigmoid activation function.

        Input:
            Z: Activation input, Z.shape == (dimension_1, dimension_2).

        Returns:
            ReLU function of the input.
            output.shape == (dimension_1, dimension_2).
        """
        return 1.0 / (1 + lib.exp(-Z))

    def _sigmoid_derivative(self, Z):
        """
        Implements the derivative of Sigmoid activation function.

        Input:
            Z: Activation input, Z.shape == (dimension_1, dimension_2).

        Returns:
            Derivative of the ReLU function of the input.
            output.shape == (dimension_1, dimension_2).
        """
        return self._sigmoid(Z) * (1 - self._sigmoid(Z))

    def _relu(self, Z):
        """
        Implements ReLU (Rectified Linear Unit) activation function.

        Input:
            Z: Activation input, Z.shape == (dimension_1, dimension_2).

        Returns:
            ReLU function of the input.
            output.shape == (dimension_1, dimension_2).
        """
        return lib.maximum(Z, 0)

    def _relu_derivative(self, Z):
        """
        Implements the derivative of ReLU (Rectified Linear Unit) activation function.

        Input:
            Z: Activation input, Z.shape == (dimension_1, dimension_2).

        Returns:
            Derivative of the ReLU function of the input.
            output.shape == (dimension_1, dimension_2).
        """
        return Z > 0

    def _leaky_relu(self, Z, alpha=0.01):
        """
        Implements Leaky ReLU (Leaky Rectified Linear Unit) activation function.

        Input:
            Z: Activation input, Z.shape == (dimension_1, dimension_2).
            alpha: Slope of function's negative values domain.

        Returns:
            Leaky ReLU function of the input.
            output.shape == (dimension_1, dimension_2).
        """
        return lib.where(Z > 0, Z, alpha * Z)

    def _leaky_relu_derivative(self, Z, alpha=0.01):
        """
        Implements the derivative of ReLU (Rectified Linear Unit) activation function.

        Input:
            Z: Activation input, Z.shape == (dimension_1, dimension_2).
            alpha: Slope of function's negative values domain.

        Returns:
            Derivative of the Leaky ReLU function of the input.
            output.shape == (dimension_1, dimension_2).
        """
        return lib.where(Z > 0, 1, alpha)

    def _initialize_weights(self):
        """
        Initiates weights with random normal distribution for the model.

        Input:
            None

        Returns:
            None
        """
        W1 = rand_gen.normal(
            self._init_weights_mu, self._init_weights_sigma, self._model_architecture['W1'])
        B1 = lib.zeros(shape=(self._model_architecture['B1']))
        W2 = rand_gen.normal(
            self._init_weights_mu, self._init_weights_sigma, self._model_architecture['W2'])
        B2 = lib.zeros(shape=(self._model_architecture['B2']))
        W3 = rand_gen.normal(
            self._init_weights_mu, self._init_weights_sigma, self._model_architecture['W3'])
        B3 = lib.zeros(shape=(self._model_architecture['B3']))

        self._params = {"W1": W1, "B1": B1,
                        "W2": W2, "B2": B2, "W3": W3, "B3": B3}

    def _forward_propagation(self, X, dropout_p=0):
        """
        Implements forward propagation pass through the network.

        Input:
            X: The dataset input, X.shape == (batch_size, n_samples).
            dropout_p: Dropout probability, default: No dropout.

        Returns:
            Dictionary of forward pass result.
        """
        W1, B1 = self._params['W1'], self._params['B1']
        W2, B2 = self._params['W2'], self._params['B2']
        W3, B3 = self._params['W3'], self._params['B3']

        IN = (rand_gen.rand(*X.shape) < (1 - self._input_noise_p)) / \
            (1 - self._input_noise_p)
        Z1 = W1.dot(X * IN) + B1
        D1 = (rand_gen.rand(*Z1.shape) < (1 - dropout_p)) / (1 - dropout_p)
        A1 = self._activation_cb(Z1) * D1
        Z2 = W2.dot(A1) + B2
        D2 = (rand_gen.rand(*Z2.shape) < (1 - dropout_p)) / (1 - dropout_p)
        A2 = self._activation_cb(Z2) * D2
        Z3 = W3.dot(A2) + B3
        A3 = self._softmax(Z3)

        result = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}

        return result

    def _backward_propagation(self, X, Y, forward_result):
        """
        Implements backward propagation pass through the network.

        Input:
            X: The dataset input, X.shape == (batch_size, n_samples).
            Y: The dataset labels, Y.shape == (num_of_classes, n_samples).
            forward_result: Dictionary of forward pass result.

        Returns:
            Dictionary of the calculated gradients.
        """
        batch_size = X.shape[1]

        W1, B1 = self._params['W1'], self._params['B1']
        W2, B2 = self._params['W2'], self._params['B2']
        W3, B3 = self._params['W3'], self._params['B3']

        Z1, A1 = forward_result['Z1'], forward_result['A1']
        Z2, A2 = forward_result['Z2'], forward_result['A2']
        Z3, A3 = forward_result['Z3'], forward_result['A3']

        dZ3 = A3 - Y
        dW3 = (1. / batch_size) * (dZ3.dot(A2.T) + self._reg * W3)
        dB3 = (1. / batch_size) * lib.sum(dZ3, axis=1, keepdims=True)
        dZ2 = W3.T.dot(dZ3) * self._activation_derivative_cb(Z2)
        dW2 = (1. / batch_size) * \
            (dZ2.dot(self._activation_cb(Z1).T) + self._reg * W2)
        dB2 = (1. / batch_size) * lib.sum(dZ2, axis=1, keepdims=True)
        dZ1 = W2.T.dot(dZ2) * self._activation_derivative_cb(Z1)
        dW1 = (1. / batch_size) * (dZ1.dot(X.T) + self._reg * W1)
        dB1 = (1. / batch_size) * lib.sum(dZ1, axis=1, keepdims=True)

        gradients = {'dW3': dW3, 'dB3': dB3, 'dW2': dW2,
                     'dB2': dB2, 'dW1': dW1, 'dB1': dB1}

        return gradients

    def _update_rule(self, gradients):
        """
        Implements Mini-Batch Gradient Descent update rule step.

        Input:
            gradients: Dictionary of last step calculated gradients.

        Returns:
            None
        """
        self._params['W1'] -= self._lr * gradients['dW1']
        self._params['B1'] -= self._lr * gradients['dB1']
        self._params['W2'] -= self._lr * gradients['dW2']
        self._params['B2'] -= self._lr * gradients['dB2']
        self._params['W3'] -= self._lr * gradients['dW3']
        self._params['B3'] -= self._lr * gradients['dB3']

    def _lr_step_decay(self, epoch):
        """
        Updates learning rate, using learning rate step decay.

        The minimum learning rate will be bounded by 0.001.

        Input:
            epoch: Current epoch of the algorithm.

        Returns:
            None
        """
        self._lr = self._initial_lr * lib.power(self._early_stop_drop, lib.floor(
            (1 + epoch) / float(self._early_stop_epochs_drop)))
        self._lr = lib.maximum(self._lr, 0.001)

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
            lib.sum(lib.square(self._params['W1'])) +
            lib.sum(lib.square(self._params['W2'])) +
            lib.sum(lib.square(self._params['W3'])))

        # Compute total loss:
        loss = round(float(cross_entropy_loss + l2_regularization_loss), 4)

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

    def fit(self, X_train, Y_train, X_validation, Y_validation, verbose=False):
        """
        Fit the model using Mini-Batch Gradient Descent algorithm.

        Input:
            X_train: Training dataset, X_train.shape == (data_dimension, n_samples).
            Y_train: Training labels,  Y_train.shape == (num_of_classes, n_samples).
            X_validation: Validation dataset, X_validation.shape == (data_dimension, m_samples).
            Y_validation: Validation labels,  Y_validation.shape == (num_of_classes, m_samples).
            verbose: Prints model performance on each epoch, default: False.

        Returns:
            The performance history training dictionary.
        """
        # Initialize variables, reset the state in case fit called multiple times.
        self._reset_state()
        n_samples = X_train.shape[1]
        batch_iterations = int(lib.ceil(n_samples / float(self._batch_size)))
        X_train, Y_train = self.shuffle(X_train, Y_train)
        X_validation, Y_validation = self.shuffle(X_validation, Y_validation)
        prime_params = None
        prime_epoch = 0
        steps = 0
        validation_accuracy_prime = -1

        if self._input_z_score_normalization is True:
            X_train = self._normalize_z_score(X_train)
            X_validation = self._normalize_z_score(X_validation)

        # Epochs training loop, it will be stopped based on range or early stop algorithm:
        for epoch in range(1, self._epochs + 1):
            low = 0
            high = self._batch_size

            # Mini-Batch Gradient Descent loop:
            for batch in range(1, batch_iterations + 1):
                # Get the current mini batch, wrap around on last iteration.
                if batch != batch_iterations:
                    X_batch = X_train[:, low:high]
                    Y_batch = Y_train[:, low:high]
                else:
                    reminder = high - n_samples
                    X_batch = lib.concatenate(
                        (X_train[:, low:n_samples], X_train[:, 0:reminder]), axis=1)
                    Y_batch = lib.concatenate(
                        (Y_train[:, low:n_samples], Y_train[:, 0:reminder]), axis=1)

                low += self._batch_size
                high += self._batch_size

                # Do one step of Gradient Descent.
                forward_result = self._forward_propagation(
                    X_batch, self._dropout_p)
                gradients = self._backward_propagation(
                    X_batch, Y_batch, forward_result)
                self._update_rule(gradients)

            # Calculate accuracy and loss for the data:
            train_predictions = self.predict(X_train)
            train_accuracy = self.evaluate(train_predictions, Y_train)
            train_loss = self._ce_loss(train_predictions, Y_train)
            validation_predictions = self.predict(X_validation)
            validation_accuracy = self.evaluate(
                validation_predictions, Y_validation)
            validation_loss = self._ce_loss(
                validation_predictions, Y_validation)

            # Dump state if needed:
            if verbose is True:
                performance_str_format = (
                    "Epoch: {epoch} | Train - Accuracy: {train_accuracy: <8} % ; Loss: {train_loss: <8} "
                    "| Validation - Accuracy: {validation_accuracy: <8} % ; Loss: {validation_loss: <8}"
                )
                performance_str = performance_str_format.format(
                    epoch=epoch, train_accuracy=train_accuracy, validation_accuracy=validation_accuracy,
                    train_loss=train_loss, validation_loss=validation_loss
                )
                print(performance_str)

            # Save performance results:
            self._performance['train_accuracy'].append(train_accuracy)
            self._performance['validation_accuracy'].append(
                validation_accuracy)
            self._performance['train_loss'].append(train_loss)
            self._performance['validation_loss'].append(validation_loss)

            # Early stop algorithm:
            if validation_accuracy > validation_accuracy_prime:
                steps = 0
                prime_params = copy.deepcopy(self._params)
                prime_epoch = epoch
                validation_accuracy_prime = validation_accuracy
            else:
                steps += 1
                if steps == self._early_stop_max_epochs:
                    break
            if self._initial_lr != 0:
                self._lr_step_decay(epoch)

        self._params = copy.deepcopy(prime_params)
        return self._performance

    def predict(self, X, output_file=None, as_labels=False):
        """
        Predict the input data X.

        Optionally save the predictions to the output_file path.

        Input:
            X: Input dataset, X.shape == (data_dimension, n_samples).
            output_file: Optional full path to an output file where to save the predictions in, default: None.
            as_labels: Return the prediction as class labels array.

        Returns:
            The predictions of the model.
            On as_labels=False: prediction.shape == (num_of_classes, n_samples).
            On as_labels=True: prediction.shape == (n_samples, 1).
        """
        if self._input_z_score_normalization is True:
            X = self._normalize_z_score(X)

        forward_result = self._forward_propagation(X)
        prediction = forward_result['A3']

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
        accuracy_percentage = (
            lib.sum(lib.array([labels == Y_hat])) / float(labels.size)) * 100
        accuracy = round(float(accuracy_percentage), 4)

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
        accuracy = round(float(accuracy_percentage), 4)

        return accuracy

    @staticmethod
    def cross_validate(model, X, Y, k_fold=10, verbose=False, plot=False):
        """
        K Fold cross validation of the model.

        Resets global rand_gen, np_rand_gen Random State generators.

        Input:
            model: The model to validate.
            X: Training input dataset, X.shape == (data_dimension, n_samples).
            Y: Training labels dataset, Y.shape == (num_of_classes, n_samples).
            k_fold: Number of folds in KFold algorithm.
            verbose: Print detailed model performance for each fold.
            plot: Plot model performance for each fold.

        Returns:
            Averaged accuracy based on K Fold algorithm.
        """
        global rand_gen, np_rand_gen

        n_samples = X.shape[1]
        n_samples_validation = n_samples / k_fold
        n_samples_train = n_samples - n_samples_validation
        fold_stats = []
        X, Y = model.shuffle(X, Y)

        for fold in range(k_fold):
            rand_gen = lib.random.RandomState(seed=random_generator_seed)
            np_rand_gen = numpy.random.RandomState(seed=random_generator_seed)

            _model = MultiClassNeuralNetwork(
                data_dim=model._data_dim,
                activation_func=model._activation_func,
                l1_hidden_size=model._l1_hidden_size,
                l2_hidden_size=model._l2_hidden_size,
                num_of_classes=model._num_of_classes,
                lr=model._lr,
                initial_lr=model._initial_lr,
                epochs=model._epochs,
                batch_size=model._batch_size,
                reg=model._reg,
                input_noise_p=model._input_noise_p,
                dropout_p=model._dropout_p,
                early_stop_max_epochs=model._early_stop_max_epochs,
                input_z_score_normalization=model._input_z_score_normalization,
                init_weights_mu=model._init_weights_mu,
                init_weights_sigma=model._init_weights_sigma
            )

            sliding_window_left = fold * n_samples_validation
            sliding_window_right = (fold + 1) * n_samples_validation

            X_validation = X[:, sliding_window_left:sliding_window_right]
            Y_validation = Y[:, sliding_window_left:sliding_window_right]

            X_train_left = X[:, 0:sliding_window_left]
            X_train_right = X[:, sliding_window_right:n_samples]
            Y_train_left = Y[:, 0:sliding_window_left]
            Y_train_right = Y[:, sliding_window_right:n_samples]

            X_train = lib.concatenate((X_train_left, X_train_right), axis=1)
            Y_train = lib.concatenate((Y_train_left, Y_train_right), axis=1)

            performance = _model.fit(
                X_train, Y_train, X_validation, Y_validation, verbose)
            predictions = _model.predict(X_validation)
            accuracy = _model.evaluate(predictions, Y_validation)
            fold_stats.append(accuracy)

            if plot is True:
                print("")
                _model.plot(performance)
                print("")

            print("\nFold ({fold}) test accuracy: {accuracy} %\n".format(
                fold=fold + 1, accuracy=accuracy))

        total_avg_accuracy = round(float(lib.average(fold_stats)), 2)

        return total_avg_accuracy


def parse_cli(static_args=None):
    """
    Parse command line options.

    Input:
        static_args: Pass list of args for testing/environments without CLI.

    Returns:
        Parsed arguments.
    """
    activation_functions = ['sigmoid', 'relu', 'leaky_relu']
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
                        help=("Load the model from the supplied full path Pickle serialized file."
                              "Note: The module used should be compitable with CuPy/NumPy on the current runtime. "
                              "See the note at the top of application code."))
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
                        choices=activation_functions,
                        default='relu',
                        help="Activation function name, default: relu.")
    parser.add_argument('-l1', '--l1-hidden-size',
                        dest='l1_hidden_size',
                        type=int, default=1500,
                        help="Number of neurons in the first hidden layer, default: 1500.")
    parser.add_argument('-l2', '--l2-hidden-size',
                        dest='l2_hidden_size',
                        type=int, default=1500,
                        help="Number of neurons in the second hidden layer, default: 1500.")
    parser.add_argument('-c', '--num-of-classes',
                        dest='num_of_classes',
                        type=int, default=10,
                        help="Number of categorial class of the data, default: 10.")
    parser.add_argument('-g', '--lr',
                        dest='lr',
                        type=float,
                        default=0,
                        help="Learning rate for the optimization algorithm, default: 0.")
    parser.add_argument('-ir', '--initial-lr',
                        dest='initial_lr',
                        type=float,
                        default=0.1,
                        help=("Initial learning rate for the optimization algorithm, default: 0.1."
                              " This feature has higher priority than the static lr feature."))
    parser.add_argument('-e', '--epochs',
                        dest='epochs',
                        type=int,
                        default=300,
                        help="Max number of epochs to do, the algorithm might stop before, due to early stop, default: 300.")
    parser.add_argument('-b', '--batch-size',
                        dest='batch_size',
                        type=int,
                        default=65,
                        help="Batch size to use for the Mini-Batch Gradient Descent, default: 65.")
    parser.add_argument('-r', '--reg',
                        dest='reg',
                        type=float,
                        default=0.1,
                        help="Regularization factor, default: 0.1.")
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

                        default=18,
                        help=("Maximum number of epochs without substantial improvement in model "
                              "accuracy before stopping the training, default: 18."))
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
                        default=0.01,
                        help="Standard deviation of the normal distribution for model weights initialization, default: 0.01.")

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
    header = "Using {lib} as the acceleration library for mathematical operation".format(
        lib=lib.__name__)
    print(f"{'':-^135}\n{' {header} ':-^78}\n{'':-^135}".format(header=header))

    args = parse_cli(static_args)
    model = None

    if args.fit is True:
        print("Loading the dataset...")
        train_X, train_Y = MultiClassNeuralNetwork.load_dataset(
            args.train_file_path)
        validation_X, validation_Y = MultiClassNeuralNetwork.load_dataset(
            args.validation_file_path)
        if train_X is None or validation_X is None:
            return 1

        if args.quiet is False:
            print(f"{'':-^135}\n{' Data shapes ':-^135}\n{'':-^135}")
            print("train_X shape: ", train_X.shape)
            print("train_Y shape: ", train_Y.shape)
            print("validation_X shape: ", validation_X.shape)
            print("validation_Y shape: ", validation_Y.shape)

        model = MultiClassNeuralNetwork(
            data_dim=train_X.shape[0],
            activation_func=args.activation_func,
            l1_hidden_size=args.l1_hidden_size,
            l2_hidden_size=args.l2_hidden_size,
            num_of_classes=args.num_of_classes,
            lr=args.lr,
            initial_lr=args.initial_lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            reg=args.reg,
            input_noise_p=args.input_noise_p,
            dropout_p=args.dropout_p,
            early_stop_max_epochs=args.early_stop_max_epochs,
            input_z_score_normalization=args.input_z_score_normalization,
            init_weights_mu=args.init_weights_mu,
            init_weights_sigma=args.init_weights_sigma
        )

        if args.quiet is False:
            print(model)
            print(f"{'':-^135}\n{' Model Fitting ':-^135}\n{'':-^135}")
        else:
            print("Fitting the model, please wait...")

        performance = model.fit(
            train_X, train_Y, validation_X, validation_Y, not args.quiet)
        print(f"{'':-^135}")

        if args.plot is True:
            model.plot(performance)

        predictions = model.predict(validation_X)
        accuracy = model.evaluate(predictions, validation_Y)
        print("Total validation accuracy: {accuracy} [%]".format(
            accuracy=accuracy))

        size_bytes = MultiClassNeuralNetwork.dump_model(model, args.dump_path)
        if size_bytes is None:
            return 1
        size_MB = size_bytes / 10**6
        print("Dump model file: {file}\nDump model size: {size} [MB]".format(
            file=args.dump_path, size=size_MB))
    elif args.load_path is not None:
        model = MultiClassNeuralNetwork.load_model(args.load_path)
        if model is None:
            return 1
        print("Loaded model file: {file}\n".format(file=args.load_path))
    elif args.compute_accuracy is True:
        print("Commputing accuracy...")
        accuracy = MultiClassNeuralNetwork.evaluate_from_file(
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
            train_X, train_Y = MultiClassNeuralNetwork.load_dataset(
                args.train_file_path)
            validation_X, validation_Y = MultiClassNeuralNetwork.load_dataset(
                args.validation_file_path)
            if train_X is None or validation_X is None:
                return 1
        print(f"{'':-^135}\n{' Model KFold Cross Validation ':-^135}\n{'':-^135}")
        X = lib.concatenate((train_X, validation_X), axis=1)
        Y = lib.concatenate((train_Y, validation_Y), axis=1)
        kfold_accuracy = MultiClassNeuralNetwork.cross_validate(
            model, X, Y, k_fold=args.k_fold, verbose=(not args.quiet), plot=args.plot)
        print("Total KFold cross validation accuracy: {accuracy} [%]".format(
            accuracy=kfold_accuracy))

    if args.predict is True:
        print("Predicting...")
        test_X, _ = MultiClassNeuralNetwork.load_dataset(
            dataset_path=args.test_file_path, without_labels=True)
        if test_X is None:
            return 1
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
    #     '/content/model_testing/cupy_trained_model_dump.bin',
    #     '--train-file-path',
    #     '/content/model_testing/train.csv',
    #     '--validation-file-path',
    #     '/content/model_testing/validate.csv',
    #     '--load-path',
    #     '/content/model_testing/cupy_trained_model_dump.bin',
    #     '--compute-accuracy',
    #     '--predict',
    #     '--prediction-file-path',
    #     '/content/model_testing/accuracy_test/validation_prediction.txt',
    #     '--labels-file-path',
    #     '/content/model_testing/accuracy_test/validation_labels.txt',
    #     '--test-file-path',
    #     '/content/model_testing/validate.csv',
    #     '--k-fold',
    #     '10'
    # ]

    rc = main(static_args)
    if static_args is None:
        exit(rc)
