from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import math
from itertools import combinations_with_replacement
#from mlfromscratch.utils import normalize, polynomial_features
#from mlfromscratch.utils import train_test_split, polynomial_features
#from mlfromscratch.utils import mean_squared_error, Plot
#from mlfromscratch.supervised_learning import LinearRegression

from plot import Plot

def k_fold_cross_validation_sets(X, y, k, shuffle=True):
    """ Split the data into k sets of training / test data """
    if shuffle:
        X, y = shuffle_data(X, y)

    n_samples = len(y)
    left_overs = {}
    n_left_overs = (n_samples % k)
    if n_left_overs != 0:
        left_overs["X"] = X[-n_left_overs:]
        left_overs["y"] = y[-n_left_overs:]
        X = X[:-n_left_overs]
        y = y[:-n_left_overs]

    X_split = np.split(X, k)
    y_split = np.split(y, k)
    sets = []
    for i in range(k):
        X_test, y_test = X_split[i], y_split[i]
        X_train = np.concatenate(X_split[:i] + X_split[i + 1:], axis=0)
        y_train = np.concatenate(y_split[:i] + y_split[i + 1:], axis=0)
        sets.append([X_train, X_test, y_train, y_test])

    # Add left over samples to last set as training samples
    if n_left_overs != 0:
        np.append(sets[-1][0], left_overs["X"], axis=0)
        np.append(sets[-1][2], left_overs["y"], axis=0)

    return np.array(sets)

def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def mean_squared_error(y_true, y_pred):
    """ Returns the mean squared error between y_true and y_pred """
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse

def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test

def polynomial_features(X, degree):
    n_samples, n_features = np.shape(X)

    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    
    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))
    
    for i, index_combs in enumerate(combinations):  
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new


class l1_regularization():
    """ Regularization for Lasso Regression.
        l1_norm = alpha * (1/2) * sum(abs(wi))
    """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)  #这个地方好像写错了，是不是应该改为return self.alpha * np.linalg.norm(w, 1)

    def grad(self, w):
        return self.alpha * np.sign(w)

class l2_regularization():
    """ Regularization for Ridge Regression. 
        l2_norm = alpha * (1/2) * sum((wi)^2)
    """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def __call__(self, w):
        return self.alpha * 0.5 *  w.T.dot(w)

    def grad(self, w):
        return self.alpha * w

class l1_l2_regularization():
    """ Regularization for Elastic Net Regression 
        l2_norm = alpha * (1/2) * sum((wi)^2)
    """
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contr = self.l1_ratio * np.linalg.norm(w)
        l2_contr = (1 - self.l1_ratio) * 0.5 * w.T.dot(w) 
        return self.alpha * (l1_contr + l2_contr)

    def grad(self, w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr) 

class Regression(object):
    """ Base regression model. Models the relationship between a scalar dependent variable y and the independent 
    variables X. 
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        """ Initialize weights randomly [-1/N, 1/N] """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        self.training_errors = []
        self.initialize_weights(n_features=X.shape[1])

        # Do gradient descent for n_iterations
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)
            # Calculate l2 loss
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))
            self.training_errors.append(mse)
            # Gradient of l2 loss w.r.t w
            grad_w = -(y - y_pred).dot(X) + self.regularization.grad(self.w)
            # Update the weights
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred

class LinearRegression(Regression):
    """Linear model.
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If 
        false then we use batch optimization by least squares.
    """
    def __init__(self, n_iterations=100, learning_rate=0.001, gradient_descent=True):
        self.gradient_descent = gradient_descent
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations,
                                            learning_rate=learning_rate)
    def fit(self, X, y):
        # If not gradient descent => Least squares approximation of w
        if not self.gradient_descent:
            # Insert constant ones for bias weights
            X = np.insert(X, 0, 1, axis=1)
            # Calculate weights by least squares (using Moore-Penrose pseudoinverse)
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression, self).fit(X, y)

class LassoRegression(Regression):
    """Linear regression model with a regularization factor which does both variable selection 
    and regularization. Model that tries to balance the fit of the model with respect to the training 
    data and the complexity of the model. A large regularization factor with decreases the variance of 
    the model and do para.
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01):
        self.degree = degree
        self.regularization = l1_regularization(alpha=reg_factor)
        super(LassoRegression, self).__init__(n_iterations, 
                                            learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(LassoRegression, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(LassoRegression, self).predict(X)

class PolynomialRegression(Regression):
    """Performs a non-linear transformation of the data before fitting the model
    and doing predictions which allows for doing non-linear regression.
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, degree, n_iterations=3000, learning_rate=0.001):
        self.degree = degree
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(PolynomialRegression, self).__init__(n_iterations=n_iterations,
                                                learning_rate=learning_rate)

    def fit(self, X, y):
        X = polynomial_features(X, degree=self.degree)
        super(PolynomialRegression, self).fit(X, y)

    def predict(self, X):
        X = polynomial_features(X, degree=self.degree)
        return super(PolynomialRegression, self).predict(X)

class RidgeRegression(Regression):
    """Also referred to as Tikhonov regularization. Linear regression model with a regularization factor.
    Model that tries to balance the fit of the model with respect to the training data and the complexity
    of the model. A large regularization factor with decreases the variance of the model.
    Parameters:
    -----------
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, reg_factor, n_iterations=1000, learning_rate=0.001):
        self.regularization = l2_regularization(alpha=reg_factor)
        super(RidgeRegression, self).__init__(n_iterations, 
                                            learning_rate)

class PolynomialRidgeRegression(Regression):
    """Similar to regular ridge regression except that the data is transformed to allow
    for polynomial regression.
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, degree, reg_factor, n_iterations=3000, learning_rate=0.01, gradient_descent=True):
        self.degree = degree
        self.regularization = l2_regularization(alpha=reg_factor)
        super(PolynomialRidgeRegression, self).__init__(n_iterations, 
                                                        learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(PolynomialRidgeRegression, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(PolynomialRidgeRegression, self).predict(X)

class ElasticNet(Regression):
    """ Regression where a combination of l1 and l2 regularization are used. The
    ratio of their contributions are set with the 'l1_ratio' parameter.
    Parameters:
    -----------
    degree: int
        The degree of the polynomial that the independent variable X will be transformed to.
    reg_factor: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    l1_ration: float
        Weighs the contribution of l1 and l2 regularization.
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, degree=1, reg_factor=0.05, l1_ratio=0.5, n_iterations=3000, 
                learning_rate=0.01):
        self.degree = degree
        self.regularization = l1_l2_regularization(alpha=reg_factor, l1_ratio=l1_ratio)
        super(ElasticNet, self).__init__(n_iterations, 
                                        learning_rate)

    def fit(self, X, y):
        X = normalize(polynomial_features(X, degree=self.degree))
        super(ElasticNet, self).fit(X, y)

    def predict(self, X):
        X = normalize(polynomial_features(X, degree=self.degree))
        return super(ElasticNet, self).predict(X)


def main_linear():

    X, y = make_regression(n_samples=100, n_features=1, noise=20)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    n_samples, n_features = np.shape(X)

    model = LinearRegression(n_iterations=100)

    model.fit(X_train, y_train)
    
    # Training error plot
    n = len(model.training_errors)
    training, = plt.plot(range(n), model.training_errors, label="Training Error")
    plt.legend(handles=[training])
    plt.title("Error Plot")
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Iterations')
    plt.show()

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print ("Mean squared error: %s" % (mse))

    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap('viridis')

    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(366 * X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.suptitle("Linear Regression")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.show()


def main_lasso():
    # Load temperature data
    data = pd.read_csv('TempLinkoping2016.txt', sep="\t")

    time = np.atleast_2d(data["time"].values).T
    temp = data["temp"].values

    X = time # fraction of the year [0, 1]
    y = temp

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    poly_degree = 13

    model = LassoRegression(degree=15, 
                            reg_factor=0.05,
                            learning_rate=0.001,
                            n_iterations=4000)
    model.fit(X_train, y_train)

    # Training error plot
    n = len(model.training_errors)
    training, = plt.plot(range(n), model.training_errors, label="Training Error")
    plt.legend(handles=[training])
    plt.title("Error Plot")
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Iterations')
    plt.show()

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print ("Mean squared error: %s (given by reg. factor: %s)" % (mse, 0.05))

    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap('viridis')

    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(366 * X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.suptitle("Lasso Regression")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.show()

def main_ridge():
    # Load temperature data
    data = pd.read_csv('TempLinkoping2016.txt', sep="\t")

    time = np.atleast_2d(data["time"].values).T
    temp = data["temp"].values

    X = time # fraction of the year [0, 1]
    y = temp

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    poly_degree = 15

    # Finding regularization constant using cross validation
    lowest_error = float("inf")
    best_reg_factor = None
    print ("Finding regularization constant using cross validation:")
    k = 10
    for reg_factor in np.arange(0, 0.1, 0.01):
        cross_validation_sets = k_fold_cross_validation_sets(
            X_train, y_train, k=k)
        mse = 0
        for _X_train, _X_test, _y_train, _y_test in cross_validation_sets:
            model = PolynomialRidgeRegression(degree=poly_degree, 
                                            reg_factor=reg_factor,
                                            learning_rate=0.001,
                                            n_iterations=10000)
            model.fit(_X_train, _y_train)
            y_pred = model.predict(_X_test)
            _mse = mean_squared_error(_y_test, y_pred)
            mse += _mse
        mse /= k

        # Print the mean squared error
        print ("\tMean Squared Error: %s (regularization: %s)" % (mse, reg_factor))

        # Save reg. constant that gave lowest error
        if mse < lowest_error:
            best_reg_factor = reg_factor
            lowest_error = mse

    # Make final prediction
    model = PolynomialRidgeRegression(degree=poly_degree, 
                                    reg_factor=reg_factor,
                                    learning_rate=0.001,
                                    n_iterations=10000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print ("Mean squared error: %s (given by reg. factor: %s)" % (mse, reg_factor))

    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap('viridis')

    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(366 * X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.suptitle("Polynomial Ridge Regression")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2), ("Training data", "Test data"), loc='lower right')
    plt.show()

if __name__ == "__main__":

    reg = "linear"
    
    if reg == "linear":
        main_linear()
        
    if reg == "lasso":
        main_lasso()
    
    if reg == "ridge":
        main_ridge()