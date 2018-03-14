"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
import numpy as np
from sklearn.utils import check_random_state

def make_dataset(n_points, class_prop=.5, std=1.6, random_state=None):
    """Generate a binary classification dataset of two circular gaussians

    Parameters
    ----------
    n_points: int (>0)
        Number of data points
    class_prop: float (0 < class_prop < 1, default=.5)
        The proportion of positive classes
    std: float (>0, default: 1.6)
        The standard deviation of the gaussians
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    X : array of shape [n_samples, 2]
        The input samples.

    y : array of shape [n_samples]
        The output values.
    """
    drawer = check_random_state(random_state)

    n_pos = int(n_points*class_prop)

    y = np.zeros((n_points), dtype=np.int)
    X = drawer.normal((1.5, 1.5), scale=std, size=(n_points, 2))

    X[:n_pos] *= -1
    y[:n_pos] = 1

    shuffler = np.arange(n_points)
    drawer.shuffle(shuffler)

    return X[shuffler], y[shuffler]






def make_balanced_dataset(n_points, random_state=None):
    """Generate a balanced dataset (i.e. roughly same number of positive
        and negative classes).

    Parameters
    ----------
    n_points: int (>0)
        Number of data points

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    X : array of shape [n_samples, 2]
        The input samples.

    y : array of shape [n_samples]
        The output values.
    """
    return  make_dataset(n_points, class_prop=.5, std=1.6,
                         random_state=random_state)


def make_unbalanced_dataset(n_points, random_state=None):
    """Generate an unbalanced dataset (i.e. the number of positive and
        negative classes is different).

    Parameters
    ----------
    n_points: int (>0)
        Number of data points

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Return
    ------
    X : array of shape [n_samples, 2]
        The input samples.

    y : array of shape [n_samples]
        The output values.
    """
    return  make_dataset(n_points, class_prop=.25, std=1.6,
                         random_state=random_state)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    X, y = make_unbalanced_dataset(1000)
    print("Number of positive examples:", np.sum(y))
    print("Number of negative examples:", np.sum(y==0))
    X1 = X[y==0]
    X2 = X[y==1]
    plt.scatter(X1[:,0], X1[:,1], color="orange", label="Negative", alpha=.5)
    plt.scatter(X2[:,0], X2[:,1], color="DodgerBlue", label="Positive", alpha=.5)
    plt.grid(True)
    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.legend(loc="upper left")
    plt.show()
