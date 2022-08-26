import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import linear_model

# Analysis

# Load the data
with open('times.pkl', 'rb') as f:
    times = np.array(pickle.load(f))
with open('a_list.pkl', 'rb') as f:
    A_list = np.array(pickle.load(f))
with open('ah_list.pkl', 'rb') as f:
    Ah_list = np.array(pickle.load(f))

# print(Ah_list)
# A_list[36] = 0.6  # TODO remove this outlier, just for testing purposes


def BRR(_list):
    """
    Apply Bayesian Ridge Regression
    TODO: Might want to use Gaussian Process Regression
    :param _list:
    :return:
    """
    A_list_not_none_indices = [i for i, val in enumerate(_list) if val is not None]
    A_list_none_indices = [i for i, val in enumerate(_list) if val is None]
    X = np.array(times)[A_list_not_none_indices].reshape(-1, 1)
    Y = np.array(_list)[A_list_not_none_indices]
    reg = linear_model.BayesianRidge()
    reg.fit(X, Y)
    Y_prime, std = reg.predict(times.reshape(-1, 1), return_std=True)
    return Y_prime, std, A_list_none_indices


def clean_series(input_list, list_name):
    Y_prime, std, A_list_none_indices = BRR(input_list)

    Y_imputed = input_list.copy()
    Y_imputed[A_list_none_indices] = Y_prime[A_list_none_indices]
    outlier_indices = [i for i, (y_im, y_pred, std_) in enumerate(zip(Y_imputed, Y_prime, std))
                       if abs(y_im - y_pred) > std_]  # Need to tweak this 2 param. Higher values => allow more noise

    A_list_without_outliers = input_list.copy()
    A_list_without_outliers[outlier_indices] = None

    Y_prime, _, _ = BRR(A_list_without_outliers)  # TODO show std on plot

    # outliers = [i for i, val in enumerate(Y_prime) if val]
    plt.plot(times, Y_prime, '--b', label='post-processed')
    plt.plot(times, input_list, 'r', label='measurements')
    plt.title(f'[{list_name}/sec]')
    plt.xlabel('times [sec]')
    plt.ylabel(f'[{list_name}]')
    plt.legend()
    plt.show()


clean_series(A_list, 'A')
clean_series(Ah_list, 'Ah')
