import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import linear_model


"""
Clean the times-series data obtained by running the optical capture of the device measurements (obtained using main.py)
"""


# Load the data
with open('times.pkl', 'rb') as f:
    times = np.array(pickle.load(f))
with open('a_list.pkl', 'rb') as f:
    A_list = np.array(pickle.load(f))
with open('ah_list.pkl', 'rb') as f:
    Ah_list = np.array(pickle.load(f))


def BRR(_list):
    """
    Apply Bayesian Ridge Regression
    TODO: Might want to use Gaussian Process Regression instead
    :param _list: 1-D data to run the analysis on
    :return: a Tuple of:
        1) list of predicted values for each time point
        2) corresponding stdev
        3) list of None indices
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
    """
    Run the cleaning pipeline and plot the results
    :param input_list: the list to clean
    :param list_name: 'name' of the list (used for plotting purposes)
    :return: list of predictions
    """
    # Run Bayesian Ridge Regression
    Y_prime, std, A_list_none_indices = BRR(input_list)

    # Detect outliers using the process std
    Y_imputed = input_list.copy()
    Y_imputed[A_list_none_indices] = Y_prime[A_list_none_indices]
    outlier_indices = [i for i, (y_im, y_pred, std_) in enumerate(zip(Y_imputed, Y_prime, std))
                       if abs(y_im - y_pred) > std_]

    # Impute outliers using previous results by None values
    A_list_without_outliers = input_list.copy()
    A_list_without_outliers[outlier_indices] = None

    # Run BRR again. The None values are replaced by the BRR prediction
    Y_prime, _, _ = BRR(A_list_without_outliers)

    # Plotting
    plt.fill(np.concatenate([times, times[::-1]]),
            np.concatenate([Y_prime - 1.96 * std, (Y_prime + 1.96 * std)[::-1]]),
            alpha=.4, fc='b', ec='None', label='95% confidence interval')
    plt.plot(times, Y_prime, '--b', label='post-processed')
    plt.plot(times, input_list, 'r', label='measurements')
    plt.title(f'[{list_name}/sec]')
    plt.xlabel('times [sec]')
    plt.ylabel(f'[{list_name}]')
    plt.legend()
    plt.show()
    return Y_prime


clean_series(A_list, 'A')
clean_series(Ah_list, 'Ah')
