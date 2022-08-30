import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import linear_model
import bisect


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


if times[-1] < 3600:
    print(f'Warning: Measurements span less than 1 hour. They span {int(times[-1])} [sec].')
else:
    # Truncate the measurements to 1 hour
    keep_until = bisect.bisect_left(times, 3600)  # Search for first element > 3600 (1 hour), times is sorted
    times = times[:keep_until]
    A_list = A_list[:keep_until]
    Ah_list = Ah_list[:keep_until]


# Remove values that are clearly wrong
for i in range(len(A_list)):
    if A_list[i] is not None and A_list[i] > 1:
        A_list[i] = None
    if Ah_list[i] is not None and Ah_list[i] > 2:
        Ah_list[i] = None


def regression(_list):
    """
    Apply Huber Regression for outlier robustness
    :param _list: 1-D data to run the analysis on
    :return: list of predicted values for each time point
    """
    A_list_not_none_indices = [j for j, val in enumerate(_list) if val is not None]
    X = np.array(times)[A_list_not_none_indices].reshape(-1, 1)
    Y = np.array(_list)[A_list_not_none_indices]
    reg = linear_model.HuberRegressor(epsilon=1.2)  # linear_model.Lasso(alpha=10)  # BayesianRidge()
    reg.fit(X, Y)
    Y_prime = reg.predict(times.reshape(-1, 1))
    return Y_prime


def clean_series_and_plot(input_list, list_name):
    """
    Run the cleaning pipeline and plot the results
    :param input_list: the list to clean
    :param list_name: 'name' of the list (used for plotting purposes)
    :return: list of predictions
    """
    # Run Huber regression. Robust to outliers
    Y_prime = regression(input_list)

    # Plotting
    plt.plot(times, input_list, 'r', label='measurements', linewidth=0.3)
    plt.plot(times, Y_prime, '--b', label='post-processed')
    plt.plot(times[-1], Y_prime[-1], color='cyan', marker='o', label=f'{round(Y_prime[-1], 3)}')
    plt.title(f'[{list_name}/sec]')
    plt.xlabel('times [sec]')
    plt.ylabel(f'[{list_name}]')
    plt.legend()
    plt.show()
    return Y_prime


clean_series_and_plot(A_list, 'A')
clean_series_and_plot(Ah_list, 'Ah')
