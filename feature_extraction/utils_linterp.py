from typing import List 

import numpy as np
from scipy import interpolate
import math


def apply_linear_interpolation(stimulus_matrix: np.ndarray,
        word_times: List[float],
        num_interpolation_points_per_time_unit: int = 25):
    '''Apply linear interpolation to stimulus matrix.
    Parameters:
    -----------
    stimulus_matrix : array_like : [num_words x num_sample_dims]
    word_times: List[float]: [num_words]

    Returns:
    --------
    interpolated_stimulus_matrix : array_like : [num_interpolated_timepoints x num_sample_dims]
    '''
    f = interpolate.interp1d(x=word_times, y=stimulus_matrix, axis=0)
    interpolated_times = get_interpolation_times(word_times)  # num_interpolation_values
    interpolated_stimulus_matrix = f(interpolated_times)
    return interpolated_stimulus_matrix


def get_interpolation_times(word_times: List[float],
        num_interpolation_points_per_time_unit: int = 25):
    '''Return interpolated times.'''
    return np.arange(int(math.ceil(min(word_times))), int(math.floor(max(word_times))), 1 / num_interpolation_points_per_time_unit)

