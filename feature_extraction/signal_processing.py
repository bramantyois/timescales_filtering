from typing import Dict, List

from utils import load_story_info
from hard_coded_things import featuresets_dict, train_stories, test_stories

import numpy as np

from tqdm.notebook import trange
from typing import Optional
from scipy.signal import periodogram

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.interpolate import interp1d

def upsampling(
    story_data: np.ndarray,
    word_presentation_times: np.ndarray,
    new_sample_rate: float,
    kind: Optional[str] = "linear",
):
    # get new time array
    new_times = np.arange(
        word_presentation_times[0], word_presentation_times[-1], 1 / new_sample_rate
    )

    # get new data array
    new_data = np.zeros((len(new_times), story_data.shape[1]))

    # for each feature
    for i in range(story_data.shape[1]):
        # f = interp1d(word_presentation_times, story_data[:, i], kind=kind)
        # # interpolate data
        # new_data[:, i] = f(new_times)
        new_data[:,i] = np.interp(new_times, word_presentation_times, story_data[:, i])

    return new_data, new_times


# function computing the PSD of a time-series


def compute_psd(
    story_data: np.ndarray,
    agg_method: str = "mean",
    sampling_rate: float = 0.5,
    fft_size: Optional[int] = None,
):
    """
    Compute the power spectral density of a time-series.

    Args:
    story_data (np.ndarray): The time-series data.
    agg_method (str, optional): The method to aggregate the PSD. Defaults to "mean".
    sampling_rate (float, optional): The sampling rate of the time-series. Defaults to 0.5.
    fft_size (Optional[int], optional): The size of the FFT. Defaults to None.

    Returns:
    np.ndarray: The PSD of the time-series.
    """
    # get the number of samples
    n_samples = story_data.shape[0]

    # get the number of features
    n_features = story_data.shape[1]

    # if no FFT size is given
    if fft_size is None:
        # use the next power of 2
        fft_size = 2 ** int(np.ceil(np.log2(n_samples)))

    ps = []
    for i in trange(n_features):
        f, p = periodogram(story_data[:, i], fs=sampling_rate, nfft=fft_size)
        ps.append(p)

    psd = np.vstack(ps)

    assert psd.shape[0] == n_features

    if agg_method == "max":
        psd = np.max(psd, axis=0)
    elif agg_method == "min":
        psd = np.min(psd, axis=0)
    else:  # agg_method == 'mean':
        psd = np.mean(psd, axis=0)

    return {"f": f, "psd": psd}


def upsample_story(
    story_name: str,
    # is_bling: bool,
    # is_chinese: bool,
    featureset_name: str = "mBERT_all",
    new_sr: float = 20,
    upsampling_method: str = "linear",
    story_grid_dir: str = "../data/deniz2019/en/sentence_TextGrids",
    story_trfile_dir: str = "../data/deniz2019/en/trfiles",
    
):
    # story_grid_dir = f"../data/bling/{subject_id}/moth_grids/en"# load test_stories
    story_data, word_presentation_times, tr_times, num_words_feature = load_story_info(
        story_name=story_name,
        featureset_name=featureset_name,
        trfile_dir=story_trfile_dir,
        grid_dir=story_grid_dir,
    )

   
    upsampled_story_data, new_times = upsampling(
        story_data, word_presentation_times, new_sr, kind=upsampling_method
    )

    return {
        "story_data": upsampled_story_data,
        "word_presentation_times": new_times,
        "sr": new_sr,
    }
    


def plot_psds(
    first_psd: Dict,
    second_psd: Dict,
    ax=None,
    normalized: bool = False,
    labels: List[str] = ["first", "second"],
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    first_max_val = 1
    second_max_val = 1

    if normalized:
        first_max_val = np.max(first_psd["psd"])
        second_max_val = np.max(second_psd["psd"])

    sns.lineplot(
        x=first_psd["f"][1:],
        y=first_psd["psd"][1:] / first_max_val,
        ax=ax,
        label=f"{labels[0]}",
    )
    sns.lineplot(
        x=second_psd["f"][1:],
        y=second_psd["psd"][1:] / second_max_val,
        ax=ax,
        label=f"{labels[1]}",
    )

    ax.set_xscale("log")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")

    plt.legend()
    plt.show()