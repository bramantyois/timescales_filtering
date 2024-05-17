import os

from typing import List, Optional
import argparse
import pickle

from tqdm import tqdm

import numpy as np
from scipy.signal import welch
import torch

import pandas as pd


from himalaya.ridge import RidgeCV
from himalaya.scoring import r2_score
from himalaya.backend import set_backend


from hard_coded_things import train_stories, test_stories
from signal_processing import upsample_story
from utils import get_dir
from hard_coded_things import frequency_to_period_name_dict
from config import config_plotting
from save_features import (
    save_filtered_features,
    get_save_file_breakpoints,
    get_savename_template,
)

config_plotting(context="paper", palette="muted")
all_stories = train_stories + test_stories

def timescale_regression_analysis(
    target_featureset_name: str,
    input_featureset_name: str,
    lang: str,
    train_stories: List[str],
    test_stories: List[str],
    timescale_feature_dir: str,
    backend: str = "torch_cuda",
):
    cur_backend = set_backend(backend)
    target_dir = os.path.join(timescale_feature_dir, target_featureset_name, lang)
    input_dir = os.path.join(timescale_feature_dir, input_featureset_name, lang)

    fcs = list(frequency_to_period_name_dict.keys())

    timescale_scores = {}
    for fc in tqdm(fcs):

        X_train = [
            np.load(os.path.join(input_dir, f"{s}.npz"))[str(fc)].T
            for s in train_stories
        ]
        Y_train = [
            np.load(os.path.join(target_dir, f"{s}.npz"))[str(fc)].T
            for s in train_stories
        ]

        X_train = np.concatenate(X_train, axis=0).astype(np.float32)
        Y_train = np.concatenate(Y_train, axis=0).astype(np.float32)

        X_test = [
            np.load(os.path.join(input_dir, f"{s}.npz"))[str(fc)].T
            for s in test_stories
        ]
        Y_test = [
            np.load(os.path.join(target_dir, f"{s}.npz"))[str(fc)].T
            for s in test_stories
        ]

        X_test = np.concatenate(X_test, axis=0).astype(np.float32)
        Y_test = np.concatenate(Y_test, axis=0).astype(np.float32)

        if np.allclose(X_train, Y_train):
            print("target and input are the same")
            break

        model = RidgeCV(
            alphas=np.logspace(-6, 2, 9),
            solver_params={
                "n_targets_batch": 64,
                "n_alphas_batch": 1,
                "n_targets_batch_refit": 64,
            },
        )

        model.fit(X_train, Y_train)

        with torch.no_grad():
            Y_test_pred = model.predict(X_test)

            # compute r2
            scores = r2_score(Y_test, Y_test_pred)
            
        scores = cur_backend.to_numpy(scores)

        stat = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "max": np.max(scores),
            "min": np.min(scores),
            "median": np.median(scores),
        }
        timescale_scores[frequency_to_period_name_dict[fc]] = stat

    return timescale_scores


def getparser():
    parser = argparse.ArgumentParser(description="Timescale regression analysis")
    parser.add_argument(
        "--target_featureset_name",
        type=str,
        default="BERT_all",
        help="Target feature set name",
    )
    parser.add_argument(
        "--input_featureset_name",
        type=str,
        default="mBERT_all",
        help="Input feature set name",
    )
    parser.add_argument(
        "--lang", type=str, default="zh", help="Language", choices=["zh", "en"]
    )
    parser.add_argument(
        "--timescale_feature_dir",
        type=str,
        default="timescale_features",
        help="Timescale feature directory",
    )

    return parser


if __name__ == "__main__":
    parser = getparser()
    args = parser.parse_args()
    
    backend = "torch_cuda" if torch.cuda.is_available() else "torch"
    set_backend(backend)

    timescale_scores = timescale_regression_analysis(
        args.target_featureset_name,
        args.input_featureset_name,
        args.lang,
        train_stories[1:],
        test_stories,
        args.timescale_feature_dir,
    )

    print(timescale_scores)

    save_dir = os.path.join("results", "timescale_scores")
    os.makedirs(save_dir, exist_ok=True)

    timescale_scores = pd.DataFrame(timescale_scores)
    
    save_fn = os.path.join(
        save_dir,
        f"{args.target_featureset_name}_{args.input_featureset_name}_{args.lang}.csv",
    )
    
    timescale_scores.to_csv(save_fn)
    
    print("Done")

# def upsample_stories(
#     stories: List[str],
#     new_sr: float,
#     story_grid_dir: str,
#     story_trfile_dir: str,
#     feature_set_name: str,
#     interpolation: str = "linear",
#     cache_dir: Optional[str] = None,
# ) -> np.ndarray:
#     """
#     Upsample a story by a factor of `upsample_factor` using linear interpolation.
#     """
#     story_data = {}

#     for s in stories:
#         story_data[s] = upsample_story(
#             s,
#             new_sr=new_sr,
#             upsampling_method=interpolation,
#             story_grid_dir=story_grid_dir,
#             story_trfile_dir=story_trfile_dir,
#             feature_set_name=feature_set_name,
#             cache_dir=cache_dir,
#         )

#     return story_data


# def compute_psd_per_channel(story_data: dict, story_list: List = train_stories):
#     n_neurons = story_data_bert_en[train_stories[0]]["story_data"].shape[1]
#     psds = np.zeros((n_neurons, fft_size // 2 + 1))
#     f = None
#     for n in trange(n_neurons):
#         # join for all stories
#         joint_data = np.concatenate(
#             [story_data[s]["story_data"][:, n] for s in story_list]
#         )
#         joint_data = np.nan_to_num(joint_data)
#         # compute periodogram
#         f, psds[n] = welch(joint_data, fs=sr, nperseg=fft_size, return_onesided=True)
#         # f, psds[n] = periodogram(joint_data, fs=sr, nfft=fft_size)

#     return f, psds


# def save_timescale_feature(
#     feature_set_name="mBERT_all",
#     lang="zh",
#     save_dir="timescale_features",
#     stories: List[str] = all_stories,
# ):
#     feature_save_dir = os.path.join(save_dir, feature_set_name, lang)
#     os.makedirs(feature_save_dir, exist_ok=True)

#     filtered_data = {}
#     for story in stories:
#         temps = []
#         for breakpoint_start, breakpoint_end in zip(
#             [0] + save_file_breakpoints[:-1], save_file_breakpoints
#         ):
#             save_dir = os.path.join("intermediate_outputs", feature_set_name, lang)

#             with open(
#                 get_savename_template(
#                     story_name=story,
#                     neuron_index_range=range(breakpoint_start, breakpoint_end),
#                     featureset_name=feature_set_name,
#                     step_name="filter",
#                     save_filepath=save_dir,
#                 ),
#                 "rb",
#             ) as f:
#                 filtered_stimulus_band = pickle.load(f)
#                 temps.append(filtered_stimulus_band)

#         filtered_data[story] = temps.copy()
#     # joint all features
#     freqs = list(frequency_to_period_name_dict.keys())

#     for story in list(filtered_data.keys()):
#         story_data = filtered_data[story]
#         chunks = len(story_data)
#         timescale = {}
#         for i, freq in enumerate(freqs):
#             timescale[str(freq)] = np.concatenate(
#                 [story_data[j][freq] for j in range(chunks)], axis=1
#             ).T

#         save_path = os.path.join(feature_save_dir, f"{story}.npz")

#         np.savez_compressed(save_path, **timescale)
