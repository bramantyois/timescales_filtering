"""Save filtered stimulus features."""

import glob
import numpy as np
import pandas as pd

import os
import pickle
import sys
import time

from filtering import apply_filter, apply_filter_even_grid
from utils import (
    get_mirrored_matrix,
    get_unmirrored_matrix,
    load_story_info,
)

from hard_coded_things import (
    frequency_to_period_name_dict,
    # noise_trim_length,
    # silence_length,
    test_stories,
    train_stories,
    train_stories_zh,
    test_stories_zh,
    featuresets_dict
)
from utils_rbf import apply_rbf_interpolation, get_rbf_interpolation_time
from utils_linterp import apply_linear_interpolation, get_interpolation_times

import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_savename_template(
    story_name: str,
    neuron_index_range: range,
    featureset_name: str,
    step_name: str,
    save_filepath: str = "intermediate_outputs",
):
    return os.path.join(
        save_filepath,
        f"{step_name}_{neuron_index_range[0]}_{neuron_index_range[-1]}_{featureset_name}_{story_name}.p",
    )


def get_save_file_breakpoints(num_neurons: int):
    """Get neuron split indices for saving separate files."""
    quarter_point = int(num_neurons // 4)
    breakpoints = [quarter_point * i for i in range(1, 4)] + [num_neurons]
    return breakpoints


def get_bandpass_values():
    """Return hard-coded bandpass fc and bandwiths for indexing type."""
    word_period_bounds = [
        (2, 4),
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
        (128, 256),
        (256, np.inf),
    ]
    word_frequency_bounds = [
        (1 / period_bound[0], 1 / period_bound[1])
        for period_bound in word_period_bounds
    ]
    frequency_bounds = word_frequency_bounds
    fc_values = [
        round(np.mean(frequency_bound), 10) for frequency_bound in frequency_bounds
    ]
    bandwidth_values = [
        2 * round(frequency_bound[0] - frequency_bound[1], 10)
        for frequency_bound in frequency_bounds
    ]

    fc_values[-1] = 1 / 256
    bandwidth_values[-1] = 1 / 256  # low-pass filter.
    return fc_values, bandwidth_values


def save_filtered_features(
    story_name: str,
    featureset_name: str,
    num_neurons: int,
    neuron_index_range: range = None,
    epsilon: float = 1e-7,
    use_lowpass: bool = True,
    use_highpass: bool = False,
    lowpass_fc: float = 1 / 256,
    highpass_fc: float = 1 / 4,
    story_trfile_dir: str = "../data/en/trfiles",
    story_grid_dir: str = "../data/en/sentence_TextGrids",
):
    """Save filtered stimulus features."""
    (
        stimulus_matrix,
        _,
        _,
        _,
    ) = load_story_info(
        story_name,
        featureset_name=featureset_name,
        trfile_dir=story_trfile_dir,
        grid_dir=story_grid_dir,
    )
    assert stimulus_matrix.shape[1] == num_neurons

    if neuron_index_range:
        stimulus_matrix = stimulus_matrix[:, neuron_index_range]
    fc_values, bandwidth_values = get_bandpass_values()  # [num_bands], [num_bands]
    filtered_stimulus_bands = {}
    for fc, bandwidth in zip(fc_values, bandwidth_values):
        t0 = time.time()
        fir = "bandpass"
        if use_lowpass and np.allclose(fc, lowpass_fc, atol=epsilon):
            fir = "lowpass"
        if use_highpass and np.allclose(fc, highpass_fc, atol=epsilon):
            fir = "highpass"
        print(f"Using {fir} filter for fc={fc}, bandwidth={bandwidth}")
        num_words = len(stimulus_matrix)
        mirror_length = min(num_words, int(2 / fc))
        stimulus_matrix_mirrored = get_mirrored_matrix(
            original_matrix=stimulus_matrix, mirror_length=mirror_length
        )
        print(f"Using mirrored matrix, shape {stimulus_matrix_mirrored.shape}")
        filtered_matrix = apply_filter_even_grid(
            fir=fir,
            data=stimulus_matrix_mirrored,
            fc=fc,
            bandwidth=bandwidth,
            extract_complex=False,
        )
        filtered_matrix = get_unmirrored_matrix(
            mirrored_matrix=filtered_matrix,
            mirror_length=mirror_length,
            original_num_samples=num_words,
        )
        t1 = time.time()
        print(f"Time to get filtered matrix for fc={fc}: {t1 - t0}")
        assert filtered_matrix.shape == stimulus_matrix.shape
        filtered_stimulus_bands[fc] = filtered_matrix

    with open(
        get_savename_template(
            story_name=story_name,
            neuron_index_range=neuron_index_range,
            featureset_name=featureset_name,
            step_name="filter",
        ),
        "wb",
    ) as f:
        pickle.dump(filtered_stimulus_bands, f)


def save_unfiltered_features(
    story_name: str,
    featureset_name: str,
    num_neurons: int,
    neuron_index_range: range = None,
    story_trfile_dir: str = "../data/en/trfiles",
    story_grid_dir: str = "../data/en/sentence_TextGrids",
):
    """Save unfiltered stimulus features."""
    (
        stimulus_matrix,
        _,
        _,
        _,
    ) = load_story_info(
        story_name,
        featureset_name=featureset_name,
        trfile_dir=story_trfile_dir,
        grid_dir=story_grid_dir,
    )
    assert stimulus_matrix.shape[1] == num_neurons

    if neuron_index_range:
        stimulus_matrix = stimulus_matrix[:, neuron_index_range]

    with open(
        get_savename_template(
            story_name=story_name,
            neuron_index_range=neuron_index_range,
            featureset_name=featureset_name,
            step_name="unfiltered",
        ),
        "wb",
    ) as f:
        pickle.dump(stimulus_matrix, f)
       
        
def interpolate_filtered_embeddings(
    story_name: str,
    featureset_name: str,
    neuron_index_range: range = None,
    rbf_train_story_name: str = "alternateithicatom",
    interpolation_method: str = "rbf",
    story_trfile_dir: str = "../data/en/trfiles",
    story_grid_dir: str = "../data/en/sentence_TextGrids",
):
    """Interpolate filtered stimulus embeddings."""
    print("Saving interpolated filtered embeddings.")
    _, word_presentation_times, tr_times, _ = load_story_info(
        story_name, trfile_dir=story_trfile_dir, grid_dir=story_grid_dir
    )

    with open(
        get_savename_template(
            story_name=story_name,
            neuron_index_range=neuron_index_range,
            featureset_name=featureset_name,
            step_name="filter",
        ),
        "rb",
    ) as f:
        filtered_stimulus_bands = pickle.load(f)

    t0 = time.time()

    if interpolation_method == "linear":
        filtered_stimulus_bands_interpolated = {
            fc: apply_linear_interpolation(
                stimulus_matrix=filtered_stimulus_bands[fc],
                word_times=word_presentation_times,
            )
            for fc in filtered_stimulus_bands.keys()
        }
    elif interpolation_method == "rbf":
        filtered_stimulus_bands_interpolated = {
            fc: apply_rbf_interpolation(
                vecs=filtered_stimulus_bands[fc],
                do_train_rbf=(story_name == rbf_train_story_name),
                fc=fc,
                data_time=word_presentation_times,
                save_string_addition=featureset_name
                + str(neuron_index_range)
                + str(fc),
            )
            for fc in filtered_stimulus_bands.keys()
        }  # {band_fc: interpolated_stimulus}
    t1 = time.time()
    print(f"Time to perform interpolation: {t1 - t0}")
    return filtered_stimulus_bands_interpolated


def downsample_embeddings(
    filtered_stimulus_bands_interpolated,
    story_name: str,
    featureset_name: str,
    num_neurons: int,
    neuron_index_range: range = None,
    lanczos_fc: float = 0.25,
    interpolation_method: str = "rbf",
    story_trfile_dir: str = "../data/en/trfiles",
    story_grid_dir: str = "../data/en/sentence_TextGrids",
    trim_start: int = 10,
    trim_end: int = 5,
):
    """Downsample interpolated filtered stimulus embeddings to BOLD TRs."""
    _, word_presentation_times, tr_times, num_words_feature = load_story_info(
        story_name,
        trfile_dir=story_trfile_dir,
        grid_dir=story_grid_dir,
    )

    with open(
        get_savename_template(
            story_name=story_name,
            neuron_index_range=neuron_index_range,
            featureset_name=featureset_name,
            step_name="filter",
        ),
        "rb",
    ) as f:
        filtered_stimulus_bands = pickle.load(f)

    print("Get downsampled interpolation value.")
    t1 = time.time()
    if interpolation_method == "linear":
        interpolated_times = get_interpolation_times(word_presentation_times)
    elif interpolation_method == "rbf":
        interpolated_times = get_rbf_interpolation_time(word_presentation_times)
    filtered_stimulus_bands_downsampled = {
        fc: apply_filter(
            fir="lanczos",
            data=filtered_stimulus_bands[fc],
            old_time=word_presentation_times,
            new_time=tr_times,
            fc=lanczos_fc,
        )
        for fc in filtered_stimulus_bands_interpolated.keys()
    }  # {band_fc: interpolated_stimulus_downsampled}
    filtered_stimulus_bands_interpolated_downsampled = {
        fc: apply_filter(
            fir="lanczos",
            data=filtered_stimulus_bands_interpolated[fc],
            old_time=interpolated_times,
            new_time=tr_times,
            fc=lanczos_fc,
        )
        for fc in filtered_stimulus_bands_interpolated.keys()
    }  # {band_fc: interpolated_stimulus_downsampled}
    t2 = time.time()
    print(f"Time to downsample: {t2 - t1}")
    print("Compute correlation with word rate")
    word_rate = num_words_feature.ravel()[trim_start:-trim_end]  # [TRs]
    for fc, downsampled_matrix in filtered_stimulus_bands_downsampled.items():
        x = np.concatenate(
            [downsampled_matrix.T, word_rate.reshape(1, -1)]
        )  # (num_neurons + 1) x num_trs
        corr = np.corrcoef(x)[-1, :-1]
        corr = corr[np.where(~np.isnan(corr))]
        print(
            f" no interp fc={fc}\tcorr with word rate {round(np.mean(np.abs(corr)), 2)}"
        )

    for (
        fc,
        downsampled_matrix,
    ) in filtered_stimulus_bands_interpolated_downsampled.items():
        x = np.concatenate(
            [downsampled_matrix.T, word_rate.reshape(1, -1)]
        )  # (num_neurons + 1) x num_trs
        corr = np.corrcoef(x)[-1, :-1]
        corr = corr[np.where(~np.isnan(corr))]
        print(f"fc={fc}\tcorr with word rate {round(np.mean(np.abs(corr)), 2)}")

    t3 = time.time()
    print(f"Time to compute word rate correlation: {t3 - t2}")

    print("Save filter bands.")
    with open(
        get_savename_template(
            story_name=story_name,
            neuron_index_range=neuron_index_range,
            featureset_name=featureset_name,
            step_name="rbf_downsampled",
        ),
        "wb",
    ) as f:
        pickle.dump(filtered_stimulus_bands_interpolated_downsampled, f)
        t4 = time.time()
        print(f"Time to save files: {t4 - t3}")


def interpolate_and_downsample_filtered_embeddings(
    story_name: str,
    featureset_name: str,
    num_neurons: int,
    neuron_index_range: range = None,
    lanczos_fc: float = 0.25,
    rbf_train_story_name: str = "alternateithicatom",
    interpolation_method: str = "rbf",
    save_interpolation=False,
    story_trfile_dir: str = "../data/en/trfiles",
    story_grid_dir: str = "../data/en/sentence_TextGrids",
):
    all_saved = True
    fcs_to_compute = []
    for fc in frequency_to_period_name_dict.keys():
        if get_savename_template(
            story_name=story_name,
            neuron_index_range=neuron_index_range,
            featureset_name=featureset_name,
            step_name=f"rbf_downsampled_{fc}",
        ) not in glob.glob("intermediate_outputs/*"):
            all_saved = False
            fcs_to_compute.append(fc)
    if all_saved:
        print(f"Skipping {story_name}, {neuron_index_range} units.")
        return

    print(
        f"Saving interpolated filtered embeddings for {story_name}, {neuron_index_range} units."
    )

    _, word_presentation_times, tr_times, num_words_feature = load_story_info(
        story_name,
        trfile_dir=story_trfile_dir,
        grid_dir=story_grid_dir,
    )

    if interpolation_method == "linear":
        interpolated_times = get_interpolation_times(word_presentation_times)
    elif interpolation_method == "rbf":
        interpolated_times = get_rbf_interpolation_time(word_presentation_times)

    with open(
        get_savename_template(
            story_name=story_name,
            neuron_index_range=neuron_index_range,
            featureset_name=featureset_name,
            step_name="filter",
        ),
        "rb",
    ) as f:
        filtered_stimulus_bands = pickle.load(f)

    t0 = time.time()
    for fc, filtered_stimulus_band in filtered_stimulus_bands.items():
        if fc not in fcs_to_compute:
            continue

        if interpolation_method == "linear":
            interpolated_data = apply_linear_interpolation(
                stimulus_matrix=filtered_stimulus_band,
                word_times=word_presentation_times,
            )
        elif interpolation_method == "rbf":
            interpolated_data = apply_rbf_interpolation(
                vecs=filtered_stimulus_band,
                do_train_rbf=(story_name == rbf_train_story_name),
                fc=fc,
                data_time=word_presentation_times,
                save_string_addition=str(neuron_index_range) + str(fc),
            )
        t1 = time.time()
        print(f"Time to perform interpolation: {t1 - t0}, {fc} done")

        # save interpolated data
        if save_interpolation:
            with open(
                get_savename_template(
                    story_name=story_name,
                    neuron_index_range=neuron_index_range,
                    featureset_name=featureset_name,
                    step_name=f"interpolated_{fc}",
                ),
                "wb",
            ) as f:
                print(f"Saving interpolated data for {fc}")
                pickle.dump(interpolated_data, f)

        # now downsample
        downsampled_data = apply_filter(
            fir="lanczos",
            data=interpolated_data,
            old_time=interpolated_times,
            new_time=tr_times,
            fc=lanczos_fc,
        )
        x = np.concatenate(
            [np.array(downsampled_data.T), np.array(num_words_feature.reshape(1, -1))]
        )  # (num_neurons + 1) x num_trs
        corr = np.corrcoef(x)[-1, :-1]
        corr = corr[np.where(~np.isnan(corr))]
        print(f"fc={fc}\tcorr with word rate {np.mean(np.abs(corr))}")

        with open(
            get_savename_template(
                story_name=story_name,
                neuron_index_range=neuron_index_range,
                featureset_name=featureset_name,
                step_name=f"rbf_downsampled_{fc}",
            ),
            "wb",
        ) as f:
            pickle.dump(downsampled_data, f)


def interpolate_and_downsample_unfiltered_embeddings(
    story_name: str,
    featureset_name: str,
    num_neurons: int,
    neuron_index_range: range = None,
    lanczos_fc: float = 0.25,
    rbf_train_story_name: str = "alternateithicatom",
    interpolation_method: str = "rbf",
    save_interpolation=False,
    story_trfile_dir: str = "../data/en/trfiles",
    story_grid_dir: str = "../data/en/sentence_TextGrids",
):
    all_saved = True

    if get_savename_template(
        story_name=story_name,
        neuron_index_range=neuron_index_range,
        featureset_name=featureset_name,
        step_name=f"unfiltered_downsampled",
    ) not in glob.glob("intermediate_outputs/*"):
        all_saved = False
    if all_saved:
        print(f"Skipping {story_name}, {neuron_index_range} units.")
        return

    print("Saving interpolated unfiltered embeddings.")
    
    _, word_presentation_times, tr_times, num_words_feature = load_story_info(
        story_name,
        trfile_dir=story_trfile_dir,
        grid_dir=story_grid_dir,
    )
    
    if interpolation_method == "linear":
        interpolated_times = get_interpolation_times(word_presentation_times)
    elif interpolation_method == "rbf":
        interpolated_times = get_rbf_interpolation_time(word_presentation_times)
     
    t1 = time.time()
    with open(
        get_savename_template(
            story_name=story_name,
            neuron_index_range=neuron_index_range,
            featureset_name=featureset_name,
            step_name="unfiltered",
        ),
        "rb",
    ) as f:
        unfiltered_stimulus = pickle.load(f)
       
    if interpolation_method == "linear":
        interpolated_data = apply_linear_interpolation(
            stimulus_matrix=unfiltered_stimulus,
            word_times=word_presentation_times,
        )
    elif interpolation_method == "rbf":
        interpolated_data = apply_rbf_interpolation(
            vecs=unfiltered_stimulus,
            do_train_rbf=(story_name == rbf_train_story_name),
            fc=fc,
            data_time=word_presentation_times,
            save_string_addition=str(neuron_index_range) + str(fc),
        )

    # now downsample
    downsampled_data = apply_filter(
        fir="lanczos",
        data=interpolated_data,
        old_time=interpolated_times,
        new_time=tr_times,
        fc=lanczos_fc,
    )
    x = np.concatenate(
        [np.array(downsampled_data.T), np.array(num_words_feature.reshape(1, -1))]
    )  # (num_neurons + 1) x num_trs
    corr = np.corrcoef(x)[-1, :-1]
    corr = corr[np.where(~np.isnan(corr))]
    print(f"corr with word rate {np.mean(np.abs(corr))}")
    
    with open(
        get_savename_template(
            story_name=story_name,
            neuron_index_range=neuron_index_range,
            featureset_name=featureset_name,
            step_name=f"unfiltered_downsampled",
        ),
        "wb",
    ) as f:
        pickle.dump(downsampled_data, f)


def extract_features(
    featureset_name,
    num_neurons,
    story_name,
    interpolation_method="rbf",
    save_interpolation=False,
    story_trfile_dir="../data/en/trfiles",
    story_grid_dir="../data/en/sentence_TextGrids",
):
    """Call function to extract filtered features for each story."""
    save_file_breakpoints = get_save_file_breakpoints(num_neurons)

    print(
        f'Saving filtered features for {story_name}, word bands indexed, {time.strftime("%c")}'
    )
    for breakpoint_start, breakpoint_end in zip(
        [0] + save_file_breakpoints[:-1], save_file_breakpoints
    ):
        save_filtered_features(
            story_name,
            featureset_name=featureset_name,
            num_neurons=num_neurons,
            neuron_index_range=range(breakpoint_start, breakpoint_end),
            story_grid_dir=story_grid_dir,
            story_trfile_dir=story_trfile_dir,
        )
        interpolate_and_downsample_filtered_embeddings(
            story_name=story_name,
            featureset_name=featureset_name,
            num_neurons=num_neurons,
            neuron_index_range=range(breakpoint_start, breakpoint_end),
            interpolation_method=interpolation_method,
            save_interpolation=save_interpolation,
            story_grid_dir=story_grid_dir,
            story_trfile_dir=story_trfile_dir,
        )


def extract_unfiltered_features(
    featureset_name,
    num_neurons,
    story_name,
    interpolation_method="rbf",
    save_interpolation=False,
    story_trfile_dir="../data/en/trfiles",
    story_grid_dir="../data/en/sentence_TextGrids",
):
    """Call function to extract unfiltered features for each story."""
    save_file_breakpoints = get_save_file_breakpoints(num_neurons)

    print(
        f'Saving unfiltered features for {story_name}, word bands indexed, {time.strftime("%c")}'
    )
    for breakpoint_start, breakpoint_end in zip(
        [0] + save_file_breakpoints[:-1], save_file_breakpoints
    ):
        save_unfiltered_features(
            story_name,
            featureset_name=featureset_name,
            num_neurons=num_neurons,
            neuron_index_range=range(breakpoint_start, breakpoint_end),
            story_grid_dir=story_grid_dir,
            story_trfile_dir=story_trfile_dir,
        )
        interpolate_and_downsample_unfiltered_embeddings(
            story_name=story_name,
            featureset_name=featureset_name,
            num_neurons=num_neurons,
            neuron_index_range=range(breakpoint_start, breakpoint_end),
            interpolation_method=interpolation_method,
            save_interpolation=save_interpolation,
            story_grid_dir=story_grid_dir,
            story_trfile_dir=story_trfile_dir,
        )
            

def get_feature(
    story_name: str,
    featureset_name: str,
    frequency: float,
):
    """Get filtered feature saved in extract_features()."""
    save_file_breakpoints = get_save_file_breakpoints(num_neurons)
    features_subsets = []
    for breakpoint_start, breakpoint_end in zip(
        [0] + save_file_breakpoints[:-1], save_file_breakpoints
    ):
        neuron_index_range = range(breakpoint_start, breakpoint_end)
        with open(
            get_savename_template(
                story_name=story_name,
                neuron_index_range=neuron_index_range,
                featureset_name=featureset_name,
                step_name=f"rbf_downsampled_{frequency}",
            ),
            "rb",
        ) as f:
            downsampled_data = pickle.load(f)
        features_subsets.append(downsampled_data)
    features = np.concatenate(features_subsets, axis=1)
    features = features - features.mean(0)  # Mean-center features.
    return features  # [num_TRs x num_dims]


def get_unfiltered_feature(
    story_name: str,
    featureset_name: str,
):
    """Get unfiltered feature saved in extract_features()."""
    save_file_breakpoints = get_save_file_breakpoints(num_neurons)
    features_subsets = []
    for breakpoint_start, breakpoint_end in zip(
        [0] + save_file_breakpoints[:-1], save_file_breakpoints
    ):
        neuron_index_range = range(breakpoint_start, breakpoint_end)
        with open(
            get_savename_template(
                story_name=story_name,
                neuron_index_range=neuron_index_range,
                featureset_name=featureset_name,
                step_name=f"unfiltered_downsampled",
            ),
            "rb",
        ) as f:
            downsampled_data = pickle.load(f)
        features_subsets.append(downsampled_data)
    features = np.concatenate(features_subsets, axis=1)
    features = features - features.mean(0)  # Mean-center features.
    return features  # [num_TRs x num_dims]


def save_features_dicts(
    featureset_name: str,
    save_path: str,
    num_neurons: int,
    trim_start: int = 10,
    trim_end: int = 5,
):
    """Save train and test feature dicts."""
    train_features_dict = {}
    test_features_dict = {}

    train_features_untrimmed_dict = {}
    test_features_untrimmed_dict = {}

    train_features_dict_meta = []
    test_features_dict_meta = []

    for frequency, timescale_name in frequency_to_period_name_dict.items():
        train_feature_trimmed = []
        train_feature_untrimmed = {}
        for i, story_name in enumerate(train_stories):
            feature = get_feature(
                featureset_name=featureset_name,
                story_name=story_name,
                frequency=frequency,
            )
            # )[trim_start:-trim_end]

            train_features_dict_meta.append(
                {
                    "timescale_name": timescale_name,
                    "index": i,
                    "story_name": story_name,
                    "feature_len": feature.shape[0],
                }
            )
            train_feature_trimmed.append(feature[trim_start:-trim_end])
            train_feature_untrimmed[story_name] = feature

        test_feature_trimmed = []
        test_feature_untrimmed = {}
        for i, story_name in enumerate(test_stories):
            feature = get_feature(
                featureset_name=featureset_name,
                story_name=story_name,
                frequency=frequency,
            )
            # )[trim_start:-trim_end]

            test_features_dict_meta.append(
                {
                    "timescale_name": timescale_name,
                    "index": i,
                    "story_name": story_name,
                    "feature_len": feature.shape[0],
                }
            )

            test_feature_trimmed.append(feature[trim_start:-trim_end])
            test_feature_untrimmed[story_name] = feature

        train_feature_trimmed = np.concatenate(train_feature_trimmed, axis=0)
        test_feature_trimmed = np.concatenate(test_feature_trimmed, axis=0)

        train_features_dict[timescale_name] = train_feature_trimmed
        test_features_dict[timescale_name] = test_feature_trimmed

        train_features_untrimmed_dict[timescale_name] = train_feature_untrimmed
        test_features_untrimmed_dict[timescale_name] = test_feature_untrimmed

        # print(train_feature_trimmed.shape, test_feature_trimmed.shape)
        assert (
            train_feature_trimmed.shape[1]
            == test_feature_trimmed.shape[1]
            == num_neurons
        ), f"train_feature_trimmed.shape[1] = {train_feature_trimmed.shape[1]}, test_feature_trimmed.shape[1] = {test_feature_trimmed.shape[1]}, num_neurons = {num_neurons}"

    print("saving compressed dict")
    np.savez_compressed(
        os.path.join(save_path), train=train_features_dict, test=test_features_dict
    )

    print("saving untrimmed dict")
    np.savez_compressed(
        os.path.join(save_path.replace(".npz", "_untrimmed.npz")),
        train=train_features_untrimmed_dict,
        test=test_features_untrimmed_dict,
    )

    # save meta data as csv
    train_meta_df = pd.DataFrame.from_dict(train_features_dict_meta)
    # select only "2_4_words" timescale since all others have same length
    train_meta_df = train_meta_df[train_meta_df["timescale_name"] == "2_4_words"]
    # sort by index
    train_meta_df = train_meta_df.sort_values(by=["index"])
    # now get start and end indices
    train_meta_df["start"] = train_meta_df.groupby("index").cumcount()
    train_meta_df["end"] = train_meta_df["start"] + train_meta_df["feature_len"]

    test_meta_df = pd.DataFrame.from_dict(test_features_dict_meta)
    # select only "2_4_words" timescale since all others have same length
    test_meta_df = test_meta_df[test_meta_df["timescale_name"] == "2_4_words"]
    # sort by index
    test_meta_df = test_meta_df.sort_values(by=["index"])
    # now get start and end indices
    test_meta_df["start"] = test_meta_df.groupby("index").cumcount()
    test_meta_df["end"] = test_meta_df["start"] + test_meta_df["feature_len"]

    train_meta_df.to_csv(os.path.join(save_path.replace(".npz", "_train_meta.csv")))
    test_meta_df.to_csv(os.path.join(save_path.replace(".npz", "_test_meta.csv")))


def save_unfiltered_features_dicts(
    featureset_name: str,
    save_path: str,
    num_neurons: int,
    trim_start: int = 10,
    trim_end: int = 5,
):
    """Save train and test feature dicts."""
    train_features_dict = {}
    test_features_dict = {}

    train_features_untrimmed_dict = {}
    test_features_untrimmed_dict = {}

    train_features_dict_meta = []
    test_features_dict_meta = []

    for i, story_name in enumerate(train_stories):
        feature = get_unfiltered_feature(
            featureset_name=featureset_name,
            story_name=story_name,
        )
        # )[trim_start:-trim_end]

        train_features_dict_meta.append(
            {
                "index": i,
                "story_name": story_name,
                "feature_len": feature.shape[0],
            }
        )
        train_features_dict[story_name] = feature
        train_features_untrimmed_dict[story_name] = feature

    for i, story_name in enumerate(test_stories):
        feature = get_unfiltered_feature(
            featureset_name=featureset_name,
            story_name=story_name,
        )
        # )[trim_start:-trim_end]

        test_features_dict_meta.append(
            {
                "index": i,
                "story_name": story_name,
                "feature_len": feature.shape[0],
            }
        )
        test_features_dict[story_name] = feature
        test_features_untrimmed_dict[story_name] = feature

    # save meta data as csv
    train_meta_df = pd.DataFrame.from_dict(train_features_dict_meta)
    # sort by index
    train_meta_df = train_meta_df.sort_values(by=["index"])
    # now get start and end indices
    train_meta_df["start"] = train_meta_df.groupby("index").cumcount()
    train_meta_df["end"] = train_meta_df["start"] + train_meta_df["feature_len"]

    test_meta_df = pd.DataFrame.from_dict(test_features_dict_meta)
    # sort by index
    test_meta_df = test_meta_df.sort_values(by=["index"])
    # now get start and end indices
    test_meta_df["start"] = test_meta_df.groupby("index").cumcount()
    test_meta_df["end"] = test_meta_df["start"] + test_meta_df["feature_len"]

    train_meta_df.to_csv(os.path.join(save_path.replace(".npz", "_train_meta.csv")))
    test_meta_df.to_csv(os.path.join(save_path.replace(".npz", "_test_meta.csv")))

    print("saving compressed dict")
    np.savez_compressed(
        os.path.join(save_path), train=train_features_dict, test=test_features_dict
    )
    
    

def get_parser():
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--task",
    #     type=str,
    #     default="build_features",
    #     help="Task to run",
    #     choices=["build_features", "interpolate_features"],
    # )

    parser.add_argument(
        "--featureset_name",
        type=str,
        default="BERT_all",
        help="Featureset name",
        choices=[
            "BERT_all",
            "BERT_10",
            "BERT_100",
            "mBERT_all",
            "mBERT_10",
            "mBERT_100",
            "chinese_BERT_all",
        ],
    )
    parser.add_argument(
        "--interpolation_method",
        type=str,
        default="linear",
        help="Interpolation method",
        choices=["linear", "rbf"],
    )

    parser.add_argument(
        "--save_interpolation",
        action="store_true",
        help="Whether to save interpolated features",
    )

    parser.add_argument(
        "--is_bling",
        action="store_true",
        help="Whether bling data is being used",
    )

    parser.add_argument(
        "--subject_id",
        type=str,
        default="COL",
    )

    parser.add_argument(
        "--is_chinese",
        action="store_true",
        help="Whether Chinese data is being used",
    )

    parser.add_argument(
        "--trim_start",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--trim_end",
        type=int,
        default=5,
    )
    
    parser.add_argument(
        "--dont_extract_filtered",
        action="store_true",
        help="Whether to extract filtered features",
    )
    
    parser.add_argument(
        "--extract_unfiltered",
        action="store_true",
        help="Whether to extract unfiltered features",
    )

    return parser


if __name__ == "__main__":

    parser = get_parser().parse_args()

    for save_dir in ["outputs", "intermediate_outputs", "best_alphas"]:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # clean intermediate outputs
    for f in glob.glob("intermediate_outputs/*"):
        os.remove(f)
    # clean best alphas
    for f in glob.glob("best_alphas/*"):
        os.remove(f)

    featureset_name = str(parser.featureset_name)
    interpolation_method = str(parser.interpolation_method)
    save_interpolation = bool(parser.save_interpolation)

    num_neurons = 9984

    story_grid_dir = "../data/en/sentence_TextGrids"
    story_trfile_dir = "../data/en/trfiles"

    if parser.is_bling:
        if parser.is_chinese:
            story_grid_dir = f"../data/bling/grids_w_punctuation/zh"
            story_trfile_dir = f"../data/bling/{parser.subject_id}/trfiles/zh"
        else:

            story_grid_dir = "../data/deniz2019/en/sentence_TextGrids"
            story_trfile_dir = f"../data/bling/{parser.subject_id}/trfiles/en"
    else:
        story_grid_dir = "../data/deniz2019/en/sentence_TextGrids"
        story_trfile_dir = "../data/deniz2019/en/trfiles"

    stories = train_stories + test_stories

    if not parser.dont_extract_filtered:
        for story_name in stories:
            extract_features(
                featureset_name=featureset_name,
                # use_lowpass=True,
                num_neurons=num_neurons,
                story_name=story_name,
                interpolation_method=interpolation_method,
                save_interpolation=save_interpolation,
                story_grid_dir=story_grid_dir,
                story_trfile_dir=story_trfile_dir,
            )

        lang_appendix = "zh" if parser.is_chinese else "en"

        save_features_dicts(
            featureset_name=featureset_name,
            save_path=f"./outputs/timescales_{featureset_name}_{lang_appendix}_{parser.subject_id.lower()}.npz",
            num_neurons=num_neurons,
            trim_start=parser.trim_start,
            trim_end=parser.trim_end,
        )
    
    if parser.extract_unfiltered:
        lang_appendix = "zh" if parser.is_chinese else "en"

        for story_name in stories:
            extract_unfiltered_features(
                featureset_name=featureset_name,
                num_neurons=num_neurons,
                story_name=story_name,
                interpolation_method=interpolation_method,
                save_interpolation=save_interpolation,
                story_grid_dir=story_grid_dir,
                story_trfile_dir=story_trfile_dir,
            )
            
            # open words embeddings file
            model_name = featuresets_dict[featureset_name][0][1]['model_name']
            
            word_embeddings_fn = f"outputs/word_embeddings_{model_name}.npy"
            words_fn = f"outputs/words_{model_name}.npy"
            
            word_embeddings = np.load(word_embeddings_fn)
            words = np.load(words_fn)
            
            # save it as npz
            np.savez_compressed(
                f"outputs/word_embeddings_{featureset_name}_{story_name}_{lang_appendix}.npz",
                word_embeddings=word_embeddings,
                words=words,
            )

        save_unfiltered_features_dicts(
            featureset_name=featureset_name,
            save_path=f"./outputs/timescales_unfiltered_{featureset_name}_{lang_appendix}_{parser.subject_id.lower()}.npz",
            num_neurons=num_neurons,
            trim_start=parser.trim_start,
            trim_end=parser.trim_end,
        )
