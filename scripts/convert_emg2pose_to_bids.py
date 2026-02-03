# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to convert the emg2pose dataset from HDF5 to BIDS format.
EMG data is stored as EMG type, and joint angles are stored as MISC channels.
https://bids-specification.readthedocs.io/en/bep-032-cleanup/modality-specific-files/electromyography.html
"""

from pathlib import Path

import click
import h5py
import mne
from mne.io.constants import FIFF
import mne_bids
import numpy as np
import pandas as pd
import tqdm

mne.set_log_level("WARNING")


def load_hdf5_data(file_path: Path) -> dict:
    """Load data from an emg2pose HDF5 file."""
    with h5py.File(file_path, "r") as f:
        data = f["emg2pose/timeseries"][:]
        return {
            "time": data["time"],
            "emg": data["emg"],
            "joint_angles": data["joint_angles"],
        }


def get_ik_failure_annotations(joint_angles: np.ndarray, sfreq: float) -> mne.Annotations:
    """Create BAD_IK annotations for periods with IK failures.

    IK failures are detected as samples where all joint angles are close to zero.
    Contiguous IK failure periods are merged into single annotations.
    """
    # Detect IK failures (all joints close to zero)
    is_zero = np.isclose(joint_angles, 0).all(axis=1)

    if not is_zero.any():
        return mne.Annotations([], [], [])

    # Find contiguous blocks of IK failures
    # Pad with False to detect edges at start/end
    padded = np.concatenate([[False], is_zero, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # Convert to time
    onsets = starts / sfreq
    durations = (ends - starts) / sfreq
    descriptions = ["BAD_IK"] * len(onsets)

    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)


def get_mne_raw(
    file_path: Path,
    stage: str,
) -> mne.io.Raw:
    """Read an HDF5 file and create an MNE Raw object.

    EMG channels are stored as EMG type, joint angles as MISC type.
    The stage (task type) is stored as an annotation spanning the entire recording.
    """
    # Load data
    data_dict = load_hdf5_data(file_path)

    sfreq = 2000.0  # Hz

    # Create channel names for EMG (16 channels)
    emg_ch_names = [f"emg{i}" for i in range(16)]

    # Create channel names for joint angles (20 per hand)
    joint_names = [f"joint{i}" for i in range(20)]

    # Combine all channel names
    all_ch_names = emg_ch_names + joint_names

    # Set channel types: EMG channels and joint angles as 'misc'
    ch_types = ["emg"] * 16 + ["misc"] * 20

    # Combine data: EMG then joint angles
    # Shape: (n_channels, n_samples)
    # EMG data appears to be in microvolts, convert to Volts for MNE
    # Joint angles are in radians - scale by 1e6 to compensate for EDF's
    # assumption that MISC channels are in Volts (written as µV, read back as V)
    data = np.concatenate([
        data_dict["emg"].T * 1e-6,         # (16, n_samples) - µV to V
        data_dict["joint_angles"].T * 1e6,  # (20, n_samples) - radians, scaled for EDF
    ], axis=0)

    # Create MNE info and Raw object
    info = mne.create_info(ch_names=all_ch_names, sfreq=sfreq, ch_types=ch_types)
    # Set unit for joint angle channels to radians
    for ch_name in joint_names:
        ch_idx = all_ch_names.index(ch_name)
        info["chs"][ch_idx]["unit"] = FIFF.FIFF_UNIT_RAD

    raw = mne.io.RawArray(data, info)

    # Add annotation for the stage/task type
    stage_annotation = mne.Annotations(
        onset=[0.0],
        duration=[raw.times[-1]],
        description=[f"stage/{stage}"],
    )

    # Add BAD_IK annotations for IK failure periods
    ik_annotations = get_ik_failure_annotations(data_dict["joint_angles"], sfreq)

    # Combine annotations
    raw.set_annotations(stage_annotation + ik_annotations)

    return raw


def convert_to_bids(
    subject_idx: int,
    session_idx: int,
    recording_idx: int,
    file_path: Path,
    stage: str,
    side: str,
    bids_root: str,
) -> None:
    """Convert an emg2pose recording to BIDS format."""
    raw = get_mne_raw(file_path, stage)

    bids_path = mne_bids.BIDSPath(
        subject=f"{subject_idx + 1:02d}",
        session=f"{session_idx + 1:02d}",
        task="emg2pose",
        acquisition=side,  # left or right hand
        run=f"{recording_idx + 1:02d}",
        datatype="emg",
        root=bids_root,
    )
    mne_bids.write_raw_bids(
        raw=raw,
        bids_path=bids_path,
        overwrite=True,
        format="EDF",
        allow_preload=True,
        emg_placement="Other",
    )

    # Update channels.tsv to set units to "rad" for joint angle channels
    channels_tsv = bids_path.copy().update(suffix="channels", extension=".tsv").fpath
    channels_df = pd.read_csv(channels_tsv, sep="\t")
    joint_mask = channels_df["name"].str.startswith("joint")
    channels_df.loc[joint_mask, "units"] = "rad"
    channels_df.to_csv(channels_tsv, sep="\t", index=False)


@click.command()
@click.option(
    "--dataset-root",
    type=str,
    default=Path.home().joinpath("emg2pose_dataset_mini"),
    help="Original dataset root directory",
)
@click.option(
    "--bids-root",
    type=str,
    default=Path.home().joinpath("emg2pose_bids_data"),
    help="BIDS dataset root directory",
)
def main(dataset_root: str, bids_root: str):
    dataset_root = Path(dataset_root)
    bids_root = Path(bids_root)

    # Read metadata
    df = pd.read_csv(dataset_root / "metadata.csv")

    # Filter to only files that exist in the dataset
    existing_files = set(f.stem for f in dataset_root.glob("*.hdf5"))
    df = df[df["filename"].isin(existing_files)].copy()

    # Get unique users
    users = sorted(df["user"].unique())

    for subject_idx, user in enumerate(users):
        user_df = df[df["user"] == user]
        sessions = sorted(user_df["session"].unique())

        for session_idx, session in enumerate(sessions):
            session_df = user_df[user_df["session"] == session]

            # Sort by start time and process each recording
            recordings = session_df.sort_values("start")
            for recording_idx, (_, row) in enumerate(tqdm.tqdm(
                recordings.iterrows(),
                desc=f"User {subject_idx + 1}, Session {session_idx + 1}",
                total=len(recordings),
            )):
                file_path = dataset_root / f"{row['filename']}.hdf5"

                if not file_path.exists():
                    print(f"Warning: Missing file {file_path}")
                    continue

                convert_to_bids(
                    subject_idx=subject_idx,
                    session_idx=session_idx,
                    recording_idx=recording_idx,
                    file_path=file_path,
                    stage=row["stage"],
                    side=row["side"],
                    bids_root=bids_root,
                )


if __name__ == "__main__":
    main()
