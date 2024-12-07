# emg2pose

A dataset of Surface electromyography (sEMG) recordings paired with ground-truth, motion-capture recordings of the hands. Data loading, baseline model training, and baseline model evaluation code are provided.

<p align="center">
  <img src="https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_overview.png" alt="EMG2Pose Overview" width="75%">
</p>


## Data
The entire dataset has $25,253$ HDF5 files, each consisting of time-aligned, 2kHz sEMG and joint angles for a single hand in a single stage. Each stage is ~1 minute. There are $193$ participants, spanning $370$ hours and $29$ stages. `emg2pose.data.Emg2PoseSessionData` offers a programmatic read-only interface into the HDF5 session files.

The full dataset statistics are as follows:

<p align="center">
  <img src="images/dataset_stats.png" alt="Dataset statistics" width="75%">
</p>

The `metadata.csv` file includes the following information for each HDF5 file:

| Column             | Description |
|--------------------|-------------|
| `user`              | Anonymized user ID |
| `session`           | Recording session (there are multiple stages per recording session) |
| `stage`             | Name of stage |
| `side`              | Hand side (`left` or `right`) |
| `moving_hand`       | Whether the hand is prompted to move during the stage |
| `held_out_user`     | Whether the user is held out from the training set |
| `held_out_stage`    | Whether the stage is held out from the training set |
| `split`             | `train`, `test`, or `val` |
| `generalization`    | Type of generalization; across user (`user`), stage (`stage`), or across user and stage (`user_stage`) |

## Setup

### Environment and Dependencies

```shell
# Clone the repo, setup environment, and install local package
# NOTE: the facebookresearch github repo will be available for the camera-ready version
git clone git@github.com:facebookresearch/emg2pose.git ~/emg2pose
cd ~/emg2pose
conda env create -f environment.yml

# Activate the environment
conda activate emg2pose

# Install the emg2pose package
pip install -e .

# Install the UmeTrack package (for forward kinematics and mesh skinning)
pip install -e emg2pose/UmeTrack
```

## Getting Started (Small, Sanity-Check Dataset)

The full dataset is $431$ GiB -- which can be cumbersome for a quick start. As a solution, we
also host a smaller (~ $600$ MiB) version of the dataset which can be downloaded and used to run
a sanity-check version of the train and eval logic.

### (Optional) Download Just the Metadata CSV (5 MiB)

The `emg2pose_metadata.csv` file described above can be downloaded on its own using the following endpoint.

NOTE: this metadata file is also included in each of the dataset downloads

```shell
# Download (just) the metadata.csv file to ~/emg2pose_metadata.csv
cd ~ && curl https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv -o emg2pose_metadata.csv
```

### Download a Smaller Version of the Dataset (~600 MiB)

```shell
# Download a mini (600 MiB) version of the dataset
cd ~ && curl "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar" -o emg2pose_dataset_mini.tar

# Unpack the tar to ~/emg2pose_dataset_mini
tar -xvf emg2pose_dataset_mini.tar
```

### Sanity Check Train / Eval

To run a sanity-check training workflow over the small, sanity-check version of the
dataset, please use the following command.

This runs training for the `tracking_vemg2pose` experiment for $5$ epochs as a sanity check.
It also runs evaluation on the validation and test splits -- again as a sanity check.

```shell
python -m emg2pose.train \
train=True \
eval=True \
experiment=tracking_vemg2pose \
trainer.max_epochs=5 \
data_split=mini_split \
data_location="${HOME}/emg2pose_dataset_mini"
```

## Getting Started (Full Dataset)

Above, we provided instructions for working with a smaller version of the dataset as a means
of sanity checking the main entrypoint (`train.py`). Here, we show how to get started with
the whole dataset.

### Download the Full Dataset (431 GiB)

```shell
# Download the full (431 GiB) version of the dataset, extract to ~/emg2pose_dataset
cd ~ && curl https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar -o emg2pose_dataset.tar

# Unpack the tar to ~/emg2pose_dataset
tar -xvf emg2pose_dataset.tar
```

### Train on the Full Dataset

To launch an example, full training run for the `vemg2pose (tracking)` setting, use the following:

```shell
python -m emg2pose.train \
train=True \
eval=True \
experiment=tracking_vemg2pose \
data_location="${HOME}/emg2pose_dataset"
```

The `experiment` CLI option supports the following experiments (see `config/experiment` files):
* `tracking_vemg2pose`
* `regression_vemg2pose`
* `regression_neuropose`

## Downloading Pre-trained Checkpoints

We provide pre-trained checkpoints (as `.ckpt` files) for the following:

1. vemg2pose (tracking, regression settings)
2. neuropose (regression setting)

To download and unpack these checkpoints, run the following.

```shell
# Download checkpoints
cd ~ && curl "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz" -o emg2pose_model_checkpoints.tar.gz

# Unpack to ~/emg2pose_model_checkpoints
tar -xvzf emg2pose_model_checkpoints.tar.gz
```

## Evaluation / Testing

To run basic evaluation for the validation / test splits, use the following:

Note that the `experiment` option to this script should match the checkpoint's experiment.

```shell
# Run train.py with train=False to isolate basic evaluation logic
python -m emg2pose.train \
train=False \
eval=True \
data_location="${HOME}/emg2pose_dataset" \
experiment=tracking_vemg2pose \
checkpoint="${HOME}/emg2pose_model_checkpoints/tracking_vemg2pose.ckpt"
```

To run analyses for different modes of generalization and to generate a `.csv` file with results, use
the following script.

Note that the `experiment` option to this script should match the checkpoint's experiment.

```shell
python -m emg2pose.test_analysis \
data_location="${HOME}/emg2pose_dataset" \
experiment=tracking_vemg2pose \
checkpoint="${HOME}/emg2pose_model_checkpoints/tracking_vemg2pose.ckpt"
```

## Notebook and Visualization

Check out the Jupyter Notebook in `notebooks/getting_started.ipynb` for a brief walkthrough of data
loading, inference, and data visualization.

## License

emg2pose is CC-BY-NC-SA-4.0 licensed, as found in the LICENSE file.

emg2pose is also licensed subject to the licenses of its code dependencies.

UmeTrack is licensed under Attribution-NonCommercial 4.0 International, as found in the emg2pose/UmeTrack/LICENSE and [GitHub](https://github.com/facebookresearch/UmeTrack/blob/main/LICENSE).
