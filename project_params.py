"""
Project default parameters

Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Luca Bondi(Robert Bosch GmbH), Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

# Horizontal field of view [deg]
fov = 120

# Width of frame
frame_width = 1280

# Output regions, e.g. number of discrete azimuth angles
num_regions = 5

# Sample rate [Hz]
sr = 24000

# Analysis window length [samples]
win_length = 960  # 40ms

# Analysis window hop [samples]
hop_length = 480  # 20ms

# Number of Log Mel coefficients
n_mels = 64

# Number of GCC-Phat coefficients
gcc_nc = 64  # Same as Log Mel with DCASE architecture

# Input window duration [s]
in_dur = 4

# Labels time resolution [s]
labels_period = 0.5

# Batch size
batch_size = 128

# Point Sources, True => pointwise sources, else boxwise
point_sources = True

# If True, use only the largest box (area-wise) per frame, class, and time per file
# (in the video annotations)
index_max_box_only = False

# If false, remove class distinction all together (i.e. treat car and truck the same)
class_distinction = True

# Maximum number of training epochs
max_epochs = 500

# Initial learning rate
lr = 1e-3

# Early stop patience [epochs]
patience = 100

# ReduceLROnPlateau patience
rlrop_patience = 50

# Random seed
seed = 42

# Loss function

loss = 'WeightedBinaryCrossentropy'

# Fold key
fold_key = 'location_id'


# Baseline DCASE SELD ACCDOA architecture
dcase_seld_params = dict(
    # Number of CNN nodes, constant for each layer
    nb_cnn2d_filt=64,
    # CNN time pooling, length of list = number of CNN layers, list value = pooling per layer
    t_pool_size=[5, 5, 1],
    # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer
    f_pool_size=[4, 4, 2],
    # RNN contents, length of list = number of layers, list value = number of nodes
    rnn_size=[128, 128],
    # FNN contents, length of list = number of layers, list value = number of nodes
    fnn_size=[128],
    # Dropout rate, constant for all layers
    dropout_rate=0.05,
)

