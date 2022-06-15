"""Generate training jobs scripts.
Quick Start:
.. code-block:: python
    # make sure you have the dependencies
    pip install slurmjobs numpy fire
    # NOTE: slurmjobs does not require slurm - it's just templating
    # this will generate folders for all job formats
    python jobs.py
    # or you can just generate for one type (shell)
    python jobs.py shell
This script supports 3 different output formats:
 - ``singularity``: Use this if you have an sbatch environment with Singularity (e.g. NYU Greene)
 - ``sbatch``: Use this if you have an sbatch environment without Singularity (e.g. like NYU Prince (R.I.P.))
 - ``shell``: Use this if you just want to run the scripts with bash. This will run wherever!
Unless you are on an HPC environment, you're probably just going to want the shell version.
This will generate shell scripts
You can train over any of the command line flags in each of the scripts. Here
we're focusing on:
 - ``num_regions``: range(3, 12, 2) - the number of horizontal regions to localize over.
 - ``folds``: range(0, 5) - The validation folds to use. The other 4 are used for training.
This will generate a grid as a product of each of these variables. So since there are 5 each, the
total number of generated jobs is 25 for training and a corresponding 25 for testing.

Copyright (c) 2022 Magdalena Fuentes, Bea Steers, Julia Wilkins
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

# import os
import slurmjobs

# You should run this script from inside localize_sound

# sets the default CLI generation format

slurmjobs.Jobs.cli = 'argparse'

# sbatch and singularity config

# sbatch arguments are placed at the top of your sbatch file.
# The output format is: ``--{key}={value}``
sbatch = {
    # 'email': f'{os.getenv("USER")}@nyu.edu',
    'mem': '96GB',
    'time': '1-23:00',
    'gres': 'gpu:1',
    'cpus_per_task': 4,
}

# job search grid config

fov = 120
num_regions = [5]
in_dur = 4
point_sources = [True, None]
filter_confirmed = [True]
audio_filtered_labels = [None]

train_dataset = 'Urbansas'
val_dataset = 'Urbansas'
folds = [0, 1, 2, 3, 4]
batch_size = 128
metric = 'val_auc_pr'
loss = ['WeightedBinaryCrossentropy']


#########################
# Job Formats

# setup the different job formats

class Singularity(slurmjobs.Singularity):
    options = dict(
        slurmjobs.Singularity.options,
        # override default options
        sif='cuda11.2.2-cudnn8-devel-ubuntu20.04.sif',
        overlay='/scratch/mf3734/share/urbansas/urbansas-5GB-200K.ext3',
        sbatch=sbatch,
    )

class Slurm(slurmjobs.Slurm):
    options = dict(
        slurmjobs.Singularity.options,
        sbatch=sbatch
    )

class Shell(slurmjobs.Shell):
    options = dict(
        slurmjobs.Shell.options,
        # setting this as True will launch jobs in the background (using nohup) which allows
        # you to launch all jobs at once. I figure that behavior may be surprising
        # so it is disabled by default. But if you want a more sbatch-like environment, you 
        # can set this as true
        background=False,
    )




#########################
# Parameter Grids

# create the training grid

# # train folds are everything not in validation
train_folds = [folds[:i] + folds[i+1:] for i in range(len(folds))]

grid = slurmjobs.Grid([
    ('point_sources', point_sources),
    ('filter_confirmed', filter_confirmed),
    ('audio_filtered_labels', audio_filtered_labels),
    (('train_folds', 'val_folds'), (train_folds, folds))
])

############################
# Generate Functions


def generate(kind, name=None):
    # get the job format class
    if kind.lower() in {'singularity', 'sing'}:
        cls = Singularity
    elif kind.lower() in {'slurm', 'sbatch'}:
        cls = Slurm
    elif kind.lower() in {'shell', 'bash'}:
        cls = Shell
    else:
        raise ValueError("Invalid job kind. Pick one of: {singularity, slurm, shell}")

    # this is where to save the output scripts
    root_dir = f'jobs/{kind}'

    # train jobs - writes to jobs/{kind}/train_model 

    batch_train = cls('python train_model.py', job_id='config_name', root_dir=root_dir, name=f'train_model{f"-{name}" if name else ""}')
    # generate the jobs and print out a summary
    run_script, job_paths = batch_train.generate(
        grid,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        overwrite=True)
    slurmjobs.util.summary(run_script, job_paths)

    # predict jobs - writes to jobs/{kind}/predict_model

    batch_predict = cls('python predict_model.py', job_id=False, root_dir=root_dir, name=f'predict_model{f"-{name}" if name else ""}')
    # create a grid for prediction using the training grid
    predict_grid = slurmjobs.LiteralGrid([
        {
            'folds': d['val_folds'],
            'config_name': batch_predict.format_job_id(d, name=f'train_model{f"-{name}" if name else ""}'),
            'point_sources': d['point_sources'],
            'filter_confirmed': d['filter_confirmed']
        }
        for d in grid
    ])
    print(predict_grid)
    # generate the jobs and print out a summary
    run_script, job_paths = batch_predict.generate(
        predict_grid, overwrite=True)
    slurmjobs.util.summary(run_script, job_paths)

def main(*kinds, name=None):
    '''Generate job formats. The generated files are located under ``jobs/{kind}``.
    Arguments:
        *kinds: The names of the job types to use. Available options:
             - ``singularity``: Use this if you have an sbatch environment with Singularity (e.g. NYU Greene)
             - ``sbatch``: Use this if you have an sbatch environment without Singularity (e.g. like NYU Prince (R.I.P.))
             - ``shell``: Use this if you just want to run the scripts with bash. This will launch the script as a 
                    background task.
        name: name extension to append to folder name (python jobs.py --name exp1 results in rain_model-exp1 folder)
    '''
    for k in kinds or ['singularity', 'shell']:
        generate(k, name)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
