# Urbansas Baseline

For a full description of the dataset visit [our website](https://beasteers.github.io/urbansas-website/).

## Installation
```bash
conda create -n urbansas python=3.8
conda activate urbansas

# download the code
git clone https://github.com/magdalenafuentes/urbansas.git
cd urbansas

# install dependencies
pip install -e .
```

## Downloading the dataset

The dataset is hosted on [Zenodo](https://zenodo.org/record/6658386#.Yq-QrfPMK74). You will soon be able to download it using [Soundata](https://soundata.readthedocs.io/en/latest/).


```python
import soundata

dataset = soundata.initialize('urbansas')
dataset.download()  # download the dataset
dataset.validate()  # validate that all the expected files are there
```

If you downloaded the dataset using your own method (e.g. directly from Zenodo), either:
 - Move the dataset `mv path/to/your/urbansas ~/sound_datasets/urbansas` 
 - Symlink the dataset `ln -s path/to/your/urbansas ~/sound_datasets/urbansas` 
 - Change the path so it points to your downloaded dataset.
   1. Go to `project_paths.py` and set `urbansas_root = "path/to/your/urbansas"`

Hint: `path/to/your/urbansas` should contain the folders `annotations/    audio/    video_2fps/`

## Setup
   
1. Create the data index.

    This will compile a JSON file containing the files in the dataset, along with their
    ground truth labels as used in this project. You will need this for evaluation and training.

    ```bash
    python index_dataset.py --datasets Urbansas
    ```

    Confirm that the file `Urbansas.json` was created in the `index/` folder.

    ```bash
    ls index/*.json
    ```

2. Cache the dataset in the format used for training in this repo by running:

    ```python
    python cache_dataset.py
    ```


## Training the model
Follow these steps if your goal is to train the model from scratch and reproduce results seen in our ICASSP paper submission. This example
trains a model with point sources (see paper). To train a boxwise model simply remove the flag `--point_sources` in all 
steps below and change the folder names accordingly. 

> Note: This code is organized for 5-fold cross validation. In the `index/`, each file object will contain a `fold` key, which tells us which fold that UID is found in. For example, when training on folds 1, 2, 3, and 4, we *leave out* fold 0 and only use it for evaluation. 

Train each fold. Take note of the `config_name` that you use. This is the name that your model checkpoints will be saved under.

> Note: if you want to reproduce the results only, skip the training and use the provided weights to make the predictions!

```bash
python train_model.py \
    --train_dataset Urbansas --val_dataset Urbansas \
    --train_folds 1 2 3 4 --val_folds 0 \
    --config_name urbansas_f0_point_sources \
    --point_sources --filter_confirmed

```

For information on batch job generation over parameter grid or training in an environment using Slurm and or Singularity, see the documentation in `jobs.py`.


## Get some predictions! 

To predict on just one left-out fold, run 

```bash
python predict_model.py \
    --config_name urbansas_f0_point_sources \
    --folds 0 --point_sources --filter_confirmed
``` 

Your predictions will be saved in `predictions/<your_config_name>`. Inside here you'll find a `.h5` file containing the predictions. 

## Evaluate

Load these up and evaluate them agains the ground truth:

```bash
python evaluate_model.py \
    --config_name urbansas_f0_point_sources \
    --point_sources
``` 

## See the results
To compute the mean results as the paper, run the following notebook.

```bash
jupyter lab notebooks/results.ipynb
```

## Visually inspect the results
You can also visually inspect some of this results using the notebook below.

```bash
jupyter lab notebooks/viz.ipynb
```

## Citation
If you are using this code or our dataset please cite the following paper:

```
"Urban sound & sight: Dataset and benchmark for audio-visual urban scene understanding."
Fuentes, M., Steers, B., Zinemanas, P., Rocamora, M., Bondi, L., Wilkins, J., Shi, Q., Hou, Y., Das, S., Serra, X. and Bello, J.P., 
in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.
```

```
@inproceedings{urbansas_2022,
  title={Urban sound \& sight: Dataset and benchmark for audio-visual urban scene understanding},
  author={Fuentes, Magdalena and Steers, Bea and Zinemanas, Pablo and Rocamora, Mart{\'\i}n and Bondi, Luca and Wilkins, Julia and Shi, Qianyi and Hou, Yao and Das, Samarjit and Serra, Xavier and others},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={141--145},
  year={2022},
  organization={IEEE}
}
```



## License

The ICASSP 2022 Urbansas Baseline is open-sourced under the BSD-3-Clause license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in ICASSP 2022 Urbansas Baseline, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

