# ICASSP 2021 Urbansas Baseline

## TODOS
- [ ] Rename TAU references to Urbansas
- [ ] Add remaining project params as arguments
- [ ] Add final results and weights to run off-the-shelf

## Getting up and running with our code (WIP)
Follow these steps if your goal is to train the model from scratch and reproduce results seen in our ICASSP paper submission. 

1. Clone the entire repository using `git clone https://github.com/magdalenafuentes/localize_sound.git`. Navigate into this repository. Now you will make a few tweaks to some of the file/folder paths to get your version up and running.

2. Set dataset and workspace path
   1. Set environment variable `WORKSPACE`, or `workspace` variable in [`project_paths.py`](project_paths.py), to the path of your work folder, where artifacts will be stored.
   2. Set environment variable `URBANSAS_ROOT`, or `urbansas_root` variable in [`project_paths.py`](project_paths.py), to the path where you've downloaded the Urbansas dataset. 
      - This folder should contain directories `audio`, `video`, and `annotations`, containing respectively `.wav`, `.mp4s`, and `.csv` files you will use for training and test.
   3. Leave the rest of the project paths as is and you should be good to go here.

3. Next, you will want to create your "index", which will pre-process the data from the video and audio annotations and create a nice JSON file that will be passed to the model in future steps:
    1. Run `python index_dataset.py --datasets URBANSAS` if you are creating the index locally.
    2. OR Run `sbatch jobs/index.sbatch` from the root if you are running the job on a server. Optionally you could also run `bash jobs/index.sbatch` if you are running interactively on a server.
4. Confirm that the file `Urbansas.json` was created in the `index/` folder.
5. Cache dataset by running `python cache_dataset.py`. For the `urbansas` dataset and a target sample rate of `24000`Hz:
   1. original files are read from `<URBANSAS_ROOT>`, e.g. `<URBANSAS_ROOT>/audio/filename.wav`
   2. cached files are stored in `<WORKSPACE>/data/Urbansas/sr-24000/`, e.g. `<WORKSPACE>/data/Urbansas/sr-24000/audio/filename.wav`
6. Next you are ready to train the model:
    1. Note: This code is organized for 5-fold cross validation. In the `index/`, each file object will contain a `fold` key, which tells us which fold that UID is found in. For example, when running the job `baseline_f0.sbatch`, this will train on folds 1, 2, 3, and 4 while *leaving out* fold 0 (read: baseline_f0.sbatch). 
    2. Time to train the model! Explore the `jobs/` folder. This is where we've set up some simple model training scripts. If you're running the model locally, you can run something like this: 
    `python train_model.py --train_dataset URBANSAS --val_dataset URBANSAS --train_folds 1 2 3 4 --val_folds 0 --config_name URBANSAS_f0_scratch`. This will train on all folds besides 0. If you're running on a server, feel free to use `sbatch jobs/baseline_f0.sbatch` from the root or other bash commands.
    3. Note that you can customize your `config_name` above by passing in a custom argument.
    4. The results of this model training iteration will be stored in `checkpoints/<your config name>`.
    5. You'll want to repeat this process for every fold, running `baseline_fX.sbatch` for each fold 0-4.
    6. **DEBUGGING**: If you're getting this error: 
    ```
    Traceback (most recent call last):
    File "train_model.py", line 230, in <module>
        main(**vars(args))
    File "train_model.py", line 145, in main
        train_batch = next(iter(train_db))
    StopIteration
    ``` 
    it is likely one of two issues. This error is being thrown because the iterator is trying to go through `train_db` and something is wrong: either your `project_paths.py` file has the wrong path leading to the `data` folder, or your `batch_size` parameter in `project_params.py` could be larger than your total number of files.

7. Time to get some predictions! 
    1. To predict on just one left-out fold, run `python predict_model.py --config_name URBANSAS_f0_scratch --dataset URBANSAS --folds 0;` (predicting on the configuration trained above). If you're on a server or are ready to predict on each fold, go ahead and run `sbatch jobs/predict.sbatch` from the root directory. You can also edit the `predict.sbatch` file to only call `predict_model.py` for whichever folds you're working with.
    2. Your predictions will be saved in `predictions/<your_config_name>`. Inside here you'll find a `.h5` file containing the predictions. Load these up and evaluate using this notebook: INSERT NOTEBOOK CLEANED UP


## License

The ICASSP 2021 Urbansas Baseline is open-sourced under the BSD-3-Clause license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in ICASSP 2021 Urbansas Baseline, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).
