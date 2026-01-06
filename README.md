## Installation
Follow the steps below to set up your environment.  
For Qulacs to work with multiple cores follow the extra steps extra steps, otherwise just do the pip install's.  
Recommended Python version is `3.12`.

```bash
python3.12 -m venv myvenv  
source myvenv/bin/activate
```

Default qulacs simulator
```bash
pip install qulacs
```

Fast qulacs simulators (linux) use similar commands with homebrew for macos
```bash
sudo apt install gcc-14 g++-14  
sudo apt install libboost-all-dev

export C_COMPILER=gcc-14  
export CXX_COMPILER=g++-14  
export QULACS_OPT_FLAGS="-mtune=native"  
pip install git+https://github.com/qulacs/qulacs.git  
```

other dependencies
```bash
pip install tequila-basic  
pip install pyscf  
pip install torch-cluster  
pip install torch_geometric  
```

## Project structure
variational-parameter-modeling/  
├── README.md  
├── code/  
│   ├── train.py  
│   └── *.py  
├── data/  
│   └── *.csv # (in subfolders, includes datasets used in thesis)  
└──*.pth   # Model files saved/loaded outside main folder  

From the main folder you can run the following functionalities as python modules.


## Training 
You can define the details in the file `train.py` (model, dataset, evaluation). Then run the training as a module.
During Training the best model will get saved to the main folder, for later usage.
```
python -m code.train
```

## Testing Model Performance
Here you can plot the dataset and model performance of your trained models. Select the dataset and the model that you want to evaluate in the file `test_model.py`.
```
python -m code.test_model
```

## Generating Datasets
Define MAX_ATOMS, AMOUNT and the geometry (structure) of molecules you want to use, which will produces different outcomes (dataset quality, learnability of model). The Dataset is the main driver for model performance.
```
python -m code.generate_dataset
```