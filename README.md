## Installation
Follow the steps below to set up your environment.  
For Qulacs to work with multiple cores do these extra steps. Otherwise just do the pip install's.

sudo apt install python3.12-venv

python3.12 -m venv myvenv  
source myvenv/bin/activate

sudo apt install gcc-14 g++-14  
sudo apt install libboost-all-dev

export C_COMPILER=gcc-14  
export CXX_COMPILER=g++-14  
export QULACS_OPT_FLAGS="-mtune=native"  
pip install git+https://github.com/qulacs/qulacs.git  

pip install tequila-basic  
pip install pyscf  
pip install torch-cluster  
pip install torch_geometric  

## Project structure
variational-parameter-modeling/  
├── README.md  
├── code/  
│   ├── train.py  
│   └── *.py  
├── data/  
│   └── *.csv # (in subfolders, includes datasets used in thesis)  
└──*.pth   # Model files saved/loaded outside main folder  



## Training 
python -m train

## Testing Model Performance
Here u can plot the dataset and model performance of ur trained models. Select the dataset and model u want to evaluate
python -m test_model

## Generating Datasets
Define MAX_ATOMS, AMOUNT of molecules u want to use.
Changing the do_minimize can produce a lot of different outcomes (dataset quality, learnability of model).
The Dataset is the main driver for model performance.
python -m generate_dataset