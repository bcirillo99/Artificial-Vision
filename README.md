# Artificial-Vision

## Set-up
### Create a conda environment
First create a new conda environment and activate it
```
conda create --name cenv python=3.9
conda activate cenv
```
### GPU setup
Then install CUDA and cuDNN with conda
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```
Configure the system paths
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
### Install TensorFlow
Install tensorflow
```
pip install --upgrade pip
pip install tensorflow
```
If by using the program receive the following error "ImportError: cannot import name 'dtensor' from 'tensorflow.compat.v2.experimental'" downgrad keras to:
```
pip install keras==2.6.*
```
### Install requirements
```
pip install -r requirements.txt
```