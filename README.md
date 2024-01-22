# Multi-expert system for Age estimation

The repo contains code and papers for the university exam _Artificial Vision_. The exam was in the form of a competition (rules and evaluation metrics are in the pdf _contest.pdf_) whose goal was to develop methods based on modern Deep Convolutional Neural Networks (DCNN) for age estimation from images of faces (methods were restricted to a single neural network or small ensembles consisting of at most 3 classifiers).
What we did first was to find the best classifier among different models, and then build a multi-expert system, composed by an ensamble of three of these "weak" classifiers; we used an _RUSBoost_ algorithm to train it and find the best decision rule, involving the three of them, according to the "degree of expertise" of each one. The pdf file *Report_GTA.pdf* explains the various decisions and analyses that led to the implementation of this system.

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
