# cnn_vti_inversion
Determination of the elastic parameters of a VTI medium from sonic logging data using deep learning

************************
Maksim Bazulin (maksim.bazulin@skoltech.ru), Denis Sabitov, Marwan Charara

Codes prepared for Computers and Geosciences journal
************************

## Description

```
dataset10
```
The folder contains 10 shot gathers. This data can not be used for a proper neural network training but can be used for the testing. 


```
dataset10/dataset_log_file_TD.txt
```
The text file that contains parameters of the medium related to the shot gathers, where columns denote respectively:
file_number rho vp vs eps gamma delta

```
main_train.py
```
The script is devoted to the neural network construction and training.  

```
main_test.py
```
The script is devoted to the neural network testing.  

```
form_lists.py
```
The script is used for the list construction for the inputs and outputs for the neural network. 

```
make_plot.ipynb
```
The script can be used to visualize shot gathers from dataset10 folder.

```
train_output
```
This is the folder, where you can find outputs of the `main_train.py` and `main_test.py`. This folder usually contains 'model.h5' - the file of the trained neural network weigths. This file is too large for uploading it, so do not hesitate to contact me if you need a sample of such a file. 
