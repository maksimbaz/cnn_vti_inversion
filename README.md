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
The folder contains 10 shot gather. This data can not be used for a proper neural network training but can be used for the testing. 

```
dataset10/dataset_log_file_TD.txt
```
text file that contains parameters of the medium related to the shot gathers, where columns denotes respectevly:
file_number rho vp vs eps gamma delta

```
main_train.py
```
Script is devoted to the neural network construction and training.  

```
main_test.py
```
Script is devoted to the neural network testing.  

```
form_lists.py
```
The script is used for the list construction the inputs and outputs for the neural netowrk. 

```
make_plot.ipynb
```
Script can be used to visualize shot gathers from Dataset4000 folder.
