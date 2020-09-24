# cnn_vti_inversion
Determination of the elastic parameters of a VTI medium from sonic logging data using deep learning

```
dataset10
```
The folder contains 10 shot gather. This data can not be used for a proper neural network training but can be used for the testing. 
dataset_log_file_TD.txt - text file that contains parameters of the medium related to the shot gathers:
number rho vp vs eps gamma delta

```
main_train.ipynb
```
Script is devoted to the neural network construction and training.  

```
main_test.ipynb
```
Script is devoted to the neural network testing.  

```
make_plot.ipynb
```
Script can be used to visualize shot gathers from Dataset4000 folder.

```
form_lists.py
```
The script is used for the list construction the inputs and outputs for the neural netowrk. 

Computer code should provide the following information:

1. A readme.txt file (or equivalent) providing the name of the program, the title of the manuscript along with the author details. This will assist in correctly assigning the program code and associated files to the correct submission.
2. A user manual or instruction guide that provides information on how to use the program.
3. The source code for any programs that have been written.
4. Test data that can be used to assure that the program is working correctly. Test data should not be overly large so that there are problems downloading the program code and data.
5. Output files should also be provided that will allow a user to check if a compiled program is working properly.
6. Executable program code is not encouraged because of difficulty in transmitting .exe files past Virus scanners and the limited life of executable code.
7. All files should be compressed into .zip or .gz format, which will then be placed on the Computers& Geosciences FTP site for download once the manuscript has been accepted and published.
