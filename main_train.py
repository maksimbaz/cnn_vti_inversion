# This script is used for the neural network construction and training.

import numpy as np
import pickle
import os
import random
import time
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.utils import multi_gpu_model, plot_model
from keras.callbacks import LambdaCallback
from keras import regularizers, optimizers

train_output_folder = './train_output/'

# figure fonts 
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 16}
matplotlib.rc('font', **font)

# load lists
with open('list_dataset_filepaths', 'rb') as fp:
    list_dataset_filepaths = pickle.load(fp) # list of the paths to samples form the dataset
with open('list_parameters', 'rb') as fp:
    list_parameters= pickle.load(fp) # list of parameters (neural network output)
# load coefficient shot gather (input) normalization
with open('max_seism_value', 'rb') as fp:
    max_seism_value = pickle.load(fp) # this parameter is used for the input dataset normalization

datset_size = len(list_dataset_filepaths)
assert len(list_dataset_filepaths) == len(list_parameters)
print('datset size:', datset_size)

# reading block
# function for the dataset reading from file 
filename_r_time = './dataset10/seism_time.bin' # each sample (shot gather) has the same time size, which is saved in this file
time_full = np.fromfile (filename_r_time)
mean_timestep = len (time_full)

num_of_rec_in_group = 21
epoch_number = 0 # the value changes during training process
def read_x_data(list_dataset_filepaths):
#     np.random.seed()
    global epoch_number, time_full, mean_timestep
    seismogram = np.zeros((len(list_dataset_filepaths), num_of_rec_in_group, mean_timestep))
    amp_map = np.zeros((num_of_rec_in_group, mean_timestep)) # amplitude map (moving average) fot noise adding
    N = 220 # width of the window usded for the amplitude map calculation
    gain = np.exp(-4e5*time_full[:]**2)*1e2/(epoch_number+1)+1
    gain /= max_seism_value
    for ifile, file_path in enumerate(list_dataset_filepaths):
        filename_r = file_path
        seism_read = np.fromfile(filename_r)
        for irec in range(num_of_rec_in_group):
            seismogram[ifile, irec, :] = seism_read[irec*mean_timestep:(irec+1)*mean_timestep]*gain
            amp_map[irec, :] = np.convolve(abs(seismogram[ifile, irec, :]), np.ones((N))/N, mode='same')
        noise = np.random.rand(num_of_rec_in_group, mean_timestep)/5-0.1
        seismogram[ifile, :, :] = seismogram[ifile, :, :] + noise*amp_map[:,:]
        if seismogram.shape[1]*seismogram.shape[2]*8 != os.path.getsize(filename_r):
            print('error! smth wrong with reading')
    return seismogram.reshape(seismogram.shape[0], seismogram.shape[1],seismogram.shape[2], 1) #channels last

# funtion used for the Keras fit_generator
def dataset_loader(list_dataset_filepaths, list_parameters, batch_size):
    L=len(list_dataset_filepaths)
    #this line is just to make the generator infinite, keras needs that
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            x_dataset = read_x_data(list_dataset_filepaths[batch_start:limit])
            y_dataset = np.array(list_parameters[batch_start:limit])
            batch_start += batch_size
            batch_end += batch_size
            yield (x_dataset, y_dataset) #a tuple with two numpy arrays with batch_size samples

# output array (desired parameters) has to be normalized
def normalize_list_parameters(list_parameters):
    if datset_size != len(list_parameters):
        print('error! smth wrong with dataset size')
    list_parameters_numpy = np.asarray(list_parameters, dtype=np.float32)
    rho_max = np.max(list_parameters_numpy[:,0])
    vp_max = np.max(list_parameters_numpy[:,1])
    vs_max = np.max(list_parameters_numpy[:,2])
    eps_max = np.max(list_parameters_numpy[:,3])
    gamma_max = np.max(list_parameters_numpy[:,4])
    delta_max = np.max(list_parameters_numpy[:,5])
    list_parameters_numpy[:,0] /= rho_max
    list_parameters_numpy[:,1] /= vp_max
    list_parameters_numpy[:,2] /= vs_max
    list_parameters_numpy[:,3] /= eps_max
    list_parameters_numpy[:,4] /= gamma_max
    list_parameters_numpy[:,5] /= delta_max
    rho_mean = np.mean(list_parameters_numpy[:,0])
    vp_mean = np.mean(list_parameters_numpy[:,1])
    vs_mean = np.mean(list_parameters_numpy[:,2])
    eps_mean = np.mean(list_parameters_numpy[:,3])
    gamma_mean = np.mean(list_parameters_numpy[:,4])
    delta_mean = np.mean(list_parameters_numpy[:,5])
    list_parameters_numpy[:,0] -= rho_mean
    list_parameters_numpy[:,1] -= vp_mean
    list_parameters_numpy[:,2] -= vs_mean
    list_parameters_numpy[:,3] -= eps_mean
    list_parameters_numpy[:,4] -= gamma_mean
    list_parameters_numpy[:,5] -= delta_mean
    list_parameters_normalized = [] 
    for i in range(datset_size):
        list_parameters_normalized.append( [ list_parameters_numpy[i,0], list_parameters_numpy[i,1], list_parameters_numpy[i,2], list_parameters_numpy[i,3], list_parameters_numpy[i,4], list_parameters_numpy[i,5] ] )
    return list_parameters_normalized, rho_max, vp_max, vs_max, eps_max, gamma_max, delta_max, rho_mean, vp_mean, vs_mean, eps_mean, gamma_mean, delta_mean

# save normalization coefficients to the file
# we will need when using trained neural network
list_parameters, rho_max, vp_max, vs_max, eps_max, gamma_max, delta_max, rho_mean, vp_mean, vs_mean, eps_mean, gamma_mean, delta_mean = normalize_list_parameters(list_parameters)
normalization_param_list = []
normalization_param_list.append('rho_max='+'{}'.format(rho_max)+'\n')
normalization_param_list.append('vp_max='+'{}'.format(vp_max)+'\n')
normalization_param_list.append('vs_max='+'{}'.format(vs_max)+'\n')
normalization_param_list.append('eps_max='+'{}'.format(eps_max)+'\n')
normalization_param_list.append('gamma_max='+'{}'.format(gamma_max)+'\n')
normalization_param_list.append('delta_max='+'{}'.format(delta_max)+'\n')
normalization_param_list.append('rho_mean='+'{}'.format(rho_mean)+'\n')
normalization_param_list.append('vp_mean='+'{}'.format(vp_mean)+'\n')
normalization_param_list.append('vs_mean='+'{}'.format(vs_mean)+'\n')
normalization_param_list.append('eps_mean='+'{}'.format(eps_mean)+'\n')
normalization_param_list.append('gamma_mean='+'{}'.format(gamma_mean)+'\n')
normalization_param_list.append('delta_mean='+'{}'.format(delta_mean)+'\n')
with open(train_output_folder + "normalization_param_list.txt", "w") as f_write:
    for lineWrite in normalization_param_list:
        f_write.write(lineWrite)

# split data to train and validation subsets
np.random.seed()
list_filepaths_train, list_filepaths_valid, true_parameters_train, true_parameters_valid = train_test_split(list_dataset_filepaths, list_parameters, test_size=0.1)
print('train dataset size:', len(list_filepaths_train))
print('validation dataset size:', len(list_filepaths_valid))

# check shapes of the x and y dataset
x_dataset_example = read_x_data(list_filepaths_train[9:10])
y_dataset_example = np.array(true_parameters_train[9:10])
print ('x_dataset shape (batch(=1), num_of_rec_in_group, timesteps, channels(=1)):', x_dataset_example.shape)
print ('y_dataset shape (batch(=1), dim[vp ,vs]):', y_dataset_example.shape)

# initialize the convolutional neural network model
with tf.device('/cpu:0'):
    model = Sequential()
    model.add(Conv2D(filters=50, input_shape=(21,5500,1), kernel_size=(6,6), strides=(1,1), padding='same', activation='relu'))
    model.add(Conv2D(filters=50, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'))
    model.add(Conv2D(filters=50, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(Conv2D(filters=75, kernel_size=(3,3), strides=(1,2), padding='valid', activation='relu'))    
    model.add(Conv2D(filters=75, kernel_size=(3,3), strides=(1,2), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=100, kernel_size=(2,2), strides=(1,2), padding='same', activation='relu'))
    model.add(Conv2D(filters=100, kernel_size=(2,2), strides=(1,2), padding='valid', activation='relu'))
    model.add(Conv2D(filters=100, kernel_size=(2,2), strides=(1,2), padding='valid', activation='relu'))
    model.add(Flatten())
    model.add( Dense(2500, activation='relu') )
    model.add(Dropout(0.3))
    model.add( Dense(750, activation='relu') )
    model.add( Dense(200, activation='relu') )
    model.add( Dense(6) )
    model.add(Activation('linear'))
    print('model initialized')
    model.summary()

# or load pretrained model
# model = load_model(train_output_folder + 'model.h5')

# compile model
batch_size=8;
nb_epoch=250;
print('nb_epoch:', nb_epoch)
print('steps_per_epoch:', np.ceil(datset_size/batch_size))
print('validation_steps:', np.ceil(len(list_filepaths_valid)/batch_size))
parallel_model = multi_gpu_model(model, gpus=4)
parallel_model.compile(loss='mean_squared_error', optimizer=optimizers.Adadelta())

# functions which gets epoch number during training process
def get_epoch(epoch):
    global epoch_number
    epoch_number = epoch

GetEpoch_callback = LambdaCallback(on_epoch_begin=lambda epoch,logs: get_epoch(epoch))

# training
start_time = time.time()
history=parallel_model.fit_generator(dataset_loader(list_filepaths_train, true_parameters_train, batch_size), 
                                     steps_per_epoch=np.ceil(datset_size/batch_size), epochs=nb_epoch, verbose=1, 
                                     validation_data=dataset_loader(list_filepaths_valid, true_parameters_valid, batch_size), validation_steps=np.ceil(len(list_filepaths_valid)/batch_size), callbacks=[GetEpoch_callback])
model.save(train_output_folder + 'model.h5')
done_time = time.time()
elapsed_time = done_time - start_time
print('elapsed time:', elapsed_time)

# plot training and validation loss function values
print(history.history.keys())
# summarize history for loss
fig=plt.figure(figsize=(12, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(history.history['loss'][3:], linewidth=2)
plt.plot(history.history['val_loss'][3:],'--', linewidth=2)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

print('train_loss for the last training epoch:', history.history['loss'][-1])
print('valid_loss for the last training epoch:', history.history['val_loss'][-1])

# check predictions for the validation dataset
predictions_valid = parallel_model.predict_generator(dataset_loader(list_filepaths_valid, true_parameters_valid, batch_size), steps=np.ceil(len(list_filepaths_valid)/batch_size), verbose=1)

# put here normalization coefficients, if you uploaded complete model
# rho_max=3839.0
# vp_max=6489.0
# vs_max=3999.0
# eps_max=0.25956490635871887
# gamma_max=0.24065768718719482
# delta_max=0.2931036949157715
# rho_mean=0.6626886129379272
# vp_mean=0.6760891079902649
# vs_mean=0.7519354820251465
# eps_mean=0.30609753727912903
# gamma_mean=0.19618447124958038
# delta_mean=0.2660829424858093

# convert predictions to real values
for i in range(len(predictions_valid)):
    #unMEAN
    predictions_valid[i][0] += rho_mean
    predictions_valid[i][1] += vp_mean
    predictions_valid[i][2] += vs_mean
    predictions_valid[i][3] += eps_mean
    predictions_valid[i][4] += gamma_mean
    predictions_valid[i][5] += delta_mean
    true_parameters_valid[i][0] += rho_mean
    true_parameters_valid[i][1] += vp_mean
    true_parameters_valid[i][2] += vs_mean
    true_parameters_valid[i][3] += eps_mean
    true_parameters_valid[i][4] += gamma_mean
    true_parameters_valid[i][5] += delta_mean
    #unMAX
    predictions_valid[i][0] *= rho_max
    predictions_valid[i][1] *= vp_max
    predictions_valid[i][2] *= vs_max
    predictions_valid[i][3] *= eps_max
    predictions_valid[i][4] *= gamma_max
    predictions_valid[i][5] *= delta_max
    true_parameters_valid[i][0] *= rho_max
    true_parameters_valid[i][1] *= vp_max
    true_parameters_valid[i][2] *= vs_max
    true_parameters_valid[i][3] *= eps_max
    true_parameters_valid[i][4] *= gamma_max
    true_parameters_valid[i][5] *= delta_max

# plot predictions vs true values
true_parameters_valid = np.array(true_parameters_valid)
predictions_valid = np.array(predictions_valid)
#one_png
fig_res, ax_res = plt.subplots(6,1)
fig_res.set_size_inches(10, 50)
ax_res[0].set(xlabel='Reference CNN output', ylabel= 'Calculated CNN output', title=r'$\rho, kg/m^3$')
ax_res[0].scatter(true_parameters_valid[:,0], predictions_valid[:,0], facecolors='none', edgecolors='b')
ax_res[0].locator_params(nbins=6)
ax_res[1].set(xlabel='Reference CNN output', ylabel= 'Calculated CNN output', title=r'$V_{p_0}, m/s$')
ax_res[1].scatter(true_parameters_valid[:,1], predictions_valid[:,1], facecolors='none', edgecolors='b')
ax_res[1].locator_params(nbins=6)
ax_res[2].set(xlabel='Reference CNN output', ylabel= 'Calculated CNN output', title=r'$V_{s_0}, m/s$')
ax_res[2].scatter(true_parameters_valid[:,2], predictions_valid[:,2], facecolors='none', edgecolors='b')
ax_res[2].locator_params(nbins=6)
ax_res[3].set(xlabel='Reference CNN output', ylabel= 'Calculated CNN output', title=r'$\varepsilon$')
ax_res[3].scatter(true_parameters_valid[:,3], predictions_valid[:,3], facecolors='none', edgecolors='b')
ax_res[3].locator_params(nbins=6)
ax_res[4].set(xlabel='Reference CNN output', ylabel= 'Calculated CNN output', title=r'$\gamma$')
ax_res[4].scatter(true_parameters_valid[:,4], predictions_valid[:,4], facecolors='none', edgecolors='b')
ax_res[4].locator_params(nbins=6)
ax_res[5].set(xlabel='Reference CNN output', ylabel= 'Calculated CNN output', title=r'$\delta$')
ax_res[5].scatter(true_parameters_valid[:,5], predictions_valid[:,5], facecolors='none', edgecolors='b')
ax_res[5].locator_params(nbins=6)
plt.savefig(train_output_folder + 'predictions_all.png')