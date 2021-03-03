import numpy as np
import pickle
import os

dataset_directory = './dataset10/'

# create x(train) and y(teacher) dataset list
def create_XY_lists(dataset_ditectory, log_file_name, num_of_samples_to_read):
    list_dataset_filepaths = [] # x-list: path to each file of dataset
    list_parameters = [] # y-list: rho vp vs eps gamma delta 
    file_path_y = os.path.join(dataset_ditectory, log_file_name)
    with open(file_path_y, 'r') as file_read_y:
        for i in range(num_of_samples_to_read+1):
            data_y = file_read_y.readline()
            if i == 0:
                if int(data_y) != num_of_samples_to_read:
                    print('achtung! smth wrong with dataset size and read samples')
            else:
                data_y = data_y.split('\t')
                list_parameters.append( [ float(data_y[1]), float(data_y[2]), float(data_y[3]), float(data_y[4]), float(data_y[5]), float(data_y[6]) ] )
                filename_r = dataset_ditectory + 'seism' + str(data_y[0]) + '.bin'
                list_dataset_filepaths.append(filename_r)
    return list_dataset_filepaths, list_parameters

list_dataset_filepaths, list_parameters = create_XY_lists(dataset_directory, 'dataset_log_file_TD.txt', 10)

assert len(list_dataset_filepaths) == len(list_parameters)

num_of_rec = 13

filename_r_time = dataset_directory + 'seism_time.bin'
time_full = np.fromfile (filename_r_time)
num_of_timesteps = len(time_full)


def read_seismogram_monomode(filepath):
    seism_read = np.fromfile(filepath)
    seismogram = np.zeros((num_of_rec, num_of_timesteps))
    for irec in range(num_of_rec):
        seismogram[irec, :] = seism_read[irec*num_of_timesteps:(irec+1)*num_of_timesteps]
    if seismogram.shape[0]*seismogram.shape[1]*8 != os.path.getsize(filepath):
        print('achtung! smth wrong with reading')
    return seismogram


def clear_lists_from_nan_samples(list_dataset_filepaths, list_parameters):
    list_dataset_filepaths_new = []
    list_parameters_new = []
    count_nan_seismograms = 0
    max_seism_value = 0 
    for i, item in enumerate(list_dataset_filepaths):
        print(i, 'of', len(list_dataset_filepaths))
        seismogram = read_seismogram_monomode(item)
        if (np.any(np.isnan(seismogram[:,:])) == True) or np.any(abs(seismogram[:,:])>1200):
            count_nan_seismograms+=1
            print('nan bitch is detected!', count_nan_seismograms)
            print(item)
            print('-------')
        else:
            max_seism_value_temp = np.max(abs(seismogram[:,:]))
            if max_seism_value_temp > max_seism_value:
                max_seism_value = max_seism_value_temp
            list_dataset_filepaths_new.append(item)
            list_parameters_new.append(list_parameters[i])
    print('|||||| num_of_nan_samples:', count_nan_seismograms, '||||||')
    return list_dataset_filepaths_new, list_parameters_new, max_seism_value


print('please, wait! clearing NaN seismograms...')
list_dataset_filepaths, list_parameters, max_seism_value = clear_lists_from_nan_samples(list_dataset_filepaths, list_parameters)
print('clearing NaN seismograms done!')

# exclude seismograms, where vs<1500 (they are only in egd1200)
list_parameters_arr = np.array(list_parameters)
numbers = np.where(list_parameters_arr[:,2] < 1500)
print(numbers)
numbers = np.array(numbers)
numbers = numbers[0]
for inum in numbers:
    list_parameters[inum] = list_parameters[inum-1]
    list_dataset_filepaths[inum] = list_dataset_filepaths[inum-1]

with open('max_seism_value', 'wb') as fp:
    pickle.dump(max_seism_value, fp)
with open('list_dataset_filepaths', 'wb') as fp:
    pickle.dump(list_dataset_filepaths, fp)
with open('list_parameters', 'wb') as fp:
    pickle.dump(list_parameters, fp)
