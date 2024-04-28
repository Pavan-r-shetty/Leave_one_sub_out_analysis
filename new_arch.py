import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as kb
import ml_util_4_head
from scipy import signal
import os

import pandas as pd

import importlib
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError, MeanSquaredError

# Leave-One-Subject-Out (LOSO) Cross-Validation
subjects = [ 2, 3, 4, 5, 6, 7, 8, 9, 10]  # List of all subjects
# subjects = [3, 4]  # List of all subjects

sides = ['LEFT', 'RIGHT']
trial_nums = [1,2,3,4,5,6,7,12]
# trial_nums = [1]

root_folder = "latest_dataset_for_4_head"

import tensorflow as tf


def construct_model_2023v2(window_size,
                                      filter_sizes,
                                      kernel_sizes,
                                      dilations,
                                      num_channels=8,
                                      batch_norm_insertion_pts=[2],
                                      sp_dense_sizes=[20, 10],
                                      ss_dense_sizes=[20, 10],
                                      v_dense_sizes=[20, 10],
                                      r_dense_sizes=[20, 10],
                                      do_fix_input_dim=False):
    if len(filter_sizes) != len(kernel_sizes)+1:
        raise ValueError(
            'Must provide one more filter size than kernel size--last kernel size is calculated')
    current_output_size = window_size  # Track for final conv layer

    #Use None in dim 0 to allow variable input length.
    #Use window_size to fix size--helpful for debugging dimensions
    if do_fix_input_dim:
        input_layer = tf.keras.layers.Input(
            shape=(window_size, num_channels), name='my_input_layer')
    else:
        input_layer = tf.keras.layers.Input(
            shape=(None, num_channels), name='my_input_layer')

    # outputs = []
    z = input_layer
    # Conv1D for each head
    for layer_idx in range(len(kernel_sizes)):
        z = tf.keras.layers.Conv1D(filters=filter_sizes[layer_idx], kernel_size=kernel_sizes[layer_idx],
                                    dilation_rate=dilations[layer_idx], activation='relu')(z)
        if layer_idx in batch_norm_insertion_pts:
            z = tf.keras.layers.BatchNormalization()(z)
        current_output_size = current_output_size - \
            dilations[layer_idx]*kernel_sizes[layer_idx] + dilations[layer_idx]
    if current_output_size < 1:
        raise ValueError('layers shrink the cnn too much')
    else:
        print('adding final conv layer of kernel size: ', current_output_size)
        z = tf.keras.layers.Conv1D(
            filters=filter_sizes[-1], kernel_size=current_output_size, activation='relu')(z)
    for num_neurons in sp_dense_sizes:
        z = tf.keras.layers.Dense(num_neurons, activation='relu')(z)
    output_stance_phase = tf.keras.layers.Dense(1, name='stance_phase_output')(z)    


    current_output_size = window_size      
    z = input_layer
    # Conv1D for each head
    for layer_idx in range(len(kernel_sizes)):
        z = tf.keras.layers.Conv1D(filters=filter_sizes[layer_idx], kernel_size=kernel_sizes[layer_idx],
                                    dilation_rate=dilations[layer_idx], activation='relu')(z)
        if layer_idx in batch_norm_insertion_pts:
            z = tf.keras.layers.BatchNormalization()(z)
        current_output_size = current_output_size - \
            dilations[layer_idx]*kernel_sizes[layer_idx] + dilations[layer_idx]
    if current_output_size < 1:
        raise ValueError('layers shrink the cnn too much')
    else:
        print('adding final conv layer of kernel size: ', current_output_size)
        z = tf.keras.layers.Conv1D(
            filters=filter_sizes[-1], kernel_size=current_output_size, activation='relu')(z)
    for num_neurons in ss_dense_sizes:
        z = tf.keras.layers.Dense(num_neurons, activation='relu')(z)
    output_stance_swing = tf.keras.layers.Dense(
        1, activation='sigmoid', name='stance_swing_output')(z)
    
    current_output_size = window_size
    z = input_layer
    # Conv1D for each head
    for layer_idx in range(len(kernel_sizes)):
        z = tf.keras.layers.Conv1D(filters=filter_sizes[layer_idx], kernel_size=kernel_sizes[layer_idx],
                                    dilation_rate=dilations[layer_idx], activation='relu')(z)
        if layer_idx in batch_norm_insertion_pts:
            z = tf.keras.layers.BatchNormalization()(z)
        current_output_size = current_output_size - \
            dilations[layer_idx]*kernel_sizes[layer_idx] + dilations[layer_idx]
    if current_output_size < 1:
        raise ValueError('layers shrink the cnn too much')
    else:
        print('adding final conv layer of kernel size: ', current_output_size)
        z = tf.keras.layers.Conv1D(
            filters=filter_sizes[-1], kernel_size=current_output_size, activation='relu')(z)
    for num_neurons in v_dense_sizes:
        z = tf.keras.layers.Dense(num_neurons, activation='relu')(z)
    velocity = tf.keras.layers.Dense(1, name='velocity_output')(z) 

    z = input_layer
    current_output_size = window_size
    # Conv1D for each head
    for layer_idx in range(len(kernel_sizes)):
        z = tf.keras.layers.Conv1D(filters=filter_sizes[layer_idx], kernel_size=kernel_sizes[layer_idx],
                                    dilation_rate=dilations[layer_idx], activation='relu')(z)
        if layer_idx in batch_norm_insertion_pts:
            z = tf.keras.layers.BatchNormalization()(z)
        current_output_size = current_output_size - \
            dilations[layer_idx]*kernel_sizes[layer_idx] + dilations[layer_idx]
    if current_output_size < 1:
        raise ValueError('layers shrink the cnn too much')
    else:
        print('adding final conv layer of kernel size: ', current_output_size)
        z = tf.keras.layers.Conv1D(
            filters=filter_sizes[-1], kernel_size=current_output_size, activation='relu')(z)
    for num_neurons in r_dense_sizes:
        z = tf.keras.layers.Dense(num_neurons, activation='relu')(z)
    ramp = tf.keras.layers.Dense(1, name='ramp_output')(z)  

    model = tf.keras.Model(inputs=[input_layer], outputs=[
    output_stance_phase, output_stance_swing, velocity, ramp])

    return model

            



def systematic_sample(data, keep=1, total=5):
    """
    Reduces the data by keeping a specific number of instances for every group.
    
    Args:
    data (numpy array): The dataset to be reduced.
    keep (int): Number of instances to keep in each group.
    total (int): Total number of instances in each group.
    
    Returns:
    numpy array: The reduced dataset.
    """
    indices = np.arange(len(data))
    selected_indices = indices[::total][:keep * (len(indices) // total)]
    return data[selected_indices]





subject_wise_mean_rmse = {}
subject_wise_valid_loss = {}


for test_subject in subjects:
    print(f"Training with subject {test_subject} as test subject...")

    # ... (same setup as in your original code)
    subjects_to_train_with = [subj for subj in subjects if subj != test_subject]
    window_size = 200

    num_snapshots_in_sequence = 300
    sequence_len = num_snapshots_in_sequence + window_size - 1

    
    training_instances = np.empty(shape=[0, sequence_len, 12], dtype=np.float32)
    files_to_train_with = ml_util_4_head.get_files_to_use(root_folder, subjects_to_train_with, sides, trial_nums)

    # ... (load data and process it as in your original code)

    for myfile in files_to_train_with:
        data = ml_util_4_head.load_file(myfile)
        ss_col = data[:,-4]
        ss_col = ml_util_4_head.manipulate_ss_col(ss_col)
        data[:,-4] = ss_col

        # MAX 
        ramp_col = data[:,-2]
        # tolerance = 0.005  # Define a tolerance range of 0.2 +/- 0.01
        # ramp_col[np.abs(ss_col - 0.2) > tolerance] = -100
        ramp_col[ss_col==0]=-100
        data[:,-2] = ramp_col

        # print(myfile)
        num_rows, num_cols = data.shape
        # print(num_cols)
        num_rows_to_drop = num_rows % sequence_len

        data = data[0:-num_rows_to_drop]
        new_num_rows, num_cols = data.shape
        num_sequences = new_num_rows/sequence_len
        new_data_shape = (int(num_sequences), sequence_len, num_cols)
        new_instances = data.reshape(new_data_shape)

        training_instances = np.append(training_instances, new_instances, axis=0)


    # Make 0 -> -1

    
    # Print shape before sampling
    print("Shape before sampling:", training_instances.shape)

    # Sample 50% of the data

    training_instances = systematic_sample(training_instances, keep=1, total=2)
    print("Shape after 50% sampling:", training_instances.shape)

    shuffled_training_instances = tf.random.shuffle(training_instances) 
    num_channels = 8
    x = shuffled_training_instances[:, :, :num_channels]
    y_v = shuffled_training_instances[:, window_size-1:,-1]
    y_r = shuffled_training_instances[:, window_size-1:,-2]
    y_sp = shuffled_training_instances[:, window_size-1:,-4]
    y_ss = shuffled_training_instances[:, window_size-1:,-3]


    split_fraction = 0.8
    split_num = int(np.rint(split_fraction*x.shape[0]))
    x_train = x[:split_num,:,:]
    x_train = tf.cast(x_train, tf.float32)
    y_sp_train = y_sp[:split_num,:]
    y_ss_train = y_ss[:split_num,:]
    y_r_train = y_r[:split_num,:]
    y_v_train = y_v[:split_num,:]

    x_valid = x[split_num:,:,:]
    y_sp_valid = y_sp[split_num:,:]
    y_ss_valid = y_ss[split_num:,:]
    y_r_valid = y_r[split_num:,:]
    y_v_valid = y_v[split_num:,:]





    do_fix_input_dim=False

    tf.keras.backend.clear_session()
    
    model = construct_model_2023v2(window_size=window_size,
    filter_sizes=[44,44,44,44], kernel_sizes=[8,8,8], dilations=[6,6,6], num_channels=8,
    batch_norm_insertion_pts=[0,1], sp_dense_sizes=[23,23,23,23], ss_dense_sizes=[23,23,23,23], v_dense_sizes=[23,23,23,23], r_dense_sizes=[23,23,23,23], do_fix_input_dim=do_fix_input_dim)

# 8,0.3406950941516293,"{'filter_sizes': (44,), 'kernel_sizes': (8,), 'cnn_depth': (4,), 'dilations': (6,), 'dense_depth': (4,), 'dense_width': (23,)}"



    print(model.summary())
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=[ml_util_4_head.custom_loss, 'binary_crossentropy', ml_util_4_head.custom_loss, ml_util_4_head.custom_loss_for_ramp], loss_weights=[4,1,0.3,0.1], optimizer=optimizer)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)
    mc = tf.keras.callbacks.ModelCheckpoint(f'4_head_model_leave_one_sub_out/test_subject_{test_subject}.h5', monitor='val_loss',
                                            mode='min', save_best_only=True, verbose=1)   
    history = model.fit(x=x_train, y=[y_sp_train, y_ss_train, y_v_train, y_r_train], 
        batch_size=32, epochs=50, validation_data=(x_valid,[y_sp_valid, y_ss_valid, y_v_valid, y_r_valid]), 
        callbacks=[es, mc], verbose=1)
    
    # history_dict = history.history

    subject_wise_valid_loss[test_subject] = {


        'epoch': range(1, len(history.history['loss']) + 1),
        'val_loss_sp': history.history['val_stance_phase_output_loss'],
        'val_loss_ss': history.history['val_stance_swing_output_loss'],
        'val_loss_v': history.history['val_velocity_output_loss'],
        'val_loss_r': history.history['val_ramp_output_loss']
    }

    df_loss = pd.DataFrame.from_dict(subject_wise_valid_loss[test_subject])
    # Construct the directory path
    directory_path = 'loss'

    try:
        os.makedirs(directory_path)

    except:
        pass

    csv_file_path = os.path.join(directory_path, f"{test_subject}.csv")
    # Save to CSV
    df_loss.to_csv(csv_file_path, index=False)
    
    



    rmses = []
    rmses_vel = []
    rmses_isp = []
    rmses_ramp = []



    num_channels = 8

    files_to_test_with = ml_util_4_head.get_files_to_use(root_folder, [test_subject], sides, trial_nums)
    for myfile in files_to_test_with:
        data = ml_util_4_head.load_file(myfile)

        ss_col_test = data[window_size-1:,-4]


        # MAX 
        ramp_col_test = data[window_size-1:,-2]
        ramp_col_test[ss_col_test==0]=-100
        data[window_size-1:,-2] = ramp_col_test
    


        x_test = tf.expand_dims(data[:,:num_channels], axis=0)
        x_test=tf.cast(x_test, dtype = tf.float32)
        y_sp_test = data[window_size-1:,-4]
        y_ss_test = data[window_size-1:,-3]
        y_v_test = data[window_size-1:,-1]
        y_r_test = data[window_size-1:,-2]


        model_outputs = model.predict(x=x_test)    
        y_sp_predict, y_ss_predict, y_v_predict, y_r_predict = tf.squeeze(model_outputs)

        if  test_subject==1:
            # Adjust the slicing indices based on the condition and filter where y_sp_test != -1
            valid_indices = (y_sp_test[13000:-900] != -1) & (y_ss_test[13000:-900] == True)
            err = y_sp_predict[13000:-900][valid_indices] - y_sp_test[13000:-900][valid_indices]
        else:
            # Keep the original slicing if test_subject is not 1 and filter where y_sp_test != -1
            valid_indices = (y_sp_test[7000:-900] != -1) & (y_ss_test[7000:-900] == True)
            err = y_sp_predict[7000:-900][valid_indices] - y_sp_test[7000:-900][valid_indices]


        rmse = np.sqrt(np.mean(np.square(err))) 
        rmses.append(rmse)

        if test_subject==1:

            err_isp = y_ss_predict[13000:-900]-y_ss_test[13000:-900]
        else:
            err_isp = y_ss_predict[7000:-900]-y_ss_test[7000:-900]
        
            # Mask nan values
            err_isp = err_isp[~np.isnan(err_isp)]
            rmse_isp = np.sqrt(np.mean(np.square(err_isp))) 
            rmses_isp.append(rmse_isp)



        err_vel = y_v_predict - y_v_test
        err_vel = err_vel[y_v_test!=-1].numpy()
        rmse_vel = np.sqrt(np.mean(np.square(err_vel)))
        rmses_vel.append(rmse_vel)


        # Mask where y_r_test is not -100
        mask = y_r_test != -100

        # Apply mask to calculate RMSE
        err_ramp = y_r_predict[mask] - y_r_test[mask]
        rmse_ramp = np.sqrt(np.mean(np.square(err_ramp)))
        rmses_ramp.append(rmse_ramp)


    mean_rmses_sp = np.mean(rmses)
    mean_rmses_isp = np.mean(rmses_isp)
    mean_rmses_vel = np.mean(rmses_vel)
    mean_rmses_ramp = np.mean(rmses_ramp)

    subject_wise_mean_rmse[test_subject] = {
    'mean_rmses_sp': mean_rmses_sp,
    'mean_rmses_isp': mean_rmses_isp,
    'mean_rmses_vel': mean_rmses_vel,
    'mean_rmses_ramp': mean_rmses_ramp}




# Convert dictionary to DataFrame for easy CSV writing
df = pd.DataFrame.from_dict(subject_wise_mean_rmse, orient='index')

# Save to CSV
df.to_csv('subject_wise_mean_rmse.csv')

# Assuming subject_wise_mean_rmse is your dictionary with subject-wise mean RMSEs

# Initialize variables to sum up the means for each metric
total_mean_rmses_sp = 0
total_mean_rmses_isp = 0
total_mean_rmses_vel = 0
total_mean_rmses_ramp = 0

num_subjects = len(subject_wise_mean_rmse)

# Sum up the means for each metric
for values in subject_wise_mean_rmse.values():
    total_mean_rmses_sp += values['mean_rmses_sp']
    total_mean_rmses_isp += values['mean_rmses_isp']
    total_mean_rmses_vel += values['mean_rmses_vel']
    total_mean_rmses_ramp += values['mean_rmses_ramp']

# Calculate the overall means
overall_mean_rmses_sp = total_mean_rmses_sp / num_subjects
overall_mean_rmses_isp = total_mean_rmses_isp / num_subjects
overall_mean_rmses_vel = total_mean_rmses_vel / num_subjects
overall_mean_rmses_ramp = total_mean_rmses_ramp / num_subjects

# You can print these values or store them as needed
print(f'Overall Mean RMSE SP: {overall_mean_rmses_sp}')
print(f'Overall Mean RMSE ISP: {overall_mean_rmses_isp}')
print(f'Overall Mean RMSE VEL: {overall_mean_rmses_vel}')
print(f'Overall Mean RMSE RAMP: {overall_mean_rmses_ramp}')

# Calculate overall mean for each metric
overall_means = df.mean()

# Optionally, convert overall_means to DataFrame for saving to CSV
overall_means_df = pd.DataFrame(overall_means, columns=['Overall Mean']).transpose()

# Save overall means to CSV
overall_means_df.to_csv('overall_mean_rmse.csv', index=False)



