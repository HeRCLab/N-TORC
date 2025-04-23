import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import plotly
from tensorflow import keras
from sklearn.metrics import mean_squared_error
import random as rn
import optuna
import datetime
from math import isnan
from bisect import bisect

# Check if GPU is available, otherwise throw an exception
# assert tf.test.is_gpu_available()
# assert tf.test.is_built_with_cuda()
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
print('tf.config.list_physical_devices GPU = ', tf.config.list_physical_devices('GPU'))
for device in gpu_devices:
    print('device = ', device)
    tf.config.experimental.set_memory_growth(device, True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print('passed GPU test')


keras.backend.set_learning_phase(0)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED'] = '0'
rn.seed(1702058893996)
np.random.seed(42)
tf.random.set_seed(1234)

def write_to_file(data, filename):
    f = open(filename+".txt", "w")
    f.write(str(data).replace('[','').replace(']',''))
    f.close()
def append_to_file(data, filename):
    f = open(filename + ".txt", "a")
    f.write(str(data))
    f.close()

pin_data_folderpath = 'dataset/pin_location_data/'
acc_data_folderpath = 'dataset/acceleration_signal_data/'


now = datetime.datetime.now()
print()
print('start time = ', now)
print()
IF_USE_SLIDING_WINDOW_DATASET = False
feature_len = 16
NUM_EPOCHS = 250
LSTM_UNITS = 64
BATCH_SIZE = 128
NUM_EPOCHS_FINAL = 45
NUM_EPOCHS = NUM_EPOCHS_FINAL
optuna_n_trials = 650
optuna_n_trials_current = 0
optuna_n_trials_MAX = 360
NUM_CLASSESS = 256
optuna_sampler_tag = 'BoTorchSampler'
out_file_tag = 'results/'+optuna_sampler_tag + '_v1'
result_path = 'results/main_results_'+optuna_sampler_tag
rmse_cost_path = 'results/rmse_cost_'+optuna_sampler_tag

def create_model(n_CNN_layers, CNN_filters_size, n_LSTM_layers, LSTM_units, n_MLP_layers, MLP_units, feature_len):
    print('inside create_model testing')
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(feature_len, 1)))

    # CNN
    kernel_size = 3
    for i in range(n_CNN_layers):
        model.add(tf.keras.layers.Conv1D(filters=CNN_filters_size[i], kernel_size=kernel_size, activation='relu', padding='same',
                                         data_format='channels_last'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))

    # lstm
    for i in range(n_LSTM_layers):
        model.add(tf.keras.layers.LSTM(LSTM_units[i], return_sequences=True, stateful=False))

    # to make suitable for MLP and o/p layer
    model.add(tf.keras.layers.Flatten())

    # MLP
    for i in range(n_MLP_layers):
        model.add(tf.keras.layers.Dense(MLP_units[i], activation="relu"))


    # final
    model.add(tf.keras.layers.Dense(NUM_CLASSESS, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def signaltonoise(sig, pred, dB=True):
    print(sig.shape)
    print(pred.shape)
    noise = sig - pred
    a_sig = np.sqrt(np.mean(np.square(sig)))
    a_noise = np.sqrt(np.mean(np.square(noise)))
    snr = (a_sig/a_noise)**2
    if(not dB):
        return snr
    return 10*np.log10(snr)

def preprocess(acc_path, pin_path, feature_len):
    f = open(acc_path)
    data_acc = f.read().split(', ')
    f.close()

    f = open(pin_path)
    data_pin = f.read().split(', ')
    f.close()

    data_acc = list(map(float, data_acc))
    data_pin = list(map(float, data_pin))

    acc = np.array(data_acc)
    pin = np.array(data_pin)

    for i in range(len(pin)):
        if (isnan(pin[i])):
            pin[i] = pin[i - 1]

    X_std = np.std(acc)
    y_std = np.std(pin)
    acc = acc / X_std
    pin = pin / y_std

    if not IF_USE_SLIDING_WINDOW_DATASET:
        ds = feature_len
        acc = acc[:(acc.size - 1)]
        X = np.reshape(acc[:acc.size // ds * ds], (acc.size // ds, ds))
        pin = pin.tolist()
        list_len = len(pin)
        src = []
        for indx, stepindx in enumerate(range(0, len(pin), ds)):
            if stepindx + ds >= list_len:
                break
            src.append(pin[stepindx + ds])

        y = np.array(src)

        print('0. X = ', X.shape)
        print('0. y = ', y.shape)
        X_to_ret = X
        y_to_ret = y

    else:
        X = acc
        y = pin

        if IF_USE_SLIDING_WINDOW_DATASET:
            X_data = []

            X = X.tolist()
            y = y.tolist()

            acc_data_list = X
            pin_data_list = y
            for index in range(0, len(X) - feature_len):
                src_segment = acc_data_list[index:index + feature_len]
                X_data.append(src_segment)

            y_data = pin_data_list[feature_len:]
            X = np.array(X_data)
            y = np.array(y_data)
            X_to_ret = X
            y_to_ret = y
    return (X_to_ret, y_to_ret, 0, X_std, y_std)

test_dataset_list = [
    'Random_Index_Set_Random_Dwell_set9_test2',
    'Random_Index_Set_Random_Dwell_set3_test6',
    'Random_Index_Set_Random_Dwell_set5_test9',
    'Slow_Postional_Displacement_10steps_test4',
    'Slow_Postional_Displacement_60steps_test7',
    'Slow_Postional_Displacement_30steps_test3',
    'Standard_Index_Set_test5',
    'Standard_Index_Set_test16',
    'Standard_Index_Set_test10'
]


IF_USE_SHUFFLED_SPLIT_DATA = False # True
def process_predictions(optuna_suggested_param_tag, y_for_train, predictions_train, y_for_test, predictions_test):
    snr_train = signaltonoise(y_for_train, predictions_train)
    snr_train = round(float(snr_train), 4)
    print('snr_train = ', snr_train)
    rmse_train = mean_squared_error(y_for_train, predictions_train) ** 0.5
    rmse_train = round(float(rmse_train), 4)
    print('rmse_train = ', rmse_train)
    snr_test = signaltonoise(y_for_test, predictions_test)
    snr_test = round(float(snr_test), 4)
    print('snr_test = ', snr_test)
    rmse_test = mean_squared_error(y_for_test, predictions_test) ** 0.5
    rmse_test = round(float(rmse_test), 4)
    print('rmse_test = ', rmse_test)
    result_tag = ('for '+'all_data_trial_test_' + optuna_suggested_param_tag + ' snr = ' + str(snr_test) +' rmse = ' + str(rmse_test)+ '\n')
    append_to_file(result_tag, result_path)
    result_tag = ('for '+'all_data_trial_train_' + optuna_suggested_param_tag + ' snr = ' + str(snr_train) +' rmse = ' + str(rmse_train)+ '\n')
    append_to_file(result_tag, result_path)
    return rmse_test


def get_SNR_rmse(y_test, ypred, y_std_dev=1):
    y_test = y_test.flatten() * y_std_dev
    ypred = ypred.flatten() * y_std_dev
    snr = signaltonoise(y_test, ypred)
    snr = round(float(snr), 4)
    rmse = mean_squared_error(y_test, ypred) ** 0.5
    rmse = round(float(rmse), 4)
    return snr, rmse

def process_predictions_datasetwise(y_sep, ypred_sep, filename_sep, tag):
    snr = signaltonoise(y_sep, ypred_sep)
    snr = round(float(snr), 4)
    print('snr = ', snr)
    rmse = mean_squared_error(y_sep, ypred_sep) ** 0.5
    rmse = round(float(rmse), 4)
    print('tag =', tag,' filename_sep = ', filename_sep, ' snr = ', snr, ' rmse = ', rmse)

    return rmse, snr
def process_datasetwise_prediction(model, X_y_Full_seperate_datasets, optuna_suggested_param_tag, buckets_maps_discrete_to_original_pin_val):
    rmse_orig_list = []


    datasetwise_train_snr_list = []
    datasetwise_test_snr_list = []
    datasetwise_train_rmse_list = []
    datasetwise_test_rmse_list = []

    for xypair in X_y_Full_seperate_datasets:
        X_sep = xypair['X']
        current_y_orig = xypair['current_y_orig']
        filename_sep = xypair['filename']
        print('separate evaluation for dataset ', filename_sep)
        X_sep = np.expand_dims(X_sep, 2)
        print('X_sep.shape =', X_sep.shape)

        ypred_sep = model.predict(X_sep, verbose=2)
        print('ypred_sep.shape 1 =', ypred_sep.shape)
        ypred_sep = np.argmax(ypred_sep, axis=1)
        print('ypred_sep.shape 2 =', ypred_sep.shape)
        ypred_sep = regenerate_pin_locations_from_discrete_vals(ypred_sep, buckets_maps_discrete_to_original_pin_val)
        ypred_sep = np.array(ypred_sep)
        print('ypred_sep.shape 3 =', ypred_sep.shape)


        current_y_orig = np.array(current_y_orig)

        rmse_orig, snr_orig = process_predictions_datasetwise(current_y_orig, ypred_sep, filename_sep, 'with_orig')
        rmse_orig_list.append(rmse_orig)

        if filename_sep in test_dataset_list:
            datasetwise_test_snr_list.append(snr_orig)
            datasetwise_test_rmse_list.append(rmse_orig)
        else:
            datasetwise_train_snr_list.append(snr_orig)
            datasetwise_train_rmse_list.append(rmse_orig)

    print('avg rmse_orig_list =', sum(rmse_orig_list) / len(rmse_orig_list))
    avg_datasetwise_train_snr_list = round(sum(datasetwise_train_snr_list) / len(datasetwise_train_snr_list), 4)
    avg_datasetwise_train_rmse_list = round(sum(datasetwise_train_rmse_list) / len(datasetwise_train_rmse_list), 4)

    avg_datasetwise_test_snr_list = round(sum(datasetwise_test_snr_list) / len(datasetwise_test_snr_list), 4)
    avg_datasetwise_test_rmse_list = round(sum(datasetwise_test_rmse_list) / len(datasetwise_test_rmse_list), 4)
    result_tag = optuna_suggested_param_tag + ' avg_datasetwise_train_snr = '+str(avg_datasetwise_train_snr_list)+' avg_datasetwise_train_rmse = '+str(avg_datasetwise_train_rmse_list)\
                 +' avg_datasetwise_test_snr = '+str(avg_datasetwise_test_snr_list)+\
                 ' avg_datasetwise_test_rmse = '+str(avg_datasetwise_test_rmse_list)+ '\n'
    append_to_file(result_tag, result_path)


def get_model_cost(n_CNN_layers, CNN_filters_size, n_LSTM_layers, LSTM_units, n_MLP_layers, MLP_units, input_len):
    kernel_size = 3
    cnn_cost = 0
    last_dense_layer_input_len = input_len
    last_dense_layer_input_len_final = input_len

    final_output_len = 1
    for i in range(n_CNN_layers):
        cnn_cost = cnn_cost + input_len*kernel_size*CNN_filters_size[i]
        input_len = input_len/2
        last_dense_layer_input_len = input_len
        last_dense_layer_input_len_final = CNN_filters_size[-1]*last_dense_layer_input_len

    lstm_cost = 0
    if n_LSTM_layers >= 1:
        lstm_layer1_cost = 4*(last_dense_layer_input_len*LSTM_units[0] + LSTM_units[0]*LSTM_units[0]) + 3*LSTM_units[0]

        n_LSTM_layers = n_LSTM_layers - 1
        temp_lstm_cost = 0
        for i in range(n_LSTM_layers):
            temp_lstm_cost = temp_lstm_cost + 4*(LSTM_units[i+1]*LSTM_units[i+1] + LSTM_units[i+1]*LSTM_units[i+1]) + 3*LSTM_units[i+1]

        lstm_cost = lstm_layer1_cost + temp_lstm_cost
        last_dense_layer_input_len_final = last_dense_layer_input_len*LSTM_units[-1]

    mlp_cost = 0
    mlp_input_len = last_dense_layer_input_len_final
    for i in range(n_MLP_layers):
        mlp_cost = mlp_cost + mlp_input_len*MLP_units[i]
        mlp_input_len = MLP_units[i]
        last_dense_layer_input_len = MLP_units[i]

    total_cost = cnn_cost + lstm_cost + mlp_cost + last_dense_layer_input_len*final_output_len
    print('total_cost =', total_cost)
    total_cost = round(total_cost/1000000, 4)
    return total_cost

def regenerate_pin_locations_from_discrete_vals(discrete_y, buckets_maps_discrete_to_original_pin_val):
    bucketed_pin_vals = []
    for val in discrete_y:
        bucketed_pin_vals.append(buckets_maps_discrete_to_original_pin_val[val])

    return bucketed_pin_vals

def get_rmse_for_optuna(model, X_train, y_train, X_test, y_test, optuna_suggested_param_tag, final_X_y_Full_seperate_datasets, buckets_maps_discrete_to_original_pin_val):
    history = model.fit(X_train, y_train, epochs=NUM_EPOCHS_FINAL, verbose=2)
    print('history.history[accuracy] = ', history.history['accuracy'])
    # evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('test_acc = ', test_acc)
    print('test_loss = ', test_loss)
    predictions_train = model.predict(X_train, verbose=2)

    print('y_for_train.shape = ', y_train.shape)
    print('predictions_train.shape = ', predictions_train.shape)
    predictions_train = np.argmax(predictions_train, axis=1)
    print('predictions_train = ', predictions_train)
    predictions_test = model.predict(X_test, verbose=2)
    predictions_test = np.argmax(predictions_test, axis=1)
    print('y_for_test.shape = ', y_test.shape)
    print('predictions_test.shape = ', predictions_test.shape)
    y_train_ref = np.argmax(y_train, axis=1)
    y_test_ref = np.argmax(y_test, axis=1)
    print('UNQ len(set(y_train_ref)) = ', len(set(y_train_ref)))
    print('UNQ len(set(predictions_train)) = ', len(set(predictions_train)))
    print('UNQ len(set(y_test_ref)) = ', len(set(y_test_ref)))
    print('UNQ len(set(predictions_test)) = ', len(set(predictions_test)))

    y_train_ref = regenerate_pin_locations_from_discrete_vals(y_train_ref, buckets_maps_discrete_to_original_pin_val)
    y_test_ref = regenerate_pin_locations_from_discrete_vals(y_test_ref, buckets_maps_discrete_to_original_pin_val)
    predictions_train = regenerate_pin_locations_from_discrete_vals(predictions_train,
                                                                    buckets_maps_discrete_to_original_pin_val)
    predictions_test = regenerate_pin_locations_from_discrete_vals(predictions_test,
                                                                   buckets_maps_discrete_to_original_pin_val)
    y_train_ref = np.array(y_train_ref)
    y_test_ref = np.array(y_test_ref)
    predictions_train = np.array(predictions_train)
    predictions_test = np.array(predictions_test)
    print('y_train_ref = ', y_train_ref)
    print('predictions_train = ', predictions_train)
    print('y_test_ref = ', y_test_ref)
    print('predictions_test = ', predictions_test)

    append_to_file('\n', result_path)
    rmse_test = process_predictions(optuna_suggested_param_tag, y_train_ref, predictions_train, y_test_ref, predictions_test)
    process_datasetwise_prediction(model, final_X_y_Full_seperate_datasets, optuna_suggested_param_tag, buckets_maps_discrete_to_original_pin_val)
    return rmse_test

IF_SKIP_VALS = False
max_pin_list = []
min_pin_list = []
def preprocess_vClassification(acc_path, pin_path, feature_len):
    f = open(acc_path)
    data_acc = f.read().split(', ')
    f.close()

    f = open(pin_path)
    data_pin = f.read().split(', ')
    f.close()

    data_acc = list(map(float, data_acc))
    data_pin = list(map(float, data_pin))

    if IF_SKIP_VALS:
        data_acc_new = []
        take_val = True
        skip_vals = 32
        list_len = len(data_acc)
        for indx, stepindx in enumerate(range(0, len(data_acc), skip_vals)):
            if stepindx + skip_vals >= list_len:
                break
            if take_val:
                data_acc_new.append(data_acc[stepindx:stepindx + skip_vals])
                take_val = False
            else:
                take_val = True
        data_acc = data_acc_new
        data_pin_new = []
        take_val = True
        list_len = len(data_pin)
        for indx, stepindx in enumerate(range(0, len(data_pin), skip_vals)):
            if stepindx + skip_vals >= list_len:
                break
            if take_val:
                data_pin_new.append(data_pin[stepindx:stepindx + skip_vals])
                take_val = False
            else:
                take_val = True

        data_pin = data_pin_new

        acc = np.array(data_acc).reshape(1, -1)[0]
        pin = np.array(data_pin).reshape(1, -1)[0]

    else:
        acc = np.array(data_acc)
        pin = np.array(data_pin)

    print('acc.shape = ', acc.shape)
    print('pin.shape = ', pin.shape)

    for i in range(len(pin)):
        if (isnan(pin[i])):
            pin[i] = pin[i - 1]

    X_std = np.std(acc)
    y_std = np.std(pin)
    ds = feature_len
    acc = acc[:(acc.size - 1)]
    X = np.reshape(acc[:acc.size // ds * ds], (acc.size // ds, ds))

    pin = pin.tolist()

    max_pin = max(pin)
    min_pin = min(pin)
    print('max_pin = ', max_pin)
    print('min_pin = ', min_pin)
    pin_range = max_pin - min_pin
    print('pin_range = ', pin_range)
    max_pin_list.append(max_pin)
    min_pin_list.append(min_pin)


    list_len = len(pin)
    src = []
    for indx, stepindx in enumerate(range(0, len(pin), ds)):
        if stepindx + ds >= list_len:
            break
        src.append(pin[stepindx + ds])  # DONE

    y = np.array(src)
    X_to_ret = X
    y_to_ret = y
    return (X_to_ret, y_to_ret, 0, X_std, y_std)


acc_files = os.listdir(acc_data_folderpath)
def prepare_dataset_for_each_trial(feature_len, acc_files):
    X_y_Full_seperate_datasets = []
    for acc_file in acc_files:
        print('running for acc_file = ', acc_file)
        filename = os.path.splitext(acc_file)[0]
        common_filename_part = filename.split('_acc_new')[0]
        pin_file = common_filename_part + '_pin_new' + '.txt'
        print('pin file found = ', os.path.exists(pin_data_folderpath + pin_file))
        print('current acc file path = ', acc_data_folderpath + acc_file)
        print('current pin file path = ', pin_data_folderpath + pin_file)
        (X, y, t, X_std, y_std) = preprocess_vClassification(acc_data_folderpath + acc_file, pin_data_folderpath + pin_file,
                                             feature_len)
        temp_X_y_pair = {'X': X, 'y': y, 'filename': common_filename_part, 'X_std': X_std, 'y_std': y_std}
        X_y_Full_seperate_datasets.append(temp_X_y_pair)

    X_Full = []
    y_Full = []
    X_std_dev_Full = []
    y_std_dev_Full = []
    for j in range(len(acc_files)):
        xypair = X_y_Full_seperate_datasets[j]
        X = xypair['X']
        y = xypair['y']
        print('9. X.shape = ', X.shape)
        print('9. y.shape = ', y.shape)
        filename = xypair['filename']
        X_std = xypair['X_std']
        y_std = xypair['y_std']
        if filename in test_dataset_list:
            print('skipping training for ', filename)
            continue
        X_Full.extend(X.tolist())
        y_Full.extend(y.tolist())
        X_std_dev_Full.append(X_std)
        y_std_dev_Full.append(y_std)
    print('len(X_std_dev_Full) = ', len(X_std_dev_Full))
    print('max(y_Full) = ', max(y_Full))
    print('min(y_Full) = ', min(y_Full))
    print('max_pin_list = ', max_pin_list)
    print('min_pin_list = ', min_pin_list)
    max_max_pin_list = max(max_pin_list)
    min_min_pin_list = min(min_pin_list)
    print('MAX = max_pin_list = ', max_max_pin_list)
    print('MIN = min_pin_list = ', min_min_pin_list)
    pin_list_range = max_max_pin_list - min_min_pin_list
    print('pin_list_range = ', pin_list_range)
    pin_val_slot = pin_list_range / NUM_CLASSESS
    print('pin_val_slot = ', pin_val_slot)
    print('Total vals in pin = ', len(y_Full))
    print('Unique vals in pin = ', len(set(y_Full)))
    print('np.array(y_Full).shape = ', np.array(y_Full).shape)
    discrete_y_Full = []
    breakpoints = []
    breakpoints.append(0.0)
    for i in range(NUM_CLASSESS + 1):
        breakpoints.append(min_min_pin_list + i * pin_val_slot)

    buckets = {}
    for i in y_Full:
        buckets.setdefault(breakpoints[bisect(breakpoints, i)], []).append(i)
    buckets = dict(sorted(buckets.items()))
    buckets_maps = {}
    i = 0
    for key in buckets:
        buckets_maps[key] = i
        i += 1

    buckets_maps_discrete_to_original_pin_val = {v: k for k, v in buckets_maps.items()}

    X_Full_train = []
    y_Full_train = []
    X_discrete_y_Full_seperate_datasets = []
    for x_y_pair in X_y_Full_seperate_datasets:
        current_filename = x_y_pair['filename']
        current_X = x_y_pair['X']
        current_y_orig = x_y_pair['y']
        current_X_std = x_y_pair['X_std']
        current_y_std = x_y_pair['y_std']
        print('ysep 11 ', np.array(current_y_orig).shape)
        current_discrete_y = []
        current_y = current_y_orig.tolist()
        for y_val in current_y:
            for key in buckets:
                slot_vals = buckets[key]
                if y_val in slot_vals:
                    class_val = [0] * NUM_CLASSESS
                    class_val[buckets_maps[key]] = 1
                    current_discrete_y.append(class_val)
                    break

        temp_X_discrete_y_pair = {'X': current_X, 'y': current_discrete_y, 'filename': current_filename,
                                  'X_std': current_X_std, 'y_std': current_y_std, 'current_y_orig': current_y_orig}
        X_discrete_y_Full_seperate_datasets.append(temp_X_discrete_y_pair)
        if current_filename in test_dataset_list:
            print('skipping training for ', current_filename)
            continue
        X_Full_train.extend(current_X.tolist())
        y_Full_train.extend(current_discrete_y)

    X_Full_train = np.array(X_Full_train)
    y_Full_train = np.array(y_Full_train)
    print('X_Full_train.shape = ', X_Full_train.shape)
    print('y_Full_train.shape = ', y_Full_train.shape)
    X_Full, y_Full = X_Full_train, y_Full_train
    test_size = 0.3

    if not IF_USE_SHUFFLED_SPLIT_DATA:
        X_train, X_test, y_train, y_test = train_test_split(X_Full, y_Full, test_size=test_size, shuffle=False,
                                                            random_state=42)  # 70% training and 30% test
    else:
        # shuffle data
        X_train, X_test, y_train, y_test = train_test_split(X_Full, y_Full, test_size=test_size,
                                                            random_state=42)  # 70% training and 30% test


    print('final. X_train = ', X_train.shape)
    print('final. y_train = ', y_train.shape)
    print('final. X_test = ', X_test.shape)
    print('final. y_test = ', y_test.shape)
    # new try
    X_train_trn = np.expand_dims(X_train, 2)
    y_train_trn = y_train

    y_test_trn = y_test
    X_test_trn = np.expand_dims(X_test, 2)

    print('final 2. X_train_trn = ', X_train_trn.shape)
    print('final 2. y_train_trn = ', y_train_trn.shape)
    print('final 2. X_test_trn = ', X_test_trn.shape)
    print('final 2. y_test_trn = ', y_test_trn.shape)
    X_for_train, y_for_train = X_train_trn, y_train_trn
    X_for_test, y_for_test = X_test_trn, y_test_trn
    return X_for_train, y_for_train, X_for_test, y_for_test, X_discrete_y_Full_seperate_datasets, buckets_maps_discrete_to_original_pin_val

def objective(trial):
    global optuna_n_trials_current
    print('optuna_n_trials_current =', optuna_n_trials_current)

    rmse = 2.0
    model_cost = 3.1
    model_cost_constraint = model_cost - 3.01
    rmse_constraint = rmse - 0.5
    trial.set_user_attr("constraint", (rmse_constraint, model_cost_constraint))

    if optuna_n_trials_current >= optuna_n_trials_MAX:
        print('optuna_n_trials_current >= optuna_n_trials_MAX')
        raise optuna.TrialPruned()

    append_to_file('\n',result_path)
    append_to_file('trial no = ',result_path)
    append_to_file(optuna_n_trials_current, result_path)


    n_CNN_layers = trial.suggest_int('n_CNN_layers', 0, 5)
    CNN_filters_size = []
    CNN_filters_size_temp = []
    CNN_filters_size_1 = trial.suggest_int('CNN_filters_size_1', 16, 256)
    CNN_filters_size_temp.append(CNN_filters_size_1)
    CNN_filters_size_temp.append(CNN_filters_size_1)
    CNN_filters_size_temp.append(CNN_filters_size_1)
    CNN_filters_size_temp.append(CNN_filters_size_1)
    CNN_filters_size_temp.append(CNN_filters_size_1)
    for i in range(n_CNN_layers):
        CNN_filters_size.append(CNN_filters_size_temp[i])


    n_LSTM_layers = trial.suggest_int('n_LSTM_layers', 0, 3)

    LSTM_units = []
    LSTM_units_temp = []
    LSTM_units_1 = trial.suggest_int('LSTM_units_1', 16, 425)
    LSTM_units_temp.append(LSTM_units_1)
    LSTM_units_temp.append(LSTM_units_1)
    LSTM_units_temp.append(LSTM_units_1)
    LSTM_units_temp.append(LSTM_units_1)
    LSTM_units_temp.append(LSTM_units_1)
    for i in range(n_LSTM_layers):
        LSTM_units.append(LSTM_units_temp[i])

    n_MLP_layers = trial.suggest_int('n_MLP_layers', 0, 5)
    feature_len = trial.suggest_int('feature_len', 128, 750)

    IFNOLAYERS=False
    if n_CNN_layers==0 and n_LSTM_layers==0 and n_MLP_layers==0:
        n_MLP_layers=1
        IFNOLAYERS=True

    MLP_units = []
    MLP_units_temp = []
    MLP_units_1 = trial.suggest_int('MLP_units_1', 4, 512)
    MLP_units_temp.append(MLP_units_1)
    MLP_units_temp.append(MLP_units_1)
    MLP_units_temp.append(MLP_units_1)
    MLP_units_temp.append(MLP_units_1)
    MLP_units_temp.append(MLP_units_1)


    for i in range(n_MLP_layers):
        MLP_units.append(MLP_units_temp[i])


    model_cost = get_model_cost(n_CNN_layers, CNN_filters_size, n_LSTM_layers, LSTM_units, n_MLP_layers, MLP_units, feature_len)
    print('n_CNN_layers =', n_CNN_layers, ' CNN_filters_size =', CNN_filters_size, ' n_LSTM_layers =', n_LSTM_layers,
          ' LSTM_units =', LSTM_units, ' n_MLP_layers =', n_MLP_layers, ' MLP_units =', MLP_units, ' feature_len =',
          feature_len, ' model_cost =', model_cost)

    rmse = 2.0
    model_cost_constraint = model_cost - 3.01
    rmse_constraint = rmse - 0.5
    trial.set_user_attr("constraint", (rmse_constraint, model_cost_constraint))

    if model_cost > 3.0:
        print('model_cost =', model_cost, ' optuna_TrialPruned()')
        raise optuna.TrialPruned()


    final_X_train, final_y_train, final_X_test, final_y_test, final_X_y_Full_seperate_datasets, buckets_maps_discrete_to_original_pin_val = prepare_dataset_for_each_trial(feature_len, acc_files)
    model = create_model(n_CNN_layers=n_CNN_layers, CNN_filters_size=CNN_filters_size, n_LSTM_layers=n_LSTM_layers, LSTM_units=LSTM_units,
                         n_MLP_layers=n_MLP_layers, MLP_units=MLP_units, feature_len=feature_len)
    print('model summary =', model.summary())
    optuna_n_trials_current = optuna_n_trials_current + 1


    optuna_suggested_param_tag = 'n_CNN_layers=' + str(n_CNN_layers) + '#CNN_filters_size=' + str(
        CNN_filters_size) + '#n_LSTM_layers=' + str(n_LSTM_layers) + \
                                 '#LSTM_units=' + str(LSTM_units) + '#n_MLP_layers=' + str(
        n_MLP_layers) + '#MLP_units=' + str(MLP_units) + '#feature_len=' + str(feature_len) +'#model_cost='+str(model_cost)+ \
                                 '#optuna_sampler_tag='+optuna_sampler_tag+'#IFNOLAYERS=' + str(IFNOLAYERS) + ' '
    rmse = get_rmse_for_optuna(model, final_X_train, final_y_train, final_X_test, final_y_test, optuna_suggested_param_tag, final_X_y_Full_seperate_datasets, buckets_maps_discrete_to_original_pin_val)

    rmse_cost_log = 'rmse=' + str(rmse) + '=model_cost=' + str(model_cost) + '\n'
    append_to_file(rmse_cost_log, rmse_cost_path)
    print('trial_optuna_suggested_param_tag = ', optuna_suggested_param_tag, ' rmse = ', rmse)
    model_cost_constraint = model_cost - 3.01
    rmse_constraint = rmse - 0.5
    trial.set_user_attr("constraint", (rmse_constraint, model_cost_constraint))
    return rmse, model_cost

def constraints(trial):
    return trial.user_attrs["constraint"]

if __name__ == '__main__':
    NUM_EPOCHS_list = [NUM_EPOCHS_FINAL]
    write_to_file('\n', result_path)

    optuna_sampler = optuna.integration.BoTorchSampler(
        constraints_func=constraints,
        n_startup_trials=10,
        device='cuda'
    )
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=optuna_sampler,
    )
    print(f"Sampler is {study.sampler.__class__.__name__}")
    study.optimize(objective, n_trials=optuna_n_trials)
    best_params = study.best_trials
    print()
    print('best_params = ', best_params)
    print(f"Number of trials on the Pareto front: {len(best_params)}")
    print()

    axes = optuna.visualization.plot_pareto_front(study, target_names=['RMSE', 'Cost'])
    axes.write_image(out_file_tag + ".png")
    axes.write_image(out_file_tag + ".pdf")
    plotly.offline.plot(axes, filename=out_file_tag + '.html')

print('Done All')
now = datetime.datetime.now()
print('end time = ', now)
