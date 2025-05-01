import numpy as np
import optuna
import plotly
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.ticker as ticker

model_cost_list = []
datasetwise_test_rmse_list = []
model_all_params_list = []

n_trials = 360
dir_path = 'results/'
main_results_all_info_path = dir_path + 'main_results_BoTorchSampler.txt'
out_file_tag = dir_path + 'name_of_saved_pareto_front_plots'
FILEPATH = 'optimal_models/'

# IMPORTANT NOTES ******************
# TO filter-out VALUES BEYOND EXPECTED RANGE For ZOOMED IN Plot
# FOR ZOOM OUT SKIP for model_cost > 0.7 AND RMSE > 0.75
# FOR ZOOM IN SKIP for model_cost > ~0.055 AND RMSE > 0.75
RMSE_LIMIT = 0.75
MODEL_COST_LIMIT = 0.71

# def prep_values_for_pareto_front_from_csv(filepath):
#     allparams_data = read_csv(filepath)
#     model_cost_list = allparams_data['model_cost'].tolist()
#     datasetwise_test_rmse_list = allparams_data['datasetwise_test_rmse'].tolist()
#     model_all_params_list = allparams_data['all_model_params'].tolist()
#     return model_cost_list, datasetwise_test_rmse_list, model_all_params_list

model_config_dict_list = []

def gen_model_details(filepath):
    f = open(filepath)
    lines = f.readlines()
    f.close()
    line_details_model_cost = 0.0
    line_details_rmse = 0.0
    for line in lines:
        if line.startswith('for all_data_trial_test_n_CNN_layers='):
            line_details_model_cost = line.split('#model_cost=')[-1].split('#optuna_sampler_tag=')[0]
            line_details_model_cost = float(line_details_model_cost)
            line_details_rmse = line.split()[-1]
            line_details_rmse = float(line_details_rmse)

        if line.startswith('n_CNN_layers='):
            line = line.strip()
            line_details = line.split('#')
            n_CNN_layers = line_details[0].split('=')[-1]
            CNN_filters_size = line_details[1].split('=')[-1]
            n_LSTM_layers = line_details[2].split('=')[-1]
            LSTM_units = line_details[3].split('=')[-1]
            n_MLP_layers = line_details[4].split('=')[-1]
            MLP_units = line_details[5].split('=')[-1]
            feature_len = line_details[6].split('=')[-1]

            model_config_dict = {'n_CNN_layers': n_CNN_layers,
                                 'CNN_filters_size': CNN_filters_size,
                                 'n_LSTM_layers': n_LSTM_layers,
                                 'LSTM_units': LSTM_units,
                                 'n_MLP_layers': n_MLP_layers,
                                 'MLP_units': MLP_units,
                                 'feature_len': feature_len,
                                 'model_cost': line_details_model_cost,
                                 'rmse': line_details_rmse
                                 }
            if line_details_model_cost > MODEL_COST_LIMIT:
                continue
            if line_details_rmse > RMSE_LIMIT:
                continue
            model_config_dict_list.append(model_config_dict)

    print('model_config_dict_list =', model_config_dict_list)
    print(len(model_config_dict_list))
    return len(model_config_dict_list)

def objective(trial):
    trial.set_user_attr('n_CNN_layers', str(model_config_dict_list[trial.number]['n_CNN_layers']))
    trial.set_user_attr('CNN_filters_size', str(model_config_dict_list[trial.number]['CNN_filters_size']))

    trial.set_user_attr('n_LSTM_layers', str(model_config_dict_list[trial.number]['n_LSTM_layers']))
    trial.set_user_attr('LSTM_units', str(model_config_dict_list[trial.number]['LSTM_units']))

    trial.set_user_attr('n_MLP_layers', str(model_config_dict_list[trial.number]['n_MLP_layers']))
    trial.set_user_attr('MLP_units', str(model_config_dict_list[trial.number]['MLP_units']))

    trial.set_user_attr('feature_len', str(model_config_dict_list[trial.number]['feature_len']))

    rmse = model_config_dict_list[trial.number]['rmse']
    model_cost = model_config_dict_list[trial.number]['model_cost']

    return rmse, model_cost

pareto_optimal_rmse_cost_list = []
def parse_gen_pareto_optimal_rmse_cost_list(best_params):
    best_params = str(best_params)
    best_params_details = best_params.split(', values=')
    best_params_details = best_params_details[1:]
    for param in best_params_details:
        param = param.split(', datetime_start=')[0]
        param = param.replace('[','').replace(']','')
        param = param.split(', ')
        param = list(map(float, param))
        print('param =', param)
        print('param 0 =', param[0])
        print('param 1 =', param[1])

        line_details_rmse = param[0]
        line_details_model_cost = param[1]

        if line_details_model_cost > MODEL_COST_LIMIT:
            continue
        if line_details_rmse > RMSE_LIMIT:
            continue
        pareto_optimal_rmse_cost_list.append(param)

    return

def create_model(n_CNN_layers, CNN_filters_size, n_LSTM_layers, LSTM_units, n_MLP_layers, MLP_units, feature_len):
    print('inside create_model testing')
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(feature_len, 1)))

    # CNN
    kernel_size = 3
    for i in range(n_CNN_layers):
        model.add(tf.keras.layers.Conv1D(filters=CNN_filters_size[i], kernel_size=kernel_size, activation='relu',
                                         padding='same',
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
    model.add(tf.keras.layers.Dense(256))
    model.compile(loss='mse', optimizer='adam')


    return model

def get_layer_units_from_str(layer_count, layer_details):
    filters_size_1 = layer_details.replace('[', '').replace(']', '').replace(',', '')
    filters_size_1 = list(map(int, filters_size_1.split()))
    if layer_count > 0:
        filters_size_1 = filters_size_1[0]
    else:
        filters_size_1 = 0

    return filters_size_1

def prepare_pareto_front_network(network_details, network_count):
    n_CNN_layers = int(network_details['n_CNN_layers'])
    print('n_CNN_layers =', n_CNN_layers)
    CNN_filters_size = []
    CNN_filters_size_temp = []

    CNN_filters_size_1 = get_layer_units_from_str(n_CNN_layers, network_details['CNN_filters_size'])

    CNN_filters_size_temp.append(CNN_filters_size_1)
    CNN_filters_size_temp.append(CNN_filters_size_1)
    CNN_filters_size_temp.append(CNN_filters_size_1)
    CNN_filters_size_temp.append(CNN_filters_size_1)
    CNN_filters_size_temp.append(CNN_filters_size_1)

    for i in range(n_CNN_layers):
        CNN_filters_size.append(CNN_filters_size_temp[i])
    print('CNN_filters_size =', CNN_filters_size)

    n_LSTM_layers = int(network_details['n_LSTM_layers'])
    LSTM_units = []
    LSTM_units_temp = []

    LSTM_units_1 = get_layer_units_from_str(n_LSTM_layers, network_details['LSTM_units'])

    LSTM_units_temp.append(LSTM_units_1)
    LSTM_units_temp.append(LSTM_units_1)
    LSTM_units_temp.append(LSTM_units_1)
    LSTM_units_temp.append(LSTM_units_1)
    LSTM_units_temp.append(LSTM_units_1)

    for i in range(n_LSTM_layers):
        LSTM_units.append(LSTM_units_temp[i])
    print('LSTM_units =', LSTM_units)
    n_MLP_layers = int(network_details['n_MLP_layers'])
    feature_len = int(network_details['feature_len'])
    MLP_units_1 = get_layer_units_from_str(n_MLP_layers, network_details['MLP_units'])

    MLP_units = []
    MLP_units_temp = []

    MLP_units_temp.append(MLP_units_1)
    MLP_units_temp.append(MLP_units_1)
    MLP_units_temp.append(MLP_units_1)
    MLP_units_temp.append(MLP_units_1)
    MLP_units_temp.append(MLP_units_1)

    for i in range(n_MLP_layers):
        MLP_units.append(MLP_units_temp[i])
    print('MLP_units =', MLP_units)
    optuna_suggested_param_tag = 'n_CNN_layers=' + str(n_CNN_layers) + '#CNN_filters_size=' + str(
        CNN_filters_size) + '#n_LSTM_layers=' + str(n_LSTM_layers) + \
                                 '#LSTM_units=' + str(LSTM_units) + '#n_MLP_layers=' + str(
        n_MLP_layers) + '#MLP_units=' + str(MLP_units) + '#feature_len=' + str(feature_len) + ' '

    print('trial_optuna_suggested_param_tag = ', optuna_suggested_param_tag)
    model = create_model(n_CNN_layers=n_CNN_layers, CNN_filters_size=CNN_filters_size, n_LSTM_layers=n_LSTM_layers,
                         LSTM_units=LSTM_units,
                         n_MLP_layers=n_MLP_layers, MLP_units=MLP_units, feature_len=feature_len)

    json_string = model.to_json()


    FILENAME = 'network_' + str(feature_len) + '_' + str(n_CNN_layers) + '_' + str(CNN_filters_size_1) + '_' + str(
        n_LSTM_layers) + '_' \
               + str(LSTM_units_1) + '_' + str(n_MLP_layers) + '_' + str(MLP_units_1) + '_' + str(network_count)
    print('FILENAME =', FILENAME)
    with open(FILEPATH + FILENAME + ".json", "w") as outfile:
        outfile.write(json_string)
    model.save_weights(FILEPATH + FILENAME + '.weights.h5')


def parse_save_pareto_front_networks(best_params):
    network_count = 0
    for item in best_params:
        print('pareto-front network params =', item.user_attrs)
        print('pareto-front network values =', item.values)
        prepare_pareto_front_network(item.user_attrs, network_count)
        network_count += 1

# def append_to_file(data, filename):
#     f = open(filename, "a")
#     f.write(str(data))
#     f.close()

def create_pareto_front():
    global n_trials
    n_trials = gen_model_details(main_results_all_info_path)
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=n_trials, timeout=300)
    best_params = study.best_trials
    parse_gen_pareto_optimal_rmse_cost_list(best_params)
    parse_save_pareto_front_networks(best_params)
    print('best_params = ', best_params)
    print(f"Number of trials on the Pareto front: {len(best_params)}")

    axes = optuna.visualization.plot_pareto_front(study, target_names=['RMSE', 'Cost'])
    axes.write_image(out_file_tag + ".png")
    axes.write_image(out_file_tag + ".pdf")
    plotly.offline.plot(axes, filename=out_file_tag + '.html')
    return

create_pareto_front()

datapoints = []
for ii in range(0, n_trials):
    rmse = model_config_dict_list[ii]['rmse']
    model_cost = model_config_dict_list[ii]['model_cost']
    if model_cost > MODEL_COST_LIMIT:
        continue
    if rmse > RMSE_LIMIT:
        continue

    temp_data = []
    temp_data.append(rmse)
    temp_data.append(model_cost)
    datapoints.append(temp_data)

datapoints_new = []
for i in datapoints:
    if i not in pareto_optimal_rmse_cost_list:
        datapoints_new.append(i)

datapoints_orig = datapoints
datapoints = datapoints_new
datapoints = np.array(datapoints)
datapoints_orig = np.array(datapoints_orig)

pareto_optimal_rmse_cost_list = np.array(pareto_optimal_rmse_cost_list)

ehsan_result = [[721100, 0.0056]]
ehsan_result = np.array(ehsan_result)
joud1_result = [[152800, 0.0302]]
joud1_result = np.array(joud1_result)
joud2_result = [[216000, 0.7087]]
joud2_result = np.array(joud2_result)

markersize = 5
markersize_optimal = 5
fontsize = 12

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(ticker.EngFormatter())

h = plt.plot(datapoints[:, 0], datapoints[:, 1], '.k', markersize=markersize, marker='x', label='Non Pareto-optimal')
hp = plt.plot(pareto_optimal_rmse_cost_list[:, 0], pareto_optimal_rmse_cost_list[:, 1], '.r', marker='o', markersize=markersize_optimal, label='Pareto optimal')

hp2 = plt.plot(ehsan_result[:, 0], ehsan_result[:, 1], '.c', marker='s', markersize=markersize_optimal, label='Ehasn et. al')
hp3 = plt.plot(joud1_result[:, 0], joud1_result[:, 1], '.m', marker='p', markersize=markersize_optimal, label='Joud et. al - model 1')
hp4 = plt.plot(joud2_result[:, 0], joud2_result[:, 1], '.g', marker='h', markersize=markersize_optimal, label='Joud et. al - model 2')

plt.xlabel('Accuracy (RMSE)', fontsize=fontsize)
plt.ylabel('# Multiplies', fontsize=fontsize)

plt.xlim(0, 0.75)
_ = plt.legend(loc=1, numpoints=1)
fig_name = '_pareto_py'
plt.savefig(out_file_tag+fig_name+'.png')
plt.savefig(out_file_tag+fig_name+'.pdf')
plt.savefig(out_file_tag+fig_name+'.svg')

