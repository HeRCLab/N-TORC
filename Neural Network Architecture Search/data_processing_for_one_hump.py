import numpy as np
def write_to_file(data, filename):
    f = open(filename+".txt", "w")
    f.write(str(data).replace('[','').replace(']',''))
    f.close()

acc_vals = []
pin_vals = []
pred_pin_vals = []
time_vals = []

dir_path = '/path/to/directory/containing/required/data/files/'
filepath_acc = dir_path + 'acceleration_signal_filename'+'.txt'
filepath_pin = dir_path + 'reference_pin_location_filename'+'.txt'
filepath_pred_pin_bad = dir_path + 'less_accurate_model_predicted_pin_location_filename'+'.txt'
filepath_pred_pin_good = dir_path + 'more_accurate_model_predicted_pin_location_filename'+'.txt'
filepath_time = dir_path + 'time_values_filename'+'.txt'
out_dir = dir_path + '/path/to/output/files/'

def read_data(filepath):
    f = open(filepath)
    lines = f.read().split(', ')
    f.close()
    data_vals = list(map(float, lines))
    return data_vals

acc_vals = read_data(filepath_acc)
pin_vals = read_data(filepath_pin)
pred_pin_vals = read_data(filepath_pred_pin_bad)
time_vals = read_data(filepath_time)
pred_pin_vals_rmse3 = read_data(filepath_pred_pin_good)
pin_vals_new = np.interp(np.arange(0, len(pin_vals), 0.00602409639), np.arange(0, len(pin_vals)), pin_vals)
pred_pin_vals_new = np.interp(np.arange(0, len(pred_pin_vals), 0.00602409639), np.arange(0, len(pred_pin_vals)), pred_pin_vals)
pred_pin_vals_rmse3_new = np.interp(np.arange(0, len(pred_pin_vals_rmse3), 0.00195048486), np.arange(0, len(pred_pin_vals_rmse3)), pred_pin_vals_rmse3)

left_x_val = 1020
right_x_val = 1130
pin_vals_new = pin_vals_new.tolist()
pin_vals_new_hump = pin_vals_new[left_x_val*166:right_x_val*166]
write_to_file(pin_vals_new_hump, out_dir + 'reference_pin_vals_new_hump')


pred_pin_vals_new = pred_pin_vals_new.tolist()
pred_pin_vals_new_hump = pred_pin_vals_new[left_x_val*166:right_x_val*166]
write_to_file(pred_pin_vals_new_hump, out_dir + 'pred_pin_vals_new_hump_less_accurate_model')

pred_pin_vals_rmse3_new = pred_pin_vals_rmse3_new.tolist()
pred_pin_vals_rmse3_new_hump = pred_pin_vals_rmse3_new[left_x_val*166:right_x_val*166]
write_to_file(pred_pin_vals_rmse3_new_hump, out_dir + 'pred_pin_vals_new_hump_more_accurate_model')

acc_vals_hump = acc_vals[left_x_val*166:right_x_val*166]
write_to_file(acc_vals_hump, out_dir + 'acceleration_signal_vals_new_hump')

time_vals_hump = time_vals[left_x_val*166:right_x_val*166]
write_to_file(time_vals_hump, out_dir + 'time_vals_new_hump')
