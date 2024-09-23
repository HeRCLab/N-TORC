% import the data
data = importfile_data("higgs_sep22_exploded.csv");
data.("model_number") = (1:size(data,1))';
%predicted_conv_data=prediction("conv1d_predictions_from.csv");
%predicted_lstm_data=importfile2("lstm_predictions_from.csv");
predicted_dense_data=predict_dense("dense_predictions_from.csv");
[conv_resources,lstm_resources,dense_resources,data] = extract_training_data(data,0);
[conv_latency,lstm_latency,dense_latency,data] = extract_training_data(data,1);

collapsed_conv_resources = collapse_table(conv_resources,...
            ["reuse_factor","n_inputs_resources","sequence_length_resource"],...
            ["bram_18k","lut","ff","dsp48e"]);
conv_latency_collapsed = collapse_table(conv_latency, ...
    ["reuse_factor", "n_inputs_latency", "sequence_length_latency"], ...
    ["latency_min", "latency_max"]);
lstm_resources_collapsed = collapse_table(lstm_resources, ...
    ["reuse_factor", "n_inputs_resources", "sequence_length_resource","lstm_size"], ...
    ["bram_18k", "lut", "ff", "dsp48e"]);
lstm_latency_collapsed = collapse_table(lstm_latency, ...
    ["reuse_factor", "n_inputs_latency", "sequence_length_latency","lstm_size"], ...
    ["latency_min", "latency_max"]);
dense_resources_collapsed = collapse_table(dense_resources, ...
    ["reuse_factor", "n_inputs_resources", "sequence_length_resource","dense_size_resource"], ...
    ["bram_18k", "lut", "ff", "dsp48e"]);
dense_latency_collapsed = collapse_table(dense_latency, ...
    ["reuse_factor", "n_inputs_latency", "sequence_length_latency","dense_size_latency"], ...
    ["latency_min", "latency_max"]);
% writetable(collapsed_conv_resources, 'collapsed_conv_resources.csv');
% writetable(conv_latency_collapsed, 'conv_latency_collapsed.csv');
% writetable(lstm_resources_collapsed, 'lstm_resources_collapsed.csv');
% writetable(lstm_latency_collapsed, 'lstm_latency_collapsed.csv');
% writetable(dense_resources_collapsed, 'dense_resources_collapsed.csv');
% writetable(dense_latency_collapsed, 'dense_latency_collapsed.csv');

% writetable(conv_resources, 'conv_resources.csv');
% writetable(conv_latency, 'conv_latency.csv');
% writetable(lstm_latency, 'lstm_latency.csv');
% writetable(lstm_resources, 'lstm_resources.csv');
% writetable(dense_resources, 'dense_resources.csv');
% writetable(dense_latency, 'dense_latency.csv');


%%%%%%%%%
%% 
% % Fixed parameters
% fixed_n_inputs = 134;
% fixed_sequence_length = 1;
% fixed_cnn_filters = 32;
% 
% % Filter the actual data
% filtered_data = collapsed_conv_resources( ...
%     collapsed_conv_resources.n_inputs_resources == fixed_n_inputs & ...
%     collapsed_conv_resources.sequence_length_resource == fixed_sequence_length & ...
%     collapsed_conv_resources.cnn_filters_resources == fixed_cnn_filters, :);
% filtered_data_latency=conv_latency( conv_latency_collapsed.n_inputs_latency == fixed_n_inputs & ...
%     conv_latency_collapsed.sequence_length_latency == fixed_sequence_length & ...
%     conv_latency_collapsed.cnn_filters_latency == fixed_cnn_filters, :);
% 
% % Filter the predicted data
% predicted_filtered_data = predicted_conv_data( ...
%     predicted_conv_data.cnn_in == fixed_n_inputs & ...
%     predicted_conv_data.sequence_length ...
%     == fixed_sequence_length & ...
%     predicted_conv_data.cnn_filters == fixed_cnn_filters, :);
% 
% % Extract the reuse factors and LUT/BRAM values
% reuse_factors = log2(filtered_data.correct_reuse_factor_resource);
% lut_values = filtered_data.lut;
% bram_values = filtered_data.bram_18k;
% ff_values = filtered_data.ff;
% dsp48e_values = filtered_data.dsp48e;
% latency_max_values = filtered_data_latency.latency_max;
% predicted_lut_values = predicted_filtered_data.predicted_lut;
% predicted_bram_values = predicted_filtered_data.predicted_bram;
% predicted_ff_values = predicted_filtered_data.predicted_ff;
% predicted_dsp48e_values = predicted_filtered_data.predicted_dsp;
% predicted_latency_max_values = predicted_filtered_data.predicted_latency_max;
% predicted_reuse_factors = log2(predicted_filtered_data.reuse_factor);
% 
% % Find the common reuse factors
% [common_reuse_factors, idx_actual, idx_predicted] = intersect(reuse_factors, predicted_reuse_factors);
% 
% % Actual and predicted data for the common reuse factors
% common_lut_values_actual = lut_values(idx_actual);
% common_bram_values_actual = bram_values(idx_actual);
% common_ff_values_actual = ff_values(idx_actual);
% common_dsp48e_values_actual = dsp48e_values(idx_actual);
% common_latency_max_actual = latency_max_values(idx_actual);
% 
% common_lut_values_predicted = predicted_lut_values(idx_predicted);
% common_bram_values_predicted = predicted_bram_values(idx_predicted);
% common_ff_values_predicted = predicted_ff_values(idx_predicted);
% common_dsp48e_values_predicted = predicted_dsp48e_values(idx_predicted);
% common_latency_max_predicted = predicted_latency_max_values(idx_predicted);
% 
% % Plot 1: LUT vs BRAM
% figure;
% yyaxis left;
% plot(common_reuse_factors, common_lut_values_actual, '-o', 'DisplayName', 'Actual LUT');
% hold on;
% plot(common_reuse_factors, common_lut_values_predicted, '--o', 'DisplayName', 'Predicted LUT');
% ylabel('LUT');
% 
% yyaxis right;
% plot(common_reuse_factors, common_bram_values_actual, '-x', 'DisplayName', 'Actual BRAM');
% plot(common_reuse_factors, common_bram_values_predicted, '--x', 'DisplayName', 'Predicted BRAM');
% ylabel('BRAM');
% 
% xlabel(' log2 Reuse Factor for CNN');
% title(' Reuse Factor vs LUT and BRAM (n\_in=134, cnn\_filter=32, sequence\_length=1)');
% grid on;
% legend show;
% hold off;
% 
% % Plot 2: FF vs DSP48E
% figure;
% yyaxis left;
% plot(common_reuse_factors, common_ff_values_actual, '-o', 'DisplayName', 'Actual FF');
% hold on;
% plot(common_reuse_factors, common_ff_values_predicted, '--o', 'DisplayName', 'Predicted FF');
% ylabel('FF');
% 
% yyaxis right;
% plot(common_reuse_factors, common_dsp48e_values_actual, '-x', 'DisplayName', 'Actual DSP48E');
% plot(common_reuse_factors, common_dsp48e_values_predicted, '--x', 'DisplayName', 'Predicted DSP48E');
% ylabel('DSP48E');
% 
% xlabel('log2 Reuse Factor for CNN');
% title('Reuse Factor vs FF and DSP48E (n\_in=134, cnn\_filter=32, sequence\_length=1)');
% grid on;
% legend show;
% hold off;
% 
% % Plot 3: Latency_Max vs Correct Reuse Factor
% figure;
% plot(common_reuse_factors, common_latency_max_actual, '-o', 'DisplayName', 'Actual Latency Max');
% hold on;
% plot(common_reuse_factors, common_latency_max_predicted, '--o', 'DisplayName', 'Predicted Latency Max');
% ylabel('Latency Max for CNN');
% xlabel('log2 Reuse Factor for CNN');
% title('Reuse Factor vs Latency Max (n\_in=134, cnn\_filter=32, sequence\_length=1)');
% grid on;
% legend show;
% hold off;
% 
% %%%%%LSTM Fixed parameters for plot
% fixed_n_inputs_lstm=32;
% fixed_sequence_length_lstm=67;
% fixed_lstm_size=11;
% 
% % Filter the actual data for LSTM
% filtered_data_lstm = lstm_resources_collapsed( ...
%     lstm_resources_collapsed.n_inputs_resources == fixed_n_inputs_lstm & ...
%     lstm_resources_collapsed.sequence_length_resource == fixed_sequence_length_lstm & ...
%     lstm_resources_collapsed.lstm_size == fixed_lstm_size, :);
% 
% filtered_data_latency_lstm = lstm_latency_collapsed( lstm_latency_collapsed.n_inputs_latency == fixed_n_inputs_lstm & ...
%     lstm_latency_collapsed.sequence_length_latency == fixed_sequence_length_lstm & ...
%     lstm_latency_collapsed.lstm_size == fixed_lstm_size, :);
% 
% % Filter the predicted data for LSTM
% predicted_filtered_data_lstm = predicted_lstm_data( ...
%     predicted_lstm_data.n_in == fixed_n_inputs_lstm & ...
%     predicted_lstm_data.sequence_length == fixed_sequence_length_lstm & ...
%     predicted_lstm_data.lstm_size == fixed_lstm_size, :);
% 
% % Extract the reuse factors and LUT/BRAM values for LSTM
% reuse_factors_lstm = log2(filtered_data_lstm.correct_reuse_factor_lstm_resource);
% reuse_factors_lstm_latency=log2(filtered_data_latency_lstm.correct_reuse_factor_lstm_latency);
% lut_values_lstm = filtered_data_lstm.lut;
% bram_values_lstm = filtered_data_lstm.bram_18k;
% ff_values_lstm = filtered_data_lstm.ff;
% dsp48e_values_lstm = filtered_data_lstm.dsp48e;
% latency_max_values_lstm = filtered_data_latency_lstm.latency_max;
% predicted_lut_values_lstm = predicted_filtered_data_lstm.predicted_lut;
% predicted_bram_values_lstm = predicted_filtered_data_lstm.predicted_bram;
% predicted_ff_values_lstm = predicted_filtered_data_lstm.predicted_ff;
% predicted_dsp48e_values_lstm = predicted_filtered_data_lstm.predicted_dsp;
% predicted_latency_max_values_lstm = predicted_filtered_data_lstm.predicted_latency_max;
% predicted_reuse_factors_lstm = log2(predicted_filtered_data_lstm.reuse_factor);
% 
% % Find the common reuse factors for LSTM
% [common_reuse_factors_lstm, idx_actual_lstm, idx_predicted_lstm] = intersect(reuse_factors_lstm, predicted_reuse_factors_lstm);
% [common_reuse_factors_lstm_latency, idx_actual_lstm_latency, idx_predicted_lstm_latency]=intersect(reuse_factors_lstm_latency, predicted_reuse_factors_lstm);
% 
% 
% % Actual and predicted data for the common reuse factors for LSTM
% common_lut_values_actual_lstm = lut_values_lstm(idx_actual_lstm);
% common_bram_values_actual_lstm = bram_values_lstm(idx_actual_lstm);
% common_ff_values_actual_lstm = ff_values_lstm(idx_actual_lstm);
% common_dsp48e_values_actual_lstm = dsp48e_values_lstm(idx_actual_lstm);
% common_latency_max_actual_lstm = latency_max_values_lstm(idx_actual_lstm);
% 
% common_lut_values_predicted_lstm = predicted_lut_values_lstm(idx_predicted_lstm);
% common_bram_values_predicted_lstm = predicted_bram_values_lstm(idx_predicted_lstm);
% common_ff_values_predicted_lstm = predicted_ff_values_lstm(idx_predicted_lstm);
% common_dsp48e_values_predicted_lstm = predicted_dsp48e_values_lstm(idx_predicted_lstm);
% common_latency_max_predicted_lstm = predicted_latency_max_values_lstm(idx_predicted_lstm);
% 
% % Plot 1: LUT vs BRAM for LSTM
% figure;
% yyaxis left;
% plot(common_reuse_factors_lstm, common_lut_values_actual_lstm, '-o', 'DisplayName', 'Actual LUT');
% hold on;
% plot(common_reuse_factors_lstm, common_lut_values_predicted_lstm, '--o', 'DisplayName', 'Predicted LUT');
% ylabel('LUT');
% 
% yyaxis right;
% plot(common_reuse_factors_lstm, common_bram_values_actual_lstm, '-x', 'DisplayName', 'Actual BRAM');
% plot(common_reuse_factors_lstm, common_bram_values_predicted_lstm, '--x', 'DisplayName', 'Predicted BRAM');
% ylabel('BRAM');
% 
% xlabel('log2 Reuse Factor for LSTM');
% title('Reuse Factor vs LUT and BRAM for LSTM (n\_in=32, lstm\_size=11, sequence\_length=67)' );
% grid on;
% legend show;
% hold off;
% 
% % Plot 2: FF vs DSP48E for LSTM
% figure;
% yyaxis left;
% plot(common_reuse_factors_lstm, common_ff_values_actual_lstm, '-o', 'DisplayName', 'Actual FF');
% hold on;
% plot(common_reuse_factors_lstm, common_ff_values_predicted_lstm, '--o', 'DisplayName', 'Predicted FF');
% ylabel('FF');
% 
% yyaxis right;
% plot(common_reuse_factors_lstm, common_dsp48e_values_actual_lstm, '-x', 'DisplayName', 'Actual DSP48E');
% plot(common_reuse_factors_lstm, common_dsp48e_values_predicted_lstm, '--x', 'DisplayName', 'Predicted DSP48E');
% ylabel('DSP48E');
% 
% xlabel('log2 Reuse Factor for LSTM');
% title('Reuse Factor vs FF and DSP48E (n\_in=32, lstm\_size=11, sequence\_length=67)');
% grid on;
% legend show;
% hold off;
% 
% % Plot 3: Latency_Max vs Correct Reuse Factor for LSTM
% figure;
% plot(common_reuse_factors_lstm_latency, common_latency_max_actual_lstm, '-o', 'DisplayName', 'Actual Latency Max');
% hold on;
% plot(common_reuse_factors_lstm_latency, common_latency_max_predicted_lstm, '--o', 'DisplayName', 'Predicted Latency Max');
% ylabel('Latency Max for LSTM');
% xlabel('log2 Reuse Factor for LSTM');
% title('Reuse Factor vs Latency Max (n\_in=32, lstm\_size=11, sequence\_length=67)');
% grid on;
% legend show;
% hold off;
% 
%%%%% Dense Fixed parameters for plot
fixed_n_inputs_dense = 19;
fixed_sequence_length_dense = 1;
fixed_dense_size = 19;

% Filter the actual data for Dense
filtered_data_dense = dense_resources_collapsed( ...
    dense_resources_collapsed.n_inputs_resources == fixed_n_inputs_dense & ...
    dense_resources_collapsed.sequence_length_resource == fixed_sequence_length_dense & ...
    dense_resources_collapsed.dense_size_resource == fixed_dense_size, :);

filtered_data_latency_dense = dense_latency_collapsed( dense_latency_collapsed.n_inputs_latency == fixed_n_inputs_dense & ...
    dense_latency_collapsed.sequence_length_latency == fixed_sequence_length_dense & ...
    dense_latency_collapsed.dense_size_latency == fixed_dense_size, :);

% Filter the predicted data for Dense
predicted_filtered_data_dense = predicted_dense_data( ...
    predicted_dense_data.n_in == fixed_n_inputs_dense & ...
    predicted_dense_data.sequence_length == fixed_sequence_length_dense & ...
    predicted_dense_data.dense_size == fixed_dense_size, :);

% Extract the reuse factors and LUT/BRAM values for Dense
reuse_factors_dense = log2(filtered_data_dense.correct_reuse_factor_dense_resource);
reuse_factors_dense_latency = log2(filtered_data_latency_dense.correct_reuse_factor_dense_latency);
lut_values_dense = filtered_data_dense.lut;
bram_values_dense = filtered_data_dense.bram_18k;
ff_values_dense = filtered_data_dense.ff;
dsp48e_values_dense = filtered_data_dense.dsp48e;
latency_max_values_dense = filtered_data_latency_dense.latency_max;
predicted_lut_values_dense = predicted_filtered_data_dense.predicted_lut;
predicted_bram_values_dense = predicted_filtered_data_dense.predicted_bram;
predicted_ff_values_dense = predicted_filtered_data_dense.predicted_ff;
predicted_dsp48e_values_dense = predicted_filtered_data_dense.predicted_dsp;
predicted_latency_max_values_dense = predicted_filtered_data_dense.predicted_latency_max;
predicted_reuse_factors_dense = log2(predicted_filtered_data_dense.reuse_factor);

% Find the common reuse factors for Dense
[common_reuse_factors_dense, idx_actual_dense, idx_predicted_dense] = intersect(reuse_factors_dense, predicted_reuse_factors_dense);
[common_reuse_factors_dense_latency, idx_actual_dense_latency, idx_predicted_dense_latency] = intersect(reuse_factors_dense_latency, predicted_reuse_factors_dense);

% Actual and predicted data for the common reuse factors for Dense
common_lut_values_actual_dense = lut_values_dense(idx_actual_dense);
common_bram_values_actual_dense = bram_values_dense(idx_actual_dense);
common_ff_values_actual_dense = ff_values_dense(idx_actual_dense);
common_dsp48e_values_actual_dense = dsp48e_values_dense(idx_actual_dense);
common_latency_max_actual_dense = latency_max_values_dense(idx_actual_dense);

common_lut_values_predicted_dense = predicted_lut_values_dense(idx_predicted_dense);
common_bram_values_predicted_dense = predicted_bram_values_dense(idx_predicted_dense);
common_ff_values_predicted_dense = predicted_ff_values_dense(idx_predicted_dense);
common_dsp48e_values_predicted_dense = predicted_dsp48e_values_dense(idx_predicted_dense);
common_latency_max_predicted_dense = predicted_latency_max_values_dense(idx_predicted_dense);

% Plot 1: LUT vs BRAM for Dense
figure;
yyaxis left;
plot(common_reuse_factors_dense, common_lut_values_actual_dense, '-o', 'DisplayName', 'Actual LUT');
hold on;
plot(common_reuse_factors_dense, common_lut_values_predicted_dense, '--o', 'DisplayName', 'Predicted LUT');
ylabel('LUT');

yyaxis right;
plot(common_reuse_factors_dense, common_bram_values_actual_dense, '-x', 'DisplayName', 'Actual BRAM');
plot(common_reuse_factors_dense, common_bram_values_predicted_dense, '--x', 'DisplayName', 'Predicted BRAM');
ylabel('BRAM');

xlabel('log2 Reuse Factor for Dense');
title('Reuse Factor vs LUT and BRAM for Dense (n\_in=19, dense\_size=19, sequence\_length=1)' );
grid on;
legend show;
hold off;

% Plot 2: FF vs DSP48E for Dense
figure;
yyaxis left;
plot(common_reuse_factors_dense, common_ff_values_actual_dense, '-o', 'DisplayName', 'Actual FF');
hold on;
plot(common_reuse_factors_dense, common_ff_values_predicted_dense, '--o', 'DisplayName', 'Predicted FF');
ylabel('FF');

yyaxis right;
plot(common_reuse_factors_dense, common_dsp48e_values_actual_dense, '-x', 'DisplayName', 'Actual DSP48E');
plot(common_reuse_factors_dense, common_dsp48e_values_predicted_dense, '--x', 'DisplayName', 'Predicted DSP48E');
ylabel('DSP48E');

xlabel('log2 Reuse Factor for Dense');
title('Reuse Factor vs FF and DSP48E (n\_in=19, dense\_size=19, sequence\_length=1)');
grid on;
legend show;
hold off;

% Plot 3: Latency_Max vs Correct Reuse Factor for Dense
figure;
plot(common_reuse_factors_dense, common_latency_max_actual_dense, '-o', 'DisplayName', 'Actual Latency Max');
hold on;
plot(common_reuse_factors_dense, common_latency_max_predicted_dense, '--o', 'DisplayName', 'Predicted Latency Max');
ylabel('Latency Max for Dense');
xlabel('log2 Reuse Factor for Dense');
title('Reuse Factor vs Latency Max (n\_in=19, dense\_size=19, sequence\_length=1)');
grid on;
legend show;
hold off;




% function [train_x, train_y, test_x, test_y, val_x, val_y] = prep_data_for_training(dataset, input_cols, output_cols)
%     n = size(dataset,1);
%     idx_scrambled = randperm(n);
%     idxTrain = idx_scrambled(1:floor(n*.8));
%     idxTest = idx_scrambled(ceil(n*.8):n);
%     idxValidation = [];
% 
%     train_x = dataset(idxTrain,input_cols);
%     train_y = dataset(idxTrain,output_cols);
%     test_x = dataset(idxTest,input_cols);
%     test_y = dataset(idxTest,output_cols);
%     val_x = dataset(idxValidation,input_cols);
%     val_y = dataset(idxValidation,output_cols);
% end

%%
% [idxTrain,idxValidation,idxTest] = trainingPartitions(size(collapsed_conv_resources,1),0.8,0,0.2);

% resources for conv
% input_cols = ["cnn_filters_resources","sequence_length_resource","n_inputs_resources","reuse_factor"];
% output_cols = ["dsp48e","lut","ff","bram_18k"];
% %[train_x, train_y, test_x, test_y, val_x, val_y] = prep_data_for_training(collapsed_conv_resources, input_cols, output_cols);
% [train_x_resources, train_y_resources, test_x_resources, test_y_resources, val_x_resources, val_y_resources] = prep_data_for_training(collapsed_conv_resources, input_cols, output_cols);
% [model_0,train_error_0,test_error_0] = train_regressor(train_x_resources,train_y_resources,test_x_resources,test_y_resources,val_x_resources,val_y_resources);
% 
% 
% 
% 
% % resources for lstm
% input_cols = ["lstm_size","sequence_length_resource","n_inputs_resources","reuse_factor"];
% output_cols = ["dsp48e","lut","ff","bram_18k"];
% %[train_x, train_y, test_x, test_y, val_x, val_y] = prep_data_for_training(lstm_resources_collapsed, input_cols, output_cols);
% %[model,train_error,test_error] = train_regressor(train_x,train_y,test_x,test_y,val_x,val_y);
% [train_x_resources, train_y_resources, test_x_resources, test_y_resources, val_x_resources, val_y_resources] = prep_data_for_training(lstm_resources_collapsed, input_cols, output_cols);
% [model_1,train_error_1,test_error_1] = train_regressor(train_x_resources,train_y_resources,test_x_resources,test_y_resources,val_x_resources,val_y_resources);
% 
% 
% % resources for dense
% input_cols = ["dense_size_resource","sequence_length_resource","n_inputs_resources","reuse_factor"];
% output_cols = ["dsp48e","lut","ff","bram_18k"];
% % [train_x, train_y, test_x, test_y, val_x, val_y] = prep_data_for_training(dense_resources_collapsed, input_cols, output_cols);
% %[model,train_error,test_error] = train_regressor(train_x,train_y,test_x,test_y,val_x,val_y);
% [train_x_resources, train_y_resources, test_x_resources, test_y_resources, val_x_resources, val_y_resources] = prep_data_for_training(dense_resources_collapsed, input_cols, output_cols);
% [model_2,train_error_2,test_error_2] = train_regressor(train_x_resources,train_y_resources,test_x_resources,test_y_resources,val_x_resources,val_y_resources);
% 
% 
% % latency for conv
% input_cols = ["cnn_filters_latency","sequence_length_latency","n_inputs_latency","reuse_factor"];
% output_cols = ["latency_min","latency_max"];
% %[train_x, train_y, test_x, test_y, val_x, val_y] = prep_data_for_training(conv_latency_collapsed, input_cols, output_cols);
% %[model,train_error,test_error] = train_regressor(train_x,train_y,test_x,test_y,val_x,val_y);
% [train_x_latency, train_y_latency, test_x_latency, test_y_latency, val_x_latency, val_y_latency] = prep_data_for_training(conv_latency_collapsed, input_cols, output_cols);
% [model_3,train_error_3,test_error_3] = train_regressor(train_x_latency,train_y_latency,test_x_latency,test_y_latency,val_x_latency,val_y_latency);
% %[model,train_error,test_error] = train_random_forest(train_x,train_y,test_x,test_y,val_x,val_y);
% 
% % latency for lstm
% input_cols = ["lstm_size","sequence_length_latency","n_inputs_latency","reuse_factor"];
% output_cols = ["latency_min","latency_max"];
% %[train_x, train_y, test_x, test_y, val_x, val_y] = prep_data_for_training(lstm_latency_collapsed, input_cols, output_cols);
% %[model,train_error,test_error] = train_regressor(train_x,train_y,test_x,test_y,val_x,val_y);
% [train_x_latency, train_y_latency, test_x_latency, test_y_latency, val_x_latency, val_y_latency] = prep_data_for_training(lstm_latency_collapsed, input_cols, output_cols);
% [model_4,train_error_4,test_error_4] = train_regressor(train_x_latency,train_y_latency,test_x_latency,test_y_latency,val_x_latency,val_y_latency);
% %[model,train_error,test_error] = train_random_forest(train_x,train_y,test_x,test_y,val_x,val_y);
% 
% 
% % latency for dense
% input_cols = ["dense_size_latency","sequence_length_latency","n_inputs_latency","reuse_factor"];
% output_cols = ["latency_min","latency_max"];
% %[train_x, train_y, test_x, test_y, val_x, val_y] = prep_data_for_training(dense_latency_collapsed, input_cols, output_cols);
% %[model,train_error,test_error] = train_regressor(train_x,train_y,test_x,test_y,val_x,val_y);
% [train_x_latency, train_y_latency, test_x_latency, test_y_latency, val_x_latency, val_y_latency]= prep_data_for_training(dense_latency_collapsed, input_cols, output_cols);
% [model_5,train_error_5,test_error_5] = train_regressor(train_x_latency,train_y_latency,test_x_latency,test_y_latency,val_x_latency,val_y_latency);
% 
% function [model, train_error, test_error, val_error] = train_regressor(train_x, train_y, test_x, test_y, val_x, val_y)
%     % Convert data tables to arrays
%     train_x_array = table2array(train_x);
%     train_y_array = table2array(train_y);
%     test_x_array = table2array(test_x);
%     test_y_array = table2array(test_y);
%     val_x_array = table2array(val_x);
%     val_y_array = table2array(val_y);
% 
%     % Train ensemble regression model using fitrensemble
%     model = fitrensemble(train_x_array, train_y_array, 'Method', 'Bag','NumLearningCycles',100); 
%     % Train error
%     train_y_hat = predict(model, train_x_array);
%     train_error = sqrt(mean((train_y_array - train_y_hat).^2));  % RMSE
%     % Test error
%     test_y_hat = predict(model, test_x_array);
%     test_error = sqrt(mean((test_y_array - test_y_hat).^2));  % RMSE
%     % Validation error
%     val_y_hat = predict(model, val_x_array);
%     val_error = sqrt(mean((val_y_array - val_y_hat).^2));  % RMSE
% end
% 
% function collapsed_table = collapse_table(table, input_columns, output_columns)
%     n = 1;
%     for i = 2:size(table, 1)
%         column_match = 1;
%         if any(table{i, input_columns} ~= table{i-1, input_columns})
%             column_match = 0;
%         end
%         if ~column_match
%             if n>1
%                 standard_deviation = std(table{(i-n):(i-1), output_columns});
%                 for j = 1:numel(standard_deviation)
%                     table{i-1, strcat(output_columns{j}, "_std")} = standard_deviation(j);
%                 end
% 
%             end
%             n = 1;
%         else
%             n = n + 1;
%         end
%     end
%     collapsed_table = table;
% end
function collapsed_table = collapse_table(table, input_columns, output_columns)
    n = 1;
    i = 2;
    while i <= size(table, 1)
        column_match = 1;
        % compare the values in the input_columns for current and previous
        % rows
        for col = input_columns
            if table{i, col} ~= table{i-1, col}
                column_match = 0;
                break;
            end
        end

        if ~column_match
            % calculate the mean for output_columns for the previous
            % blocks
            if n > 1
                mean_values = mean(table{(i-n):(i-1), output_columns});
                for j = 1:numel(output_columns)
                    table{i-1, output_columns{j}} = mean_values(j);
                end
                % delete similar rows
                table((i-n):(i-2), :) = []; %first row of the group:the row just before matching block
                i = i - (n - 1);  %n-1 no. of rows deleted
            end
            n = 1; 
        else
            n = n + 1; 
        end
        i = i + 1; %indexing after deleting rows
    end
    collapsed_table = table;
end


% function adjusted_reuse_factor = adjust_reuse_factor(desired_reuse_factor,n_in,n_out)
% 
% 	valid_reuse_factors = [];
% 
% 	min_dist = 1e10;
% 
% 	for r = 1:(n_in * n_out)
% 
% 		mult_limit = ceil(n_in * n_out / min([n_in r]));
% 
% 		valid = (mod(mult_limit,n_out)==0 || r>=n_in) && ...
% 			(mod(r,n_in)==0 || r<n_in) && ...
% 			(mod(n_in * n_out,r)==0);
% 
% 		if valid
% 			valid_reuse_factors = [valid_reuse_factors r];
% 		end
% 
% 		dist = abs(r-desired_reuse_factor);
% 		if dist < min_dist
% 			min_dist = dist;
% 			adjusted_reuse_factor = r;
% 		end
% 
%   end
%     if isempty(valid_reuse_factors)
%         adjusted_reuse_factor=desired_reuse_factor;
%     end
% 
% end
function is_valid = validate_reuse_factor(n_in, n_out, rf)
    multfactor = min(n_in, rf);
    multiplier_limit = ceil((n_in * n_out) / multfactor);
    
    is_valid = (mod(multiplier_limit, n_out) == 0) || (rf >= n_in);
    is_valid = is_valid && ((mod(rf, n_in) == 0) || (rf < n_in));
    
    is_valid = is_valid && (mod(n_in * n_out, rf) == 0);
end

function valid_reuse_factors = get_valid_reuse_factors(n_in, n_out)
    max_rf = n_in * n_out;
    valid_reuse_factors = [];
    for rf = 1:max_rf
        is_valid = validate_reuse_factor(n_in, n_out, rf);
        if is_valid
            valid_reuse_factors = [valid_reuse_factors, rf];
        end
    end
end

function closest_rf = get_closest_reuse_factor(valid_rf, chosen_rf)
   
    diffs = abs(valid_rf - chosen_rf);
    [~, idx] = min(diffs);
   
    closest_rf = valid_rf(idx);
end


function [conv_table,lstm_table,dense_table,data] = extract_training_data(data,resources_or_latency)

    if ~resources_or_latency
        search_key = "resource_instance";
    else
        search_key = "latency_instance";
    end

    % add a column with the layer index to express lexographic ordering of
    % layers across the rows of the table (different for resources and
    % latency due to the differences in ordering of the report files
    for i=1:size(data,1)
        resource_instance = data{i,search_key};
        
        if isempty(regexp(resource_instance,'Instance'))
            idx1 = regexp(resource_instance,'\d+_U\d+\s*$','match');
            idx2 = regexp(idx1,'^\d+','match');
            data{i,"layer_index"} = str2num(idx2);
        else
            data{i,"layer_index"} = -1;
        end
    end

    % sort rows in order of neural network parameters and layer number
    data=sortrows(data,["inputs","cnn_layers","cnn_filters","lstm_layers","lstm_size","dense_layers","dense_size","fit_status","reuse_factor","layer_index"]);

    % compute input sizes for individual layers
   for i = 1:size(data, 1)
    % Add column to table for input size for individual layers
    resource_instance = data{i, "resource_instance"};
    latency_instance = data{i, "latency_instance"};
    if (~resources_or_latency && ~isempty(regexp(resource_instance, 'Instance'))) || ...
       (resources_or_latency && ~isempty(regexp(latency_instance, 'Instance')))
        layer_idx = 0;
        n_cnn = data{i, "cnn_layers"};
        n_lstm = data{i, "lstm_layers"};
        n_dense = data{i, "dense_layers"} + 1;
    else
        % Translate layer number to layer type and layer number within the type
        if (~resources_or_latency && (~isempty(regexp(resource_instance, "conv")) || ...
           ~isempty(regexp(resource_instance, "lstm")) || ~isempty(regexp(resource_instance, "dense")))) || ...
           (resources_or_latency && (~isempty(regexp(latency_instance, "conv")) || ...
           ~isempty(regexp(latency_instance, "lstm")) || ~isempty(regexp(latency_instance, "dense"))))
            user_reuse_factor = data{i, "reuse_factor"};
           
            if layer_idx < n_cnn
                % This is a CNN layer
                cnn_layer_num = layer_idx;
                %n_in = data{i, "cnn_filters"} * 3; % kernel_size=3
                n_out = data{i, "cnn_filters"};
                % Get valid reuse factors and closest reuse factor
               
                if cnn_layer_num == 0
                    n_inputs = data{i, "inputs"};
                    n_in=  3;
                    valid_reuse_factors = get_valid_reuse_factors(n_in, n_out);
                    correct_reuse_factor = get_closest_reuse_factor(valid_reuse_factors, user_reuse_factor);
                    if ~resources_or_latency
                        data{i, "sequence_length_resource"} = 1;
                        data{i, "n_inputs_resources"} = n_inputs;
                   
                    
                        data{i, "correct_reuse_factor_resource"} = correct_reuse_factor;
                        data{i,"cnn_filters_resources"}=n_out;
                       
                    else
                        data{i, "sequence_length_latency"} = 1;
                        data{i, "n_inputs_latency"} = n_inputs;
                    
                        data{i, "correct_reuse_factor_latency"} = correct_reuse_factor;
                        data{i,"cnn_filters_latency"}=n_out;
                     
                    end
                else
                    n_inputs = data{i, "cnn_filters"};
                    n_in=data{i, "cnn_filters"} * 3; % kernel_size=3
                    valid_reuse_factors = get_valid_reuse_factors(n_in, n_out);
                    correct_reuse_factor = get_closest_reuse_factor(valid_reuse_factors, user_reuse_factor);
                    
                    if ~resources_or_latency
                        data{i, "n_inputs_resources"} = n_inputs;
                        data{i, "sequence_length_resource"} = data{i, "inputs"} / 2^(cnn_layer_num - 1);
                         
                        data{i, "correct_reuse_factor_resource"} = correct_reuse_factor;
                        data{i,"cnn_filters_resources"}=n_out;
                         
                    else
                        data{i, "n_inputs_latency"} = n_inputs;
                        data{i, "sequence_length_latency"} = data{i, "inputs"} / 2^(cnn_layer_num - 1);
                    
                        data{i, "correct_reuse_factor_latency"} = correct_reuse_factor;
                        data{i,"cnn_filters_latency"}=n_out;
                      
                    end
                end
            elseif layer_idx < (n_cnn + n_lstm)
                % This is an LSTM layer
                lstm_layer_num = layer_idx - n_cnn;
                if lstm_layer_num == 0
                    n_inputs = data{i, "cnn_filters"};
                else
                    n_inputs = data{i, "lstm_size"};
                end
                n_out_lstm = data{i, "lstm_size"} * 4;
                % Get valid reuse factors and closest reuse factor
                valid_reuse_factors_lstm = get_valid_reuse_factors(n_inputs, n_out_lstm);
                correct_reuse_factor_lstm = get_closest_reuse_factor(valid_reuse_factors_lstm, user_reuse_factor);
                if ~resources_or_latency
                    data{i, "n_inputs_resources"} = n_inputs;
                    data{i, "sequence_length_resource"} = data{i, "inputs"} /2^n_cnn;
               
                    data{i, "correct_reuse_factor_lstm_resource"} = correct_reuse_factor_lstm;
                    
                else
                    data{i, "n_inputs_latency"} = n_inputs;
                    data{i, "sequence_length_latency"} = data{i, "inputs"} /2^n_cnn;
                
                    data{i, "correct_reuse_factor_lstm_latency"} = correct_reuse_factor_lstm;
                    
                end
            else
                % This is a Dense layer
                dense_layer_num = layer_idx - n_cnn - n_lstm;
                % Default
                %n_inputs = data{i, "dense_size"};
                %n_inputs=1;
                dense_size = data{i, "dense_size"};
                if dense_layer_num == 0 && n_lstm > 0
                    n_inputs = data{i, "lstm_size"}*data{i, "inputs"} / 2^n_cnn;
                    %sequence_length = data{i, "inputs"} / 2^n_cnn;
                     sequence_length = 1;
                
                elseif dense_layer_num == 0 && n_lstm == 0 % No LSTMs
                    n_inputs = data{i, "cnn_filters"}*data{i, "inputs"} / 2^n_cnn;
                    %sequence_length = data{i, "inputs"} / 2^n_cnn;
                     sequence_length = 1;
                else
                        % For other dense layers
                    n_inputs = data{i, "dense_size"};
                    %sequence_length = data{i, "inputs"} / 2^n_cnn;
                    sequence_length = 1;
                end
                if dense_layer_num == n_dense - 1
                    dense_size = 1;
                    sequence_length = 1;
                end
                n_out_dense = data{i, "dense_size"};
                % Get valid reuse factors and closest reuse factor
                valid_reuse_factors_dense = get_valid_reuse_factors(n_inputs, n_out_dense);
                correct_reuse_factor_dense = get_closest_reuse_factor(valid_reuse_factors_dense, user_reuse_factor);
                if ~resources_or_latency
                    data{i, "dense_size_resource"} = dense_size;
                    data{i, "sequence_length_resource"} = sequence_length;
                    data{i, "n_inputs_resources"} = n_inputs;
               
                    data{i, "correct_reuse_factor_dense_resource"} = correct_reuse_factor_dense;

                    
                    
                else
                    data{i, "dense_size_latency"} = dense_size;
                    data{i, "sequence_length_latency"} = sequence_length;
                    data{i, "n_inputs_latency"} = n_inputs;
                data{i, "correct_reuse_factor_dense_latency"} =user_reuse_factor;
                    
                end
            end
            layer_idx = layer_idx + 1;
        else
            % Non-compute layers such as ReLU, pooling, etc.
            if ~resources_or_latency
                data{i, "n_inputs_resources"} = -1;
            else
                data{i, "n_inputs_latency"} = -1;
            end
        end
    end
   end
%%
    conv_table = filter_table(data,"conv",resources_or_latency);
    lstm_table = filter_table(data,"lstm",resources_or_latency);
    dense_table = filter_table(data,"dense",resources_or_latency);
end
%%

function filtered_table = filter_table(input_table,type,resource_or_latency)
    % create separate tables for each layer type and delete unneeded rows
    if ~resource_or_latency
        matching_rows=regexp(input_table{:,"resource_instance"},type);
    else
        matching_rows=regexp(input_table{:,"latency_instance"},type);
    end
    matching_rows_logical=zeros(size(matching_rows,1),1);
    for i=1:size(matching_rows,1)
        if isempty(matching_rows{i})
            matching_rows_logical(i)=0;
        else
            matching_rows_logical(i)=1;
        end
    end

    if ~resource_or_latency
        if strcmp(type,"conv")
            cols = ["model_number","fit_status","layer_index","reuse_factor","correct_reuse_factor_resource","n_inputs_resources","cnn_filters_resources","sequence_length_resource","bram_18k","lut","ff","dsp48e"];
        elseif strcmp(type,"lstm")
            cols = ["model_number","fit_status","layer_index","reuse_factor","correct_reuse_factor_lstm_resource","n_inputs_resources","sequence_length_resource","lstm_size","bram_18k","lut","ff","dsp48e"];
        else
            cols = ["model_number","fit_status","layer_index","reuse_factor","correct_reuse_factor_dense_resource","n_inputs_resources","sequence_length_resource","dense_size_resource","bram_18k","lut","ff","dsp48e"];
        end
    else
        if strcmp(type,"conv")
            cols = ["model_number","fit_status","layer_index","reuse_factor","correct_reuse_factor_latency","cnn_filters_latency","n_inputs_latency","sequence_length_latency","latency_min","latency_max"];
        elseif strcmp(type,"lstm")
            cols = ["model_number","fit_status","layer_index","reuse_factor","correct_reuse_factor_lstm_latency","n_inputs_latency","sequence_length_latency","reuse_factor","n_inputs_latency","lstm_size","latency_min","latency_max"];
        else
            cols = ["model_number","fit_status","layer_index","reuse_factor","correct_reuse_factor_dense_latency","n_inputs_latency","sequence_length_latency","dense_size_latency","reuse_factor","latency_min","latency_max"];
        end
    end

    filtered_table = input_table(find(matching_rows_logical),cols);
    filtered_table = sortrows(filtered_table,[4,5,6,7]);

   
   
    
  
 
end

function conv1d_predictions_from = prediction(filename, dataLines)
%IMPORTFILE Import data from a text file
%  CONV1D_PREDICTIONS_FROM = IMPORTFILE(FILENAME) reads data from text
%  file FILENAME for the default selection.  Returns the data as a table.
%
%  CONV1D_PREDICTIONS_FROM = IMPORTFILE(FILE, DATALINES) reads data for
%  the specified row interval(s) of text file FILENAME. Specify
%  DATALINES as a positive scalar integer or a N-by-2 array of positive
%  scalar integers for dis-contiguous row intervals.
%
%  Example:
%  conv1d_predictions_from = importfile("/MATLAB Drive/conv1d_predictions_from.csv", [2, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 22-Sep-2024 01:57:18

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [2, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 11);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["layer_name", "reuse_factor", "sequence_length", "cnn_filters", "cnn_in", "predicted_lut", "predicted_ff", "predicted_bram", "predicted_dsp", "predicted_latency_min", "predicted_latency_max"];
opts.VariableTypes = ["string", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "layer_name", "WhitespaceRule", "preserve");
opts = setvaropts(opts, "layer_name", "EmptyFieldRule", "auto");

% Import the data
conv1d_predictions_from = readtable(filename, opts);

end
function sep19_combined_csv_file = importfile_data(filename, dataLines)
%IMPORTFILE Import data from a text file
%  SEP19_COMBINED_CSV_FILE = IMPORTFILE(FILENAME) reads data from text
%  file FILENAME for the default selection.  Returns the data as a table.
%
%  SEP19_COMBINED_CSV_FILE = IMPORTFILE(FILE, DATALINES) reads data for
%  the specified row interval(s) of text file FILENAME. Specify
%  DATALINES as a positive scalar integer or a N-by-2 array of positive
%  scalar integers for dis-contiguous row intervals.
%
%  Example:
%  sep19_combined_csv_file = importfile("/MATLAB Drive/sep19_combined_csv_file.csv", [2, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 22-Sep-2024 09:37:06

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [2, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 26);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["inputs", "cnn_layers", "cnn_filters", "lstm_layers", "lstm_size", "dense_layers", "dense_size", "fit_status", "min_latency", "max_latency", "reuse_factor", "min_gflops_GFLOPS_", "max_gflops_GFLOPS_", "total_cost", "total_lut", "total_ff", "total_dsp48e", "total_bram_18k", "resource_instance", "bram_18k", "lut", "ff", "dsp48e", "latency_instance", "latency_min", "latency_max"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "string", "double", "double", "double", "double", "string", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["resource_instance", "latency_instance"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["fit_status", "resource_instance", "latency_instance"], "EmptyFieldRule", "auto");

% Import the data
sep19_combined_csv_file = readtable(filename, opts);

end


function lstm_predictions_from = importfile2(filename, dataLines)
%IMPORTFILE Import data from a text file
%  LSTM_PREDICTIONS_FROM = IMPORTFILE(FILENAME) reads data from text
%  file FILENAME for the default selection.  Returns the data as a table.
%
%  LSTM_PREDICTIONS_FROM = IMPORTFILE(FILE, DATALINES) reads data for
%  the specified row interval(s) of text file FILENAME. Specify
%  DATALINES as a positive scalar integer or a N-by-2 array of positive
%  scalar integers for dis-contiguous row intervals.
%
%  Example:
%  lstm_predictions_from = importfile("/MATLAB Drive/lstm_predictions_from.csv", [2, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 21-Sep-2024 23:00:12

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [2, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 11);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["layer_name", "reuse_factor", "sequence_length", "lstm_size", "n_in", "predicted_lut", "predicted_ff", "predicted_bram", "predicted_dsp", "predicted_latency_min", "predicted_latency_max"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "layer_name", "TrimNonNumeric", true);
opts = setvaropts(opts, "layer_name", "ThousandsSeparator", ",");

% Import the data
lstm_predictions_from = readtable(filename, opts);

end


function dense_predictions_from = predict_dense(filename, dataLines)
%IMPORTFILE Import data from a text file
%  DENSE_PREDICTIONS_FROM = IMPORTFILE(FILENAME) reads data from text
%  file FILENAME for the default selection.  Returns the data as a table.
%
%  DENSE_PREDICTIONS_FROM = IMPORTFILE(FILE, DATALINES) reads data for
%  the specified row interval(s) of text file FILENAME. Specify
%  DATALINES as a positive scalar integer or a N-by-2 array of positive
%  scalar integers for dis-contiguous row intervals.
%
%  Example:
%  dense_predictions_from = importfile("/MATLAB Drive/dense_predictions_from.csv", [2, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 22-Sep-2024 11:59:21

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [2, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 11);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["layer_name", "reuse_factor", "dense_size", "n_in", "sequence_length", "predicted_lut", "predicted_ff", "predicted_bram", "predicted_dsp", "predicted_latency_min", "predicted_latency_max"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "layer_name", "TrimNonNumeric", true);
opts = setvaropts(opts, "layer_name", "ThousandsSeparator", ",");

% Import the data
dense_predictions_from = readtable(filename, opts);

end
