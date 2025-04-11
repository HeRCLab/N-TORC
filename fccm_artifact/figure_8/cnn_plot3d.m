bar_size = 1;

% Set values for CNN
cols = ["latency", "lut", "ff", "dsp48e", "bram"];
titles = ["Latency in Cycles", "LUT Usage", "FF Usage", "DSP Usage", "BRAM Usage"];
layer_cnn = "conv1d";
reuse_factor_cnn = [16, 48, 96, 384];
rf_cnn = categorical([16, 48, 96, 384]);  % Set reuse factors as categorical
data_cnn = readtable("combined_latency_resources.csv");

% Set values for LSTM
layer_lstm = "lstm";
reuse_factor_lstm = [32, 64, 128];
lstm_sizes = [16, 32];
rf_lstm = categorical([32, 64, 128]);  % Set reuse factors as categorical
data_lstm = readtable("combined_lstm_latency_resources.csv");

% Set values for Dense layer
layer_dense = "dense";
reuse_factor_dense = [32, 64, 128];
dense_sizes = [32, 64];
rf_dense = categorical(reuse_factor_dense);  % Set reuse factors as categorical
data_dense = readtable("combined_dense_latency_resources.csv");

% Create figure with tiled layout for all three layers: CNN, LSTM, and Dense
figure;
tiledlayout(3, 5, 'Padding', 'normal', 'TileSpacing', 'normal');

% Set global properties for LaTeX font
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesFontSize', 10, 'defaultAxesFontName', 'Times');

% Plot CNN metrics
x_vals_cnn = [3, 6];  % x-axis positions for CNN filters 16 and 32
for idx = 1:numel(cols)
    col = cols(idx);
    col2 = col;

    % Use correct predicted column names for CNN (adjust for each metric)
    if col == "latency"
        col2 = "predicted_latency_max";
    elseif col == "dsp48e"
        col2 = "predicted_dsp";
    else
        col2 = "predicted_" + col;
    end

    filters16_pred = zeros(1, numel(reuse_factor_cnn));
    filters16_actual = zeros(1, numel(reuse_factor_cnn));
    filters32_pred = zeros(1, numel(reuse_factor_cnn));
    filters32_actual = zeros(1, numel(reuse_factor_cnn));

    % Extract relevant data for each reuse factor
    for i = 1:numel(reuse_factor_cnn)
        data_temp = data_cnn(data_cnn.reuse_factor == reuse_factor_cnn(i), :);
        data_temp16 = data_temp(data_temp.cnn_filters == 16, :);
        data_temp32 = data_temp(data_temp.cnn_filters == 32, :);

        filters16_actual(1, i) = mean(data_temp16{:, col});
        filters16_pred(1, i) = mean(data_temp16{:, col2});
        filters32_actual(1, i) = mean(data_temp32{:, col});
        filters32_pred(1, i) = mean(data_temp32{:, col2});
    end

    % Create subplot for each metric
    nexttile;
    hold on;
    title(layer_cnn + " " + titles(idx));
    xlabel(['Number of' newline 'Filters'], 'FontSize', 12);
    ylabel('Reuse Factor', 'FontSize', 12);
    zlabel(titles(idx), 'FontSize', 12);

    % Plot bars for 16 filters at x = 3 (Actual and Predicted with different colors)
    h_16 = bar3(rf_cnn, [filters16_actual', filters16_pred'], bar_size, 'grouped');
    for ii = 1:length(h_16)
        XData = get(h_16(ii), 'XData');
        set(h_16(ii), 'XData', XData + (x_vals_cnn(1) - 1) * ones(size(XData)));
    end

    % Plot bars for 32 filters at x = 6 (Actual and Predicted with different colors)
    h_32 = bar3(rf_cnn, [filters32_actual', filters32_pred'], bar_size, 'grouped');
    for ii = 1:length(h_32)
        XData = get(h_32(ii), 'XData');
        set(h_32(ii), 'XData', XData + (x_vals_cnn(2) - 1) * ones(size(XData)));
    end

    % Set x-axis labels to represent filter sizes
    set(gca, 'XTick', x_vals_cnn, 'XTickLabel', {'16', '32'});
    set(gca, 'YTickLabel', string(rf_cnn));

    % Set legend with LaTeX interpreter
    legend({'Actual', 'Predicted'}, 'FontSize', 12);
    view([130 29]);
end

% Plot LSTM metrics
x_vals_lstm = [3, 6];  % x-axis positions for LSTM sizes 16 and 32
for idx = 1:numel(cols)
    col = cols(idx);
    col2 = col;

    % Use correct predicted column names for LSTM (adjust for each metric)
    if col == "latency"
        col2 = "predicted_latency_max";
    elseif col == "dsp48e"
        col2 = "predicted_dsp";
    else
        col2 = "predicted_" + col;
    end

    lstm16_pred = zeros(1, numel(reuse_factor_lstm));
    lstm16_actual = zeros(1, numel(reuse_factor_lstm));
    lstm32_pred = zeros(1, numel(reuse_factor_lstm));
    lstm32_actual = zeros(1, numel(reuse_factor_lstm));

    % Extract relevant data for each reuse factor
    for i = 1:numel(reuse_factor_lstm)
        data_temp = data_lstm(data_lstm.reuse_factor == reuse_factor_lstm(i), :);
        data_temp16 = data_temp(data_temp.lstm_units == 16, :);
        data_temp32 = data_temp(data_temp.lstm_units == 32, :);

        lstm16_actual(1, i) = mean(data_temp16{:, col});
        lstm16_pred(1, i) = mean(data_temp16{:, col2});
        lstm32_actual(1, i) = mean(data_temp32{:, col});
        lstm32_pred(1, i) = mean(data_temp32{:, col2});
    end

    % Create subplot for each metric
    nexttile;
    hold on;
    title(layer_lstm + " " + titles(idx));
    xlabel('LSTM Units', 'FontSize', 12);
    ylabel('Reuse Factor', 'FontSize', 12);
    zlabel(titles(idx), 'FontSize', 12);

    % Plot bars for LSTM size 16 at x = 3 (Actual and Predicted with different colors)
    h_16 = bar3(rf_lstm, [lstm16_actual', lstm16_pred'], bar_size, 'grouped');
    for ii = 1:length(h_16)
        XData = get(h_16(ii), 'XData');
        set(h_16(ii), 'XData', XData + (x_vals_lstm(1) - 1) * ones(size(XData)));
    end

    % Plot bars for LSTM size 32 at x = 6 (Actual and Predicted with different colors)
    h_32 = bar3(rf_lstm, [lstm32_actual', lstm32_pred'], bar_size, 'grouped');
    for ii = 1:length(h_32)
        XData = get(h_32(ii), 'XData');
        set(h_32(ii), 'XData', XData + (x_vals_lstm(2) - 1) * ones(size(XData)));
    end

    % Set x-axis labels to represent LSTM sizes
    set(gca, 'XTick', x_vals_lstm, 'XTickLabel', {'16', '32'});
    set(gca, 'YTickLabel', string(rf_lstm));

    % Set legend with LaTeX interpreter
    legend({'Actual', 'Predicted'}, 'FontSize', 12);
    view([130 29]);
end

% Plot Dense metrics
x_vals_dense = [3, 6];  % x-axis positions for Dense sizes 32 and 64
for idx = 1:numel(cols)
    col = cols(idx);
    col2 = col;

    % Use correct predicted column names for Dense (adjust for each metric)
    if col == "latency"
        col2 = "predicted_latency_max";
    elseif col == "dsp48e"
        col2 = "predicted_dsp";
    else
                col2 = "predicted_" + col;
    end

    dense32_pred = zeros(1, numel(reuse_factor_dense));
    dense32_actual = zeros(1, numel(reuse_factor_dense));
    dense64_pred = zeros(1, numel(reuse_factor_dense));
    dense64_actual = zeros(1, numel(reuse_factor_dense));

    % Extract relevant data for each reuse factor
    for i = 1:numel(reuse_factor_dense)
        data_temp = data_dense(data_dense.reuse_factor == reuse_factor_dense(i), :);
        data_temp32 = data_temp(data_temp.dense_units == 32, :);
        data_temp64 = data_temp(data_temp.dense_units == 64, :);

        dense32_actual(1, i) = mean(data_temp32{:, col});
        dense32_pred(1, i) = mean(data_temp32{:, col2});
        dense64_actual(1, i) = mean(data_temp64{:, col});
        dense64_pred(1, i) = mean(data_temp64{:, col2});
    end

    % Create subplot for each metric
    nexttile;
    hold on;
    title(layer_dense + " " + titles(idx));
    xlabel('Dense Units', 'FontSize', 12);
    ylabel('Reuse Factor', 'FontSize', 12);
    zlabel(titles(idx), 'FontSize', 12);

    % Plot bars for Dense size 32 at x = 3 (Actual and Predicted with different colors)
    h_32 = bar3(rf_dense, [dense32_actual', dense32_pred'], bar_size, 'grouped');
    for ii = 1:length(h_32)
        XData = get(h_32(ii), 'XData');
        set(h_32(ii), 'XData', XData + (x_vals_dense(1) - 1) * ones(size(XData)));
    end

    % Plot bars for Dense size 64 at x = 6 (Actual and Predicted with different colors)
    h_64 = bar3(rf_dense, [dense64_actual', dense64_pred'], bar_size, 'grouped');
    for ii = 1:length(h_64)
        XData = get(h_64(ii), 'XData');
        set(h_64(ii), 'XData', XData + (x_vals_dense(2) - 1) * ones(size(XData)));
    end

    % Set x-axis labels to represent dense sizes
    set(gca, 'XTick', x_vals_dense, 'XTickLabel', {'32', '64'});
    set(gca, 'YTickLabel', string(rf_dense));

    % Set legend with LaTeX interpreter
    legend({'Actual', 'Predicted'}, 'FontSize', 12);
    view([130 29]);
end

exportgraphics(gcf, "combined_lstm_cnn_dense_3d_plots.pdf", 'ContentType', 'vector');


