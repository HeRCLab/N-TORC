% read ground truth data
conv_resources = readtable("data/cnn/collapsed_conv_resources.csv");
lstm_resources = readtable("data/lstm/lstm_resources_collapsed.csv");
dense_resources = readtable("data/dense/dense_resources_collapsed.csv");

conv_latency = readtable("data/cnn/conv_latency_collapsed.csv");
lstm_latency = readtable("data/lstm/lstm_latency_collapsed.csv");
dense_latency = readtable("data/dense/dense_latency_collapsed.csv");

% remove reuse factor 1, since these synthesize the latency optimized
% version of the hls4ml code
dense_latency(dense_latency{:,"correct_reuse_factor_dense_latency"}==1,:) = [];

% open figure
close all;
n=1;

legend_text={};

subplot(3,2,n);n=n+1;
legend_text = plot_resources (conv_resources{:,"n_inputs_resources"} .* 3, ...
                conv_resources{:,"cnn_filters_resources"}, ... 
                conv_resources{:,"correct_reuse_factor_resource"},...
                conv_resources{:,"lut"}, ...
                "LUTs", ...
                "conv",...
                "n\_out=",...
                "conv1d layer cost",...
                legend_text, ...
                conv_resources{:,"cnn_filters_resources"} * 3,...
                1,0);

legend_text={};
subplot(3,2,n);n=n+1;
legend_text = plot_resources (conv_latency{:,"n_inputs_latency"} * 3,...
                conv_latency{:,"cnn_filters_latency"}, ...
                conv_latency{:,"correct_reuse_factor_latency"},...
                conv_latency{:,"latency_max"}, ...
                "latency (cycles)", ...
                "conv",...
                "sequence length=",...
                "conv1d layer latency",...
                legend_text, ...
                conv_latency{:,"sequence_length_latency"},...
                1,1);

legend_text = {};
subplot(3,2,n);n=n+1;
legend_text = plot_resources (lstm_resources{:,"n_inputs_resources"},...
                lstm_resources{:,"lstm_size"} * 4, ...
                lstm_resources{:,"correct_reuse_factor_lstm_resource"},...
                lstm_resources{:,"lut"}, ...
                "LUTs", ...
                "LSTM",...,
                "n\_out=",...
                "LSTM layer cost",...
                legend_text, ...
                lstm_resources{:,"lstm_size"} * 4,...
                1,0);

legend_text = {};
subplot(3,2,n);n=n+1;
legend_text = plot_resources (lstm_latency{:,"n_inputs_latency"},...
                lstm_latency{:,"lstm_size"} * 4, ...
                lstm_latency{:,"correct_reuse_factor_lstm_latency"},...
                lstm_latency{:,"latency_max"}, ...
                "latency (cycles)", ...
                "LSTM",...
                "sequence length=",...
                "LSTM layer latency",...
                legend_text, ...
                lstm_latency{:,"sequence_length_latency"},...
                1,1);

legend_text = {};
subplot(3,2,n);n=n+1;
legend_text = plot_resources (dense_resources{:,"n_inputs_resources"},...
                dense_resources{:,"dense_size_resource"}, ...
                dense_resources{:,"correct_reuse_factor_dense_resource"},...
                dense_resources{:,"lut"}, ...
                "LUTs", ...
                "dense",...
                "n\_in=",...
                "dense layer cost",...
                legend_text, ...
                dense_resources{:,"n_inputs_resources"},...
                1,0);

hold off;
legend_text = {};
subplot(3,2,n);n=n+1;
legend_text = plot_resources (dense_latency{:,"n_inputs_latency"},...
                dense_latency{:,"dense_size_latency"}, ...
                dense_latency{:,"correct_reuse_factor_dense_latency"},...
                dense_latency{:,"latency_max"}, ...
                "latency (cycles)", ...
                "dense",...
                "n\_in=",...
                "dense layer latency",...
                legend_text, ...
                dense_latency{:,"n_inputs_latency"},...
                1,1);

function legend_text = plot_resources (n_in,n_out,rf,resources,resource_name,layer_name,legend_prefix,title_text,legend_text,class_col,logscale,use_only_rf)

    %figure;
    hold on;

    % clear dense layers with one neuron
    idx_to_clear = or(n_in==3,n_out==1);
    n_in(idx_to_clear)=[];
    n_out(idx_to_clear)=[];
    rf(idx_to_clear)=[];
    resources(idx_to_clear)=[];
    class_col(idx_to_clear)=[];

    if use_only_rf==0
        block_factor = n_in .* n_out ./ rf;
    else
        block_factor = rf;
    end
    unique_block_factors = unique(block_factor);
    
    unique_classes = unique(class_col);
    
    % set up plot colors
    colors = lines(size(unique_classes,1));
    
    mu_vector=[];
    sig_vector=[];
    bf_vector=[];
    class_vector=[];
    n_vector=[];
    for class = unique_classes'
        for bf = unique_block_factors'
            idx_a = and((block_factor == bf),(class_col == class));
            n = sum(idx_a);
            if sum(idx_a) > 0
                class_vector = [class_vector class];
                mu=mean(resources(idx_a));
                sig=std(resources(idx_a));
                mu_vector = [mu_vector mu];
                sig_vector = [sig_vector sig];
                bf_vector = [bf_vector bf];
                n_vector = [n_vector n];
            end
        end
    end

    traces=[];

    n=1;
    for class = unique_classes'
        mu_vector2 = mu_vector(class==class_vector);
        sig_vector2 = sig_vector(class==class_vector);
        bf_vector2 = bf_vector(class==class_vector);
        n_vector2 = n_vector(class==class_vector);
        [~,idx_sort] = sort(bf_vector2);
        text = legend_prefix+class;
        p1=plot(log2(bf_vector2(idx_sort)),mu_vector2(idx_sort),'o-','LineWidth',1,'DisplayName',text);
        p2=errorbar(log2(bf_vector2(idx_sort)),mu_vector2(idx_sort),sig_vector2(idx_sort),'LineWidth',1);
        traces=[traces p1];
        p1.Color = colors(n,:);
        p2.Color = colors(n,:);
        n = n+1;
        %legend_text = [legend_text layer_name+class layer_name+class];
        %bar(log2(bf_vector2(idx_sort)),n_vector2(idx_sort));
    end
    if logscale
        yscale('log');
    end
    if use_only_rf==0
        xlabel('log2(block factor)');
    else
        xlabel('log2(reuse factor)');
    end
    ylabel(resource_name);
    legend(traces);
    title(title_text);
    hold off;
end

% data = data
% ind_col = indenpendent variable(s)
% class_col = critical for training separate models (can be [])
% dep_col = dependent variable
% modeltype = 1 for linear, 2 for order-2 polynomial, 3 for exponential 4 for NN
% class_col = the column(s) corresponding to each independent model
% ind_name = independent variable name
% dep_name = dependent variable name
% dep_col = the column that we're predicting
% plot_var = which independent variable we want to plot against
% normalize = use standard normalization
function [] = piecewise_model (data,ind_col,class_col,dep_col,modeltype,ind_name,dep_name,plot_col,normalize)
    
    % one figure for each call
    figure;
    hold on;
    legend_val={};
    n=1;

    if modeltype == 1 && size(ind_col,2)>1
        modeltype = 5;
    end

    if ~isempty(class_col)
        % train separate models for each class
        classes_to_model = unique(data(:,class_col),'rows');
        n_max = size(classes_to_model,1);
        for i=1:size(classes_to_model,1)
            % extract class data
            idx = find(data(:,class_col)==classes_to_model(i));
            data_class = data(idx,:);
            % create a model
            [model,pred,mu,sig] = create_model(data_class,ind_col,dep_col,modeltype,normalize);
            % add legend entries
            legend_val=[legend_val,"data "+sprintf("%g ",classes_to_model(i,:)),"predicted "+sprintf("%g ",classes_to_model(i,:)),"model "+sprintf("%g ",classes_to_model(i,:))];
            % plot results
            [n,legend_val] = plot_results (n,n_max,legend_val,plot_col,ind_col,dep_col,data_class,mu,sig,normalize,pred,model,modeltype,ind_name,dep_name,[]);
            n=n+1;
        end

    else
        % train one model but evaluate for each class
        [model,pred,mu,sig] = create_model(data,ind_col,dep_col,modeltype,normalize);

        % treat non-plotted independent variables as classes
        % start with all independent variables
        classes = ind_col;
        % remove those that we're plotting against
        classes(find(ind_col==plot_col))=[];
        plot_classes = unique(data(:,classes),'rows');
        n_max = size(plot_classes,1);
        n=1;
        for j=1:size(plot_classes,1)
            idx = find(data(:,classes)==plot_classes(j,:));
            data_filtered  = data(idx,:);
            legend_val=[legend_val,"data "+sprintf("%g ",plot_classes(j,:)),"predicted "+sprintf("%g ",plot_classes(j,:)),"model "+sprintf("%g ",plot_classes(j,:))];
            pred_filtered = pred(idx,1);
            [n,legend_val] = plot_results (n,n_max,legend_val,plot_col,ind_col,dep_col,data_filtered,mu,sig,normalize,pred_filtered,model,modeltype,ind_name,dep_name,plot_classes(j,:));
            n=n+1;
        end
    end      
end

function [model,pred,mu,sig] = create_model(data_class,ind_col,dep_col,modeltype,normalize)
    % optionally normalize data
    mu = mean(data_class);
    sig = std(data_class);
    if normalize
        data_norm = (data_class - mu)./sig;
    else
        data_norm = data_class;
    end

    if modeltype==1
        model = polyfit(data_norm(:,ind_col),data_norm(:,dep_col),1);
        pred_norm = polyval(model,data_norm(:,ind_col));
        
    elseif modeltype==2
        model = polyfit(data_norm(:,ind_col),data_norm(:,dep_col),2);
        pred_norm = polyval(model,data_norm(:,ind_col));

    elseif modeltype==3
        g = fittype('a+b*exp(-c*x)');
        a_start = min(data_norm(:,dep_col));
        b_start = max(data_norm(:,dep_col)) - min(data_norm(:,dep_col));
        model = fit(data_norm(:,ind_col),data_norm(:,dep_col),g,'Start',[a_start,b_start,10],'Lower',[1000,1000,0],'Upper',[Inf,Inf,1]);
        pred_norm = model.a + model.b .* exp(-model.c .* data_norm(:,ind_col));

    elseif modeltype==4
        % not sure how to interpolate with multivariate
        [model,pred_norm]=fit_nn(data_norm(:,ind_col),data_norm(:,dep_col));

    else
        % linear regression
        model = data_norm(:,ind_col) \ data_norm(:,dep_col);
        pred_norm = data_norm(:,ind_col) * model;
    end

    if normalize
        pred = pred_norm .* sig(:,dep_col) + mu(:,dep_col);
    else
        pred = pred_norm;
    end

    %fprintf("rmse = %0.4e\n",mean((pred-data_class(:,dep_col)).^2)^.5);
    percent_error = (data_class(:,dep_col)-pred)./(data_class(:,dep_col)+.01);
    fprintf("rmse = %0.4f%%\n",mean(percent_error.^2)^.5*100);
end

function [n,legend_val] = plot_results (n,n_max,legend_val,ind_col,orig_ind_col,dep_col,data_class,mu,sig,norm,pred,model,modeltype,ind_name,dep_name,class)

    colormap = lines(n_max);
    p1 = plot(data_class(:,ind_col),data_class(:,dep_col),'o','LineWidth',2);
    hold on;
    p2 = plot(data_class(:,ind_col),pred,'x','LineWidth',2);
    xlabel(ind_name);
    ylabel(dep_name);
    %title("model results");
    %plot(data_class(:,1),pred,'-');

    % TODO: support > 2 independent variables
    if numel(ind_col) == numel(orig_ind_col)
        sweep = min(data_class(:,ind_col)):.1:max(data_class(:,ind_col));
        rf = sweep;
    elseif ind_col(1) == orig_ind_col(1)
        sweep = min(data_class(:,ind_col)):.1:max(data_class(:,ind_col));
        rf = [sweep;ones(size(sweep)) * class];
    else
        sweep = min(data_class(:,ind_col)):.1:max(data_class(:,ind_col));
        rf = [ones(size(sweep)) * class;sweep];
    end

    if norm
        rf_norm = (rf - mu(orig_ind_col)') ./ sig(orig_ind_col)';
    else
        rf_norm = rf;
    end

    if modeltype==1 || modeltype==2
        fitted_smooth = polyval(model,rf_norm);
    elseif modeltype==3
        fitted_smooth = model.a + model.b * exp(-model.c .* rf_norm);
    elseif modeltype==4
        fitted_smooth = predict(model,rf_norm');
    else
        fitted_smooth = rf_norm' * model;
    end

    if norm
        fitted_smooth = fitted_smooth .* sig(dep_col) + mu(dep_col);
    end

    p3 = plot(sweep,fitted_smooth,'LineWidth',2);
    %plot(rf,interp1(data_class(:,ind_col),data_class(:,dep_col),rf,"spline"),':','LineWidth',2);
    %legend({"data","predicted","model","spline interpolation"});
    
    legend(legend_val);
    %hold off;
    
    if isempty(p1)
        1;
    end
    p1.Color = colormap(n,:);
    p2.Color = colormap(n,:);
    p3.Color = colormap(n,:);
end

% int 1 for linear augmentation
function [mynet,pred] = fit_nn (x,y)

    % NN
    layers = [featureInputLayer(size(x,2)) fullyConnectedLayer(50) tanhLayer fullyConnectedLayer(1)];
    epochs = 1000;
    opts = trainingOptions('adam', ...
            'MaxEpochs',epochs, ...
            'InitialLearnRate',.01, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',1000, ...
            'LearnRateDropFactor',0.2, ...
            'Verbose',0, ...
            'minibatchsize',size(x,1));

    mynet = trainnet(x,y,layers,"mse",opts);
    pred = predict(mynet,x);
end

function found = found(found_classes,class)
    found = [];
    for i=1:size(found_classes,1)
        if all(found_classes(i,:) == class)
            found = [found i];
        end
    end
end

function [count, mean, M2, variance] = update_stats (count, mean, M2, new_value)
    count = count+1;
    delta = new_value - mean;
    mean = mean + delta / count;
    delta2 = new_value - mean;
    M2 = M2 + delta * delta2;
    variance = M2 / count;
end

function data = add_data_mult_and_nout (data,to_add)
    for i=1:size(to_add,1)
        found=0;
        multiplier_limit_found = to_add{i,"n_in"} * to_add{i,"n_out"} / min(to_add{i,"n_in"},to_add{i,"correct_reuse_factor_resource"});
        block_factor_found = to_add{i,"n_in"} * to_add{i,"n_out"} / to_add{i,"correct_reuse_factor_resource"};
        n_out_found =  to_add{i,"n_out"};
        n_in_found =  to_add{i,"n_in"};

        for j=1:size(data,1)

            % merge criteria
            multiplier_limit_alreadyproc = data(j,12);
            n_out_alreadyproc = data(j,3);
            n_in_alreadyproc = data(j,2);
            block_factor_alreadyproc = data(j,11);
            if (block_factor_found == block_factor_alreadyproc) &&...
                n_out_found == n_out_alreadyproc

                [data(j,4), data(j,6), data(j,7), var] = update_stats (data(j,4), data(j,6), data(j,7), to_add{i,"lut"});
                data(j,8) = var^.5;

                n = data(j,4)-1;
                data(j,5) = n/(n+1) * data(j,5) + 1/(n+1) * to_add{i,"bram_18k"};
                %data(j,6) = n/(n+1) * data(j,6) + 1/(n+1) * to_add{i,"lut"};
                data(j,9) = n/(n+1) * data(j,9) + 1/(n+1) * to_add{i,"ff"};
                data(j,10) = n/(n+1) * data(j,10) + 1/(n+1) * to_add{i,"dsp48e"};
                %data(j,4) = n+1;
                found = 1;
            end
        end
    
        if ~found
            data = [data;...
                to_add{i,"correct_reuse_factor_resource"},...
                to_add{i,"n_in"},...
                to_add{i,"n_out"},...
                1,...
                to_add{i,"bram_18k"},...
                to_add{i,"lut"},...
                0,... % lut_M2
                0,... % lut_std
                to_add{i,"ff"},...
                to_add{i,"dsp48e"},...
                block_factor_found,...
                multiplier_limit_found];
        end
    end
end

function data = add_data_mult_and_nout_latency (data,to_add)
    for i=1:size(to_add,1)
        found=0;
        multiplier_limit_found = to_add{i,"n_in"} * to_add{i,"n_out"} / min(to_add{i,"n_in"},to_add{i,"correct_reuse_factor_lstm_latency"});
        n_out_found =  to_add{i,"n_out"};
        sequence_length_found = to_add{i,"sequence_length_latency"};
        workload_found = to_add{i,"n_in"} * to_add{i,"n_out"} * to_add{i,"sequence_length_latency"};

        for j=1:size(data,1)

            % merge criteria
            multiplier_limit_alreadyproc = data(j,9);
            sequence_length_alreadyproc = data(j,4);
            workload_alreadyproc = data(j,10);

            if (multiplier_limit_found == multiplier_limit_alreadyproc) &&...
                workload_found == workload_alreadyproc

                [data(j,5), data(j,6), data(j,7), var] = update_stats (data(j,4), data(j,6), data(j,7), to_add{i,"latency_min"});
                data(j,8) = var^.5;

                found = 1;
            end
        end
    
        if ~found
            data = [data;...
                to_add{i,"correct_reuse_factor_lstm_latency"},...
                to_add{i,"n_in"},...
                to_add{i,"n_out"},...
                to_add{i,"sequence_length_latency"},...
                1,...
                to_add{i,"latency_min"},...
                0,... % latency_M2
                0,... % latency_std
                multiplier_limit_found,...
                workload_found];
        end
    end
end

