function [tra, val, tes] = get_data(tes_percent, val_percent, data_set)
    if(strcmp('handwritten',data_set))
        y_idx = 65; x_start = 1; x_end = 64;
        all_data = importdata('Handwritten\optdigits.tra');
        all_data = [all_data; importdata('Handwritten\optdigits.tes')];
        all_data(:, x_start:x_end) = unit_std_0_mu_normalize_columns(all_data(:,x_start:x_end));
    end
    if(strcmp('bc',data_set))
        y_idx = 11; x_start = 2; x_end = 10;
        all_data = importdata('BreastCancer/breast-cancer-wisconsin_.data');
        all_data(:, x_start:x_end) = unit_std_0_mu_normalize_columns(all_data(:,x_start:x_end));
    end
   
    if(strcmp('forestTypes',data_set))
        y_idx = 28; 
        x_start = 1; x_end = 9;
        all_data = importdata('ForestTypes/training.csv');
        all_data.textdata = all_data.textdata(2:end,1);
        tmp = importdata('ForestTypes/testing.csv');
        tmp.textdata = tmp.textdata(2:end,1);
        all_data.data = [all_data.data; tmp.data];
        all_data.textdata = [all_data.textdata; tmp.textdata];
        
        all_data.data = [all_data.data zeros(size(all_data.data,1),1)];
        classes = unique(all_data.textdata);
        %The first class is already labelled 0, classes are labeled from 0 to k-1
        for k = 2:size(classes,1)
            all_data.data(strcmp(all_data.textdata,classes(k)),y_idx) = k-1;
        end
        all_data = all_data.data;
        all_data(:, x_start:x_end) = unit_std_0_mu_normalize_columns(all_data(:,x_start:x_end));
    end
    
    ho_part = cvpartition(all_data(:,y_idx), 'HoldOut', tes_percent/100);
    tra_data = all_data(training(ho_part),:);
    tes_data = all_data(test(ho_part),:);
    kfold_part = cvpartition(tra_data(:,y_idx), 'KFold', floor((1/val_percent)*100));

    for i = 1:size(kfold_part.TrainSize,2)
        idx = training(kfold_part,i);
        y =  tra_data(idx,y_idx);
        classes = unique(y);
        y_mat = zeros(size(y,1), size(classes,1));
        for k = 1:size(classes,1)
            y_mat(:,k) = (y == classes(k));
            y_mat(y_mat(:,k) == 0,k) = -1;
        end
        if(size(y_mat,2) == 2)
            y_mat = y_mat(:,1);
        end
        tra(i) = struct('x', tra_data(idx,x_start:x_end), 'y', y_mat, 'y_label', tra_data(idx,y_idx));
        
        y =  tra_data(not(idx),y_idx);
        classes = unique(y);
        y_mat = zeros(size(y,1), size(classes,1));
        for k = 1:size(classes,1)
            y_mat(:,k) = (y == classes(k));
            y_mat(y_mat(:,k) == 0,k) = -1;
        end
        if(size(y_mat,2) == 2)
            y_mat = y_mat(:,1);
        end
        val(i) = struct('x', tra_data(not(idx),x_start:x_end), 'y', y_mat, 'y_label', tra_data(not(idx),y_idx));
    end
    
    y = tes_data(:,y_idx);
    classes = unique(y);
    y_mat = zeros(size(y,1), size(classes,1));
    for k = 1:size(classes,1)
        y_mat(:,k) = (y == classes(k));
        y_mat(y_mat(:,k) == 0,k) = -1;
    end
    if(size(y_mat,2) == 2)
        y_mat = y_mat(:,1);
    end
    tes = struct('x', tes_data(:,x_start:x_end), 'y', y_mat, 'y_label', tes_data(:,y_idx));
end