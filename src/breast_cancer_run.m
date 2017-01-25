
data_splits = [90 10; 90 30; 70 10; 70 30; 50 10; 50 30];
results = zeros(size(data_splits,1), 6);
res_i = 1;
for ds_idx = 1:size(data_splits,1)
    tes_p = data_splits(ds_idx,1);
    val_p = data_splits(ds_idx,2);
    figure
    [tra, val, tes] = get_data(tes_p,val_p,'bc');
    cv_err = zeros(length(tra),1);

    outlier_costs = logspace(-2,2, 5);
    krnl_params = 0.5:0.5:5;
    err = zeros(length(outlier_costs), length(krnl_params));
    C = 0;
    sigma = 0;
    min_err = 1;
    for i = 1:length(outlier_costs)
        for j = 1:length(krnl_params)
            cv_err = zeros(length(tra),1);
            for cvi = 1:length(tra)
               [theta,cv_err(cvi),K,K0] = train_kernel_breast_cancer_SVM(tra(cvi), val(cvi), @rbf_k, outlier_costs(i), krnl_params(j), true);
            end
            cv_err = mean(cv_err);
            err(i,j) = cv_err;
            if(cv_err<min_err)
                min_err = cv_err;
                C = outlier_costs(i);
                sigma = krnl_params(j);
            end
        end
    end
    surf(krnl_params, log10(outlier_costs),err);

    ylabel('Log_1_0 of Outlier penalty cost, C');
    xlabel('Kernel parameter, sigma');
    zlabel('Cross-validation error');
    title(['Surface plot showing the fine tuning paramaters C and sigma']);
    
    savefig(['breast_cancer' int2str(tes_p)  '_'  int2str(val_p) '.fig']);
    
    [theta,cv_err(1),K,K0] = train_kernel_breast_cancer_SVM(tra(1), val(1), @rbf_k, C, sigma, true);
    ytes_computed= kernel_dot_product(theta, tra(1).x, tes.x, @rbf_k, sigma);
    tes_err = compute_error(tes.y, ytes_computed);
    
    results(res_i, :) = [tes_p val_p C sigma min_err tes_err];
    disp(['training%- ' int2str(100 - tes_p) ', validation%- ' int2str(val_p) ', C- ' num2str(C) ', sigma- ' num2str(sigma) ', Cross Validation accuracy- ' num2str(1-min_err)  ', Test accuracy- ' num2str(1-tes_err)])
    res_i = res_i +1;
end








