function [theta,cv_err, K, K0] = train_kernel_breast_cancer_SVM(tra,val,krnl,C, krnl_param, use_diff_error_cost)
    e = 1e-6;
    [theta, cv_err, K, K0] = train_kernel_LLS(tra, val, krnl, C, krnl_param);
    tra_cov_matrix = zeros(size(tra.x,1)+1, size(tra.x,1)+1, size(tra.x,1));
    for i = 1:size(tra.x,1)
       tra_cov_matrix(:, :, i) = K(i,:)' * K(i,:); 
    end
    
    y = tra.y;
    cost = 0;
    itr = 0;
    while 1==1
        z = abs(1 - y .* (K * theta));  
        filter = (z < e);
        z(filter,1) = e;
        
        if(use_diff_error_cost)
            class_ratio = sum(y == 1) / sum(y == -1);
            C1 = C / class_ratio;
            C2 = C * class_ratio;
        else 
            C1 = C;
            C2 = C;
        end
        
        hinge_loss_cost = 0;
        for i = 1:size(y,1)
            if(y(i) == 1)
                hinge_loss_cost = hinge_loss_cost + C1* (1/(4*z(i))) * ((1- y(i) * K(i,:) * theta) + z(i))^2;
            else
                hinge_loss_cost = hinge_loss_cost + C2 *(1/(4*z(i))) * ((1- y(i) * K(i,:) * theta) + z(i))^2;
            end
        end
        new_cost = 0.5 * (theta' * K0 * theta) +  hinge_loss_cost;
        cost_change = abs(cost - new_cost)/abs(cost);
        if(cost_change < 1e-3 || itr == 250)
            break;
        end
        cost = new_cost;

        inv_sum1 = 0;
        other_sum1 = 0;
        inv_sum2 = 0;
        other_sum2 = 0;
        for i = 1:size(y,1)
            if(y(i) == 1)
              inv_sum1 = inv_sum1 + C1 *  ( (tra_cov_matrix(:, :,i)) / (2*z(i)) );
              other_sum1 = other_sum1 + C1 * ( ((1+z(i)) / (2*z(i))) * ( y(i) .* K(i,:)') );
            else
              inv_sum2 = inv_sum2 + C2 * ( (tra_cov_matrix(:, :,i)) / (2*z(i)) );
              other_sum2 = other_sum2 + C2 * ( ((1+z(i)) / (2*z(i))) * ( y(i) .* K(i,:)') );
            end
        end
        theta = (pinv (K0 + inv_sum1 + inv_sum2 ) * (other_sum1 + other_sum2));
        itr = itr + 1;
    end 
   

    yval_computed= kernel_dot_product(theta, tra.x, val.x, krnl, krnl_param);
    cv_err = compute_error(val.y, yval_computed);
end
 