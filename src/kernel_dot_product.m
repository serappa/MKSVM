function dotp = kernel_dot_product(theta, xtra, x, kernel_func, kernel_param)
    %theta has bias subsumed
    
    kernel_mat = kernel_func(xtra,x, kernel_param);
    
    %To encorporate adding bias that is present in theta
    kernel_mat = [kernel_mat ; ones(1,size(kernel_mat,2))];
   
    dotp = theta' * kernel_mat;
    dotp = dotp';
    
end