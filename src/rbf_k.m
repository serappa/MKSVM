function x = rbf_k(x1, x2, sigma)
    n = size(x1,1);
    m = size(x2,1);
    %produce a nxm kernel dot product matrix of every feature vector in x1
    %with every other feature vector in x2
    %Note, that gram matrix is produced when x1 = x2
    x = zeros(n,m);
    if (isequal(x1,x2))
        %Gram matrix case
        for i = 1:n
            for j = i:m
                x(i,j) = exp (- (norm(x1(i,:)' - x2(j,:)')^2/( 2*(sigma^2))) );
                x(j,i) = x(i,j);
            end
        end
    end
    for i = 1:n
        for j = 1:m
            x(i,j) = exp (- (norm(x1(i,:)' - x2(j,:)')^2/( 2*(sigma^2))) );
        end
    end
    
end