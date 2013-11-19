function [V r res] = RNNSC(data, U, restart, V, useGpu)
    c = 0;
    eps = 1e-2;
    if restart
        V = rand(size(U, 2), size(data, 2));
    end
    if useGpu
        data = gpuArray((data));
        U = gpuArray((U));
        V = gpuArray((V));
    end
    lambdaV = 1;
    
    r = ones(size(data));
    oldRes = 1e10;
    UV = U * V;
    while true
        c = c + 1;
		V = V .* ((U' * (data ./ r)) ./ (U' * (UV ./ r) + lambdaV));
        UV = U * V;
        r = (abs(data - UV));
        temp = r;
        idx = r < eps;
        r(idx) = eps;
        if mod(c, 10) == 1
            res = gather(sum(sum(temp .^ 2 ./ r)) + lambdaV * sum(sum(V)));
            if abs(oldRes - res) / oldRes < 1e-3 || c > 1000
                break;
            end
            oldRes = res;
        end
    end
    disp(c);
    r = (abs(data - UV));
    r(r < eps) = eps;
    if useGpu
        V = gather(V);
        r = gather(r);
        res =  gather(sum(sum(temp .^ 2 ./ r)) + lambdaV * sum(sum(V)));
    else
        res =  sum(sum(temp .^ 2 ./ r)) + lambdaV * sum(sum(V));
    end
end