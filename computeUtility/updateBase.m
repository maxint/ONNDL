function [A, B, U] = updateBase(A, B, U, Y, opt)
	step = 0.2;
    maxIter = 100;
    tol = 1e-2;
    
    m = size(U, 1);
	TA = A;
	TB = B;
	oldRes = 1e10;
	c = 0;
    [V, r, res] = RNNSC(Y, U, 1, 0, false && opt.useGpu);
	while true
 		c = c + 1;
        oldU = U;
        for i = 1 : m
			TA{i} = A{i} * opt.forgetFactor + V * diag(1 ./ r(i, :)) * V';
			TB{i} = B{i} * opt.forgetFactor + V * diag(1 ./ r(i, :)) * Y(i, :)';
			grad =  U(i, :) * TA{i}' - TB{i}';
            U(i, :) = U(i, :) - step * grad;
		end
		U(U < 0) = 0;
        temp = sqrt(sum(U .* U));
        U(:, temp == 0) = oldU(:, temp == 0);
 		U = bsxfun(@rdivide, U, (temp > 1) .* temp + (temp <= 1));	
        if c > maxIter 
            break;
        end
        [V, r, res] = RNNSC(Y, U, 0, V, false);
        if res > oldRes
            step = step / 2;
            U = oldU;
            continue;
        end
        disp(res);
        if abs(res - oldRes)  < tol
            break;
        end
        oldRes = res;
    end
	A = TA;
	B = TB;
end