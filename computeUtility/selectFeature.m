function sel = selectFeature(AA_pos, AA_neg, param, total)
% function P = selectFeature(AA_pos, AA_neg, param)
% obtain the projection matrix P

% input --- 
% AA_pos: the normalized positive templates
% AA_neg: the normalized negative templates
% param: the parameters for sparse representation

% output ---
% P: the projection matrix

%*************************************************************
%% Copyright (C) Wei Zhong.
%% All rights reserved.
%% Date: 05/2012

A = [AA_pos AA_neg];
L = [ones(size(AA_pos,2),1); (-1)*ones(size(AA_neg,2),1)];     % the label for each template, +1 for positive templates and -1 for negative templates
param.lambda = 0.002;
param.L = length(L);
[w c]=LogisticR(A', L, param.lambda, []);
w = full(w);
sel = (w~=0);
while sum(sel) < total * 0.5 && param.lambda > 1e-4
    opts.x0=w;               % warm-start of x
    opts.c0=c;
    param.lambda = param.lambda * 0.75;
    [w c]=LogisticR(A', L, param.lambda, []);
    w = full(w);
    sel = (w~=0);
end

end