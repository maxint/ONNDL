function param = estwarp_condens(frm, tmpl, param, opt)
n = opt.numsample;
sz = size(tmpl.mean);
N = sz(1)*sz(2);

if ~isfield(param,'param')
  param.param = repmat(affparam2geom(param.est(:)), [1,n]);
else
  cumconf = cumsum(param.conf);
  idx = floor(sum(repmat(rand(1,n),[n,1]) > repmat(cumconf,[1,n])))+1;
  param.param = param.param(:,idx);
end
param.param = param.param + randn(6,n).*repmat(opt.affsig(:),[1,n]);
wimgs = warpimg(frm, affparam2mat(param.param), sz);
data = reshape(wimgs,[N,n]);
data = bsxfun(@rdivide, data, sqrt(sum(data .* data)));

oriData = data;
data = opt.P' * data;
base = opt.P' * tmpl.basis;
tic;
[coef, ~, ~] = RNNSC(data, base, 1, 0, true && opt.useGpu);
toc 
diff = sum(base(:, 1 : opt.maxbasis) * coef(1 : opt.maxbasis, :)) - sum(base(:, opt.maxbasis + 1 : end) * coef(opt.maxbasis + 1 : end, :));
param.coef = coef;
  

param.conf = exp(double(diff) ./opt.condenssig)';

param.conf = param.conf ./ sum(param.conf);
[maxprob,maxidx] = max(param.conf);
if maxprob == 0
    error('overflow!');
end
param.est = affparam2mat(param.param(:,maxidx));
param.wimg = reshape(oriData(:,maxidx), sz);
param.err = reshape(oriData(:,maxidx) - tmpl.basis * coef(:, maxidx), sz);
param.recon = reshape(tmpl.basis(:, 1 : opt.maxbasis) * coef(1 : opt.maxbasis, maxidx), sz);
if exist('coef', 'var')
    param.bestCoef = coef(:,maxidx);
end
