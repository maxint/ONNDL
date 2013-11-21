function param = estwarp_condens(frm, tmpl, param, opt)
% estimate the warp parameters

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
data = bsxfun(@rdivide, data, sqrt(sum(data .* data))); % normalization

oriData = data;
data = opt.P' * data;
base = opt.P' * tmpl.basis;

% optimize ----------------------------------------------------------------
tic;
[coef, ~, ~] = RNNSC(data, base, 1, 0, true && opt.useGpu);
toc 

% difference between object particles and negative particles
diff = sum(base(:, 1 : opt.maxbasis) * coef(1 : opt.maxbasis, :)) - sum(base(:, opt.maxbasis + 1 : end) * coef(opt.maxbasis + 1 : end, :));
param.coef = coef;

% convert difference to likehood (confidence)
param.conf = exp(double(diff) ./opt.condenssig)';

% get the particle with maximum likehood
param.conf = param.conf ./ sum(param.conf); % normalize
[maxprob,maxidx] = max(param.conf);
if maxprob == 0
    error('overflow!');
end

% calculate the result ----------------------------------------------------
param.est = affparam2mat(param.param(:,maxidx)); % 6x1 warp parameters
param.wimg = reshape(oriData(:,maxidx), sz); % warped image
param.err = reshape(oriData(:,maxidx) - tmpl.basis * coef(:, maxidx), sz); % error
param.recon = reshape(tmpl.basis(:, 1 : opt.maxbasis) * coef(1 : opt.maxbasis, maxidx), sz); % reconstructed patch
if exist('coef', 'var')
    param.bestCoef = coef(:,maxidx);
end
