%% load data
load('AL_data_2000_BP.mat')

%% create precision matrix
spde.C = speye(prod(sz));
spde.G = igmrfprec(sz,1);
spde.G2 = spde.G*spde.G;

%% Initial plots
%indicators of observed grid cells
I_obs = reshape(sum(A,1),sz);
I_grid = reshape(sum(A_grid,1),sz);

%plot covariates
figure(1)
for i=2:size(B_grid,2)
  subplot(2,3,i-1)
  imagesc(longitude, latitude([2 1]), reshape(A_grid'*B_grid(:,i),sz), ...
    'alphadata', I_grid)
  axis xy tight
  title(names{i})
  colorbar
end

%plot observations and their covariates
figure(2)
subplot(3,3,1)
imagesc(longitude, latitude([2 1]), reshape(A'*Y,sz), 'alphadata', I_obs)
axis xy tight
title('Pollen data')
colorbar
for i=2:size(B,2)
  subplot(3,3,i)
  imagesc(longitude, latitude([2 1]), reshape(A'*B(:,i),sz), ...
    'alphadata', I_obs)
  axis xy tight
  title(names{i})
  colorbar
end

%% Restore variables
B = B_org;
B_grid = Bg_org;
names = names_org;

%% Save variables
B_org = B;
Bg_org = B_grid;
names_org = names;
%% Remove covariates
%keep_idx = [1, 2, 3, 4, 5, 6, 7];
keep_idx = [1, 3, 4, 7];
B = B(:, keep_idx);
B_grid = B_grid(:, keep_idx);
names = names(keep_idx);

%% Fit model

%we need a global variable for x_mode to reuse it
%between optimisation calls
global x_mode;
x_mode = [];
%attempt to estimate parameters (optim in optim...)
%subset to only observed points here
% [0 0] - two parameters to estimate, lets start with 0 in log-scale.
alpha = 2;
par = fminsearch( @(theta) GMRF_negloglike_Be(theta, Y, A, B, spde, 1e-6, alpha), [0 0]);
%conditional mean is now given be the mode
E_xy = x_mode;

%use the taylor expansion to compute posterior precision
%you need to reuse some of the code from GMRF_negloglike_NG 
%to create inputs for this function call
tau = exp(par(1));
kappa2 = exp(par(2));

if alpha==1
  Q_x = tau*(kappa2 * spde.C + spde.G);
elseif alpha==2
  Q_x = tau*(kappa2^2 * spde.C + 2*kappa2 * spde.G + spde.G2);
else
  error('Unknown alpha')
end

Qbeta = 1e-6 * speye(size(B, 2));
Qall = blkdiag(Q_x, Qbeta);
Aall = [A B];

[f, ~, Q_xy] = GMRF_taylor_Be(E_xy, Y, Aall, Qall);

%% Analyse betas
e = [zeros(size(Q_xy,1)-size(B,2), size(B,2)); eye(size(B,2))];
V_beta0 = e'*(Q_xy\e);
beta_sig = sqrt(diag(V_beta0));
beta = E_xy(end-(size(B,2)-1):end);

pval = @(x, sig) normcdf(abs(x), 0, sig);

for i = 1:length(beta)
    pv = 2*(1 - pval(beta(i), beta_sig(i)));
    fprintf(1, '[%s]: significance level: %1.4f, significant?: %d\n', ...
    string(names(i)), pv, pv < 0.05);
end
fprintf(1, '\n');

%%
p = @(z) exp(z) ./ (1 + exp(z));
E_mean = [zeros(size(A_grid)) B_grid] * E_xy;
P_mean = p(E_mean);

E_spat = [A_grid zeros(size(B_grid)) ] * E_xy;
P_spat = p(E_spat);

E_all = [A_grid B_grid] * E_xy;
P_all = p(E_all);

%% Save predictions
SAR_mean_sign = P_mean;

%% Plot all predictions
barrange = [-7 5];
subplot(131)
imagesc(reshape(A_grid'*E_mean,sz), ...
    'alphadata', I_grid)
colorbar
caxis(barrange);
axis tight
title('Mean component')

subplot(132)
imagesc(reshape(A_grid'*E_spat,sz), ...
    'alphadata', I_grid)
colorbar
caxis(barrange);
axis tight
title('Spatial component')

subplot(133)
imagesc(reshape(A_grid'*E_all,sz), ...
    'alphadata', I_grid)
colorbar
caxis(barrange);
axis tight
title('Full model')

set(gcf, 'Position',  [170,430,1370,400])
%%
varx = Q_xy_var(1:end-size(B,2));
imagesc(reshape(log(varx),sz), ...
    'alphadata', I_grid)
colorbar

%%

mean_modes = zeros(1000, 1);
mean_vars = zeros(1000, 1);
Q_xy_x = Q_xy(1:end-size(B,2), 1:end-size(B,2));
xs = zeros(1000, size(Q_xy_x, 1));

R = chol(Q_xy_x);
for i = 1:1000
    x = R \ randn(size(R,1),1);
    xs(i,:) = x;
end

mean(xs, 'all')

Q_xy_var = var(xs, 1, 1)';



%%
confint_upper = p(A_grid'*E_all + 1.96*sqrt(Q_xy_var));
confint_lower = p(A_grid'*E_all - 1.96*sqrt(Q_xy_var));

caxis_limits = [0 1];
figure()
subplot(131)
imagesc(reshape(confint_lower,sz), ...
    'alphadata', I_grid)
colorbar
caxis(caxis_limits);
axis tight
title('Lower confidence interval')
subplot(132)
imagesc(reshape(A_grid'*P_all,sz), ...
    'alphadata', I_grid)
colorbar
caxis(caxis_limits);
axis tight
title('Predictions')
subplot(133)
imagesc(reshape(confint_upper,sz), ...
    'alphadata', I_grid)
colorbar
caxis(caxis_limits);
axis tight
title('Upper confidence interval')
set(gcf, 'Position',  [170,430,1370,400])
%% Simulate to find variance

data = E_mean;
mean_modes = zeros(1000, 1);
mean_vars = zeros(1000, 1);

for i = 1:1000
    P = randperm(length(data), round(length(data) / 10));
    mean_vars(i) = var(data(P), 1);
    mean_modes(i) = mean(data(P));
end

pred_var = mean(mean_vars)
pred_mode = mean(mean_modes)

i = -4*sqrt(pred_var / 1000):0.001:4*sqrt(pred_var / 1000);
ynpdf = normpdf(pred_mode + i, pred_mode, sqrt(pred_var / 200));

figure(1)
hold on;
plot(pred_mode + i, ynpdf);
histogram(mean_modes, 'Normalization', 'pdf');
hold off;
figure(2)
normplot(mean_modes);

%% Validation

E_all_val = [A B] * E_xy;

P_all_val = p(E_all_val);

logit = @(x) log(x ./ (1 - x));

Y_all_val = Y;
res = Y_all_val - P_all_val;

rss = res'*res;
tss = (Y_all_val - mean(Y_all_val))'*(Y_all_val - mean(Y_all_val));

r2 = 1 - rss/tss;
n = length(Y_all_val);
pf = size(B,2);
r2adj = 1 - (1-r2)*(n-1)/(n-pf-1)


%% Plot diff
figure()
subplot(221)
imagesc(reshape(A_grid'*(CAR_mean_all-CAR_mean_sign),sz), ...
    'alphadata', I_grid)
colorbar
caxis([-0.2 0.2]);
axis tight
title('CAR: all - significant')
subplot(222)
imagesc(reshape(A_grid'*(SAR_mean_all-SAR_mean_sign),sz), ...
    'alphadata', I_grid)
colorbar
caxis([-0.2 0.2]);
axis tight
title('SAR: all - significant')
subplot(223)
imagesc(reshape(A_grid'*(CAR_mean_all-SAR_mean_all),sz), ...
    'alphadata', I_grid)
colorbar
caxis([-0.2 0.2]);
axis tight
title('CAR - SAR : all')
subplot(224)
imagesc(reshape(A_grid'*(CAR_mean_sign-SAR_mean_sign),sz), ...
    'alphadata', I_grid)
colorbar
caxis([-0.2 0.2]);
axis tight
title('CAR - SAR : significant')