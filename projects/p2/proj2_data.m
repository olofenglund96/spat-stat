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

%% Fit model

%we need a global variable for x_mode to reuse it
%between optimisation calls
global x_mode;
x_mode = [];
%attempt to estimate parameters (optim in optim...)
%subset to only observed points here
% [0 0] - two parameters to estimate, lets start with 0 in log-scale.
par = fminsearch( @(theta) GMRF_negloglike_Be(theta, Y, A, B, spde), [0 0]);
%conditional mean is now given be the mode
E_xy = x_mode;

%use the taylor expansion to compute posterior precision
%you need to reuse some of the code from GMRF_negloglike_NG 
%to create inputs for this function call
[~, ~, Q_xy] = GMRF_taylor_Be(E_xy, Y);
