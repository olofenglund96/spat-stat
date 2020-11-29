%%
i = 0:80;
hold on;

%fun = @(dist, sig, kappa, nu) matern_covariance(dist, sig, kappa, nu);
%fun = @(dist, sig, kappa, nu) exponential_covariance(dist, sig, kappa);
%fun = @(dist, sig, kappa, nu) matern_covariance(dist, sig, kappa, nu);
fun = @(dist, sig, kappa, nu) spherical_covariance(dist, sig, kappa);


subplot(2,2,1);
for j = 0:3:60
    r = fun(i, 1, j, 1);
    plot(i, r);
    title('kappa');
    hold on;
end

subplot(2,2,2);
for j = 0:0.1:1
    r = fun(i, j, 0.1, 1);
    plot(i, r);
    title('sigma2');
    hold on;
end

subplot(2,2,3);
for j = 0:0.5:5
    r = fun(i, 1, 0.1, j);
    plot(i, r);
    title('nu');
    hold on;
end

%%

% First use matern_covariance to create a Sigma-covariance matrix.
% and set mu=(a constant mean).
figure(1)
[u1,u2] = ndgrid(1:50,1:60);
D = distance_matrix([u1(:), u2(:)]);
mu = 10;
Sigma = matern_covariance(D, 5, 0.5, 2);
N = 50*60;
sz = [50 60];
R = chol(Sigma); % Calculate the Cholesky factorisation
eta = mu+R'*randn(N,1); % Simulate a sample
eta_image = reshape(eta,sz); %reshape the column to an image
imagesc(eta_image)

%%
figure(2)
sigma_epsilon = 0.1;
y=eta + randn(N,1)*sigma_epsilon;

z=y-mu;
plot(D,z*z','.k');

%%
hold on;
d = linspace(0, max(D, [], 'all'), 10^3);
r = matern_covariance(d, 5, 0.1, 2);
plot(d, r, '-r');

%%
[rhat,s2hat,m,n,d, varioest]=covest_nonparametric(D,z,15, max(D, [], 'all')/2);

plot(d,rhat,'-',0,s2hat,'o')
%plot(d,varioest,'-')

%%
par = covest_ls(rhat, s2hat, m, n, d);

figure(1)
[u1,u2] = ndgrid(1:50,1:60);
D = distance_matrix([u1(:), u2(:)]);

Sigma = matern_covariance(D, 5, 0.5, 2);
Sigma_hat = matern_covariance(D, par(1), par(2), par(3));

%%
mu = 10;
N = 50*60;
sz = [50 60];
nums = randn(N,1);

R = chol(Sigma); % Calculate the Cholesky factorisation
eta = mu+R'*nums; % Simulate a sample
eta_image = reshape(eta,sz); %reshape the column to an image

Rh = chol(Sigma_hat); % Calculate the Cholesky factorisation
etah = mu+Rh'*nums; % Simulate a sample
eta_imageh = reshape(etah,sz); %reshape the column to an image

subplot(2, 1, 1);
imagesc(eta_image);
subplot(2, 1, 2);
imagesc(eta_imageh);

%%
par = covest_ls(rhat, s2hat, m, n, d);
i = linspace(0, max(D, [], 'all'), 10^3);
r = matern_covariance(i, 5, 0.5, 2);
r_hat = matern_covariance(i, par(1), par(2), par(3));
hold on;
plot(i, r, '-r');
plot(i, r_hat, '-b');
par

%%
par = covest_ls(rhat, s2hat, m, n, d, 'matern', [0, 0, 2, 0]);
i = linspace(0, max(D, [], 'all'), 10^3);
r = matern_covariance(i, 5, 0.5, 2);
r_hat = matern_covariance(i, par(1), par(2), par(3));
hold on;
plot(i, r, '-r');
plot(i, r_hat, '-b');
par

%%
p = 0.1;
I_obs = (rand(sz)<=p);

%add nugget to the covariance matrix
Sigma_yy = Sigma + sigma_epsilon^2*eye(size(Sigma));
%and divide into observed/unobserved
Sigma_uu = Sigma_yy(~I_obs, ~I_obs);
Sigma_uo = Sigma_yy(~I_obs, I_obs);
Sigma_oo = Sigma_yy(I_obs, I_obs);
y_o = y(I_obs);
y_u = y(~I_obs);

X = ones(prod(sz),1);
X_u = X(~I_obs);
X_o = X(I_obs);

mui = mu;

y_rec = nan(sz);
y_rec(I_obs) = y_o;
y0save = y_rec;
y_rec(~I_obs) = X_u*mui + Sigma_uo*inv(Sigma_oo)*(y_o - X_o*mui);

%%
figure(1)
imagesc(reshape(y,sz));
figure(2)
subplot(1, 2, 1);
imagesc(y0save);
subplot(1, 2, 2);
imagesc(y_rec);