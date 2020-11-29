%%
rmsmin = 10000;
v = 1:6;
for i = v
    varset = nchoosek(v,i);
    for j = 1:size(varset, 1)
        vars = varset(j,:);
        % vars = [%1
        %         2
        %         3
        %         4
        %         5
        %         %6
        %         ];

        X_i = [ones(size(X, 1), 1) X(:, vars)];
        beta = regress(Y, X_i);
        eta = Y-X_i*beta;

        sigma2 = norm(eta)^2 / (size(X,1) - size(X,2));

        %pick out the relevant parts of the grid
        I_land = ~any(isnan(X_grid),2);
        grid = [ones(size(X_grid(I_land,:), 1), 1) X_grid(I_land, vars)];
        %create a matrix holding "predicitons"
        mu = nan(sz);
        %do "predicitons"
        E = grid*beta;
        %and fit these into the relevant points
        mu(I_land) = E;
        mu_ui = mu;
        mu_li = mu;
        mu_ui(I_land) = mu_ui(I_land) + sqrt(sigma2)*1.96;
        mu_li(I_land) = mu_li(I_land) - sqrt(sigma2)*1.96;

        rms = @(e) sqrt((1 / size(e,1)) * e'*e);

        X_valid_i = [ones(size(X_valid, 1), 1) X_valid(:, vars)];

        if rms(X_valid_i*beta - Y_valid) < rmsmin
            rmsmin = rms(X_valid_i*beta - Y_valid)
            vars
        end
    end
end

%%
vars = [1 2 3 4 5];

X_i = [ones(size(X, 1), 1) X(:, vars)];
beta = regress(Y, X_i);
eps = Y-X_i*beta;

sigma2 = norm(eps)^2 / (size(X,1) - size(X,2));

%pick out the relevant parts of the grid
grid = [ones(size(X_grid(I_land,:), 1), 1) X_grid(I_land, vars)];
%create a matrix holding "predicitons"
mu = nan(sz);
%do "predicitons"
E = grid*beta;
%and fit these into the relevant points
mu(I_land) = E;
mu_ui = mu;
mu_li = mu;
mu_ui(I_land) = mu_ui(I_land) + sqrt(sigma2)*1.96;
mu_li(I_land) = mu_li(I_land) - sqrt(sigma2)*1.96;

max(mu, [], 'all')
min(mu, [], 'all')

rms = @(e) sqrt((1 / size(e,1)) * e'*e);
%%
Vbeta = sigma2*inv((X_i'*X_i));
Vy = sum((X_valid_i*Vbeta).*X_valid_i,2) + sigma2;

stand_res = (Y_valid - X_valid_i*beta) ./ sqrt(Vy);
%%

X_valid_i = [ones(size(X_valid, 1), 1) X_valid(:, vars)];
epsv = Y_valid-X_valid_i*beta;
rms(epsv)
%%
i = linspace(-10*sigma2, 10*sigma2);
hold on;
histogram(stand_res, 'Normalization', 'pdf');
plot(i, normpdf(i, 0, sqrt(var(stand_res))))
title('Distribution of standardised residuals from OLS');
legend(["Histogram of \epsilon", "Normal distribution with \mu = 0, \sigma = \surd{V(\epsilon)}"])
%%
figure(2);
subplot(1,3,1);
imagesc([4.9 30.9], [71.1 55.4], mu, 'alphadata', I_img)
axis xy; hold on
plot(Border(:,1),Border(:,2), '-')
hold off
title('Predictions')
colorbar
caxis([min(mu_li, [], 'all'), max(mu_ui, [], 'all')])
subplot(1,3,2);
imagesc([4.9 30.9], [71.1 55.4], mu_ui, 'alphadata', I_img)
axis xy; hold on
plot(Border(:,1),Border(:,2), '-')
hold off
title('Upper prediction interval')
colorbar
caxis([min(mu_li, [], 'all'), max(mu_ui, [], 'all')])
subplot(1,3,3);
imagesc([4.9 30.9], [71.1 55.4], mu_li, 'alphadata', I_img)
axis xy; hold on
plot(Border(:,1),Border(:,2), '-')
hold off
title('Lower prediction interval')
colorbar
caxis([min(mu_li, [], 'all'), max(mu_ui, [], 'all')])

%%
D = distance_matrix(X_i(:,[2 3]));
z = eps*eps';
zm = mean(D*z, 2);
plot(D, z,'.k');
hold on;
%%
[rhat,s2hat,m,n,d, varioest]=covest_nonparametric(D,eps,300, (max(D, [],'all'))/2);
plot(d,rhat,'-',0,s2hat,'o')
%plot(d,varioest,'-')

%%
grid = [ones(size(X_grid(I_land,:), 1), 1) X_grid(I_land, vars)];
X_spat = X_i(:,[2 3]);
grid_spat = grid(:, [2 3]);
D_kk = distance_matrix(X_spat);
D_uk = distance_matrix(grid_spat, X_spat);
D_ku = distance_matrix(X_spat, grid_spat);


%%
par = covest_ml(D_kk, eps, 'matern', [s2hat, 0, 0.5, 0]);

%%
z = eps;
%%
Rhat = zeros(101, 100); 
for i = 1:100
    index = randperm(length(z)); 
    z = z(index);
    [rhat, s2hat, m, n, d, varioest] = covest_nonparametric(D, z, 100,(max(D, [],'all'))/2);
    Rhat(:,i) = rhat';
end
%%
Rhat_mean = mean(Rhat,2); 
plot(d,Rhat_mean,'o')
%%
% Confidence interval 
Rhat_var = var(Rhat,0,2); 
conf_lo = Rhat_mean-1.96*sqrt(Rhat_var); 
conf_up =  Rhat_mean+1.96*sqrt(Rhat_var); 
plot(d,Rhat_mean,'o', d, conf_lo, 'r-', d, conf_up, 'r-') 
hold on 
plot(d, Rhat, 'k.') 
hold on 
plot(d, A, 'b-') 
%%
A = quantile(Rhat,[0.025 0.5 0.975], 2); 
plot(d,A)

%%
[rhat, s2hat, m, n, d, varioest] = covest_nonparametric(D, eps, 200,(max(D, [],'all'))/2);
covfun = matern_covariance(d, par(1), par(2), par(3));
plot(d,rhat,'o', d, covfun)


%%
Sigma_kk = matern_covariance(D_kk, par(1), par(2), par(3));
Sigma_uk = matern_covariance(D_uk, par(1), par(2), par(3));
Sigma_ku = matern_covariance(D_ku, par(1), par(2), par(3));

Sigma_kk = Sigma_kk + par(4)*eye(size(Sigma_kk));
R = chol(Sigma_kk + eye(size(Sigma_kk))*1e-5);
beta_krig = (X_i'/R*X_i)\(X_i'/R*Y);

%%
Sigma_yy_uk = Sigma_uk + par(4)*eye(size(Sigma_uk));
Sigma_yy_ku = Sigma_ku + par(4)*eye(size(Sigma_ku));

%%
%create a matrix holding "predicitons"
mu_krig = nan(sz);
%do "predicitons"
E = grid*beta_krig + Sigma_yy_uk/R*(Y - X_i*beta_krig);
%and fit these into the relevant points
mu_krig(I_land) = E;
max(mu_krig, [], 'all')
min(mu_krig, [], 'all')
%%
valid_spat = X_valid_i(:, [2 3]);
D_val_uk = distance_matrix(valid_spat, X_spat);
D_val_ku = distance_matrix(X_spat, valid_spat);

Sigma_val_uk = matern_covariance(D_val_uk, par(1), par(2), par(3)) + par(4)*eye(size(D_val_uk));
Sigma_val_ku = matern_covariance(D_val_ku, par(1), par(2), par(3)) + par(4)*eye(size(D_val_ku));

Val_preds = X_valid_i*beta_krig + Sigma_val_uk/R*(Y - X_i*beta_krig);
eps_krig = Val_preds - Y_valid;

mu_ui_krig(I_land) = mu_krig(I_land) + sqrt(par(4))*1.96
mu_li_krig(I_land) = mu_krig(I_land) - sqrt(par(4))*1.96

rms = @(e) sqrt((1 / size(e,1)) * e'*e);
rms(eps_krig)
X_valid_i = [ones(size(X_valid, 1), 1) X_valid(:, vars)];

%%
all_mu = [mu mu_krig];
figure(2);
subplot(1,2,1);
imagesc([4.9 30.9], [71.1 55.4], mu, 'alphadata', I_img)
axis xy; hold on
plot(Border(:,1),Border(:,2), '-')
hold off
title('Predictions using OLS model')
colorbar
caxis([min(all_mu, [], 'all'), max(all_mu, [], 'all')])
subplot(1,2,2);
imagesc([4.9 30.9], [71.1 55.4], mu_krig, 'alphadata', I_img)
axis xy; hold on
plot(Border(:,1),Border(:,2), '-')
hold off
title('Predictions using Universal Kriging model')
colorbar
caxis([min(all_mu, [], 'all'), max(all_mu, [], 'all')])
%%
imagesc([4.9 30.9], [71.1 55.4], abs(mu - mu_krig), 'alphadata', I_img)
axis xy; hold on
plot(Border(:,1),Border(:,2), '-')
hold off
title('Absolute prediction differences between the two models')
colorbar