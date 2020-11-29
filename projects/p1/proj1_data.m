%set paths to course files and/or download HA1_SE_Temp.mat from the homepage

%% load data
load HA1_temp_2015_06
%points inside of Sweden (i.e. not nan)
I_land = ~any(isnan(X_grid),2);
%reshape grid to images
X_img = reshape(X_grid, sz(1), sz(2), []);
I_img = reshape(I_land, sz);

%% Plot data
%plot observations
figure(1)
subplot(231)
scatter(X(:,1), X(:,2), 20, Y, 'filled', ...
  'markeredgecolor', 'k')
axis xy tight; hold on
%validation data
scatter(X_valid(:,1), X_valid(:,2), 20, Y_valid(:,end), 'filled', ...
  'markeredgecolor', 'r')
%borders
plot(Border(:,1),Border(:,2),'k')
hold off; colorbar
title('2015-06 temperature')

%plot elevation and other covariates
for i=3:6
  subplot(2,3,i-1)
  imagesc([4.9 30.9], [71.1 55.4], X_img(:,:,i), 'alphadata', I_img)
  axis xy; hold on
  plot(Border(:,1),Border(:,2))
  scatter(X(:,1), X(:,2), 25, X(:,i),...
    'filled','markeredgecolor','k')
  colorbar
  hold off
  title(names{i})
end

%% Example showing how to do predictions and fit them onto the grid
%pick out the relevant parts of the grid
grid = X_grid(I_land,:);
%create a matrix holding "predicitons"
mu = nan(sz);
%do "predicitons" 
E = grid*randn(size(grid,2),1);
%and fit these into the relevant points
mu(I_land) = E;
%plot
figure(2)
imagesc([4.9 30.9], [71.1 55.4], mu, 'alphadata', I_img)
axis xy; hold on
plot(Border(:,1),Border(:,2), '-')
hold off
title('Predictions')
