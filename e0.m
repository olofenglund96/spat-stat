im = imread('images/tornetrask2.jpg');
x = double(im)/255;

%%
imagesc(x(:,:,1))
colormap('gray');

%%
imagesc(x(:,:,[3,2,1]));

%%
xh = round(size(x, 2) / 2);
xn = x;
xn(:,xh:end,:) = xn(:,xh:end,[3,2,1]);

imagesc(xn);

%%
x=x/max(x(:));
img(:,:,1)=(x(:,:,1)<=0.25);
img(:,:,2)=(x(:,:,1)>0.25)&(x(:,:,1)<=0.5);
img(:,:,3)=(x(:,:,1)>0.5)&(x(:,:,1)<=0.75);
img(:,:,4)=(x(:,:,1)>0.75);

rgbim = rgbimage(img);
imagesc(rgbim);

%%
im = imread('images/lkab.jpg');
x = double(im)/255;
y=x./repmat(sum(x,3),[1,1,3]);

imagesc(y);

%%
load T_lund.mat

%%
plot(T_lund(:,1), T_lund(:,2));
datetick

%%
t = T_lund(:,1); Y = T_lund(:,2); n = length(Y);
X = [ones(n,1) sin(2*pi*t/365) cos(2*pi*t/365)];

beta = regress(Y, X);
eta = Y-X*beta;
plot(t, Y, 'b', t, X*beta, 'r');
datetick

%%
plot(t, eta);
etat = eta(2:end);
eta1 = eta(1:end-1);
alpha = regress(etat, eta1);

res = eta1-etat*alpha;
phi = var(res);

%%
nu = zeros(20, 1);
nu(1) = phi
for i = 2:20
    nu(i) = alpha*nu(i-1)
end

i = 1:20
plot(i, nu, 'o')

%%
[ycov,lags]=xcov(X,20,'biased');
plot(lags, ycov(lags));

