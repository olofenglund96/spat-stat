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
c = zeros(25, 1);
for h = 0:24
    esum = 0;
    for i = 1:(length(eta) - h)
        esum = esum + eta(i)*eta(i+h);
    end
    
    c(h+1) = 1/length(eta) * esum;
end

p = 0:24;
plot(p, c(p+1), 'bo');
xlim([0,25]);
grid on;
%%
hold on;
plot(t, eta);
etat = eta(2:end);
eta1 = eta(1:end-1);
alpha = regress(etat, eta1);

res = etat-eta1*alpha;
phi = var(res);

%%
nu = @(h) alpha.^h / (1 - alpha^2) * phi;

i = 0:25;
hold on;
plot(i, nu(i), 'r');

%%
preds = [];
invp = [];
invm = [];

offset = 180;
eta_0 = eta(offset);
pred = @(t) X(offset+t,:) * beta + (alpha.^t * eta_0)';
varp = @(t) phi * (1 - alpha.^(2*t)) / (1 - alpha.^2);


i = 1:25;
hold on;
plot(i, pred(i), 'g');
plot(i, Y(i+offset-1), 'b', i, X(i+offset-1,:)*beta, '--r');

plot(i, pred(i) + 1.96*sqrt(varp(i))', '--m', i, pred(i) - 1.96*sqrt(varp(i))', '--m');



