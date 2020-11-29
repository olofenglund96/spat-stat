%% 1. Some dependence structures
k2 = 0.01;
m = 100;
n = m;

q1 = [0 -1 0; -1 4 + k2 -1; 0 -1 0];
q2 = [0 -0.1 -10; -0.1 20.4 + k2 -0.1; -10 -0.1 0];
q3 = [0 -1 -2; -1 8 + k2 -1; -2 -1 0];

Q1 = gmrfprec([m, n], q1);
Q2 = gmrfprec([m, n], q2);
Q3 = gmrfprec([m, n], q3);

s1 = Q1 \ sparse(m*n/2 + m/2, 1, 1, size(Q1, 1),1);
s2 = Q2 \ sparse(m*n/2 + m/2, 1, 1, size(Q2, 1),1);
s3 = Q3 \ sparse(m*n/2 + m/2, 1, 1, size(Q3, 1),1);

%%
subplot(1, 3, 1);
imagesc(reshape(s1, [m,n]));

subplot(1, 3, 2);
imagesc(reshape(s2, [m,n]));

subplot(1, 3, 3);
imagesc(reshape(s3, [m,n]));

%%
mu = 5;

R1 = chol(Q1);
x1 = mu + R1 \ randn(size(R1,1),1);

R2 = chol(Q2);
x2 = mu + R2 \ randn(size(R2,1),1);

R3 = chol(Q3);
x3 = mu + R3 \ randn(size(R3,1),1);

%%
subplot(1, 3, 1);
imagesc(reshape(x1, [m,n]));

subplot(1, 3, 2);
imagesc(reshape(x2, [m,n]));

subplot(1, 3, 3);
imagesc(reshape(x3, [m,n]));

%%
Q12 = Q1^2;
s12 = Q12 \ sparse(m*n/2 + m/2, 1, 1, size(Q12, 1),1);


Q22 = Q2^2;
s22 = Q22 \ sparse(m*n/2 + m/2, 1, 1, size(Q22, 1),1);


subplot(2,2,1)
spy(Q1)
subplot(2,2,2)
spy(Q12)

subplot(2,2,3)
spy(Q2)
subplot(2,2,4)
spy(Q22)

%% 2. Interpolation
load('lab3.mat')

%%
imagesc(xmiss);

%%
y = xmiss(known);
mu = mean(y);
m = size(xmiss, 1);
n = size(xmiss, 2);

Q = gmrfprec([m, n], q2);

s = Q \ xmiss(:);


subplot(121);
surf(reshape(s, [m,n]));

subplot(122);
surf(xtrue);

%%
Y = xmiss(known);
A = sparse(1:length(Y), find(known), 1, length(Y), numel(xmiss));
Q = gmrfprec([m, n], q2);
Aall = [A ones(length(Y),1)];
Qbeta = 1e-3 * speye(1); %use SPEYE to create a SPARSE-matrix
Qall = blkdiag(Q, Qbeta);

%assume a very small observation-uncertainty
Qeps = 1e5 * speye(length(Y));
%posterior precision
Qxy = Qall + Aall'*Qeps*Aall;
p = amd(Qxy);
Qxy = Qxy(p,p);
Aall = Aall(:,p);

E_xy = Qxy \ Aall'*Qeps*Y;
E_xy(p) = E_xy;

%%

E_zy = [speye(size(Q,1)) ones(size(Q,1),1)]*E_xy;
%E_zy(known) = xmiss(known);
imagesc(reshape(E_zy, [m, n]));


%%

titan = imread('images/titan.jpg');
imshow(titan);

%%
miss = 0.99;
xtrue = double(titan) / 255;
known=(rand(size(titan))>miss);
xmiss=xtrue.*known;
m = size(xmiss, 1);
n = size(xmiss, 2);
imshow([xtrue, xmiss])
%%

Y = xmiss(known);
A = sparse(1:length(Y), find(known), 1, length(Y), numel(xmiss));
Q = gmrfprec([m, n], q1)^2;
Aall = [A ones(length(Y),1)];
Qbeta = 1e-3 * speye(1); %use SPEYE to create a SPARSE-matrix
Qall = blkdiag(Q, Qbeta);

%assume a very small observation-uncertainty
Qeps = 1e5 * speye(length(Y));
%posterior precision
Qxy = Qall + Aall'*Qeps*Aall;
p = amd(Qxy);
Qxy = Qxy(p,p);
Aall = Aall(:,p);
Rxy = chol(Qxy);

%%

E_xy = Rxy \ (Rxy' \ Aall'*Qeps*Y);
E_xy(p) = E_xy;

%%
E_zy = [speye(size(Q,1)) ones(size(Q,1),1)]*E_xy;
%E_zy(known) = xmiss(known);
subplot(121);
imshow([reshape(E_zy, [m, n]) xtrue]);
subplot(122);
surf(reshape(E_zy, [m, n])-xtrue);

