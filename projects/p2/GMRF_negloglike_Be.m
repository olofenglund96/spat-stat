function negloglike = GMRF_negloglike_Be(theta, y, A, B, spde, qbeta, alpha)
% GMRF_NEGLOGLIKE_BE  Calculate the GMRF data likelihood, non-Gaussian observations
%
% negloglike = GMRF_negloglike_Be(theta, y, A, B, spde, qbeta, alpha)
%
% theta = log([tau kappa2]);
% y = the data vector, as a column with n elements
% A = the observation matrix, sparse n-by-N
% B = covariates for the observations, n-by-Nbeta
% spde = structure with the three matrices used to construct Q: C,G,G2.
%          see igmrfprec, sparse N-by-N
% qbeta = Precision for the regression parameters (Nbeta)-by-(Nbeta)
% alpha = which Q to use (alpha=1 or 2).
%
% This is only a skeleton for Home Assignment 2.

% $Id: gmrf_negloglike_skeleton.m 4454 2011-10-02 18:29:12Z johanl $

%default values for
if nargin<6 || isempty(qbeta), qbeta=1e-6; end
if nargin<7 || isempty(alpha), alpha=1; end

%extract parameters
tau = exp(theta(1));
kappa2 = exp(theta(2));

%compute Q
if alpha==1
  Q_x = tau*(kappa2 * spde.C + spde.G);
elseif alpha==2
  Q_x = tau*(kappa2^2 * spde.C + 2*kappa2 * spde.G + spde.G2);
else
  error('Unknown alpha')
end

%combine Q_x and Qbeta and create observation matrix
Qbeta = qbeta * speye(7);
Qall = blkdiag(Q_x, Qbeta);
Aall = [A B];

%declare x_mode as global so that we start subsequent optimisations from
%the previous mode (speeds up nested optimisation).
global x_mode;
if isempty(x_mode)
  %no existing mode, use zero init vector
  x_mode = zeros(size(Qall,1),1);
end

%compute reorder
p_x = amd(Q_x);
p = amd(Qall + Aall'*Aall);
Qall = Qall(p,p);
Aall = Aall(:,p);
x_mode = x_mode(p);

%find mode
x_mode = fminNR(@(x) GMRF_taylor_Be(x, y, Aall, Qall), x_mode);

%find the Laplace approximation of the denominator
[f, ~, Q_xy] = GMRF_taylor_Be(x_mode, y, Aall, Qall);
%note that f = -log_obs + x_mode'*Q*x_mode/2.

%Compute choleskey factors
[R_x, ok_x] = chol( Q_x(p_x,p_x) );
[R_xy, ok_xy] = chol( Q_xy );
if ok_x~=0 || ok_xy~=0
  %choleskey factor fail -> (almost) semidefinite matrix -> 
  %-> det(Q) ~ 0 -> log(det(Q)) ~ -inf -> negloglike ~ inf
  %Set negloglike to a REALLY big value
  negloglike = realmax;
  return;
end

%note that f = -log_obs + x_mode'*Q*x_mode/2.
det(Q_xy)
negloglike = -f - log(sqrt(det(Q_xy)));

%inverse reorder before returning
x_mode(p) = x_mode;

%print diagnostic information (progress)
fprintf(1, 'Theta: %11.4e %11.4e; fval: %11.4e\n', ...
  theta(1), theta(2), negloglike);

