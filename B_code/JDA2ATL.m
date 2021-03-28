% Transfer Feature Learning with Joint Distribution Adaptation.  
% M. Long, J. Wang, G. Ding, J. Sun, and P.S. Yu.
% IEEE International Conference on Computer Vision (ICCV), 2013.

% Contact: Mingsheng Long (longmingsheng@gmail.com)
function [Z,P] = JDA2ATL(Xs,Xt,Ys,Yt0,options,A)
addpath(genpath('../liblinear/matlab'));
if nargin < 5
    error('Algorithm parameters should be set!');
end
if ~isfield(options,'k')
    options.k = 100;
end
if ~isfield(options,'lambda')
    options.lambda = 0.1;
end
if ~isfield(options,'ker')
    options.ker = 'primal';
end
if ~isfield(options,'gamma')
    options.gamma = 1.0;
end
if ~isfield(options,'data')
    options.data = 'default';
end
k = options.k;
lambda = options.lambda;
ker = options.ker;
gamma = options.gamma;
data = options.data;

% fprintf('JDA:  data=%s  k=%d  lambda=%f\n',data,k,lambda); %% Delete [PZH]

% Set predefined variables
X = [Xs,Xt];
X = X*diag(sparse(1./sqrt(sum(X.^2))));
[m,n] = size(X);
ns = size(Xs,2);
nt = size(Xt,2);
C = length(unique(Ys));

% Construct MMD matrix

% e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)]; % In here, I change the M_0 to M_ATL
e = [1/ns*A;-1/nt*ones(nt,1)]; 

M = e*e'*C;

if ~isempty(Yt0) && length(Yt0)==nt
    for c = reshape(unique(Ys),1,C)
        e = zeros(n,1);
        e(Ys==c) = 1/length(find(Ys==c));
        e(ns+find(Yt0==c)) = -1/length(find(Yt0==c));
        e(isinf(e)) = 0;
        M = M + e*e';
    end
end

M = M/norm(M,'fro');

% Construct centering matrix
H = eye(n)-1/(n)*ones(n,n);

% Joint Distribution Adaptation: JDA
if strcmp(ker,'primal')
    [P,~] = eigs(X*M*X'+lambda*eye(m),X*H*X',k,'SM');
    Z = P'*X;
else
    K = kernel(ker,X,[],gamma);
    [P,~] = eigs(K*M*K'+lambda*eye(m),K*H*K',k,'SM');
    Z = P'*K;
end

% fprintf('Algorithm JDA terminated!!!\n\n');

end
