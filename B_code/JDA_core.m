
function [Z,P] = JDA_core( X_src, Y_src, X_tar, Y_tar_pseudo, A, options )
%% Set options
% lambda = options.lambda;              %% lambda for the regularization
% dim = options.dim;                    %% dim is the dimension after adaptation, dim <= m
% 	kernel_type = options.kernel_type;    %% kernel_type is the kernel name, primal|linear|rbf
% 	gamma = options.gamma;                %% gamma is the bandwidth of rbf kernel

%% Construct MMD matrix
X_src = X_src';
X_tar = X_tar';
X = [X_src',X_tar'];
X = X*diag(sparse(1./sqrt(sum(X.^2))));
[ ~, n ] = size(X);
ns = size(X_src,1);
nt = size(X_tar,1);
e = [ 1/ns * ones(ns,1) ; -1/nt * ones(nt,1) ];
ea = [ 1/ns * A ; -1/nt * ones(nt,1) ];
C = length(unique(Y_src));

%%% M0
M = ea * ea' * C;  %multiply C for better normalization
% M = ea * ea' ;  %multiply C for better normalization

%%% Mc
N = 0;
if ~isempty(Y_tar_pseudo) && length(Y_tar_pseudo)==nt
    for c = reshape(unique(Y_src),1,C)
        e = zeros(n,1);
        e(Y_src==c) = 1 / length(find(Y_src==c));
        e(ns+find(Y_tar_pseudo==c)) = -1 / length(find(Y_tar_pseudo==c));
        e(isinf(e)) = 0;
        N = N + e*e';
    end
end

M = M + N;
M = M / norm(M,'fro');

% 	%% Centering matrix H
% 	H = eye(n) - 1/n * ones(n,n);
%
%% Calculation
% 	if strcmp(kernel_type,'primal')
dim = options.dim;
[P,~]  = eigs(X*M*X',dim,'LR');
P = real(P);
Z = P'*X;
%     else
%     	K = kernel_jda(kernel_type,X,[],gamma);
%     	[A,~] = eigs(K*M*K'+lambda*eye(n),K*H*K',dim,'SM');
%     	Z = A'*K;
% 	end

end