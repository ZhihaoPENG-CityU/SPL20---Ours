%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{ 
# Cite
@article{peng2020non,
  title={Non-Negative Transfer Learning With Consistent Inter-Domain Distribution},
  author={Peng, Zhihao and Jia, Yuheng and Hou, Junhui},
  journal={IEEE Signal Processing Letters},
  volume={27},
  pages={1720--1724},
  year={2020},
  publisher={IEEE}}
# Reference
[r1] Non-Negative Transfer Learning with Consistent Inter-domain Distribution, SPL20
% MATLAB R2013a
% Author£º ZhiHao PENG (13/09/2020)
% Contact: zhihapeng3-c@my.cityu.edu.hk; zhpengcn@126.com
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
% Input:
%%%%   X           :   the training data,d*ns
%%%%   Y           :   the test data,d*nt
%%%%   KXY         :   the class coefficient matrix of x_i,x_j, ns * ns
%%%%   WXY         :   the distance coefficient matrix of x_i,x_j, ns * ns
%%%%   Part        :   [Part] Struct
%%%%%%%%   r1      :   parameter of r1 to balance KXY
%%%%%%%%   r2      :   parameter of r2 to balance WXY
%%%%%%%%   u       :   Lagrangian multipliers

% Output:
%%%%   A           	:   the weight coefficient vector, ns * 1
%%%%   P           	:   the projection matrix P, d * r
%%%%   Acc			:
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ Acc, A, P ] = main_NNTL( X, XL, Y, YL, KXY, WXY, Part )


[d, n1] = size(X);
[~, n2] = size(Y);
r =  floor((90*d)/100) ;
B1 = ones(n1,1);
B2 = ones(n2,1);
I = eye(n1,n1);
P = eye( d, r );


r1 = Part.r1;
r2 = Part.r2;


A = zeros( n1, 1) ;
V = zeros( n1 ,1 );
Y2 = zeros( n1,1 );
y1 = 0;
u = Part.u ;
umax = 10^5 ;
pu = 1.01 ;
Cls = [];
Acc = [];
MaxIter = 250;  

for i = 1:MaxIter
    % Update weight vector by Step 1  in [r1] (i.e., Eq. (6))
    left = ( 2 / (n1^2 ) ) * ((( X' * P) * P') * X ) + r1 * (KXY + KXY') + r2 * (WXY + WXY') + u * ( B1 * B1' ) + u * I ;
    right = ( 2/(n1 * n2)) * ((( (X' * P) * P') * Y ) * B2) + u * B1 + u * V - y1 * B1 - Y2 ;
    A  = pinv(left) * right;
    
    % Update V by Step 3 in [r1]
    V = A + ( 1 / u ) * Y2 ;
    V = max( V ,0 );
    
    % Update projection matrix P by Step 2 in [r1]
    options.dim = r ;
    options.T = 10;
    [Z,P] = JDA2ATL(X,Y,XL,Cls,options,A);
    Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
    Zs = Z(:,1:size(X,2));
    Zt = Z(:,size(X,2)+1:end);
	% Cls is the predicted pseudo target label
    Cls = knnclassify(Zt',Zs',XL,1);
    acc = length(find(Cls==YL))/length(YL);
    Acc = [Acc;acc];
    
    % Update other by Eq. (10) in [r1]
    y1 = y1 + u * ( A' * B1 - 1 );
    Y2 = Y2 + u * ( A - V );
    u = min(  pu * u ,umax );

    con = norm(A-V, 'fro')^2;
    obj(i) = con;
	% repeat above updating steps until convergence
    if norm(A-V, 'fro')^2<10^-11 || norm(A-V, 'fro')^2 > 10^18
        break;
    end
end
end
