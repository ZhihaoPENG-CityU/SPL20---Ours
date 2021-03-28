%% We appreciate it if you use this matlab code and cite our papers.
%% Contact: zhihapeng3-c@my.cityu.edu.hk; zhpengcn@126.com
%% The BibTeX files are as follows,
%{
1- SPL20  --->
@article{peng2020non,
  title={Non-Negative Transfer Learning With Consistent Inter-Domain Distribution},
  author={Peng, Zhihao and Jia, Yuheng and Hou, Junhui},
  journal={IEEE Signal Processing Letters},
  volume={27},
  pages={1720--1724},
  year={2020},
  publisher={IEEE}
}

2- TCSVT19 --->
@article{peng2019active,
  title={Active Transfer Learning},
  author={Peng, Zhihao and Zhang, Wei and Han, Na and Fang, Xiaozhao and Kang, Peipei and Teng, Luyao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={30},
  number={4},
  pages={1022--1036},
  year={2019},
  publisher={IEEE}
}
<---
%}
%% Matlab implementation for our SPL'20 paper.
% clc,clear;
% close all;
% clear memory;
currentFolder = pwd;%
addpath(genpath(currentFolder));
tic;

load COIL_1;
XS_S = X_src;
XS_L = Y_src;
XT_S = X_tar;
XT_L = Y_tar;

pct = [0.2]; % 
% pct = [0.2, 0.4, 0.6, 0.8 ];
[d,num] = size(XS_S);

Koptions = [];
Koptions.NeighborMode = 'Supervised';
Koptions.WeightMode = 'Binary';
Koptions.gnd = XS_L;
K = constructW( XS_S',Koptions );

Woptions = [];
Woptions.NeighborMode = 'KNN';
Woptions.WeightMode = 'Binary';
W = constructW( XS_S',Woptions );

R1 = 0.1 ;
R2 = 10000 ;

acc1 = zeros(9,9);

bestAcc = 0 ;
besta = [] ;

%% Initialization
for pct_itm = 1 : length(pct)
    npct = ceil( (1-pct(pct_itm)) * num );
    for i = 1:length(R1)
        acc_result = zeros(1,9);
        for j = 1:length(R2)
            Part.r1 = R1(i);
            Part.r2 = R2(j);
            Part.u = 0.1;
            disp(['r1:',num2str(R1(i)),'    r2:',num2str(R2(j))]);
            [ Acc, A, P ] = main_NNTL( XS_S, XS_L, XT_S, XT_L, K, W, Part );
            [ XS, YS ] = pct_ATL( XS_S,XS_L,A,npct,d,num );
            [Result, ave_ls]= KNN((P'*XS)',YS',(P'*XT_S)',XT_L,1);
            acc_result(1,j) = ave_ls*100;
            disp(['acc: ',num2str(ave_ls)]);
            if ave_ls > bestAcc
                bestAcc = ave_ls;
                besta = A;
                bestr1 = R1(i);
                bestr2 = R2(j);
            end
        end
        if pct_itm == 1
            acc1(i,:) = acc_result;
        end
    end
end
toc;