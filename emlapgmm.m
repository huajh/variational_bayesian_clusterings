%%  Algtiothm       Laplacian regularized Gaussian Mixture Model by Expectation Miximum (EM-lapGMM)
%	@author         Junhao HUA
%	Create Time     2013-1-9
%   email:          huajh7 AT gmail.com
%
% References:
% [1] Xiaofei He, Deng Cai, Yuanlong Shao, Hujun Bao, and Jiawei Han,
% "Laplacian Regularized Gaussian Mixture Model for Data Clustering", IEEE
% Transactions on Knowledge and Data Engineering, Vol. 23, No. 9, pp.
% 1406-1418, 2011.

function [labelCell,model,logLRange] = emlapgmm(Data, K,option)  
%%	parameters Description:
%		M		the number of Gaussian function
%		Data 	all observed data (dim*N)
%		dim 	the Dimension of data
%		N 		the number of data
%		Alpha 	the weight vector of each Gaussian (1*K)
%		M 		the mean vector of each Gaussian (dim*K)
%		Sigma 	the Covariance matrix of each Gaussian (dim*dim*K)

    p =option.p;
    lambda = option.lambda;
    [~,N] = size(Data);
    model = InitByKmeans(Data,K);   
    W = CalcWeightbyknn(Data,p);
    epsilon = 1e-7;
    t = 0;
    maxTimes = 1000;
    %gamma = 0.9;
    DCol = full(sum(W,2));
    D = spdiags(DCol,0,N,N);
    L = D-W;
    logL = -realmax;
    logLRange = -inf(1,maxTimes);
    gamma = 0.9;
    labelCell = cell(N,1);
    while(t<maxTimes)     
    	t = t+1;             
        pkx = E_step(Data,model);
        while 1
            logold = logL;
            pkx = SmoothPosterior2(W,pkx,gamma); % laplacian process   
            model = M_step(Data,pkx);
            llnew = vbound(Data, model);
            laploss = sum(sum(pkx'*L.*pkx'));
            logL = (llnew - lambda*laploss)/N;
            %fprintf('%d %f %f %f %f\n',t,llnew,lambda*laploss,logL,gamma);
            if(logL < logold)
                gamma = 0.9*gamma;
            else
                break;
            end
        end
        logLRange(t) = logL;
        [~,label0] = max(pkx,[],2);
        labelCell{t} = label0;
        fprintf('%e %e\n',abs(logL-logold),epsilon*abs(logL));
        if(abs(logL-logold)<epsilon*abs(logL)) || t > 700
            break;
        end
    end 
    labelCell = labelCell(1:t);
    logLRange = logLRange(1:t);
    disp(['Total iteratons:',num2str(t)]);    

function Pkx = E_step(Data,model)
%% E- Step
    [~,N] = size(Data);
    [~,K] = size(model.M);
    pxk = ones(N,K);
    for i = 1:K
        % probability p(x|i) N x K
        pxk(:,i) = GaussPDF(Data,model.M(:,i),model.Sigma(:,:,i));
    end
    % calc posterior P(i|x) N*K
    pxk = repmat(model.Alpha,[N 1]).*pxk;
    Pkx = pxk./(repmat(sum(pxk,2),[1 K])+realmin);
    
function model = M_step(Data,Pkx)
%% M-step
    [dim,N] = size(Data);
    [~,M] = size(Pkx);
    PkX = sum(Pkx);
    for i=1:M
        model.Alpha(i) = PkX(i)/N;
        model.M(:,i) = Data*Pkx(:,i)/PkX(i);
        datatmp = Data-repmat(model.M(:,i),1,N);
        model.Sigma(:,:,i) = (repmat(Pkx(:,i)',dim,1).*datatmp*datatmp')/PkX(i);
        model.Sigma(:,:,i) = model.Sigma(:,:,i) +1E-5.*diag(ones(dim,1));
    end
    
function [ prob ] = GaussPDF(Data,M,Sigma)
%%  Calculate Gaussian Probability Distribution Function
    [dim,N] = size(Data);
    Data = Data'-repmat(M',N,1);
    prob = sum((Data/Sigma).*Data,2); % Data*inv(Sigma)
    prob = exp(-0.5*prob)/sqrt((2*pi)^dim*(abs(det(Sigma))+realmin));


function model = InitByKmeans(Data,K)
%%  initialization by k-means
    [dim,N] = size(Data);
    [IDX,M0] = kmeans(Data',K,'emptyaction','drop','start','uniform');    
    M0 = M0';
    Alpha0 = zeros(1,K);
    Sigma0 = zeros(dim,dim,K);
    for i=1:K
        Alpha0(i)=sum(i==IDX)/N;
        idx_temp = find(IDX==i);
        Data_tmp1 = Data(:,idx_temp)-repmat(M0(:,i),1,length(idx_temp));
        Sigma0(:,:,i) = Data_tmp1*Data_tmp1'/sum(i==IDX) +1E-5.*diag(ones(dim,1));
    end
    model.M = M0;
    model.Sigma = Sigma0;
    model.Alpha = Alpha0;    
   

function logL = vbound(Data, model)
%% calc the log likelihood
    [~,N] = size(Data);
    [~,K] = size(model.M);
    pxk = zeros(N,K);
    for i = 1:K
        % probability p(x|i) N x M
        pxk(:,i) = GaussPDF(Data,model.M(:,i),model.Sigma(:,:,i));
    end
    pxk = repmat(model.Alpha,[N 1]).*pxk;
   % Pkx = pxk./(repmat(sum(pxk,2),[1 K])+realmin);
    loglik = pxk*model.Alpha';
    loglik(loglik<realmin) = realmin;
    logL = sum(log(loglik));


function W = CalcWeightbyknn(x,p)
%%
    %@input:
    %   x(data) dim x N
    %   p       the number of nearest neighbors
    %@output:
    %   sparse matrix (N*N)
    %   W
    
    % IDX N x p
    [~,nSmp] = size(x);
    [IDX,~] = knnsearch(x',x','K',p+1,'NSMethod','kdtree','distance','euclidean');
    % Weight functoin: euclidean distance
    D = ones(nSmp,p+1);
    a = reshape(repmat((1:1:nSmp),1,p),nSmp*p,1);
    b = double(reshape(IDX(:,2:end),nSmp*p,1));
    s = reshape(D(:,2:end),nSmp*p,1);
    W = sparse(a,b,s,nSmp,nSmp);

function pkx = SmoothPosterior2(W,pkx,gamma)
%%
    if_iterator =1;
    [nSmp,k] = size(pkx);
    DCol = full(sum(W,2));
    S = spdiags(DCol.^-1,0,nSmp,nSmp)*W;
    if if_iterator
        F = pkx;
        relaF = 1;
        esp = 1e-3;
        t = 0;
        while max(max(relaF)) > esp
            t = t+1;
            for j=1:199
                F = (1-gamma)*F + gamma*W*F./repmat(DCol,1,k);
            end            
            Fold = F;
            F = (1-gamma)*F + gamma*W*F./repmat(DCol,1,k);
            if t > 50
                break;
            end
            F(F<realmin) = realmin;
            relaF = abs( Fold - F)./F;
           % fprintf('%f \n',max(max(relaF)));
        end
    else
        T = speye(size(W,1)) - gamma*S;
        T = T/(1-gamma);
        F = T\pkx;
        if min(min(F)) < 0
            F = max(0,F);
            %error('negative!');
        end
    end
    pkx = F;
 
 
function [pkxBest,LogLBest,GammaMax] = SmoothPosterior(W,X,pkx)
%%  Smooth the posterior probabilities until cnvergence
    %   find best Gamma(GammaMin,GammaMax) that maximum posterior probabilities
    %
    GammaMax = 0.99;
    GammaMin = 0;
    splitNo = 10;
    delta = (GammaMax-GammaMin)/splitNo;
    
    while delta > 1e-5
        GammaRange = GammaMin:delta:GammaMax;
        [pkxRange, LogLRange] = SmoothObj(W,X,pkx,GammaRange);
        [LogLBest, idx] = max(LogLRange);
        pkxBest = pkxRange{idx};
        maxIdx = LogLRange == LogLBest;
        if sum(maxIdx) > 1
            idx = find(maxIdx);
            GammaMin = GammaRange(idx(1));
            GammaMax = GammaRange(idx(end));
        else
            GammaMin = max(0,GammaRange(idx) - delta);
            GammaMax = GammaRange(idx) + delta;
            while GammaMax >= 1
                delta = delta/2;
                GammaMax = GammaRange(idx) + delta;
            end
        end
        delta = (GammaMax-GammaMin)/splitNo;
    end
    
function [pkxRange,LogLRange] = SmoothObj(W,X,pkx,GammaRange)
%%
    
    if_iterator = 1;
    lambda = 100;
    [nSmp,k] = size(pkx);
    LogLRange = zeros(size(GammaRange));
    pkxRange = cell(size(GammaRange));
    DCol = full(sum(W,2));
    D = spdiags(DCol,0,nSmp,nSmp);
    L = D-W;
    S = spdiags(DCol.^-1,0,nSmp,nSmp)*W;
    
    for i = 1:length(GammaRange)
        gamma = GammaRange(i);
        if gamma > 0
            if if_iterator
                F = pkx;
                iter = 0;
                relaF = 1;
                while max(max(relaF)) > 1e-3 % 1e-8
                    Fold = F;
                   % for j=1:200
                       %F = (1-gamma)*pkx + gamma*W*F./repmat(DCol,1,k);
                       F = (1-gamma)*F + gamma*W*F./repmat(DCol,1,k);
                   % end
                    relaF = abs( Fold - F)./F;                   
                    iter = iter + 1;
                end
           else
                T = speye(size(W,1)) - gamma*S;
                T = T/(1-gamma);
                F = T\pkx;
                if min(min(F)) < 0
                    F = max(0,F);
                    %error('negative!');
                end
            end
        else
            F = pkx;
        end
        model = M_step(X,F);
        llnew = vbound(X,model);
        laploss = sum(sum(F'*L.*F'));
        LogLRange(i) = llnew - lambda*laploss;
        pkxRange{i} = F;
    end


        



