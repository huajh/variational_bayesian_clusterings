%% Laplacian regularized Gaussian Mixture Model using variational Inference (VB-lapGMM)
%
%	@author         Junhao HUA
%	Create Time:	2013-1-9
%   
%
%%		

function [label,model,logLRange] = vblapgmm(Data,K,option)
%%	parameters Description:
% 		K		the number of mixing components
%		Data 	all observed data (dim*N)
%		N 		the number of data
%		dim 	the Dimension of data
%		Alpha 	the weight vector of each Gaussian (1 x K)
%		Mu 		the mean vector of each Gaussian (dim x K)
%		Sigma 	the Covariance matrix of each Gaussian (dim x dim x K)		
%
%       the initialization of prior superparameters
%       They can be set to small positive numbers to give
%       broad prior distrbutions indicating ignorance about the prior
%       Dirichlet Distribution Parameters:
%           alpha0	1
%       Wishart Distribution Parameters:
%           invW0 	dim x dim
%    		v0 		1
%       Gaussian Distribution Parameters:
%           m0 		dim x 1
%           beta0 	1
%
%       gamma: the regularization parameter 
%
    % configuration
    [dim,N] = size(Data);
    p =option.p;
    lambda = option.lambda;
    esp = 1e-7;
    logL = -realmax;
    
    W = CalcWeightbyknn(Data,p);
    DCol = full(sum(W,2));
    D = spdiags(DCol,0,N,N);
    L = D-W;
    
    prior = struct('alpha0',1,'beta0',1,'m0',mean(Data,2),'v0',dim+1,'invW0',eye(dim,dim));  
	% latent parameters : the probability of each point in each component
	%		r 		N x K
    gamma = 0.9;
    t = 0;
	maxtimes = 2000;
    logLRange = -inf(1,maxtimes);
    model.R = Initvb(Data,K);
    label = cell(N,1);
	while (t < maxtimes)
        t = t +1;
        while 1
            logL0 = logL;
            model.R = SmoothPosterior2(W,model.R,gamma); % laplacian process  
            model = MaxStep(Data,model,prior);
            model.R(model.R<realmin) = realmin;
            model.logR = log(model.R);
            llnew = vbound(Data,model,prior);            
            laploss = sum(sum(model.R'*L.*model.R'));
            logL = (llnew - lambda*laploss)/N;
           % fprintf('%d %e %e %e %f\n',t,llnew,lambda*laploss,logL,gamma);
            if(logL < logL0)
                gamma = 0.9*gamma;
            else
                break;
            end            
        end
        logLRange(t) = logL;
        [~,label0] = max(model.R,[],2);  
        label{t} = label0;
        fprintf('%d %e %e \n',t,abs(logL-logL0),esp*abs(logL));
        if abs(logL-logL0) < esp*abs(logL) || t >200
            break;
        end
        model = ExpectStep(Data,model);
    end
    label = label(1:t);
    logLRange = logLRange(1:t);
     
    disp(['Total iteratons:',num2str(t)]);


function model = MaxStep(data,model,prior)  
%%
    %	update the statistics
    % 		avgN	1 x K
    % 		avgX	dim x K
    % 		avgS	dim x dim x K
    %	update the superparameters
    %   the parameter of weight (1 x K):
    %       alpha 	1 x K
    %   the parameters of preision (dim x dim x K):
    %       invW 	dim x dim x K inv(W)
    %       V 		1 x K
    %   the parameters of mean:
    %       M 		dim x K
    %       beta 	1 x K         
    [~,K] = size(model.R);
    avgN = sum(model.R);
    aXn = data*model.R;
    avgX = bsxfun(@times,aXn,1./avgN);
    
    ws = (prior.beta0*avgN)./(prior.beta0+avgN);
    sqrtR = sqrt(model.R);
    for i = 1:K
        avgNS = bsxfun(@times,bsxfun(@minus,data,avgX(:,i)),sqrtR(:,i)');
        avgS(:,:,i) = avgNS*avgNS'/avgN(i);
        Xkm0 = avgX(:,i)-prior.m0;
        model.invW(:,:,i) = prior.invW0 +avgNS*avgNS'+ ws(i).*(Xkm0*Xkm0');
    end    
    model.alpha = prior.alpha0 + avgN;
    model.V = prior.v0 + avgN;   
    model.beta = prior.beta0 + avgN;       
    model.M = bsxfun(@times,bsxfun(@plus,prior.beta0.*prior.m0,aXn),1./model.beta);
    model.Sigma = avgS;
    

function model = ExpectStep(data,model)   
%%
    %	update the moments of parameters
    %	EQ          the expectation of Covariance matrix  N x K
    %	E_logLambda	the log expectation of precision	1 x K
    %	E_logPi 	the log expectation of the mixing proportion of the mixture components 1 x K
    [dim,N] = size(data);
    [~,K] = size(model.M);
    EQ = zeros(N,K);
    logW = zeros(1,K);
    for i=1:K
        U = chol(model.invW(:,:,i));  % Cholesky  X=R'R
        logW(i) = -2*sum(log(diag(U)));
        Q = U'\bsxfun(@minus,data,model.M(:,i));        
        EQ(:,i) = dim/model.beta(i) + model.V(i)*dot(Q,Q,1); % N x 1
    end      
    E_logLambda = sum(psi(0,bsxfun(@minus,model.V+1,(1:dim)')/2),1) + dim*log(2) + logW;
    E_logPi = psi(0,model.alpha) - psi(0,sum(model.alpha)); % 1 x K
    %	update latent parameter: r
    logRho = bsxfun(@minus,EQ,2*E_logPi + E_logLambda -dim*log(2*pi))/(-2);
    model.logR = bsxfun(@minus,logRho,logsumexp(logRho,2));
    model.R = exp(model.logR);    
  
function R0 =Initvb(data,K)
%%
    [~,N] = size(data);    
    [IDX,~] = kmeans(data',K,'emptyaction','drop','start','uniform'); 
    R0 = zeros(N,K);
    for i = 1:K
        R0(:,i) = IDX == i;
    end

function L = vbound(X, model, prior)
%%	stopping criterion
    alpha0 = prior.alpha0;
    beta0 = prior.beta0;
    m0 = prior.m0;
    v0 = prior.v0;
    
    invW0 = prior.invW0;
    % Dirichlet
    alpha = model.alpha;
    % Gaussian
    beta = model.beta;  
    m = model.M;
    % Whishart
    v = model.V;         
    invW = model.invW;  %inv(W) = V'*V
    
    R = model.R;
    logR = model.logR;

    [dim,k] = size(m);
    nk = sum(R,1);
    % pattern recognition and machine learning page496
    Elogpi = psi(0,alpha)-psi(0,sum(alpha));
    E_pz = dot(nk,Elogpi);  %10.72
    E_qz = dot(R(:),logR(:)); %10.75    
    logCoefDir0 = gammaln(k*alpha0)-k*gammaln(alpha0); % the coefficient of Dirichlet Distribution
    E_ppi = logCoefDir0+(alpha0-1)*sum(Elogpi); %10.73
    logCoefDir = gammaln(sum(alpha))-sum(gammaln(alpha));
    E_qpi = dot(alpha-1,Elogpi)+logCoefDir; %10.76
    
    U0 = chol(invW0);
    sqrtR = sqrt(R);
    xbar = bsxfun(@times,X*R,1./nk); % 10.52
    logW = zeros(1,k);
    trSW = zeros(1,k);
    trM0W = zeros(1,k);
    xbarmWxbarm = zeros(1,k);
    mm0Wmm0 = zeros(1,k);
    for i = 1:k
        U = chol(invW(:,:,i));
        logW(i) = -2*sum(log(diag(U)));      
        Xs = bsxfun(@times,bsxfun(@minus,X,xbar(:,i)),sqrtR(:,i)');
        V = chol(Xs*Xs'/nk(i));
        Q = V/U;
        trSW(i) = dot(Q(:),Q(:));  % equivalent to tr(SW)=trace(S/M)
        Q = U0/U;
        trM0W(i) = dot(Q(:),Q(:));
        q = U'\(xbar(:,i)-m(:,i));
        xbarmWxbarm(i) = dot(q,q);
        q = U'\(m(:,i)-m0);
        mm0Wmm0(i) = dot(q,q);
    end
    ElogLambda = sum(psi(0,bsxfun(@minus,v+1,(1:dim)')/2),1)+dim*log(2)+logW; % 10.65
    Epmu = sum(dim*log(beta0/(2*pi))+ElogLambda-dim*beta0./beta-beta0*(v.*mm0Wmm0))/2;
    logB0 = v0*sum(log(diag(U0)))-0.5*v0*dim*log(2)-logmvgamma(0.5*v0,dim);
    EpLambda = k*logB0+0.5*(v0-dim-1)*sum(ElogLambda)-0.5*dot(v,trM0W);
    E_logpMu_Lambda = Epmu + EpLambda; % 10.74

    Eqmu = 0.5*sum(ElogLambda+dim*log(beta/(2*pi)))-0.5*dim*k;
    logB =  -v.*(logW+dim*log(2))/2-logmvgamma(0.5*v,dim);
    HqLambda = -0.5*sum((v-dim-1).*ElogLambda-v*dim)-sum(logB);
    E_logqMu_Lambda = Eqmu - HqLambda;%10.77

    E_pX = 0.5*dot(nk,ElogLambda-dim./beta-v.*trSW-v.*xbarmWxbarm-dim*log(2*pi)); %10.71
    L = E_pX+E_pz+E_ppi+E_logpMu_Lambda-E_qz-E_qpi-E_logqMu_Lambda;
	

function pkx = SmoothPosterior2(W,pkx,gamma)
%%
    if_iterator = 1;
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
            if t > 12
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
    
    
    
    
    
    
    
    
    
    
    
