%% Laplacian regularized   infinite Student's t mixture model using variational bayesian (vb-lapismm)
%
%	@author         Junhao HUA
%	Create Time:	2013-1-12
% 
%%
	
function [label,model,logLRange] = vblapismm(Data,K,option)
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
%       Gamma2 distribution: the posterior
%       q(alpha) = Gam(eta1,eta2)
%           eta10     1
%           eta20     1 
%       the latent variable
%       the probability of each point in each component
%           R 		N x K
%       the parameters of latent variable u of Gamma distribution
%         v       1 x K
    [dim,N] = size(Data);
    prior = struct('m0',zeros(dim,1),'beta0',1e-3,'invW0',eye(dim,dim),'v0',dim+1,'eta10',0.5,'eta20',1);
    logL0 = -inf;
    esp = 1e-6;

    p =option.p;
    lambda = option.lambda;
    W = CalcWeightbyknn(Data,p);
    DCol = full(sum(W,2));
    D = spdiags(DCol,0,N,N);
    L = D-W;	
    
    maxtimes = 1000;
    logL = -realmax;
    model = Initvb(Data,K,prior);
    model = MaxStep(Data,model,prior);
    t = 0; 
    logLRange = -inf(1,maxtimes);
    label = cell(N,1);
    while t < maxtimes
        t = t+1;
        gamma = 0.9;
        while 1
            logL0 = logL;
            model.R = SmoothPosterior2(W,model.R,gamma); % laplacian process
            model.R(model.R<realmin) = realmin;
            model.logR = log(model.R);            
            model = MaxStep(Data,model,prior);
            model = updateU(model,Data);
            llnew = vbound(Data,model,prior);            
            laploss = sum(sum(model.R'*L.*model.R'));
            logL = (llnew - lambda*laploss)/N;
            fprintf('%d %e %e %e %f\n',t,llnew,lambda*laploss,logL,gamma);
            if(logL < logL0)
                gamma = 0.9*gamma;
            else
                break;
            end
        end  
        logLRange(t) = logL;
        [~,label0] = max(model.R,[],2);  
        label{t} = label0;          
        %fprintf('%e %e \n',abs(logL-logL0),esp*abs(logL));
        if abs(logL-logL0) < esp*abs(logL) || t >80
            break;
        end                  
        model = ExpectStep(Data,model);
    end
    label = label(1:t-1);
    logLRange = logLRange(1:t);  
    disp(['Total iteratons:',num2str(t)]);


function model = MaxStep(data,model,prior)     
%%
    %	update the statistics
    % 		avgN	1 x K
    % 		avgX	dim x K
    % 		avgS	dim x dim x K
    %	update the hyperparameters
    %
    %   the parameters of preision (dim x dim x K):
    %       invW 	dim x dim x K inv(W)
    %       V 		1 x K
    %   the parameters of mean:
    %       M 		dim x K
    %       beta 	1 x K 
    %   the Gamma parameters of the latent variable U
    %   Gam(U|Uv/2,Uv/2)
    %       Uv      1 x K   
    %   the hyperparameter of the beta distribution of the random variable
    %   q(piV) = Beta(vbeta1,vbeta2) 
    %       vbeta1   1 x k
    %       vbeta2   1 x k
    %   Gamma2 distribution: the posterior 
    %   q(alpha) = Gam(eta1,eta2)
    %       eta1     1
    %       eta2     1
    [~,K] = size(model.R);
    ReU = model.R.*model.eU;
    sumR = sum(model.R);
    aXn = data*ReU;
    avgN = sum(ReU);
    avgX = bsxfun(@times,aXn,1./avgN);
    % wishart
    ws = (prior.beta0*avgN)./(prior.beta0+avgN);
    sqrtRU = sqrt(ReU);
    for i = 1:K
        avgNS = bsxfun(@times,bsxfun(@minus,data,avgX(:,i)),sqrtRU(:,i)');
        Xkm0 = avgX(:,i)-prior.m0;
        model.invW(:,:,i) = prior.invW0 +avgNS*avgNS'+ ws(i).*(Xkm0*Xkm0');
    end
    model.V = prior.v0 + sumR;  
    % replace of dirichlet distribustion
    % model.alpha = prior.alpha0 + sumR;
    %
    model.vbeta1 = 1 + sumR;
    Ealpha = model.eta1/model.eta2;
    for i =1:K
        model.vbeta2(i) = Ealpha + sum(sum(model.R(:,i+1:K),2));
    end
    model.eta1 = prior.eta10 + K-1;
    E_log1_V = psi(model.vbeta2)-psi(model.vbeta1+model.vbeta2);
    model.eta2 = prior.eta20 - sum(E_log1_V(1:K-1));
    % gaussian
    model.beta = prior.beta0 + avgN;
    model.M = bsxfun(@times,bsxfun(@plus,prior.beta0.*prior.m0,aXn),1./model.beta);   
    % non-linear equations: newton method
    tmp = dot(model.elogU-model.eU,model.R)./sumR;
    model.Uv = zeros(1,K);
    for i=1:K
        model.Uv(i) = fzero(@(x)1+tmp(i)+log(x/2)-psi(0,x/2),[1e-5 1e5]); % init parameter ?
    end
   
 function model = ExpectStep(data,model)   
 %%
    %	update the moments of parameters
    %	EQ          the expectation of Covariance matrix  N x K
    %	E_logLambda	the log expectation of precision	1 x K
    %	E_logPi 	the log expectation of the mixing proportion of the mixture components 1 x K
    % 
    % latent variable
    %  R    N x K
    %  <U>,<logU>
    [dim,N] = size(data);
    [~,K] = size(model.M);
    EQ = zeros(N,K);
    logW = zeros(1,K);
    for i=1:K
        U = chol(model.invW(:,:,i));
        logW(i) = -2*sum(log(diag(U)));
        Q = U'\bsxfun(@minus,data,model.M(:,i));
        EQ(:,i) = dim/model.beta(i) + model.V(i)*dot(Q,Q,1); % N x 1
    end
    E_logLambda = sum(psi(0,bsxfun(@minus,model.V+1,(1:dim)')/2),1) + dim*log(2)+logW; % - + ?   
    %E_logPi = psi(0,model.alpha) - psi(0,sum(model.alpha)); % 1 x K  
    E_log1_V = psi(model.vbeta2)-psi(model.vbeta1+model.vbeta2);
    E_logV = psi(model.vbeta1)-psi(model.vbeta1+model.vbeta2);
    E_logPi2 = zeros(1,K);
    for i = 1:K
        E_logPi2(i) = sum(E_log1_V(1:i-1))+E_logV(i);
    end
    %	update latent parameter: R
    logRho = bsxfun(@plus,dim*model.elogU-model.eU.*EQ,2*E_logPi2 + E_logLambda -dim*log(2*pi))/2;
    model.logR = bsxfun(@minus,logRho,logsumexp(logRho,2));
    model.R = exp(model.logR);
    
    % update hyperparameters of latent parmeter U:
    a = 1/2*bsxfun(@plus,model.Uv,dim*model.R);
    b = 1/2*bsxfun(@plus,model.Uv,model.R.*EQ);  %N x K    
    model.a = a;
    model.b = b;
    % update <U>,<logU>
    model.eU = a./b;
    model.elogU = psi(0,a)-log(b);
    
function model = updateU(model,data)
%%
    % update hyperparameters of latent parmeter U:
    [dim,N] = size(data);
    [~,K] = size(model.M);
    EQ = zeros(N,K);
    logW = zeros(1,K);
    for i=1:K
        U = chol(model.invW(:,:,i));
        logW(i) = -2*sum(log(diag(U)));
        Q = U'\bsxfun(@minus,data,model.M(:,i));
        EQ(:,i) = dim/model.beta(i) + model.V(i)*dot(Q,Q,1); % N x 1
    end
    
    a = 1/2*bsxfun(@plus,model.Uv,dim*model.R);
    b = 1/2*bsxfun(@plus,model.Uv,model.R.*EQ);  %N x K    
    model.a = a;
    model.b = b;    
    % update <U>,<logU>
    model.eU = a./b;
    model.elogU = psi(0,a)-log(b);
    
function model =Initvb(data,k,prior)
%%  
    [~,N] = size(data);    
    [IDX,~] = kmeans(data',k,'emptyaction','drop','start','uniform');   
    R0 = zeros(N,k);
    for i = 1:k
        R0(:,i) = IDX == i;
    end
    model.R = R0;   
    tmp = sum(R0);
    model.eU = repmat(tmp./sum(tmp),N,1);
    model.elogU = log(model.eU);
    model.eta1 = prior.eta10;
    model.eta2 = prior.eta20;
    
    

function L = vbound(X, model, prior)        
%%	stopping criterion
    %alpha0 = prior.alpha0;
    beta0 = prior.beta0;
    m0 = prior.m0;
    v0 = prior.v0;    
    invW0 = prior.invW0;
    
    % Dirichlet
    %alpha = model.alpha;
    vbeta1 = model.vbeta1;
    vbeta2 = model.vbeta2;
    eta1 = model.eta1;
    eta2 = model.eta2;
    % Gaussian
    beta = model.beta;  
    m = model.M;
    % Whishart
    v = model.V;
    invW = model.invW;  %inv(W) = V'*V
    % gamma
    Uv =model.Uv;
    a = model.a;  % N x k
    %
    R = model.R;
   % logR = model.logR;
    eU = model.eU;
    elogU = model.elogU;
    [N,~] =size(R);
    [dim,k] = size(m);
    sumR = sum(R,1);
    ReU = model.R.*model.eU;
    avgN = sum(ReU);
    
    %   smm ->ismm
    Elog1_V = psi(vbeta2)-psi(vbeta1+vbeta2);
    ElogV = psi(vbeta1)-psi(vbeta1+vbeta2);
    Ealpha = eta1/eta2;
    Elogalpha = psi(eta1)-log(eta2);
    E_logpZ = 0;
    for i=1:k
        E_logpZ = E_logpZ + sumR(i).*sum(Elog1_V(1:i-1))+ElogV(i);
    end   
    E_logpV = (Ealpha-1)*sum(Elog1_V);
    E_logpalpha = (eta1-1)*Elogalpha-eta2*Ealpha;
    
    %reR = R;
    %reR(R<realmin) = realmin;
    %tmpR = R.*log(reR);
    %tmpR(R<realmin) = 0;
    %E_logqZ = sum(sum(tmpR),2);
    R(R<realmin) = realmin;
    E_logqZ = sum(sum(R.*log(R)),2);
    E_logqV = sum((vbeta1-1).*ElogV+(vbeta2-1).*Elog1_V,2);
    E_logqalpha = (eta1-1).*Elogalpha-eta2.*Ealpha;
    
    U0 = chol(invW0);
    xbar = bsxfun(@times,X*ReU,1./avgN); % 10.52
    logW = zeros(1,k);
    trM0W = zeros(1,k);
    xbaruLambadxbaru = zeros(1,k);
    mm0Wmm0 = zeros(1,k);
    for i = 1:k
        U = chol(invW(:,:,i));
        logW(i) = -2*sum(log(diag(U)));
        Q = U0/U;
        trM0W(i) = dot(Q(:),Q(:));
        q = U'\(xbar(:,i)-m(:,i));
        xbaruLambadxbaru(i) = dim/beta(i)+v(i).*dot(q,q);
        q = U'\(m(:,i)-m0);
        mm0Wmm0(i) = dot(q,q);
    end
    
    ElogLambda = sum(psi(0,bsxfun(@minus,v+1,(1:dim)')/2),1)+dim*log(2)+logW; % 10.65 -+ ?
    E_pX = 0.5*sum(dot(R,bsxfun(@plus,ElogLambda-dim*log(2*pi),dim*elogU-bsxfun(@times,eU,xbaruLambadxbaru)))); %10.71 / 1   
     
    Epmu = sum(dim*log(beta0/(2*pi))-beta0.*xbaruLambadxbaru)/2;
    EpLambda = 0.5*sum((v0-dim+1).*ElogLambda-dot(v,trM0W),2);
    E_logpMu_Lambda = Epmu + EpLambda; % 10.74 /2 3

    logB =  -v.*(logW+dim*log(2))/2-logmvgamma(0.5*v,dim);
    E_logqMu_Lambda = sum(dim*log(v./(2*pi))+logB+dim+(v-dim).*ElogLambda+v.*dim,2)/2;
    
    Uv2 = Uv/2;
    E_logpu = sum(N*(Uv2.*log(Uv2)-gammaln(Uv2))+(Uv2-1).*sum(elogU)-Uv2.*sum(eU),2); % right / 4    
    E_logqu = sum(sum(a.*(psi(a)-1)-gammaln(a)-elogU),2);  % / 9

    L = E_pX+E_logpMu_Lambda+E_logpu+E_logpZ+E_logpV+E_logpalpha-E_logqZ-E_logqV-E_logqalpha-E_logqMu_Lambda-E_logqu;
    
    
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
            if t > 12
                break;
            end
            F(F<realmin) = realmin;
            relaF = abs( Fold - F)./F;
            %fprintf('%f \n',max(max(relaF)));
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
    
    