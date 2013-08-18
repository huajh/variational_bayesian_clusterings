
%%
%   Student's t mixture model using variational bayesian
%
%	@author         Junhao HUA
%	Create Time:	2013-1-7
%
%   references:
%    Svensen,M.Bishop,Bohust Bayesian Mixture Modelling,2004
% 
%%
	
function [label,model,logLRange] = vbsmm(Data,K)
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

    [dim,N] = size(Data);
    prior = struct('alpha0',1e-3,'m0',zeros(dim,1),'beta0',1e-3,'invW0',eye(dim,dim),'v0',dim+1);
    logL0 = -inf;
    esp = 1e-7;

	% the latent variable
    %    the probability of each point in each component
	%		R 		N x K
    %    the parameters of latent variable u of Gamma distribution
    %       v       1 x K
    model = Initvb(Data,K);
    t = 0;
	maxtimes = 1000;    
    logLRange = -inf(1,maxtimes);    
    while t < maxtimes        
        [model,flag] = MaxStep(Data,model,prior);         
        if ~flag            
            break;
        end
        t = t +1;
        model = ExpectStep(Data,model);
        logL = vbound(Data,model,prior)/N;
        logLRange(t) = logL;
       % fprintf('%e %e \n',abs(logL-logL0),esp*abs(logL));
        if abs(logL-logL0) < esp*abs(logL)
            break;
        end       
        logL0 = logL;
    end
    logLRange = logLRange(1:t);
    [~,label] = max(model.R,[],2);   
    disp(['Total iteratons:',num2str(t)]);


function [model,flag] = MaxStep(data,model,prior)     
%%
    %	update the statistics
    % 		avgN	1 x K
    % 		avgX	dim x K
    % 		avgS	dim x dim x K
    %	update the superparameters
    %   the parameter of weight(pi) (1 x K):
    %       alpha 	1 x K
    %   the parameters of preision (dim x dim x K):
    %       invW 	dim x dim x K inv(W)
    %       V 		1 x K
    %   the parameters of mean:
    %       M 		dim x K
    %       beta 	1 x K 
    %   the Gamma parameters of the latent variable U
    %       Uv      1 x K    
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
    % dirichlet
    model.alpha = prior.alpha0 + sumR;
    % gaussian
    model.beta = prior.beta0 + avgN;       
    model.M = bsxfun(@times,bsxfun(@plus,prior.beta0.*prior.m0,aXn),1./model.beta);   
    % non-linear equations: newton method
    sumR(sumR < realmin) = realmin;
    tmp = dot(model.elogU-model.eU,model.R)./sumR;
    
    flag = true;
    if(max(tmp)>=-(1+1e-3))
        flag = false;
        return;
    end
    model.Uv = zeros(1,K);
    for i=1:K
        model.Uv(i) = fzero(@(x)1+tmp(i)+log(x/2)-psi(0,x/2),[1e-5 1e5]); % init parameter
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
    E_logPi = psi(0,model.alpha) - psi(0,sum(model.alpha)); % 1 x K    
    
    %	update latent parameter: R
    logRho = bsxfun(@plus,dim*model.elogU-model.eU.*EQ,2*E_logPi + E_logLambda -dim*log(2*pi))/2;
    model.logR = bsxfun(@minus,logRho,logsumexp(logRho,2));
    model.R = exp(model.logR);
    %disp(sum(model.R));
    % update hyperparameters of latent parmeter U:
    a = 1/2*bsxfun(@plus,model.Uv,dim*model.R);
    b = 1/2*bsxfun(@plus,model.Uv,model.R.*EQ);  %N x K    
    model.a = a;
    model.b = b;    
    % update <U>,<logU>
    model.eU = a./b;
    model.elogU = psi(0,a)-log(b);

  
function model =Initvb(data,k)
%%  
    [~,N] = size(data);
   % [IDX,~] = kmeans(data',k);     
    [IDX,~] = kmeans(data',k,'emptyaction','drop','start','uniform'); 
    R0 = zeros(N,k);
    for i = 1:k
        R0(:,i) = IDX == i;
    end
    model.R = R0;
    tmp = sum(R0);
    model.eU = repmat(tmp./sum(tmp),N,1);
    model.elogU = log(model.eU);

function L= vbound(X, model, prior)        
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
    % gamma
    Uv =model.Uv;
    a = model.a;  % N x k
    %
    R = model.R;
    logR = model.logR;
    eU = model.eU;
    elogU = model.elogU;
    [N,~] =size(R);
    [dim,k] = size(m);
    sumR = sum(R,1);
    ReU = model.R.*model.eU;
    avgN = sum(ReU);
    
    Elogpi = psi(0,alpha)-psi(0,sum(alpha));
    E_pz = dot(sumR,Elogpi);  %10.72 / 6
    E_qz = dot(R(:),logR(:)); %10.75 / 11  
    logCoefDir0 = gammaln(k*alpha0)-k*gammaln(alpha0); % the coefficient of Dirichlet Distribution
    E_ppi = logCoefDir0+(alpha0-1)*sum(Elogpi); %10.73 / 5
    logCoefDir = gammaln(sum(alpha))-sum(gammaln(alpha));
    E_qpi = dot(alpha-1,Elogpi)+logCoefDir; %10.76 / 10
    
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
    E_logpu = sum(N*(Uv2.*log(Uv2)-gammaln(Uv2))+(Uv2-1).*sum(elogU)-Uv2.*sum(eU),2); % / 4
    E_logqu = sum(sum(a.*(psi(a)-1)-gammaln(a)-elogU),2);  % / 9

    L = E_pX+E_pz+E_ppi+E_logpMu_Lambda+E_logpu-E_qz-E_qpi-E_logqMu_Lambda-E_logqu;
	

