%
%	create Guassian Mixture Data Sample
%	initialize parameters randomly
%
%@input:
%
%		N 		the number of data
%		dim 	the Dimension of data
%		M		the number of Gaussian function
%
%@output:
%
%       data 	dim-by-N
%       mu 		dim-by-M
%       Sigma   dim x dim x M
%       Aplha 	1-by-M
%       n       1 x M
%   
%       fixed = 1 or 0

function [data,mu,n] = CreateGmmSample(M,N,option)    
    dim = 2;
    if strcmp(option.type,'cirlce')
        Alpha = ones(1,M);
        var = 0.5*ones(1,M);
        mu=4*[cos((1:M)*2*pi/M);sin((1:M)*2*pi/M)];
    elseif strcmp(option.type,'toy')   
        Alpha = ones(1,M);
        var = 0.25*ones(1,M);
        a = 3;
        b = 5;
        div_cut = M+2;
        mu1 = [a*cos((2:M/2+1)*2*pi/div_cut);b*sin((2:M/2+1)*2*pi/div_cut)]; 
        mu2 = [a*cos((0:M/2-1)*2*pi/div_cut)+a;-b*sin((0:M/2-1)*2*pi/div_cut)+sqrt(b^2-a^2)]; 
        mu = [mu1,mu2];
    end
	Alpha = Alpha / norm(Alpha,1);    
    
    n=0;
	for i = 1:M
		if i~= M
			n(i) = floor(N*Alpha(i));
		else
			n(i) = N-sum(n);
        end
    end
	%
	start = 0;
	for i=1:M
		x = randn(dim,n(i));
		x = x.*var(i) + repmat(mu(:,i),1,n(i));
		data(:,(start+1):start+n(i)) = x;        
		start = start + n(i);
    end  
    if strcmp(option.type,'cirlce')
        save('circlesmp.mat','data','mu','n')
    elseif strcmp(option.type,'toy')
        save('toysmp.mat','data','mu','n')
    end



