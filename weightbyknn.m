function [ W ] = weightbyknn(x,p)
%WEIGHTBYKNN Summary of this function goes here
%   data N*dim
    [nSmp,~] = size(x);
    [IDX,~] = knnsearch(x,x,'K',p+1,'NSMethod','kdtree','distance','euclidean');
    % Weight functoin: euclidean distance
    D = ones(nSmp,p+1);
    a = reshape(repmat((1:1:nSmp),1,p),nSmp*p,1);
    b = double(reshape(IDX(:,2:end),nSmp*p,1));
    s = reshape(D(:,2:end),nSmp*p,1);
    W = sparse(a,b,s,nSmp,nSmp);
end

