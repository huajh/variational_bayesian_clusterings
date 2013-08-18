function[Y] = tanimoto(X)
%TANIMOTO  Calculate Tanimoto distance%
%   Calculates Tanimoto Distance for a logical matrix X
%   Emulates a pdist function format
%   B. Andre' Weinstock, Transform Pharmaceuticals Inc., $Date: 2006/12/06$
%   X is a logical matrix (sample x variable)
%   Y is an upper triangular square vector

Tm =zeros(size(X,1));
for a =1:size(X,1);
    Na =sum(X(a,:));
    for b =1:size(X,1);
        if a ~=b;
            Nb =sum(X(b,:));
            Nab =sum(X(a,:).*(X(b,:)));
            Tm(a,b) =1-(Nab/((Na+Nb-Nab)+eps));
        end
    end
end
Y = squareform(Tm);