function [ JSC ] = JSCBrain( GroundTruth,clust_idx,csf_gm_wm_idx)
%JACCARDSIMILARITYCOEF Summary of this function goes here
%   Detailed explanation goes here
    
    M = size(clust_idx,1);
    jsc_csf = zeros(2,M);
    jsc_gm  = zeros(2,M);
    jsc_wm  = zeros(2,M);
      
    jsc_csf(1,:) = GroundTruth == 128; % red
    jsc_csf(2,:) = clust_idx == csf_gm_wm_idx(1);
    
    jsc_gm(1,:) = 2*(GroundTruth == 254); % green 
    jsc_gm(2,:) = 2*(clust_idx == csf_gm_wm_idx(2));
    
    jsc_wm(1,:) = 3*(GroundTruth == 192);% blue
    jsc_wm(2,:) = 3*(clust_idx == csf_gm_wm_idx(3));
    
    jsc_X = jsc_csf +jsc_gm +jsc_wm;    
     
    if isempty(find(jsc_csf~=0))
        csf = 0;
    else
        csf = 1 - pdist(jsc_csf,'jaccard');
    end
    if isempty(find(jsc_gm~=0))
        gm = 0;
    else
        gm = 1 - pdist(jsc_gm,'jaccard');
    end
    if isempty(find(jsc_wm~=0))
        wm = 0;
    else
        wm = 1 - pdist(jsc_wm,'jaccard');
    end
          
    if isempty(find(jsc_X~=0))
        total = 0;
    else
        total = 1 - pdist(jsc_X,'jaccard');
    end  
    JSC = [csf gm wm total];

end

