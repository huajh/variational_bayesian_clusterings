%%
% which tissue(CSF,WM or GM) the cluster correspond to according to the means of clusters?
% the means of cluster 1 * 3
%
function idx = whichTissue(means)

    %background             0       black   (0,0,0)
    %(cerebrospinal)CSF     128     red     (255,0,0)
    %(gray matter)GM        254     green   (0,255,0)
    %(white matter)WM       192     blue    (0,0,255)
    
    [~,idx_csf] = min(means);
    [~,idx_gm] = max(means);
    idx_wm = 6 - idx_csf-idx_gm;
    idx = [idx_csf idx_gm idx_wm];