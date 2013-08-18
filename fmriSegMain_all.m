
%%  fMRI segmentation
%   dataset:IBSR
%   @author: Junhao HUA
%   @time: 1/15/2013
%%  all slice

%clc;
clear;

brainpath = '.\20Normals_T1_brain\';
segpath = '.\20Normals_T1_seg\';
scan = '12_3';

%% groundTruth
imgSegStr = strcat(segpath,scan,'.img');
Vol = spm_vol(imgSegStr);
%   Y_4D cols * #slices * rows * 1
[Y_4D,~] = spm_read_vols(Vol);
[rows,slices,cols] = size(Y_4D);
grdth = zeros(rows,cols,slices);
for i = 1:rows
    for j = 1:slices
        grdth(:,i,j) = Y_4D(:,j,i);
    end
end

%% only brain
hdrStr = strcat(brainpath,scan,'.hdr');
bucharStr = strcat(brainpath,scan,'.buchar');
fhdr = fopen(hdrStr,'r');
[confhdr,~] = fscanf(fhdr,'%d',4); % [cols rows #slices byteswap=0]
fbuchar = fopen(bucharStr,'r');
volume = reshape(fread(fbuchar,'uint8=>double'),256,256,[]); % fixed
[rows,cols,slices_Num] = size(volume);
JSC = zeros(slices_Num,4);

t = slices_Num;

logLRange = zeros(t,1000);


%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%   segment every #slice
%
% iterator time each slice

for i=1:t
    
    grdth_t = round(rot90(grdth(:,:,i)));
       
    slice = rot90(fliplr(volume(:,:,i)));    
    slice = reshape(slice(:),[],1);
    mask = find(slice ~=0);
    
    % no feature
    if isempty(mask)
        i = i-1;
        break;
    end
    interest = slice(mask);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %   change your cluster method here
    K = 3; % the cluster number is always equal to 3
    segfunc_name = 'vblapsmm';
    tic;
    [ label, model, logL ] = segfunc( segfunc_name, interest',K);
    iter(i) = size(logL,2);
    Run_Times(i) = toc;
    clust_idx = zeros(cols*rows,1);
    clust_idx(mask) = label;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    csf_gm_wm_idx = whichTissue(model.M); % convert model means to 1,2,3 index
     
    seg_result = zeros(cols*rows,3);
    seg_result(clust_idx == csf_gm_wm_idx(1),1) = 255;
    seg_result(clust_idx == csf_gm_wm_idx(2),2) = 255;
    seg_result(clust_idx == csf_gm_wm_idx(3),3) = 255;
    final_seg = reshape(seg_result, [rows cols 3]);    
    
    figure(1);imshow(final_seg); title(segfunc_name);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % compare to groundTruth
    % Jaccard similarity coefficient (JSC) or Tanimoto Similarity
    csf_gm_wm_idx = whichTissue(model.M);
    
    [ JSC(i,:) ] = JSCBrain( reshape(grdth_t,[],1),clust_idx,csf_gm_wm_idx);
          
    logLRange(i,1:iter(i)) = logL';   
    disp([i JSC(i,:)]);
    clear mask slice grdth_t interest;
end
    logLRange = logLRange(1:i,:);
    JSC = JSC(1:i,:);
    figure(2);plot(1:size(logLRange,2),logLRange);title('logLRange');
    figure(3);plot(1:size(JSC,1),JSC');title('csf,gm,wm,overall');
    figure(4);plot(iter);
    figure(5);plot(Run_Times);
    Iter_vbgmm = iter;
    RunTimes_vbgmm = Run_Times;
    avg_iter = RunTimes_vbgmm./Iter_vbgmm;
    figure(6);plot(avg_iter);
    save Iter_emgmm;
    save RunTimes_emgmm;
    JSC_vblapsmm_12_3_38_total = JSC;
    save JSC_vblapismm_12_3_38_total;

