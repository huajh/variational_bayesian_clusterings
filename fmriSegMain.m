%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  fMRI segmentation
%   dataset:IBSR
%   @author: Junhao HUA
%   @time: 1/15/2013
%   email:   huajh7@gmail.com

%%  slice_n = #

%clc;
clear;

addpath('.\spm');

brainpath = '.\20Normals_T1_brain\';
segpath = '.\20Normals_T1_seg\';

scan = '12_3';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% groundTruth
%
%   use SPM8 tools to preprocessing data
%   @function: spm_vol, spm_read_vols
%


imgSegStr = strcat(segpath,scan,'.img');
Vol = spm_vol(imgSegStr);
%   Y_4D cols * #slices * rows * 1
[Y_4D,~] = spm_read_vols(Vol);
[cols,slices,rows] = size(Y_4D);
grdth = zeros(cols,rows,slices);
for i = 1:rows
    for j = 1:slices
        grdth(:,i,j) = Y_4D(:,j,i);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% only slice # slice_n
%background             0       black   (0,0,0)
%(cerebrospinal)CSF     128     red     (255,0,0)
%(gray matter)GM        254     green   (0,255,0)
%(white matter)WM       192     blue    (0,0,255)
% grdth_t: groundturth of #slice n

slice_n = 38;
grdth_t = round(rot90(grdth(:,:,slice_n)));
grdth_rgb = zeros(256*256,3);
grdth_rgb(grdth_t == 128,1) = 255;
grdth_rgb(grdth_t == 254,2) = 255;
grdth_rgb(grdth_t == 192,3) = 255;

grdth_rgb = reshape(grdth_rgb,[256 256 3]);
figure(1);imshow(grdth_rgb);title('groundTruth');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% only brain
%
%   resolve .hdr and .buchar file
%   each slice is a 256x256 image
%
hdrStr = strcat(brainpath,scan,'.hdr');
bucharStr = strcat(brainpath,scan,'.buchar');
fhdr = fopen(hdrStr,'r');
[confhdr,~] = fscanf(fhdr,'%d',4); % [cols rows #slices byteswap=0]
fbuchar = fopen(bucharStr,'r');
volume = reshape(fread(fbuchar,'uint8=>double'),256,256,[]); % fixed
[cols,rows,slices_Num] = size(volume);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%   segment only one slice: #slice_n
%   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    slice = rot90(fliplr(volume(:,:,slice_n)));    
    figure(2);imshow(slice/255);   
    slice = reshape(slice(:),[],1);
    mask = find(slice ~=0);    
    interest = slice(mask);  % interest data
    
    %%%%%%%%%%%%%
    % optional        
   figure(3); hist(interest,unique(interest));   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %
    %   change your cluster method here
    %   the cluster number is always equal to 3
    K = 3;     
    tic;   
    segfunc_name = 'vbgmm';    
    %[ label, model, logLRangle ] = segfunc( segfunc_name, interest',K);    
    [label,model,logLRangle] = vbgmm(interest',K);
    toc;        
    clust_idx = zeros(cols*rows,1); % store cluster result here
    clust_idx(mask) = label;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % convert model means to 1,2,3 index      
    csf_gm_wm_idx = whichTissue(model.M);   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % show segment result
    seg_result = zeros(cols*rows,3);    
    seg_result(clust_idx == csf_gm_wm_idx(1),1) = 255;  % red  (255,0,0)
    seg_result(clust_idx == csf_gm_wm_idx(2),2) = 255;  % green(0,255,0)
    seg_result(clust_idx == csf_gm_wm_idx(3),3) = 255;  % blue (0,0,255)    
    final_seg = reshape(seg_result, [cols rows 3]);    
    figure(4);imshow(final_seg); 
    %title(segfunc_name);
    title(segfunc_name);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Jaccard similarity coefficient (JSC) or Tanimoto Similarity 
  
    [ JSC(i,:) ] = JSCBrain( reshape(grdth_t,[],1),clust_idx,csf_gm_wm_idx);    
    disp(JSC(i,:));    

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % optional
    % plot the log-likelihood for debug
    %    
    figure(5);plot(1:size(logLRangle,2),logLRangle');title('logL');
    



