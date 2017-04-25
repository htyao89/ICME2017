clc;clear all;close all;

addpath('./yael_v438/matlab/')	;

      feat = load('../featureExtraction/reid_train_feat.txt');
      feat = feat(:,1:1024); % global description
      feat = normr(feat);
      feat = feat';
      fprintf('Learning PCA-whitening\n');
      [~, eigvec, eigval, Xm] = yael_pca (single(feat));

      pca_dim=128;


    fprintf('load gallery feature......\n');
    [gallery_list,~] = textread('../person-520K/gallery.txt','%s %d');
    if exist('../featureExtraction/test_feat.mat')
    	tmp_feat=load('../featureExtraction/test_feat.mat');
	gallery_feat = tmp_feat.test_feat;   
	gallery_feat = gallery_feat(1:length(gallery_list),:);
    else
	gallery_feat = load('../featureExtraction/reid_gallery_feat.txt');
	gallery_feat = gallery_feat(1:length(gallery_list),:); 
    end
    gallery_fine_feat = normr(gallery_feat(:,1025:end)); gallery_fine_feat = gallery_fine_feat';
    gallery_coarse_feat = normr(gallery_feat(:,1:1024)); gallery_coarse_feat = gallery_coarse_feat';
    nTest = length(gallery_list);
 
    if 1
	gallery_coarse_feat = apply_whiten (gallery_coarse_feat, Xm, eigvec, eigval,pca_dim);	
	gallery_coarse_feat = gallery_coarse_feat';
	gallery_coarse_feat = normr(gallery_coarse_feat);
	gallery_coarse_feat = gallery_coarse_feat';
    end

    fprintf('load query feature......\n');
    [query_list,~] = textread('../person-520K/query.txt','%s %d');
    query_feat=importdata('../featureExtraction/reid_query_feat.txt');
    query_feat = query_feat(1:length(query_list),:);
    query_fine_feat = normr(query_feat(:,1025:end)); query_fine_feat = query_fine_feat';
    query_coarse_feat = normr(query_feat(:,1:1024)); query_coarse_feat = query_coarse_feat';
    nQuery = length(query_list);

    if 1
        query_coarse_feat = apply_whiten (query_coarse_feat, Xm, eigvec, eigval,pca_dim);
        query_coarse_feat = query_coarse_feat';
        query_coarse_feat = normr(query_coarse_feat);
        query_coarse_feat = query_coarse_feat';
    end


    testID = importdata('./testID.mat'); testID=testID(1:nTest); 
    testCam = importdata('./testCAM.mat'); testCam = testCam(1:nTest);

    queryID = importdata('./queryID.mat'); queryID = queryID(1:nQuery);
    queryCam = importdata('./queryCAM.mat'); queryCam = queryCam(1:nQuery);

    ap = zeros(nQuery, 1); % average precision
   CMC = zeros(nQuery, nTest);

%topKs=[100,500,1000,2500,5000,7500,10000]; % used for analysis the effect of K
topKs=[500]; % set topK=500 in our paper.
results=[];

for tk=1:length(topKs)
	topK = topKs(tk);
	for k = 1:nQuery
    		k
    		good_index = intersect(find(testID == queryID(k)), find(testCam ~= queryCam(k)))';% images with the same ID but different camera from the query
    		if isempty(good_index)
			continue;
   		end
    		junk_index1 = find(testID == -1);% images neither good nor bad in terms of bbox quality
    		junk_index2 = intersect(find(testID == queryID(k)), find(testCam == queryCam(k))); % images with the same ID and the same camera as the query
    		junk_index = [junk_index1; junk_index2]';

    		start_time = tic;
		% coarse retrieval
    		dist = distance(query_coarse_feat(:,k),gallery_coarse_feat);
    		dist = dist';
    		[~,sidx]=sort(dist);
    		sidx = sidx(1:topK);
 
    		fine_dist = distance(query_fine_feat(:,k),gallery_fine_feat(:,sidx));
    		dist(sidx)=fine_dist/100;    
    		[~, index] = sort(dist, 'ascend');
    		time_complexity(k) = toc(start_time);
    		[ap(k), CMC(k, :)] = compute_AP(good_index, junk_index, index);
	end
 
	CMC = mean(CMC);
	fprintf('single query:                                   mAP = %f, r1 precision = %f\r\n', mean(ap), CMC(1));
	results(tk,1)=mean(ap);
	results(tk,2)=CMC(1);
	results(tk,3)=mean(time_complexity);
end

results



