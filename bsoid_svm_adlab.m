function [labels,f_10fps_test] = bsoid_svm_adlab(data_test,fps,OF_mdl,smth_hstry,smth_futr)
%BSOID_SVM     Predicts mouse behavior based on the Support Vector Machine (SVM) classifier
%   
%   [LABELS,F_10FPS_TEST] = BSOID_SVM(DATA_TEST,FPS,MDL) outputs classified behaviors using the output from bsoid_mdl.
%
%   INPUTS:
%   DATA_TEST    Matrix of the positons of the 6-body parts outlining the rodent over time videotaped from the bottom looking up. 
%                Rows represents time.
%                Columns 1 & 2 tracks snout; columns 3 to 6 tracks the two front paws; Columns 7 to 10 tracks the two hind paws;
%                Columns 11 & 12 tracks the base of the tail. Tested on tracking data generated by DeepLabCut 2.0.
%   FPS    Rounded frame rate, can use VideoReader/ffmpeg(linux command) to automatically detect the input video fps. Try to match training dataset.
%   OF_MDL    Support Vector Machine Classifier Model. This is the output of bsoid_mdl.
%   SMTH_HSTRY    BOXCAR smoothing using number of frames from before. Default ~40ms before.
%   SMTH_FUTR    BOXCAR smoothing using number of frames from after. Default ~40ms after.
%
%   OUTPUTS:
%   LABELS    Predicted action based on the model, the group here matches the group number in the bsoid_gmm, 10 frame/second temporal resolution.
%   F_10FPS_TEST    The features collated for the test animal, 10 frame/second temporal resolution.
%
%   EXAMPLES:
%   clear data;
%   load test_mouse.mat OF_mdl.mat;
%   [labels,f_10fps_test] = bsoid_svm(data_test,60,OF_mdl);
%   
%   Created by Alexander Hsu, Date: 071919
%   Contact ahsu2@andrew.cmu.edu

    if nargin < 3
        error('Please input test dataset, frame rate, and the SVM model!')
    end
    if nargin < 4
        smth_hstry = round(0.05/(1/fps))-1;
        smth_futr = round(0.05/(1/fps))-1;
    end
    fprintf('Obtaining features from dataset... \n');
    for m = 1:length(data_test) % For each csv file you uploaded.
        %% Obtain features, 2 physical features and 5 time-varying signals
        clear rear_vis c2e_mp_norm mp_tp_norm c2e_tp_norm c2e_tp_ang c2e_disp tp_disp
 
        rear_vis=(data_test{m}(:,16)+data_test{m}(:,19)+data_test{m}(:,22))./3;%likehood of nose, front paws p
        c2e=(data_test{m}(:,2:3)+data_test{m}(:,5:6))/2;%center of two ears xy
        mp=data_test{m}(:,8:9);%body midpoint xy
        tp=data_test{m}(:,11:12);%tail point xy
        c2e_mp=[data_test{m}(:,2)-data_test{m}(:,8),data_test{m}(:,3)-data_test{m}(:,9)];
        mp_tp=[data_test{m}(:,8)-data_test{m}(:,11),data_test{m}(:,9)-data_test{m}(:,12)];
        c2e_tp=[data_test{m}(:,2)-data_test{m}(:,11),data_test{m}(:,3)-data_test{m}(:,12)];
 
         for i = 1:length(data_test{m})   
            c2e_mp_norm(i)=norm(c2e_mp(i,:));
            mp_tp_norm(i)=norm(mp_tp(i,:));
            c2e_tp_norm(i)=norm(c2e_tp(i,:));
         end
        rear_vis_smth{m}=transpose(movmean(rear_vis,[smth_hstry,smth_futr]));
        c2e_mp_norm_smth{m}=movmean(c2e_mp_norm,[smth_hstry,smth_futr]);
        mp_tp_norm_smth{m}=movmean(mp_tp_norm,[smth_hstry,smth_futr]);
        c2e_tp_norm_smth{m}=movmean(c2e_tp_norm,[smth_hstry,smth_futr]);
 
        for k = 1:length(data_test{m})-1    
            b_3d = [c2e_tp(k+1,:),0]; a_3d = [c2e_tp(k,:),0]; c = cross(b_3d,a_3d);
            c2e_tp_ang(k)=sign(c(3))*180/pi*atan2(norm(c),dot(c2e_tp(k,:),c2e_tp(k+1,:)));
            c2e_disp(k)=norm(data_test{m}(k+1,2:3)-data_test{m}(k,2:3));
            tp_disp(k)=norm(data_test{m}(k+1,11:12)-data_test{m}(k,11:12));
        end 
        c2e_tp_ang_smth{m}= movmean(c2e_tp_ang,[smth_hstry,smth_futr]);
        c2e_disp_smth{m} = movmean(c2e_disp,[smth_hstry,smth_futr]);
        tp_disp_smth{m} = movmean(tp_disp,[smth_hstry,smth_futr]);
        



        %% Collate 7 features. 
        feats_test{m}= [rear_vis_smth{m}(:,2:end); c2e_mp_norm_smth{m}(:,2:end); mp_tp_norm_smth{m}(:,2:end); c2e_tp_norm_smth{m}(:,2:end);...
           c2e_tp_ang_smth{m}(:,1:end);  c2e_disp_smth{m}(:,1:end);  tp_disp_smth{m}(:,1:end)];
    end
    %% For each test dataset, we will predict the behavior based on the model parameters in MDL at a temporal resolution of 10fps.
    for n = 1:length(feats_test)
        feats1 = [];
        for k = fps/10:fps/10:length(feats_test{n})
            feats1(:,end+1) = [mean(feats_test{n}(1:4,k-fps/10+1:k),2);sum(feats_test{n}(5:7,k-fps/10+1:k),2)];
        end
        f_10fps_test{n} = feats1; % Store individual animal features so you can see what the machine grouping is focusing on.
        labels{n} = predict(OF_mdl,f_10fps_test{n}'); % Predict the labels based on your model.
    end
    save('svdadlabresult.mat')
return

