%% 用于验证匹配算法得到的Homography的对应关系(渲染出来检查)
clear
%% Step1: Read Images
% Read the image westconcordorthophoto.png into the workspace.
% here to input satellite image
ortho = imread('res_show/demo/satellite.jpg');
% scale = 0.5;
% ortho = imresize(ortho,scale);
imshow(ortho)
% here to input drone image
unregistered = imread('res_show/demo/drone.jpg');
% unregistered = imresize(unregistered,scale);
imshow(unregistered)
%% Step2: Read Homography matrix
% Load transform Homography matrix t.
homo = readNPY('res_show/demo/73_Homo_292.npy');
%% Step3: Transform Unregistered Image
Rfixed = imref2d(size(ortho));
t = projective2d(homo'); % PIL和matlab之间存储图像存在转置关系
registered = imwarp(unregistered,t,'OutputView',Rfixed);
% See the result of the registration by overlaying the transformed image over the original orthophoto.
%% Step4: Show Results
imshowpair(ortho,registered,'blend');
% imshowpair(ortho,registered,'montage');
