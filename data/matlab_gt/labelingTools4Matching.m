%% 用于真值标注
clear
% input(u,v), the central coordinates of current selected matching images
u = 1086;  # x_central coordinates
v = 724;   # y_central coordinates

imgDataPath = './bestMatches4eval/'; # root_dir
imgDataDir  = dir(imgDataPath);
for i = 1:length(imgDataDir)
    imgPath = dir([imgDataPath imgDataDir(i).name]);
    if(isequal(imgDataDir(i).name,'.')||... % 去除系统自带的两个隐文件夹
       isequal(imgDataDir(i).name,'..')||...
       ~imgDataDir(i).isdir)                % 去除遍历中不是文件夹的
           continue;
    end
    imgDir  = dir([imgDataPath imgDataDir(i).name '/*.jpg']);% find all jpg format file
    %% Step 1: Read Images
    % Read the image westconcordorthophoto.png into the workspace. This
    % image is an orthophoto that has already been registered to the ground.
    % here to input satellite image
    sate_filename = imgDir(2).name;
    ortho = imread([imgDataPath imgDataDir(i).name '/' imgDir(2).name]);
    imshow(ortho)
    text(size(ortho,2),size(ortho,1)+15, ...
        'Image courtesy of Massachusetts Executive Office of Environmental Affairs', ...
        'FontSize',7,'HorizontalAlignment','right');
    % read x0 and y0 from retrival cropped satellite
    j = find('.'==sate_filename);
    imname = sate_filename(1:j-1);
    str_list = split(imname,'_');
    y0 = str2double(str_list(2,1));
    x0 = str2double(str_list(3,1));

    % Read the image westconcordaerial.png into the workspace. This image was taken from an airplane and is distorted relative to the orthophoto. 
    % Because the unregistered image was taken from a distance and the topography is relatively flat, it is likely that most of the distortion is projective.
    % here to input drone image
    unregistered = imread([imgDataPath imgDataDir(i).name '/' imgDir(1).name]);
    imshow(unregistered)
    text(size(unregistered,2),size(unregistered,1)+15, ...
        'Image courtesy of mPower3/Emerge', ...
        'FontSize',7,'HorizontalAlignment','right');
    %% Step2: Select Control Point Pairs
    [mp,fp] = cpselect(unregistered,ortho,'Wait',true);

    % Since points on cropped small satellite image only have traslated in x and y axis, their corresponding position on large satellite images can be found out
    % before compute the H, the position of points should be projected
    % fp is a 4x2 matrix, each row records labeled (x,y).
    temp = fp; % save its original pos
    fp(:,1) = fp(:,1) + x0;
    fp(:,2) = fp(:,2) + y0;

    %% Step3: Infer Geometric Transformation
    % Find the parameters of the projective transformation that best aligns the moving and fixed points by using the fitgeotrans function.
    t = fitgeotrans(mp,fp,'projective');
    t_temp = fitgeotrans(mp,temp,'projective');
    %% Step4: Transform Unregistered Image
    Rfixed = imref2d(size(ortho));
    registered = imwarp(unregistered,t,'OutputView',Rfixed);
    registered_temp = imwarp(unregistered,t_temp,'OutputView',Rfixed);
    % See the result of the registration by overlaying the transformed image over the original orthophoto.
    imshowpair(ortho,registered_temp,'blend');
    %% Step5: Transform Unregistered Image
    [x,y] = transformPointsForward(t,u,v);
    fid=fopen(['./','GT.txt'],'a');
    fprintf(fid,'%s ',imgDataDir(i).name); 
    fprintf(fid,'%.8f ',x); 
    fprintf(fid,'%.8f\r\n',y);
    fclose(fid);
end
