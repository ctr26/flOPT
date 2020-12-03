clf;
flag_record = 0;
width = 128;
image = zeros(width);
image_fusion = zeros(width*2);
radius = 5;
[x y z] = meshgrid(-width/2+1:width/2,-width/2+1:width/2,-width/2+1:width/2);

bead_1 = [0 0 0 ]; bead_1_pos = [];
bead_2 = [0 50 0 ]; bead_2_pos = [];
bead_3 = [50 0 0 ]; bead_3_pos = [];
bead_4 = [50 50 50]; bead_4_pos = [];

r_1 = sqrt((x-bead_1(1)).^2 + (y+bead_1(2)).^2 + (z-bead_1(3)).^2);
r_2 = sqrt((x-bead_2(1)).^2 + (y+bead_2(2)).^2 + (z-bead_2(3)).^2);
r_3 = sqrt((x-bead_3(1)).^2 + (y+bead_3(2)).^2 + (z-bead_3(3)).^2);
r_4 = sqrt((x-bead_4(1)).^2 + (y+bead_4(2)).^2 + (z-bead_4(3)).^2);

bead_image = flipdim(r_1<radius,1) | flipdim(r_2<radius,1) | flipdim(r_3<radius,1) | flipdim(r_4<radius,1);
%

lena = mat2gray(padarray(imread('lena-64x64.jpg'),[width/4 width/4]));
% lena = imread('lena512.bmp');

image =  mat2gray(double(bead_image));
image(:,:,width/2) = mat2gray(lena+image(:,:,width/2));
image(width/2,:,:) = squeeze(mat2gray(lena+squeeze(image(width/2,:,:))));
clear r_* bead_image x y z


 if(flag_record);v = VideoWriter('homography.avi');open(v);end
 
subplot(2,3,1);
for frame = 1:size(image)
    imshow(squeeze(mat2gray(image(:,:,frame))));title(['z: ' num2str(frame)]);
     if(flag_record);writeVideo(v,getframe(gcf));end
     drawnow;
end
sinugram=[];
 reconstruction_back_projection =double(zeros(width*4,width*4));
%%
transformed_3dref = imref3d(size(image));
transformed_3dref.XWorldLimits = transformed_3dref.XWorldLimits - mean(transformed_3dref.XWorldLimits); %size(image);
transformed_3dref.YWorldLimits = transformed_3dref.XWorldLimits - mean(transformed_3dref.XWorldLimits); %size(image);

padded_3dref = imref3d(2*size(image));
padded_3dref.XWorldLimits = padded_3dref.XWorldLimits - mean(padded_3dref.XWorldLimits);
padded_3dref.YWorldLimits = padded_3dref.XWorldLimits - mean(padded_3dref.XWorldLimits);

more_padded_3dref = imref3d(4*size(image));
more_padded_3dref.XWorldLimits = more_padded_3dref.XWorldLimits - mean(more_padded_3dref.XWorldLimits);
more_padded_3dref.YWorldLimits = more_padded_3dref.XWorldLimits - mean(more_padded_3dref.XWorldLimits);
%%
for theta = linspace(0,2*pi,128)
    
%     theta =0;
    current_slice = round(theta*127/(2*pi))+1;
    
    t_x = 0;
    t_y = 0;
    t_z = 0;
    rotation_matrix = ...
        [cos(theta) , sin(theta), 0 , t_x; ...
        -sin(theta) , cos(theta), 0 , t_y; ...
        0           , 0         , 1 , t_z ; ...
        0           , 0         , 0 , 1 ; ...
        ];
    tform=affine3d(rotation_matrix');
    
    %%
    [transformed,cb_translated_ref] = imwarp(image,transformed_3dref,tform);
    padded_transformed = padtosize(transformed,2*[width width width]);
    
    %%
%     transformed_padded = (imfuse(zeros(2*width),padded_3dref,transformed,cb_translated_ref,'Method','blend'));
%     transformed_padded =double(ones(size(transformed_padded)));
%     imshow(transformed_padded,[]);drawnow;


%%
    bead_1_pos(:,current_slice) = rotation_matrix*([bead_1 1])';
    bead_2_pos(:,current_slice) = rotation_matrix*([bead_2 1])';
    bead_3_pos(:,current_slice) = rotation_matrix*([bead_3 1])';
    bead_4_pos(:,current_slice) = rotation_matrix*([bead_4 1])';
    
    %% Find rotation matrix
    
    %    homography_matrix = []
    % RT = decompose(homography_matrix)
    %%
    

     projection = sum(transformed,2);
     padded_projection = padtosize(squeeze(projection),[width*2 width*2]);
     imshow(squeeze(padded_projection),[])
     %%
     back_projection = repmat(projection,[1 width*2 1]); %Wrong because of rotation
     padded_back_projection =  padtosize(back_projection,2*[width width width]);
     %%
     %back_projection = transformed_padded;
%     
 
    %     back_projection_background = zeros(width*4);
    %
    %     transformed_back_projection= imwarp(back_projection,padded_2dref,tform);
    [transformed_back_projection,cb_translated_ref_back_projection]= imwarp(padded_back_projection,padded_3dref,invert(tform));
    %[transformed_back_projection,cb_translated_ref_back_projection]= imwarp(back_projection,transformed_3dref,invert(tform));
    %%    
    padded_transformed_back_projection = padtosize(transformed_back_projection,[width*4 width*4 width*4]);
    %%
    %back_projection_padded = double(imfuse(zeros(4*width),more_padded_3dref,transformed_back_projection,cb_translated_ref_back_projection,'Method','blend'));
%     subplot(2,1,1); imshow(transformed_padded,[]);
%     subplot(2,1,2); imshow(back_projection_padded,[]); drawnow;
    
    %     [transformed_back_projection,~] = imwarp(image,transformed_2dref,tform);
    %
    %     size_transformed_back_projection = size(transformed_back_projection)
    %     size_back_projection_background=size(back_projection_background);
    %
    %     [r,c]=size(transformed_back_projection);
    %     xpos=2*width-floor(r/2);ypos=2*width-floor(c/2);
    %     back_projection_background(xpos:xpos+r-1,ypos:ypos+c-1)=transformed_back_projection;
    %
    
%     %%
     reconstruction_back_projection = reconstruction_back_projection + padded_transformed_back_projection;
%     
     %sinugram = vertcat(sum(transformed_padded),sinugram);
%     imshow(transformed_back_projection,[]); drawnow;
%     imshow(back_projection,[])
    %%
    
    
    subplot(2,3,1);imshow(image(:,:,width/2),[])%;hold on; plot(bead_1(1)+width/2,bead_1(2)+width/2,'r+');hold off;
    subplot(2,3,2);imshow(padded_transformed(:,:,width),[]);title('Top down view');
    hold on;
     plot(bead_1_pos(1,:)+width,bead_1_pos(2,:)+width,'r');plot(bead_1_pos(1,end)+width,bead_1_pos(2,end)+width,'r+');
     plot(bead_2_pos(1,:)+width,bead_2_pos(2,:)+width,'b');plot(bead_2_pos(1,end)+width,bead_2_pos(2,end)+width,'b+');
     plot(bead_3_pos(1,:)+width,bead_3_pos(2,:)+width,'g');plot(bead_3_pos(1,end)+width,bead_3_pos(2,end)+width,'g+');
     plot(bead_4_pos(1,:)+width,bead_4_pos(2,:)+width,'y');plot(bead_4_pos(1,end)+width,bead_4_pos(2,end)+width,'y+');
    hold off;
    subplot(2,3,3);imshow(squeeze(padded_projection),[]);title('Side view');
    hold on;
     plot(bead_1_pos(3,:)+width,bead_1_pos(2,:)+width,'r');plot(bead_1_pos(3,end)+width,bead_1_pos(2,end)+width,'r+');
     plot(bead_2_pos(3,:)+width,bead_2_pos(2,:)+width,'b');plot(bead_2_pos(3,end)+width,bead_2_pos(2,end)+width,'b+');
     plot(bead_3_pos(3,:)+width,bead_3_pos(2,:)+width,'g');plot(bead_3_pos(3,end)+width,bead_3_pos(2,end)+width,'g+');
     plot(bead_4_pos(3,:)+width,bead_4_pos(2,:)+width,'y');plot(bead_4_pos(3,end)+width,bead_4_pos(2,end)+width,'y+');
    hold off;
    subplot(2,3,4); imshow(padded_transformed_back_projection(:,:,width*2),[])
    subplot(2,3,5); imshow(reconstruction_back_projection(:,:,width*2),[])
          if(flag_record);writeVideo(v,getframe(gcf));end

     drawnow;
    
end
%%
subplot(2,3,6);
for frame = size(reconstruction_back_projection)*1/4:1:size(reconstruction_back_projection)*3/4
    imshow(squeeze(mat2gray(reconstruction_back_projection(:,:,frame))));title(['z: ' num2str(frame)]);
     if(flag_record);writeVideo(v,getframe(gcf));end
     drawnow;
end

%%



 %%cropped_reconstruction = imcrop(reconstruction_back_projection,[size(reconstruction_back_projection,1)/2-width/2 size(reconstruction_back_projection,2)/2-width/2 width width]);
%  fft_reconstruction_back_projection =fftshift(fftn(reconstruction_back_projection));
%  size_fft_reconstruction_back_projection = size(fft_reconstruction_back_projection);
%  [x y z]=meshgrid(linspace(-1,1,size(fft_reconstruction_back_projection,1)));
%  raml_lak = abs(sqrt(x.^2 + y.^2 + z.^2));
% 
%  fft_filtered_back_projection = fft_reconstruction_back_projection.*raml_lak;
%  filtered_back_projection =ifftn(ifftshift(fft_filtered_back_projection));
%% imshow(filtered_back_projection,[])
%
% %%
% % writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));
%%
if(flag_record);close(v);end