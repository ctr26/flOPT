clf;
width = 512;
image = zeros(width);
image_fusion = zeros(width*2);
radius = 10;
[x y] = meshgrid(-width/2+1:width/2,-width/2+1:width/2);

bead_1 = [0 0];
bead_2 = [0 50];
bead_3 = [100 0];
bead_4 = [100 100];

r_1 = sqrt((x-bead_1(1)).^2 + (y+bead_1(2)).^2);
r_2 = sqrt((x-bead_2(1)).^2 + (y+bead_2(2)).^2);
r_3 = sqrt((x-bead_3(1)).^2 + (y+bead_3(2)).^2);
r_4 = sqrt((x-bead_4(1)).^2 + (y+bead_4(2)).^2);

bead_1_image = flipdim(r_1<radius,1);
bead_2_image = flipdim(r_2<radius,1);
bead_3_image = flipdim(r_3<radius,1);
bead_4_image = flipdim(r_4<radius,1);

bead_image = bead_1_image | bead_2_image | flipdim(bead_3_image,1) | bead_4_image;
% New
lena = rgb2gray(imread('lena.jpg'));
lena = rgb2gray(padarray(imread('lena.jpg'),[width/4 width/4]));
lena = imread('lena512.bmp');
lena = imread('zelda-256x256.tif');
% lena = imread('zelda.png');

% image =  mat2gray(double(bead_image))+ double(mat2gray(lena));

% %% Old
% image = imread('lena.jpg');
% lena = rgb2gray();
lena = padarray(lena,[width/4 width/4]);
%  
%  image = [];
image =  mat2gray(double(bead_image))+ double(mat2gray(lena));

   %v = VideoWriter('homography.avi');open(v);

sinugram=[];
reconstruction_back_projection =double(zeros(width*4));
%%
transformed_2dref = imref2d(size(image));
transformed_2dref.XWorldLimits = transformed_2dref.XWorldLimits - mean(transformed_2dref.XWorldLimits); %size(image);
transformed_2dref.YWorldLimits = transformed_2dref.XWorldLimits - mean(transformed_2dref.XWorldLimits); %size(image);

padded_2dref = imref2d(2*size(image));
padded_2dref.XWorldLimits = padded_2dref.XWorldLimits - mean(padded_2dref.XWorldLimits);
padded_2dref.YWorldLimits = padded_2dref.XWorldLimits - mean(padded_2dref.XWorldLimits);

more_padded_2dref = imref2d(4*size(image));
more_padded_2dref.XWorldLimits = more_padded_2dref.XWorldLimits - mean(more_padded_2dref.XWorldLimits);
more_padded_2dref.YWorldLimits = more_padded_2dref.XWorldLimits - mean(more_padded_2dref.XWorldLimits);

t_x = linspace(0,100,20);

%%
for i = 1:1:20
    
        bead_1_pos = [];
    bead_2_pos = [];
    bead_3_pos = [];
    bead_4_pos = [];
    sinugram = [];
    
    
    for theta = linspace(0,2*pi,128)

    %      theta =pi/6;
    current_slice = round(theta*127/(2*pi))+1;
    

        t_x = theta*5;
        t_y = 0;

        rotation_matrix = [cos(theta), sin(theta),t_x(i); ...
            -sin(theta), cos(theta),t_y; ...
            0, 0, 1 ; ...
            ];
        tform=affine2d(rotation_matrix');
        [transformed,cb_translated_ref] = imwarp(image,transformed_2dref,tform);
        transformed_padded = (imfuse(zeros(2*width),padded_2dref,transformed,cb_translated_ref,'Method','blend'));
    %     transformed_padded =double(ones(size(transformed_padded)));
    %     imshow(transformed_padded,[]);drawnow;


    %%
        bead_1_pos(:,current_slice) = rotation_matrix*([bead_1 1])';
        bead_2_pos(:,current_slice) = rotation_matrix*([bead_2 1])';
        bead_3_pos(:,current_slice) = rotation_matrix*([bead_3 1])';
        bead_4_pos(:,current_slice) = rotation_matrix*([bead_4 1])';


         projection = sum(transformed_padded);
         back_projection = repmat((projection),width*2,1);
         %back_projection = transformed_padded;
    %     

        %     back_projection_background = zeros(width*4);
        %
        %     transformed_back_projection= imwarp(back_projection,padded_2dref,tform);
        [transformed_back_projection,cb_translated_ref_back_projection]= imwarp(back_projection,padded_2dref,invert(tform));
        back_projection_padded = double(imfuse(zeros(4*width),more_padded_2dref,transformed_back_projection,cb_translated_ref_back_projection,'Method','blend'));
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
         reconstruction_back_projection = reconstruction_back_projection + back_projection_padded;
    %     
         sinugram = vertcat(sum(transformed_padded),sinugram);
    %     imshow(transformed_back_projection,[]); drawnow;
    %     imshow(back_projection,[])
        %%


        subplot(2,3,1);imshow(image,[]);title('Raw image');%;hold on; plot(bead_1(1)+width/2,bead_1(2)+width/2,'r+');hold off;
        subplot(2,3,2);imshow(transformed_padded,[]);title('Rotating sample')
        hold on;
         plot(bead_1_pos(1,:)+width,bead_1_pos(2,:)+width,'r');plot(bead_1_pos(1,end)+width,bead_1_pos(2,end)+width,'r+');
         plot(bead_2_pos(1,:)+width,bead_2_pos(2,:)+width,'b');plot(bead_2_pos(1,end)+width,bead_2_pos(2,end)+width,'b+');
         plot(bead_3_pos(1,:)+width,bead_3_pos(2,:)+width,'g');plot(bead_3_pos(1,end)+width,bead_3_pos(2,end)+width,'g+');
         plot(bead_4_pos(1,:)+width,bead_4_pos(2,:)+width,'y');plot(bead_4_pos(1,end)+width,bead_4_pos(2,end)+width,'y+');
        hold off;
        subplot(2,3,3);imshow(sinugram,[]);title('Sinugram')
        hold on;
         plot(bead_1_pos(1,:)+width,current_slice:-1:1,'r');plot(bead_1_pos(1,end)+width,1,'r+');
         plot(bead_2_pos(1,:)+width,current_slice:-1:1,'b');plot(bead_2_pos(1,end)+width,1,'b+');
         plot(bead_3_pos(1,:)+width,current_slice:-1:1,'g');plot(bead_3_pos(1,end)+width,1,'g+');
         plot(bead_4_pos(1,:)+width,current_slice:-1:1,'y');plot(bead_4_pos(1,end)+width,1,'y+');
        hold off;
        subplot(2,3,4);
        imshow(back_projection_padded,[]);title('Back projection')
        subplot(2,3,5);
        imshow(reconstruction_back_projection,[]);title('Reconstructing image'); 
                %writeVideo(v,getframe(gcf));
         drawnow;

    end




    %%


    cropped_reconstruction = imcrop((reconstruction_back_projection),[size(reconstruction_back_projection,1)/2-width/2 size(reconstruction_back_projection,2)/2-width/2 width width]);
    fft_reconstruction_back_projection =fftshift(fft2(cropped_reconstruction));
    size_fft_reconstruction_back_projection = size(fft_reconstruction_back_projection);
    [x y]=meshgrid(linspace(-1,1,size(fft_reconstruction_back_projection,1)));
    raml_lak = abs(sqrt(x.^2 + y.^2));

    irandon_sinugram = iradon(sinugram',linspace(0,360,128),'linear','none');
    cropped_irandom = imcrop((irandon_sinugram),[size(irandon_sinugram,1)/2-width/2 size(irandon_sinugram,2)/2-width/2 width width]);
    fft_irandom = fftshift(fft2(cropped_irandom));
    fft_filtered_irandon = fft_irandom.*raml_lak;
    filtered_irandon = ifft2(ifftshift(fft_filtered_irandon));


    fft_filtered_back_projection = fft_reconstruction_back_projection.*raml_lak;
    filtered_back_projection =ifft2(ifftshift(fft_filtered_back_projection));

    imshow((mat2gray(filtered_back_projection)),[]);title('Reconstructed image');

    subplot(2,3,6);
    imshow(filtered_irandon,[]);title('iradon');
    %%
    %Line profile

    line_flopt = (filtered_back_projection(round(size(filtered_back_projection,1)/2),:));
    line_iradon = filtered_irandon(round(size(filtered_irandon,1)/2),:);

    %Textwidth 437.46112

    %% Squared difference of pixels

    square_diff_sum_filtered(i) = sum(sum((filtered_back_projection(1:end-1,1:end-1) - image).^2));
    square_diff_sum_radon(i)    = sum(sum((filtered_irandon(1:end-1,1:end-1) - image).^2));

end


% %%
% f = figure;
% set(f, 'Position', [0 0 437*0.8 (437*0.8/(1.618))])
% plot(line_flopt(128:(256+128)));
% hold on;
% plot(line_iradon(128:(256+128)));
% legend('FLOPT','Radon Transform');
% xlabel('x /px');
% ylabel('Intensity /A.U.');
% xlim([0 256])
% ylim([-100 150])
% hold off;

%%
 %writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));
 %writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));%%
 %writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));writeVideo(v,getframe(gcf));  close(v)