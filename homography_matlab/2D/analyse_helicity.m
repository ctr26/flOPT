images = 20
width = 512;
image = imread('bead_lena.tiff');
radon_file = 'filtered_irandon.tiff'
flopt_file = 'filtered_back_projection.tiff'
%%
window = [width*(1/4) width*(1/4) width/2 width/2];
image = imcrop(image,window);
imshow(image)
x_range = linspace(0,50,20);
%%
for i = 1:1:images
i
    radon_image = imread(radon_file,i);
        radon_image = imcrop(radon_image,window);
    flopt_image = imread(flopt_file,i);
        flopt_image = imcrop(flopt_image,window);
    
    %%
    square_diff_abs_filtered(i) = sum(sum(abs(mat2gray(flopt_image) - mat2gray(image))));
    square_diff_abs_radon(i)    = sum(sum(abs(mat2gray(radon_image) - mat2gray(image))));

    %% image correlation
    
    correlation_filtered(i) =  corr2(flopt_image,image);
    correlation_radon(i)    =  corr2(radon_image,image);
    
    %%
    
    norm_correlation_filtered(i) = corr2(mat2gray(flopt_image),(image));
    norm_correlation_radon(i)  = corr2(mat2gray(radon_image),(image));

    %%
    
    ssimval_filtered(i) = ssim(mat2gray(flopt_image),mat2gray(image));
    ssimval_radon(i)  = ssim(mat2gray(radon_image),mat2gray(image));

    
    %%
    ssimval_filtered(i) = ssim(mat2gray(flopt_image),mat2gray(image));
    ssimval_radon(i)  = ssim(mat2gray(radon_image),mat2gray(image));

    
    
end
%%
%     figure; plot(square_diff_abs_filtered);hold on; plot(square_diff_abs_radon);hold off;
%     legend('flopt','radon')
%     
%     figure; plot(correlation_filtered);hold on; plot(correlation_radon);hold off;
%     legend('flopt','radon')
%     
%     figure; plot(norm_correlation_filtered);hold on; plot(norm_correlation_radon);hold off;
%     legend('flopt','radon')
%     
    f=figure;
    hold on;
    plot(x_range,norm_correlation_radon,'-+');
    plot(x_range,norm_correlation_filtered,'-+');
    hold off;
    xlabel('Helical shift /px');ylabel('Correlation / A.U.')
    legend('Radon transform','flopt')
    export_pretty_fig('correlation_helicity',f)
    %%
%     figure; plot(ssimval_filtered);hold on; plot(ssimval_radon);hold off;
%     legend('flopt','radon')
%     
    