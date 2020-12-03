load('homography.mat')
%% Squared difference of pixels
    diff_sum_filtered = (filtered_back_projection(1:end-1,1:end-1) - image);
    square_diff_sum_filtered = diff_sum_filtered.^2;
    sum_square_diff_sum_filtered = sum(sum(square_diff_sum_filtered));
    
    diff_sum_radon = (filtered_irandon(1:end-1,1:end-1) - image);
    square_diff_sum_radon = diff_sum_radon.^2;
    sum_square_diff_sum_radon = sum(sum(square_diff_sum_radon));
    
    %% Plotting Histogram
    f=figure;

    [radon_hist,edges,bin] = histcounts(abs(diff_sum_radon),200);
    histogram(abs(diff_sum_radon),'facealpha',.5,'edgecolor','none')
        hold on    
    histogram(abs(diff_sum_filtered),'facealpha',.5,'edgecolor','none')
    [flopt_hist,edges,bin] = histcounts(abs(diff_sum_filtered),200);
    
    box off
    axis tight
    legend('radon','flOPT','location','northwest')
    legend boxoff
     xlabel('Absolute Difference from Source Image /A.U.')
     ylabel('Frequency /A.U.')
     %%
     export_pretty_fig('flopt_histogram',f)
    %% Plotting Divided Histogram
    figure
    hold on
    bar(radon_hist)
    bar(-flopt_hist)
    hold off
    figure
    bar(flopt_hist./radon_hist)
    hold on
    line(1:200,ones(200))
    hold off
    
    %% Plotting Difference over time
    
    load('homography_helical.mat')

    f = figure;
    %set(f, 'Position', [0 0 437*0.8 (437*0.8/(1.618))])
    
    semilogy((square_diff_sum_filtered)); hold on
    semilogy((square_diff_sum_radon)); hold off
    legend('Radon transform','flOPT');
    xlabel('Helicity as a pixel shift in x after rotation');
    ylabel('Summed absolute pixel difference from source image');
    export_pretty_fig('flopt_histogram',f)
    
    %figure;
    %plot(square_diff_sum_radon)
    


%     %%
%     figure
%     rose(log(square_diff_sum_radon(:)));hold on
%     rose(log(square_diff_sum_filtered(:))); hold off
%     
%         legend('Radon','flOPT','location','northwest')
%        
% 
%     %%
%     figure
%     hold on
%         bar(abs((diff_sum_radon(:))),20)
%         bar(abs((diff_sum_filtered(:))),20)
%     hold off
%     
%     %%
%     craigbynot=(abs(square_diff_sum_radon)./abs(square_diff_sum_filtered));
%     figure
%     histogram(craigbynot)
%     %bar(craigbynot(:))
    