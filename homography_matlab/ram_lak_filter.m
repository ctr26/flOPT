phantomData=phantom();
N=size(phantomData,1);

theta = 0:179;
N_theta = length(theta);
[R,xp] = radon(phantomData,theta);

% make a Ram-Lak filter. it's just abs(f).
N1 = length(xp);
freqs=linspace(-1, 1, N1).';
myFilter = abs( freqs );
myFilter = repmat(myFilter, [1 N_theta]);

% do my own FT domain filtering
ft_R = fftshift(fft(R,[],1),1);
filteredProj = ft_R .* myFilter;
filteredProj = ifftshift(filteredProj,1);
ift_R = real(ifft(filteredProj,[],1));

% tell matlab to do inverse FBP without a filter
I1 = iradon(ift_R, theta, 'linear', 'none', 1.0, N);

subplot(1,3,1);imagesc( real(I1) ); title('Manual filtering')
colormap(gray(256)); axis image; axis off

% for comparison, ask matlab to use their Ram-Lak filter implementation
I2 = iradon(R, theta, 'linear', 'Ram-Lak', 1.0, N);

subplot(1,3,2);imagesc( real(I2) ); title('Matlab filtering')
colormap(gray(256)); axis image; axis off

% for fun, redo the filtering wrong on purpose
% exclude high frequencies to create a low-resolution reconstruction
myFilter( myFilter > 0.1 ) = 0;
ift_R = real(ifft(ifftshift(ft_R .* myFilter,1),[],1));
I3 = iradon(ift_R, theta, 'linear', 'none', 1.0, N);

subplot(1,3,3);imagesc( real(I3) ); title('Low resolution filtering')
colormap(gray(256)); axis image; axis off