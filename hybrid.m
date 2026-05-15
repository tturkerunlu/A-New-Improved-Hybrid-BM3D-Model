function [PSNR, y_est] = hybrid(sigma, profile)

if nargin < 1 || isempty(sigma)
    sigma = 25;
end

if nargin < 2 || isempty(profile)
    profile = 'np';
end

[fname_sel, fdir_sel] = uigetfile( ...
    {'*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff', 'Görüntü Dosyaları (*.png,*.jpg,*.bmp,*.tif)'; ...
     '*.*', 'Tüm Dosyalar (*.*)'}, ...
    'Görüntü Seç');

if isequal(fname_sel, 0)
    error('Dosya seçilmedi. İşlem iptal edildi.');
end

image_path = fullfile(fdir_sel, fname_sel);
fprintf('Seçilen dosya: %s\n', image_path);

img = imread(image_path);

if isa(img, 'uint8')
    img_double = double(img) / 255;
elseif isa(img, 'uint16')
    img_double = double(img) / 65535;
else
    img_double = double(img);
    if max(img_double(:)) > 10
        img_double = img_double / 255;
    end
end

if ndims(img_double) == 2 || size(img_double, 3) == 1
    is_grayscale = true;
    fprintf('Görüntü tipi: Gri ton (tek kanal)\n');
else
    R = img_double(:,:,1);
    G = img_double(:,:,2);
    B = img_double(:,:,3);
    channel_diff = max(abs(R(:) - G(:))) + max(abs(G(:) - B(:)));
    is_grayscale = (channel_diff < 1e-6);
    if is_grayscale
        fprintf('Görüntü tipi: Gri ton (R=G=B)\n');
    else
        fprintf('Görüntü tipi: Renkli (R≠G≠B)\n');
    end
end

rng(0);

if is_grayscale
    if size(img_double, 3) == 3
        y_clean = img_double(:,:,1); 
    else
        y_clean = img_double;
    end
    z_noisy = y_clean + (sigma / 255) * randn(size(y_clean));
else
    y_clean = img_double;
    z_noisy = y_clean + (sigma / 255) * randn(size(y_clean));
end

if is_grayscale
    fprintf('BM3D uygulanıyor (sigma=%.1f, profil=%s)...\n', sigma, profile);
    [PSNR, y_est] = BM3D(y_clean, z_noisy, sigma, profile);
else
    fprintf('CBM3D uygulanıyor (sigma=%.1f, profil=%s)...\n', sigma, profile);
    [PSNR, y_est] = CBM3D(y_clean, z_noisy, sigma, profile);
end

fprintf('PSNR: %.2f dB\n', PSNR);

[fdir, fname, fext] = fileparts(image_path);
if isempty(fdir)
    fdir = '.';
end
out_path = fullfile(fdir, [fname '_denoised' fext]);

if is_grayscale
    out_img = uint8(min(max(y_est, 0), 1) * 255);
else
    out_img = uint8(min(max(y_est, 0), 1) * 255);
end

imwrite(out_img, out_path);
fprintf('Kaydedildi: %s\n', out_path);

end