function output = padtosize(input,padding)   
     array_size = size(input);     
     padded_once = padarray(input,floor((padding - array_size)/2),'both');
     array_size_once = size(padded_once);
     output = padarray(padded_once,ceil((padding - array_size_once)/2),'post');
end