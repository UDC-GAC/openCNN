echo "in_n,in_c,in_h,filt_k,filt_w,openCNN_sec,openCNN_flops,cuDNN_sec,cuDNN_flops"

for n in 32 64 96 128;
    do
        ../wgrad $n 64 56 56 64 64 3 3
    done

for n in 32 64 96 128; 
    do
        ../wgrad $n 128 28 28 128 128 3 3
    done

for n in 32 64 96 128; 
    do
        ../wgrad $n 256 14 14 256 256 3 3
    done
    
for n in 32 64 96 128; 
    do
        ../wgrad $n 512 7 7 512 512 3 3
    done
        
