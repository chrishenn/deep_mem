cuda 11.1
nvidia linux driver version >= 450.80.02*
gcc-10
cudnn 8.0.5
torch 1.8.2, torchvision 0.9.2

wget https://download.pytorch.org/libtorch/lts/1.8/cu111/libtorch-shared-with-deps-1.8.2%2Bcu111.zip
libtorch 1.8.2


conda env create -f env182.yaml

cleanup train.py
verify that env182.def builds a valid image
add dynamic number-of-blocks allocation to frnn_opt_brute (import from another version of frnn)
