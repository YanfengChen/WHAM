1.check the versions of GPU driver and CUDA platform, 
2.check the version of Ubuntu
3.follow the installation guide in INSTALL.md
4.modify the installation steps to be compatible to the versions of GPU/CUDA platform and Ubuntu os you get from checking
5.recommended compatible base packages are 
  -python 3.10, 
  -pytorch build(stable(2.9.1)+cu130) from https://download.pytorch.org/whl/cu130
  -cuda toolkit/nvcc 13.0
6.resolve the errors in installing and install WHAM successfully
7.for DPVO installation, refer to the file "DPVO_compatible_installation.md"