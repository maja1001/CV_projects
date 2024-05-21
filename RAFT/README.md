# RAFT 
An overview of the [RAFT](https://arxiv.org/pdf/2003.12039.pdf) model.

Setup conda env
 
    conda create --name raft
    conda activate raft
 
    # original    
    # conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch -y
 
    # alternative
    # conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    # conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
 
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    conda install matplotlib tensorboard scipy opencv -c conda-forge -y
    conda install ipykernel --update-deps --force-reinstall -y
    conda install pandas -y