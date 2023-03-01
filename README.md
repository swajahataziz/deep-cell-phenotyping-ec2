# Image based Cell Phenotyping on AWS EC2 Instances

<p align="center">
  <img src="./images/PyTorch DDP.png" alt="EC2 + PyTorch" height="200"/>
</p>

## Background

The ability to phenotype cells is important for biological research and drug development. Traditional phenotyping methods rely on fluorescence labeling of specific markers. However, reliance on traditional phenotyping methods may be unviable or undesirable in certain contexts. This solution builds on a deep learning approach for phenotyping disaggregated single cells with a high degree of accuracy using low-resolution bright-field and non-specific fluorescence images of the nucleus, cytoplasm, and cytoskeleton. The model trains a CNN using cell images from eight standard cancer cell-lines. The solution is based on the article published in [Nature](https://www.nature.com/articles/s42003-020-01399-x) 


## Concepts Covered
After completing this repository, you will be able to understand the following concepts: 
- Provision an [AWS EC2](https://aws.amazon.com/ec2/) and running the training of a CNN model  with [Pytorch Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- Profile model training performance using PyTorch Profilder. 
- Adapt the training script to run model training on AWS Tranium (Trn1) and DL1 instances. 


# Getting started  
### 1. Prerequisites
#### Platforms 
##### AWS EC2 instance properties   
It is recommended that the training script is executed in a multi-GPU instance such as p3.16xlarge or p3dn.24xlarge. 
##### Amazon S3   
For ease of use, download the training data from [UBC Research Data Collection](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/TDULMF) into an S3 bucket

#### Other ressources
- Python 3.11 
- PyTorch  
- Torchvision
- You can find all the additional information in the `requirements.txt` file

