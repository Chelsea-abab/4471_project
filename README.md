# **4471_project**
the codes are included in DR_DME_joint folder
the dataset of Messidor can be downloaded here:
# **DR datasets:**
Messidor-2: https://www.adcis.net/en/third-party/messidor2/

Messidor-2 labels: https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades



# referenced paper:
We follow and reproduce the CANet model in the paper 'CANet: Cross-Disease Attention Network for Joint Diabetic Retinopathy and Diabetic Macular Edema Grading' from https://ieeexplore.ieee.org/abstract/document/8892667

We also used an attention module from CBAM in the paper 'CBAM: Convolutional Block Attention Module' from 
https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html



# about the files in the folder
baseline.py is the main worker. it includes the training and validation process

baseline_individual.py is original baseline to test the individually grading of DR and DME

command.sh includes some commands to run the baseline.py

utils.py includes some fuctions to measure the scores

dataLoader.py loads the Messidor-2 dataset and do basic augmentations

cbam.py and resnet50.py includes the models

the log folder includes all the tensorboard logs when we test our models