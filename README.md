# UMKD
Code for the paper "[Uncertainty-Aware Multi-Expert Knowledge Distillation for Imbalanced Disease Grading](https://arxiv.org/abs/2505.00592)", published in MICCAI-2025.
## Datasets

Dataset   |        URL       
:--------------:|:------------------:|
眼底数据集       |   [APTOS_2019](https://www.kaggle.com/datasets/mariaherrerot/aptos2019)              
眼底数据集       |  [Eyepacs](https://zhuanlan.zhihu.com/p/683930522)        
前列腺癌数据集    |   [SICAPv2](https://zhuanlan.zhihu.com/p/686314573) 

## Results

### Teacher Performance
Teacher Model   |        Code     |    num_classes   
:--------------:|:------------------:|:--------------------:
ResNet50        |   resnet_linear_dr.py        |     眼底5 / 前列腺4             
ResNet50        |   resnet_linear_dr.py        |     眼底5 / 前列腺4          

### Student Performance 
Target Model    |     Code       |      Methids 
:--------------:|:-----------:|:-------------------:
ResNet18        |   Resnet_trainer_student.py    |      common feature learning
ResNet18        |   Resnet_trainer_student_DKD.py    |      common feature learning + DKD 
ResNet18        |   Resnet_trainer_student_SDD.py    |      common feature learning + SDD  
ResNet18        |   Resnet_trainer_student_DKD_LP.py    |      common feature learning + DKD + LowPass  
ResNet18        |   Resnet_trainer_student_DKD_REDL.py    |      common feature learning + DKD + REDL  
ResNet18        |   Resnet_trainer_student_DKD_SA.py    |      common feature learning + DKD + ShallowAlign  
ResNet18        |   Resnet_trainer_student_DKD_SPP.py    |      common feature learning + DKD + SPP  

去掉了数据预处理中的以下操作:   
transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened], p=0.8),  
transforms.RandomGrayscale(p=0.2),  
