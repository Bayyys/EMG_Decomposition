# EMG-Decomposition
Decomposing raw electromyography data into motor unit action potentials (MUAPs) can help with understanding the science of muscles and with clinical diagnoses.  

Based off project for Neuroengineerig class at UCLA

### EMG_Background.pdf
Gives background on EMG data and EMG decomposition

### EMG_Decomposition.m
MATLAB code starts with raw EMG data and outputs the shape of spikes from individual motor units through the process of filtering, detecting and aligning spikes, and k-means clustering. It also gives firing rates of each detected motor unit. 

Place data "EMG_example_2_fs_2k.csv" in the same directory to run. 
### EMG_Decomposition_Report.pdf
Analysis summary with figures and results

---

---

# 肌电图分解

将原始肌电图数据分解为运动单元动作电位（MUAPs），有助于理解肌肉科学和临床诊断。 

基于加州大学洛杉矶分校神经工程班的项目。

## EMG_Background.pdf

提供关于EMG数据和EMG分解的背景。

## EMG_Decomposition.m

MATLAB代码从原始EMG数据开始，通过过滤、检测和对齐尖峰以及K-means聚类的过程，输出单个运动单元的尖峰形状。它还给出每个检测到的运动单元的发射率。

将数据 "EMG_example_2_fs_2k.csv "放在同一目录中运行。

### EMG_Decomposition_Report.pdf

带有数字和结果的分析摘要
