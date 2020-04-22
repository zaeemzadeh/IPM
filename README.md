# Iterative Projection and Matching
implementation of the data selection algorithm proposed in: 

Alireza Zaeemzadeh, Mohsen Joneidi ( shared first authorship) , Nazanin Rahnavard, Mubarak Shah: Iterative Projection and Matching: Finding Structure-preserving Representatives and Its Application to Computer Vision. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.
[link](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zaeemzadeh_Iterative_Projection_and_Matching_Finding_Structure-Preserving_Representatives_and_Its_Application_CVPR_2019_paper.pdf)

- For a quick demo using MNIST, please run `python demo.py`.
- For active learning experiments on UCF101 video dataset see [here](https://github.com/zaeemzadeh/Active-Learning-UCF101-IPM).
## Requirements
irlb: Truncated SVD by implicitly restarted Lanczos bidiagonalization for Numpy! [code](https://github.com/bwlewis/irlbpy)


## Visualization
t-SNE visualization of two classes of UCF-101 dataset and their representatives selected by 
IPM. (left) Decision function learned by using all the
data. The goal of selection is to preserve the structure with only a
few representatives. (right) Decision function learned by using representatives
selected by IPM.


<img src="https://github.com/zaeemzadeh/Active-Learning-UCF101-IPM/blob/master/IPM_animated.gif" width="480">


## Citing IPM
If you use IPM in your research, please use the following BibTeX entry.
```
@inproceedings{zaeemzadeh2019ipm,
    title = {{Iterative Projection and Matching: Finding Structure-preserving Representatives and Its Application to Computer Vision}},
    year = {2019},
    booktitle = {Computer Vision and Pattern Recognition, 2019. CVPR 2019. IEEE Conference on},
    author = {Zaeemzadeh, Alireza and Joneidi, Mohsen and Rahnavard, Nazanin and Shah, Mubarak}
}
```

## Project Webpages
[Presentation ](https://youtu.be/OFe5z5fMUGc)

[UCF Center for Research in Computer Vision (CRCV)](https://www.crcv.ucf.edu/home/projects/iterative-projection-and-matching/)

[UCF Communications and Wireless Networks Lab (CWNlab)](http://cwnlab.eecs.ucf.edu/ipm/)

[Active Learning on UCF101 using IPM](https://github.com/zaeemzadeh/Active-Learning-UCF101-IPM)


