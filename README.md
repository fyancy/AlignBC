# AlignBC
<p align="justify">
 Aligning Binary Centers for Single-stage Monocular 3D Object Detection. This repository include the core codes of our method and some demos of monocular object detection based on [KITTI dataset](https://www.cvlibs.net/datasets/kitti/index.php).
 </p>
 
 Here are two videos for object detection for moving objects (Cars). More can be found [here](https://github.com/fyancy/AlignBC/tree/main/abc_imgs/gif).
 - Detection for close Cars on narrow roads
![image](https://github.com/fyancy/AlignBC/blob/main/abc_imgs/gif/move_short.gif)
 - Detection for close Cars on wide roads
![image](https://github.com/fyancy/AlignBC/blob/main/abc_imgs/gif/move_long.gif)

## Abstract
<p align="justify">
Precisely perceiving the environment through a 3D perspective is essential and challenging for autonomous navigation and driving. Existing techniques rely on depth from LiDAR data or disparity in stereo vision to solve the poorly presented problem of detecting far-off and occluded objects. This increases structure complexity and computation burden, especially for single-stage systems. We argue that existing well-established detectors have the intrinsic potential to detect full-scene objects, but the extrinsic capabilities are limited by the structure form and optimization. Hence, we propose a double-branch single-stage monocular 3D object detection framework that aligns binary centers of object. Structurally, we construct two symmetrical and independent detectors, respectively using different prediction manners for 3D box parameters. Functionally, two detection heads have different sensitivities for the same object due to disentangling alignment. During the training, the detection heads were trained separately to obtain specific ability and aligned to promote the convergence. At inference, predictions of two branches are filtered via depth-aware non-maximal suppression (NMS) to acquire comprehensive detection results. Extensive experiments demonstrate that the proposed method achieves the state-of-the-art performance in monocular 3D detection on the KITTI-3D benchmark.
</p>

## Proposed method
### Structure
<p align="justify">
We provide a scheme to improve the full-scene detection capability of a single-stage monocular model. The method flow is shown in Figure. A single RGB image is input to the common backbone, then two detection branches independently regress the 3D box parameters from the 2D features, and finally the comprehensive detection results are output through post-processing. Almost all anchor-free methods take a well-designed detection head as the key to improve the detection ability. To improve the diversity of detection results, we adopt different prediction ways for key points, directions and locations in the two branches. Specifically, for heatmap prediction, 2D center and 3D projected center are respectively used for truncated objects at the image edge respectively; for depth estimation, we use geometric distance decomposition for opposite and adjacent sides of 3D box; for orientation regression, we use different numbers of discrete bins. Furthermore, to align the two branches, we design a hinged disentangling loss for training and extended 3D non-maximal suppression (NMS) at inference.
</p>

<div align=center>
<img src="abc_imgs/img/structure_v3.png" width="800">
</div>
<p align="center">
Fig. Training and testing pipeline of the proposed method. 
</p>

### Training loss
We consider the 3D loss, 2D loss and alignment loss of the proposed model to minimize, as follows
$$L_\text{ABC}=\sum_{i=1}^{2}\left(L_{i,\text{kpt}}\left ( I;\phi_f,\phi_i \right ) +L_{i,\text{3D}}\left ( I;\phi_f,\phi_i \right )+L_{i,\text{2D}}\left ( I;\phi_f,\phi_i \right )\right)+L_\text{dis}\left ( \phi_f(I);\phi_1, \phi_2 \right )\text{,}$$
where $L_{i,\Box}$ denotes the loss item for the $i^\text{th}$ branch, and $\phi_f$ the parameterized backbone. For both branches, the keypoint classification loss $L_\text{kpt}$ is the penalty-reduced focal loss \cite{law2018cornernet}, $L_\text{3D}$ is the sum loss for 3D box components including dimensions ($L_\text{dim}$), location ($L_\text{loc}$) and orientation ($L_\text{ori}$), where $L_\text{dim}$ is the L1 loss for dimension regression, the orientation estimation error $L_\text{ori}$ is the MultiBin loss \cite{mousavian20173d} combined bin classification with angle regression, $L_\text{loc}$ contains the L1 loss for center offsets and the uncertainty-aware regression loss for object depth. And the disentangled box-matching loss $L_\text{dis}$, in which only the deteciton branches are updated iteratively, is given as
$$\min_{\phi_1, \phi_2}\
	\left \|\max\left(0, \big(\hat{\mathcal{B}}-\varepsilon\big)-\mathcal{B}_1\right)\right\|_1+
	\left \|\max\left(0, \mathcal{B}_2-\big(\hat{\mathcal{B}}+\varepsilon\big)\right)\right\|_1
	+\frac{1}{n_\text{dis}}\sum_{j=1}^{n_\text{dis}}\left \| \mathcal{B}_1(\Omega_j,\hat{\Omega}-\hat{\Omega}_j)-
	\mathcal{B}_2 (\Omega_j,\hat{\Omega}-\hat{\Omega}_j)\right \|_1
	\text{,}$$




## Performance


## Citations
