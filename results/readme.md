We encourage researchers to use this results as benchmark for your their _3D Human Pose and Shape Estimation_ models. The researchers shall denote their thresholding as follows: ```<occlusion_type><threshold>```. For instance, ```SOI0.75``` means the subset formed by the samples whose _Silhouette Occlusion Index_ score is above _0.75_. 

Each file contains the the list of frames ordered by the occlusion amount. Note that person id is the order of that person in the annotation list.

```
<image_filename> <person_id> <occlusion_amount>
```

For per body segment occlusions, check the _.zip_ files containing a _.json_ file for each body segment.

---
Included sequences for 3DPW are listed below. The other sequences are excluded due to not person-person occlusion or too many people at the background preventing to calculate a meaningful error.

 * courtyard_basketball_00
 * courtyard_captureSelfies_00
 * courtyard_dancing_00
 * courtyard_dancing_01
 * courtyard_giveDirections_00
 * courtyard_goodNews_00
 * courtyard_hug_00
 * courtyard_shakeHands_00
 * courtyard_warmWelcome_00
 * downtown_bar_00

---
Included sequnces for AGORA are listed below. Only the validation set is used.
 * archviz
 * hdri_50mm

--- 
All the images of the OCHuman dataset whose SMPL parameters are provided by the [EFT](https://github.com/facebookresearch/eft) are used.
