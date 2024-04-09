# Pose-Aware 3D Facial Animation Synthesis using Geometry-guided Audio-Vertices Attention

This repository contains the code for the "Pose-Aware 3D Facial Animation Synthesis using Geometry-guided Audio-Vertices Attention" paper.


## Test
The environment we used was python 3.8, a 3090 GPU, and Windows OS.

**Note** : Requirements

Before you start using our model, you need to install the requirements. To do so, we advise you to create a virtual environment. Then run `pip install -r requirements.txt`. It is worth noting that you will need to install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) manually.

Next, you will need to download some pre-trained models and pre-processing files at [here](). The file directories in the web drive and the local file directories are corresponding. Please download the files in **checkpoints**, **ds_graph** and **template** folders from the web drive and put them into the corresponding local folders respectively.

To try our model trained on VOCASET, you can run the following:
```shell
python test_voca.py
```

To try our model trained on MULTIFACE, you can run the following:
```shell
python test_multiface.py
```

If all goes well, you can see the video in the root directory.

If you need to change the audio or the 3D head model, you need to change the **wav_path** and **template_path** in test_voca.py or test_multiface.py (lines 16 and 17) and just run them again.


## Citation
If you used this code or the paper, please consider citing our work:
```
@article{li2024pose,
  title={Pose-Aware 3D Talking Face Synthesis using Geometry-guided Audio-Vertices Attention},
  author={Li, Bo and Wei, Xiaolin and Liu, Bin and He, Zhifen and Cao, Junjie and Lai, Yu-Kun},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2024},
  publisher={IEEE}
}
```
