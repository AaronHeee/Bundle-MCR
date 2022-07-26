# Bundle MCR Framework

![Introduction](./image/intro.jpg)

This is the pytorch implementation of this paper [Bundle MCR: Towards Conversational Bundle Recommendation.]() Zhankui He, Handong Zhao, Tong Yu, Sungchul Kim, Fan Du, Julian McAuley. 16th ACM Conference on Recommender Systems (RecSys '22). Oral.

**Arxiv version is on the way!**

## Bunt Implementation

**NOTE: The details of all dataset processing (and more information) are in `Appendix.pdf`.**

This is the example code for RecSys 2022 paper submission "Bundle MCR: Towards Conversational Bundle Recommendation" on dataset Steam.

We use python 3.6 and other python dependencies are listed in `requirements.txt`, you can install them with `pip install -r requirements.txt`.

## Quick Start

1. **Offline Pre-Training:** Use `bash scripts/train_offline.sh ${device_id} ${seed}`, where `${device_id}` is used to specify your GPU id, and `${seed}` is the random seed you assign. For example: 

    ```
    bash scripts/train_offline.sh 0 0
    ```
2. **Online Fine-Tuning:** Use `bash scripts/train_online.sh ${device_id} ${seed} ${pre_trained_model_path}`. The explanation of arguments are as the same as step 1, except for `${pre_trained_model_path}`, which is the `*.pt` model path to load as pre-trained Bunt for online fine-tuning, which can be found in `checkpoints` folder by default. For example:

    ```
    bash scripts/train_online.sh 0 0 checkpoints/steam/model_1.pt
    ```
3. **Collect Results:** You are free to print out your results using `python tools/results.py ${ckpt_path}`, where `${ckpt_path}` is the path of your experiment folder, such as `checkpoints/steam`.

## Data Processing (Now steam has been processed)

1. Go to `steam` folder, `cd raw/`;
2. For data interaction processing and interactions splitting, use `python 0_data_splitting.py`
3. To process attributes for Bundle MCR, use `python 1_item_attr.py`;
4. To precess categories for Bundle MCR, use `python 2_item_cate.py`.

## Bibtex

Please cite our paper if using this code, and feel free to contact [zhh004@eng.ucsd.edu](zhh004@eng.ucsd.edu) if any questions.

```text
@inproceedings{he22bundle,
  title = "Bundle MCR: Towards conversational bundle recommendation",
  author = "Zhankui He and Handong Zhao and Tong Yu and Sungchul Kim and Fan Du and Julian McAuley",
  year = "2022",
  booktitle = "RecSys"
}
```