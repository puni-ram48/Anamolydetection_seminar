# SimpleNet for Chest X-Ray Anomaly Detection

## Overview
This project extends [SimpleNet (CVPR 2023)](https://github.com/DonaldRR/SimpleNet) from industrial defect detection to medical imaging, specifically detecting pneumonia in chest X-ray images.  

- Binary classification: **Normal vs Pneumonia**  
- Grayscale-to-RGB conversion for pretrained backbones  
- Demonstrates **generalization of industrial anomaly detection to medical data**  

**Dataset:** [Chest X-Ray Pneumonia (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- 5,863 images (pediatric, 1–5 yrs)  
- Train: 1,341 images | Test: 624 images  

**Key Result:** AUROC = **0.764** (ResNet18, noise=0.005)  

---

## Installation & Usage

```bash
git clone https://github.com/puni-ram48/Anamolydetection_seminar.git
cd Anamolydetection_seminar

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train
python main.py \
    --gpu 0 --seed 42 --results_path ./results --run_name chest_xray_experiment \
    net -b resnet18 -le layer3 --noise_std 0.005 --dsc_margin 0.2 \
    dataset --batch_size 16 --resize 256 --imagesize 256 \
    -d NORMAL chest_xray ./chest_xray
````

---

## Citation

**Original SimpleNet**

```bibtex
@inproceedings{liu2023simplenet,
  title={SimpleNet: A Simple Network for Image Anomaly Detection and Localization},
  author={Liu, Zhikang and Zhou, Yiming and Xu, Yuansheng and Wang, Zilei},
  booktitle={CVPR},
  pages={20402--20411},
  year={2023}
}
```

**Dataset**

```bibtex
@misc{kermany2018labeled,
  title={Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification},
  author={Kermany, Daniel and Zhang, Kang and Goldbaum, Michael},
  year={2018},
  publisher={Mendeley Data},
  doi={10.17632/rscbjbr9sj/2}
  kaggel = {https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
}
```
