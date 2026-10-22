# XAI-ChestDiag

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23570883.svg)](https://doi.org/10.5281/zenodo.23570883)

**Explainable deep learning for emergency triaging of pulmonary diseases from chest X-rays using DenseNet121 and Grad-CAM.**

This repository contains the complete, reproducible experimental pipeline for the paper
*"Explainable Deep Learning Framework for Emergency Triaging of Pulmonary Diseases from Chest X-Rays Using DenseNet121 and Grad-CAM"*.

| Component | Implementation |
|---|---|
| Dataset | NIH ChestX-ray14 (112,120 frontal CXRs, 30,805 patients) |
| Task | Binary triage: *Abnormal* (any of the 14 findings) vs. *Normal* ("No Finding") |
| Split | Strict **patient-level** 70 / 10 / 20 (train / val / test), no patient overlap |
| Proposed model | DenseNet121 (ImageNet-pretrained) + GAP + Dropout(0.3) + sigmoid output |
| Baselines | VGG16, InceptionV3, EfficientNet-B4, **identical head, data, and training protocol** |
| Statistics | 95% CIs (DeLong for AUC, bootstrap for all metrics), pairwise **DeLong test** |
| Explainability | Grad-CAM + **Pointing Game** against NIH radiologist bounding boxes (test patients only) |
| Efficiency | Parameter count and inference throughput (images/s, batch 1 and 64) |

---

## 1. Installation

```bash
git clone https://github.com/<user>/XAI-ChestDiag.git
cd XAI-ChestDiag
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
```

## 2. Data

Download NIH ChestX-ray14 (≈45 GB) from either
* NIH: <https://nihcc.app.box.com/v/ChestXray-NIHCC>, or
* Kaggle: <https://www.kaggle.com/datasets/nih-chest-xrays/data>

and arrange it as:

```
data/nih/
├── Data_Entry_2017.csv
├── BBox_List_2017.csv
└── images/            # any sub-folder layout (images_001/images/*.png ...) is fine
```

Optional but strongly recommended (5-10× faster training):

```bash
python -m src.resize_cache --src data/nih/images --dst data/nih/images_256 --size 256
# then set  data.image_root: data/nih/images_256  in configs/default.yaml
```

## 3. Run the full experiment

**No local GPU?** Use the ready-made Kaggle notebook [`notebooks/kaggle_xai_chestdiag.ipynb`](notebooks/kaggle_xai_chestdiag.ipynb): the NIH dataset is already hosted on Kaggle and a free T4/P100 GPU is provided. Instructions are in the first cell.

**Local GPU:**

```bash
python scripts/run_all.py --seeds 42 43 44
```

or step by step:

```bash
python -m src.prepare_splits                                # patient-level split + leakage check
python -m src.train         --model densenet121 --seed 42   # repeat for each model
python -m src.evaluate      --model densenet121 --seed 42
python -m src.throughput    --batch-sizes 1 64
python -m src.compare       --seed 42                       # comparison table + DeLong tests
python -m src.pointing_game --model densenet121 --seed 42
python -m src.plots         --seed 42                       # confusion matrix, ROC, training curves
python -m src.gradcam_figure --model densenet121 --seed 42 --findings Pneumonia Infiltration Effusion
```

All hyper-parameters are in [`configs/default.yaml`](configs/default.yaml).

## 4. Outputs → paper

| File | Used for |
|---|---|
| `outputs/splits/split_summary.json` | Dataset / split description (images, patients, class counts) |
| `outputs/<model>/seed<k>/test_metrics.json` | Accuracy, sensitivity, specificity, precision, F1, AUC, confusion matrix, 95% CIs |
| `outputs/summary/comparison_seed<k>.csv` | Architecture comparison table |
| `outputs/summary/delong_seed<k>.csv` | DeLong Z and p-values |
| `outputs/summary/multi_seed_mean_sd.csv` | Mean ± SD over seeds |
| `outputs/summary/throughput.json` | Parameters and images/s |
| `outputs/densenet121/seed<k>/pointing_game*.{json,csv}` | Pointing Game accuracy (overall, per finding, centre baseline) |
| `figures/figure1_workflow.{png,pdf}` (`python figures/make_figure1.py`) | Figure 1, framework workflow |
| `outputs/figures/fig_confusion_matrix.{png,pdf}` | Confusion matrix figure |
| `outputs/figures/fig_roc_curves.{png,pdf}` | ROC curves of all models |
| `outputs/figures/fig_training_curves.{png,pdf}` | Loss / accuracy curves |
| `outputs/figures/fig_gradcam_cases.{png,pdf}` + `_images.csv` | Grad-CAM figure and the exact test images shown |

## 5. Methodological notes

* **No leakage:** partitions are created over `Patient ID`; `prepare_splits` asserts zero patient overlap.
* **Fair baselines:** every backbone uses the same GAP + Dropout + linear head, input size (224×224), optimiser, schedule and early stopping. VGG16's original fully connected layers are therefore replaced (≈14.7 M parameters).
* **Normalisation:** images are min-max scaled to [0, 1] and then standardised with ImageNet mean/std (required by the pretrained weights).
* **Grad-CAM layer:** output of the last dense block (`features.denseblock4`, 7×7×1024) for DenseNet121.
* **Pointing Game:** maximum of the heat-map vs. radiologist boxes, 15 px tolerance (Zhang et al., 2016); only images of *test* patients are used, and an image-centre baseline is reported.
* **Label noise:** NIH labels were text-mined from reports and are imperfect; results should be interpreted accordingly.
* **Statistics tests:** `python -m pytest tests -q` verifies the DeLong implementation against scikit-learn.

## 6. Citation

```bibtex
@article{xaichestdiag,
  title  = {Explainable Deep Learning Framework for Emergency Triaging of Pulmonary Diseases
            from Chest X-Rays Using DenseNet121 and Grad-CAM},
  author = {<Authors>},
  journal= {<Journal>},
  year   = {<Year>}
}
```

## License

MIT, see [LICENSE](LICENSE). The NIH ChestX-ray14 dataset is subject to its own terms of use.
