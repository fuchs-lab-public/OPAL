# A Clinical Benchmark of Public Self-Supervised Pathology Foundation Models

Repository of training recipes for the manuscript: "A Clinical Benchmark of Public Self-Supervised Pathology Foundation Models".
Manuscript link: [arxiv](https://arxiv.org/abs/2407.06508)

## Abstract
The use of self-supervised learning (SSL) to train pathology foundation models has increased substantially in the past few years. Notably, several models trained on large quantities of clinical data have been made publicly available in recent months. This will significantly enhance scientific research in computational pathology and help bridge the gap between research and clinical deployment. With the increase in availability of public foundation models of different sizes, trained using different algorithms on different datasets, it becomes important to establish a benchmark to compare the performance of such models on a variety of clinically relevant tasks spanning multiple organs and diseases. In this work, we present a collection of pathology datasets comprising clinical slides associated with clinically relevant endpoints including cancer diagnoses and a variety of biomarkers generated during standard hospital operation from two medical centers. We leverage these datasets to systematically assess the performance of public pathology foundation models and provide insights into best practices for training new foundation models and selecting appropriate pretrained models.

## Clinically Relevant Downstream Tasks

| Task      | Origin | Disease           | Slides | Scanner           |
| --------- | ------ | -------------     | -----: | ----------------- |
| Detection | MSHS   | Breast Cancer     | 1,998  | Philips Ultrafast |
| Detection | MSHS   | Oral Cancer       |   279  | Philips Ultrafast |
| Detection | MSHS   | Bladder Cancer    |   448  | Philips Ultrafast |
| Detection | MSHS   | Kidney Cancer     | 1,000  | Philips Ultrafast |
| Detection | MSHS   | Thyroid Cancer    |   710  | Philips Ultrafast |
| Detection | MSHS   | DCIS              |   233  | Philips Ultrafast |
| Detection | MSHS   | Prostate Cancer   | 1,000  | Philips Ultrafast |
| Detection | MSHS   | Colorectal Cancer |   413  | Philips Ultrafast |
| Detection | MSHS   | IBD               | 1,448  | Philips Ultrafast |

| Task      | Origin | Biomarker     | Specimen      | Slides | Scanner           |
| --------- | ------ | ------------- | ------------- | -----: | ----------------- |
| Biomarker | MSHS   | IHC ER        | Breast Cancer | 2,000  | Philips Ultrafast |
| Biomarker | MSHS   | IHC PR        | Breast Cancer | 1,986  | Philips Ultrafast |
| Biomarker | MSHS   | IHC/FISH HER2 | Breast Cancer | 2,018  | Philips Ultrafast |
| Biomarker | MSHS   | BioMe HRD     | Breast        |   563  | Philips Ultrafast |
| Biomarker | MSHS   | NGS EGFR      | LUAD          |   294  | Philips Ultrafast |
| Biomarker | MSKCC  | NGS EGFR      | LUAD          | 1,000  | Aperio AT2        |
| Biomarker | MSKCC  | NGS ALK       | LUAD          |   999  | Aperio AT2        |
| Biomarker | MSKCC  | NGS STK11     | LUAD          |   998  | Aperio AT2        |
| Biomarker | MSKCC  | NGS KRAS      | LUAD          |   998  | Aperio AT2        |
| Biomarker | MSKCC  | NGS TP53      | LUAD          |   998  | Aperio AT2        |
| Outcome   | MSKCC  | ICI Response  | NSCLC         |   454  | Aperio AT2        |

MSHS: Mount Sinai Health System
DCIS: Ductal Carcinoma In Situ
IBD: Inflammatory Bowel Disease
ER: Estrogen Receptor
PR: Progesterone Receptor
IHC: Immunohistochemistry
FISH: Fluorescence In Situ Hybridization
MSKCC: Memorial Sloan Kettering Cancer Center
LUAD: Lung Adenocarcinoma
ICI: Immene Checkpoint Inhibitors
NSCLC: Non-Small Cell Lung Cancer

## Public Pathology Foundation Models

| Model                                                               | Param. (M) | Algorithm | Training Data | Tiles (M) | Slides (K) |
| ------------------------------------------------------------------- | ---------: | --------- | ------------- | --------: | ---------: |
| [CTransPath](https://github.com/Xiyue-Wang/TransPath)               |         28 | SRCL      | TCGA, PAIP    |        16 |         32 |
| [Phikon]()                                                          |         86 | iBOT      | TCGA          |        43 |          6 |
| [UNI](https://huggingface.co/MahmoodLab/UNI)                        |        303 | DINOv2    | MGB           |       100 |        100 |
| [Virchow](https://huggingface.co/paige-ai/Virchow)                  |        631 | DINOv2    | MSKCC         |     2,000 |      1,488 |
| []()                                                                |         22 | DINO      | MSHS          |     1,600 |        423 |
| []()                                                                |         86 | DINO      | MSHS          |     1,600 |        423 |
| [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) |      1,135 | DINOv2    | PHS           |     1,300 |        171 |

MGB: Mass General Brigham
MSKCC: Memorial Sloan Kettering Cancer Center
MSHS: Mount Sinai Health System
PHS: Providence Health and Services

## Detection Benchmarks
![DINO ViT-small Checkpoints Figure](figures/plot_dinosline.png "DINO ViT-small Checkpoints")

## Biomarker Benchmarks
![DINO ViT-small Checkpoints Figure](figures/plot_dinosline.png "DINO ViT-small Checkpoints")

## Benchmark User Submitted Models
We provide a workflow to benchmark user submitted models. To submit a request follow the instructions below:
1. Submit [this form](https://forms.office.com/Pages/ResponsePage.aspx?id=YZ3odw9XsEO55GNPRi40uCeVZRc28JFPi1Agm1twtOFUMjVTRThNQVpRN1RNMldBMTNCUVZHVFFQSi4u) with the user's name and a valid email address. Optionally, a user can allow to record the results on our leaderboard by checking the relative checkbox and providing a model name.
2. The user will receive an email with a link to a secure OneDrive folder.
3. The user should upload to the provided OneDrive folder the following files:
   - `checkpoint.pth`: the binary file containing the model weights. Currently there is a 250GB limit per file. This file should be loaded with `torch.load` inside the `modules.py` files.
   - `modules.py`: python script that contains the model definition. It should also contain a function that returns the pre-trained encoder called `get_encoder`. On our end we will use the following command to load the encoder: `import modules; model=modules.get_encoder()`. The encoder should accept as input a 4-D tensor of shape (B, 3, H, W) where H and W are 224, and B is the batch size. The encoder's output should be a 2-D tensor of shape (B, F) where F is the dimensionality of the feature representation. The following snippet should run without errors.
```python
import torch
import modules
model = modules.get_encoder()
x = torch.rand(1, 3, 224, 224)
o = model(x)
assert len(x.size()) == 2
assert x.size(0) == o.size(0)
```
   - `requirements.txt` (optional): a list of packages necessary to run the model. In the backend we use the latest monai docker [container](https://hub.docker.com/r/projectmonai/monai). Your code is likely to run as is there. In case other packages are necessary, please include this file.
4. The user will receive via the provided email the results of the benchmarks as a csv file with the following columns:
   - Task
   - Task Type: Detection, Biomarker
   - Mean AUC
   - AUC Standard Deviation
5. After analysis, all data will purged. If the user opted to save the results, they will be posted in the leaderboard.