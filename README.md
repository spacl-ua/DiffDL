# DiffusionMRI-Palette: Conditional Diffusion for Parametric DTI/DKI Mapping


This is the official repo for the paper:
Phillip Martin, Maria Altbach, Ali Bilgin,
Conditional generative diffusion deep learning for accelerated diffusion tensor and kurtosis imaging,
Magnetic Resonance Imaging,
Volume 117,
2025,
110309,
ISSN 0730-725X,
https://doi.org/10.1016/j.mri.2024.110309

## Brief

This repository adapts the [Palette: Image-to-Image Diffusion Models](https://arxiv.org/pdf/2111.05826.pdf) framework for a novel medical imaging task: **generating parametric diffusion MRI metrics (DTI and DKI) from partial diffusion-weighted input**. It is specifically designed for generating scalar maps such as FA, ADC, RD, AD, MK, RK, AK, and KFA from multi-shell DWI data, using conditional denoising diffusion probabilistic models (DDPM).

This pipeline enables rapid and accurate parametric map synthesis, potentially replacing long DWI scan protocols or complex tensor/kurtosis fitting steps with a deep generative alternative.

## Implementation Details

- U-Net backbone derived from `Guided Diffusion` with optional `SR3`-style architecture.
- Uses attention at configurable feature resolutions (e.g., 16x16 or 8x8).
- Uses sinusoidal gamma embedding for denoising step conditioning.
- Applies affine transformations to condition noise prediction on time step.
- EMA (Exponential Moving Average) model tracking for improved sampling.

## Status

### Code
- Diffusion Model Core (DDPM)
- Conditional UNet for multi-channel DWI input
- Custom DTI/DKI dataset classes (FA, ADC, MK, etc.)
- Train/Test scripts
- Multi-GPU training (DDP)
- EMA support
- MAE, FID, IS metrics
- Docker/Singularity compatibility

### Task

- DTI metrics from DWI (FA, ADC)
- DKI metrics (MK, RK, AK, KFA) from multi-shell DWI

## Results

Qualitative and quantitative results show strong correlation with traditional tensor/kurtosis fitting.

## Usage

### Environment Setup
```bash
pip install -r requirements.txt
```

### Docker / Singularity (Optional HPC Setup)
```bash
docker build -t bilginlab:palette_v1   --build-arg USER_ID=$(id -u)   --build-arg GROUP_ID=$(id -g) .

docker save bilginlab:palette_v1 -o palette_docker.tar
singularity build palette_docker.sif docker-archive://palette_docker.tar
```

## Data Preparation

Each subject is a `.npy` file containing 20 slices:

| Channel Index | Signal Type        |
|---------------|--------------------|
| 0             | b=0 image          |
| 1–5           | b=1000 DWIs        |
| 6–10          | b=2000 DWIs        |
| 11–15         | b=3000 DWIs        |
| 13            | FA (target)        |
| 14–16         | ADC, AD, RD        |
| 17–19         | AK, MK, RK/KFA     |

The loader extracts DWIs for conditioning and one target scalar map per training pass.

## Configuration Example

Edit a JSON config file (e.g., `config/diffusion_dti.json`):
```jsonc
"model": {
  "which_model": {
    "name": ["models.model", "Palette"],
    "args": {
      "sample_num": 8,
      "task": "colorization",
      "ema_scheduler": { "ema_start": 1, "ema_iter": 1, "ema_decay": 0.9999 },
      "optimizers": [{ "lr": 5e-5 }]
    }
  },
  "which_networks": [{
    "name": ["models.network", "Network"],
    "args": {
      "module_name": "guided_diffusion",
      "unet": {
        "in_channel": 6,
        "out_channel": 1,
        "image_size": 192
      }
    }
  }]
}
```

## Training
```bash
python run.py -p train -c config/diffusion_dti.json
```

## Testing
```bash
python run.py -p test -c config/diffusion_dti.json
```

## Evaluation
```bash
python eval.py -s /path/to/gt -d /path/to/predictions
```
Outputs: FID, IS, MAE.


## Acknowledgements
This project is adapted from:

> **Palette: Image-to-Image Diffusion Models**  
> Y. Saharia, W. Chan, H. Chang, C. Lee, J. Ho, T. Salimans, D. Fleet, M. Norouzi.  
> NeurIPS 2022.  
> [[Paper](https://arxiv.org/pdf/2111.05826.pdf)] | [[Project](https://iterative-refinement.github.io/palette/)]

We also build upon:
- [openai/guided-diffusion](https://github.com/openai/guided-diffusion)
- [Janspiry/Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)

## Citation
```bibtex
@article{MARTIN2025110309,
title = {Conditional generative diffusion deep learning for accelerated diffusion tensor and kurtosis imaging},
journal = {Magnetic Resonance Imaging},
volume = {117},
pages = {110309},
year = {2025},
issn = {0730-725X},
doi = {https://doi.org/10.1016/j.mri.2024.110309},
url = {https://www.sciencedirect.com/science/article/pii/S0730725X2400290X},
author = {Phillip Martin and Maria Altbach and Ali Bilgin},
}
@inproceedings{palette2022,
  title={Palette: Image-to-Image Diffusion Models},
  author={Saharia, Chitwan and Chan, William and Chang, Huiwen and Lee, Chris and Ho, Jonathan and Salimans, Tim and Fleet, David J and Norouzi, Mohammad},
  booktitle={NeurIPS},
  year={2022}
}
```

## Contact
This project is maintained by Bilgin Lab at University of Arizona. For questions, please contact bilgin@arizona.edu.
