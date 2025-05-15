# StyleGuider

## About The Project
> [!NOTE]
> This repository based on the [Guide-and-Rescale](https://github.com/AIRI-Institute/Guide-and-Rescale).

>While most of the Style-Transfer methods require additional optimization steps to find embedding that represents the style of an image, which makes them computationally inefficient, this project proposes to use Diffusion Models with special energy functions alongside the Guidance Mechanism which allows control transfer of various stylistic attributes from one reference image while preserving content from another.
>
![example1](docs/panel___A%20honey%20jar%20and%20an%20orange___Neo-Figurative%20Art___04.png)
![example2](docs/panel___Ancient%20City%20Wall___Pointilism___03.png)
![example3](docs/panel___objects%20including%20a%20wine%20bottle,%20a%20wine%20glass,%20a%20bowl,%20a%20decorative%20bottle,%20and%20a%20sphere___Realism___04.png)

## Getting Started
### Prerequisites
To run the code you need to have conda environment manager. Project was tested on NVIDIA Tesla V100 32 GB.
### Installation
```bash
conda env create -f env.yaml
```
```bash
conda activate sg_env
```

## Usage
### Run demo
#### VGG Style Configuration (better content preservation, transfers only color and primitive stylisitc attributes)
```bash
python run_experiment16.py run_name=vgg_configuration
```
#### Self-Attention + V Style Guider (better style fidelity, transfers style image onto content image)
```bash
python run_experiment17.py run_name=selfattn_v_configuration
```
### Run StyleBench benchmarking
1. Download StyleBench from google drive via [link](https://drive.google.com/file/d/1Q_jbI25NfqZvuwWv53slmovqyW_L4k2r/view) and unzip it:
```bash
gdown 1Q_jbI25NfqZvuwWv53slmovqyW_L4k2r
```
```bash
unzip StyleBench.zip -d StyleBench
```

2. Prepare `yaml` config file that contains style and content pairs:
```bash
python cnt_sty_images_to_config.py --cnt StyleBench/content --sty StyleBench/style --save_to configs/samples --name stylebench --cnt_prompt_from_name
```

3. Run chosen configuration (`run_experiment16.py` or `run_experiment17.py`), but set `samples=stylebench`. Example:
```bash
python run_experiment17.py run_name=selfattn_v_configuration_stylebench samples=stylebench
```
Results of your experiment will be located at `outputs/{TODAY DATE}/{RIGHT NOW TIME}/output_imgs`

4. Evaluate metrics between stylised images and `style` / `content` from StyleBench.
To evaluate LPIPS (example):
```bash
python eval_lpips.py --content StyleBench/content --generated outputs/2025-05-15/18-57-24___selfattn_v_configuration_stylebench/output_imgs
```
To evaluate CLIP-Score (example):
```bash
python eval_clip.py --style StyleBench/style --generated outputs/2025-05-15/18-57-24___selfattn_v_configuration_stylebench/output_imgs
```
