# Instagram Filter Removal on Fashionable Images

Demo App: https://gradio.app/g/birdortyedi/instagram-filter-removal-pytorch

![][results]

> **Instagram Filter Removal on Fashionable Images**<br>
> Furkan Kınlı, Barış Özcan, Furkan Kıraç <br>
> *Accepted to NTIRE 2021 at CVPR2021* <br>
>
>**Abstract:** Social media images are generally transformed by filtering to obtain aesthetically more pleasing appearances. However, CNNs generally fail to interpret both the image and its filtered version as the same in the visual analysis of social media images. We introduce Instagram Filter Removal Network (IFRNet) to mitigate the effects of image filters for social media analysis applications. To achieve this, we assume any filter applied to an image substantially injects a piece of additional style information to it, and we consider this problem as a reverse style transfer problem. The visual effects of filtering can be directly removed by adaptively normalizing external style information in each level of the encoder. Experiments demonstrate that IFRNet outperforms all compared methods in quantitative and qualitative comparisons, and has the ability to remove the visual effects to a great extent. Additionally, we present the filter classification performance of our proposed model, and analyze the dominant color estimation on the images unfiltered by all compared methods.

## Description
The official implementation of the paper titled "Instagram Filter Removal on Fashionable Images".
We propose a method for removing Instagram filters from the images by assuming the affects of filters as the style information.

## Updates
**12/4/2021** Release of the demo app in Gradio.app

**11/4/2021** Accepted to NTIRE2021 in conjunction with CVPR2021

**5/4/2021:** Release of the code

**23/3/2021:** Submission of the paper to NTIRE 2021 at CVPR2021

## Requirements
To install requirements:

```
pip install -r requirements.txt
```

## Architecture
![][model]

## Dataset
We have collected a set of aesthetically pleasing
images that are filtered by 16 Instagram filters. **IFFI dataset**
contains 600 images and with their 16 different filtered versions for each. In particular, we have picked mostly-used
16 filters: *1977*, *Amaro*, *Brannan*, *Clarendon*, *Gingham*,
*He-Fe*, *Hudson*, *Lo-Fi*, *Mayfair*, *Nashville*, *Perpetua*, *Sutro*,
*Toaster*, *Valencia*, *Willow*, *X-Pro II*. 

We are planning to collect more images with all possible filters on Instagram.

Raw images: https://www.dropbox.com/s/t9o0uakcjt6i3rn/IFFI-dataset-raw.zip?dl=0

Dataset: https://www.dropbox.com/s/4kk06sog5zbk728/IFFI-dataset.zip?dl=0

Pix2Pix-compatible version: https://www.dropbox.com/s/87k949epwpuq3r0/IFFI-dataset-pix2pix.zip?dl=0

CycleGAN-compatible version: https://www.dropbox.com/s/cb0xtmpavkyjbik/IFFI-dataset-cycleGAN.zip?dl=0

## Training

To train IFRNet from the scratch in the paper, run this command:

```
python main.py --base_cfg config.yml --dataset IFFI --dataset_dir /path/to/dataset
```

## Evaluation

To evaluate IFRNet on IFFI dataset, run:

```
python main.py --base_cfg config.yaml -t -w ifrnet.pth --dataset IFFI --dataset_dir /path/to/dataset
```

## Citation
Not ready yet.

## Contacts
Please feel free to open an issue or to send an e-mail to ```furkan.kinli@ozyegin.edu.tr```

[results]: images/paper/results.png
[model]: images/paper/IFRNet.png