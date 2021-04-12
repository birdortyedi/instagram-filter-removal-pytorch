import requests
import os
import gradio as gr
import numpy as np
import torch
import torchvision.models as models

from configs.default import get_cfg_defaults
from modeling.build import build_model
from utils.data_utils import linear_scaling


url = "https://www.dropbox.com/s/y97z812sxa1kvrg/ifrnet.pth?dl=1"
r = requests.get(url, stream=True)
if not os.path.exists("ifrnet.pth"):
    with open("ifrnet.pth", 'wb') as f:
        for data in r:
            f.write(data)

cfg = get_cfg_defaults()
cfg.MODEL.CKPT = "ifrnet.pth"
net, _ = build_model(cfg)
net = net.eval()
vgg16 = models.vgg16(pretrained=True).features.eval()


def load_checkpoints_from_ckpt(ckpt_path):
    checkpoints = torch.load(ckpt_path, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoints["ifr"])


load_checkpoints_from_ckpt(cfg.MODEL.CKPT)


def filter_removal(img):
    arr = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    arr = torch.tensor(arr).float() / 255.
    arr = linear_scaling(arr)
    with torch.no_grad():
        feat = vgg16(arr)
        out, _ = net(arr, feat)
        out = torch.clamp(out, max=1., min=0.)
        return out.squeeze(0).permute(1, 2, 0).numpy()


title = "Instagram Filter Removal on Fashionable Images"
description = "This is the demo for IFRNet, filter removal on fashionable images on Instagram. " \
              "To use it, simply upload your filtered image, or click one of the examples to load them."
article = "<p style='text-align: center'><a href=''>Instagram Filter Removal on Fashionable Images</a> | <a href='https://github.com/birdortyedi/instagram-filter-removal-pytorch'>Github Repo</a></p>"

gr.Interface(
    filter_removal,
    gr.inputs.Image(shape=(256, 256)),
    gr.outputs.Image(),
    title=title,
    description=description,
    article=article,
    allow_flagging=False,
    examples_per_page=17,
    examples=[
        ["images/examples/98_He-Fe.jpg"],
        ["images/examples/2_Brannan.jpg"],
        ["images/examples/12_Toaster.jpg"],
        ["images/examples/18_Gingham.jpg"],
        ["images/examples/11_Sutro.jpg"],
        ["images/examples/9_Lo-Fi.jpg"],
        ["images/examples/3_Mayfair.jpg"],
        ["images/examples/4_Hudson.jpg"],
        ["images/examples/5_Amaro.jpg"],
        ["images/examples/6_1977.jpg"],
        ["images/examples/8_Valencia.jpg"],
        ["images/examples/16_Lo-Fi.jpg"],
        ["images/examples/10_Nashville.jpg"],
        ["images/examples/15_X-ProII.jpg"],
        ["images/examples/14_Willow.jpg"],
        ["images/examples/30_Perpetua.jpg"],
        ["images/examples/1_Clarendon.jpg"],
    ]
).launch()
