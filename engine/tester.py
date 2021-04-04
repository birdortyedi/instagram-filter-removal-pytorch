import os
import json
import torch
import kornia
import glog as log
import numpy as np
import itertools
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from torch.utils import data
from datasets.transforms import *
import torchvision.models as models
import torchvision.transforms as T
from lpips_pytorch import lpips

from datasets.iffi import IFFIDataset
from modeling.build import build_model
from modeling.vgg import VGG16FeatLayer
from metrics.ssim import SSIM
from utils.data_utils import linear_scaling, linear_unscaling


class Tester:
    def __init__(self, opt):
        self.opt = opt
        assert self.opt.DATASET.NAME.lower() in ["iffi"]

        self.classes = ["1997", "Amaro", "Brannan", "Clarendon", "Gingham", "He-Fe", "Hudson", "Lo-Fi", "Mayfair",
                        "Nashville", "Original", "Perpetua", "Sutro", "Toaster", "Valencia", "Willow", "X-Pro II"]

        self.model_name = "{}_{}".format(self.opt.MODEL.NAME, self.opt.DATASET.NAME) + \
                          "_{}step_{}bs".format(self.opt.TRAIN.NUM_TOTAL_STEP, self.opt.TRAIN.BATCH_SIZE) + \
                          "_{}lr_{}gpu".format(self.opt.MODEL.IFR.SOLVER.LR, self.opt.SYSTEM.NUM_GPU) + \
                          "_{}run".format(self.opt.WANDB.RUN)
        self.output_dir = os.path.join(self.opt.TEST.OUTPUT_DIR, self.model_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.transform = Compose([
            ResizeTwoInstances(self.opt.DATASET.SIZE),
            ToTensor(),
        ])
        self.to_pil = T.ToPILImage()

        self.dataset = IFFIDataset(root=self.opt.DATASET.TEST_ROOT, transform=self.transform)
        self.image_loader = data.DataLoader(dataset=self.dataset, batch_size=1, shuffle=False)

        self.net, self.mlp = build_model(self.opt)
        self.vgg16 = models.vgg16(pretrained=True).features.eval().cuda()
        self.vgg_feat = VGG16FeatLayer(self.vgg16)

        self.PSNR = kornia.losses.psnr.PSNRLoss(max_val=1.)
        self.SSIM = SSIM()  # kornia's SSIM is buggy.

        self.check_and_use_multi_gpu()
        log.info("Checkpoints loading...")
        self.load_checkpoints_from_ckpt(self.opt.MODEL.CKPT)
        self.net.eval()

    def load_checkpoints_from_ckpt(self, ckpt_path):
        checkpoints = torch.load(ckpt_path)
        self.net.load_state_dict(checkpoints["ifr"])
        if checkpoints["mlp"] is not None:
            self.mlp.load_state_dict(checkpoints["mlp"])

    def check_and_use_multi_gpu(self):
        log.info("GPU ID: {}".format(torch.cuda.current_device()))
        self.net = self.net.cuda()
        self.mlp = self.mlp.cuda()

    # def check_and_use_multi_gpu(self):
    #     if torch.cuda.device_count() > 1 and self.opt.SYSTEM.NUM_GPU > 1:
    #         log.info("Using {} GPUs...".format(torch.cuda.device_count()))
    #         self.net = torch.nn.DataParallel(self.net).cuda()
    #         self.mlp = torch.nn.DataParallel(self.mlp).cuda()
    #     else:
    #         log.info("GPU ID: {}".format(torch.cuda.current_device()))
    #         self.net = self.net.cuda()
    #         self.mlp = self.mlp.cuda()

    def eval(self):
        psnr_lst, ssim_lst, lpips_lst = list(), list(), list()
        with torch.no_grad():
            all_preds, all_targets = torch.tensor([]), torch.tensor([])
            for batch_idx, (imgs, y_imgs) in enumerate(self.image_loader):
                imgs = linear_scaling(torch.cat(imgs, dim=0).float().cuda())
                y_imgs = torch.cat(y_imgs, dim=0).float().cuda()
                y = torch.arange(0, len(self.classes)).cuda()
                all_targets = torch.cat((all_targets, y.float().cpu()), dim=0)

                vgg_feat = self.vgg16(imgs)
                output, aux = self.net(imgs, vgg_feat)
                output = torch.clamp(output, max=1., min=0.)
                y_pred = torch.argmax(self.mlp(aux), dim=-1) if self.mlp is not None else None
                all_preds = torch.cat((all_preds, y_pred.float().cpu()), dim=0) if y_pred is not None else all_preds

                print(all_preds.size())

                # ssim = self.SSIM(255. * y_imgs, 255. * output).item()
                # ssim_lst.append(ssim)
                #
                # psnr = self.PSNR(y_imgs, output).item()
                # psnr_lst.append(psnr)
                #
                # lpps = lpips(y_imgs, output, net_type='alex', version='0.1').item() / len(y_imgs)  # TODO ?? not sure working
                # lpips_lst.append(lpps)

                # batch_accuracy = round(torch.mean(torch.tensor(y == y_pred.clone().detach()).float()).item() * 100., 2)
                # log.info("{}/{}\tLPIPS: {}\tSSIM: {}\tPSNR: {}\tImage Accuracy: {}".format(batch_idx+1, len(self.image_loader), round(lpps, 3),
                #                                                                            round(ssim, 3), round(psnr, 3), batch_accuracy))

                # os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
                # for i, (y_img, img, out) in enumerate(zip(y_imgs.cpu(), linear_unscaling(imgs).cpu(), output.cpu())):
                #     self.to_pil(y_img).save(os.path.join(self.output_dir, "images", "{}_{}_real_A.png".format(batch_idx, i)))
                #     self.to_pil(img).save(os.path.join(self.output_dir, "images", "{}_{}_fake_B.png".format(batch_idx, i)))
                #     self.to_pil(out).save(os.path.join(self.output_dir, "images", "{}_{}_real_B.png".format(batch_idx, i)))

        if len(all_preds) > 0:
            acc = round((torch.sum(all_preds == all_targets).float() / len(all_preds)).item(), 3) * 100
            self.plot_cm(all_targets, all_preds, list(range(len(self.classes))))
            # self.plot_confusion_matrix(all_targets, all_preds, self.classes)
        else:
            acc = "None"
        results = {"Dataset": self.opt.DATASET.NAME, "PSNR": np.mean(psnr_lst), "SSIM": np.mean(ssim_lst), "LPIPS": np.mean(lpips_lst), "Accuracy": acc}
        log.info(results)
        # with open(os.path.join(self.output_dir, "metrics.json"), "a+") as f:
        #     json.dump(results, f)

    def infer(self):
        pass

    def plot_cm(self, tgts, preds, clss):
        sns.light_palette("seagreen", as_cmap=True)
        cm = confusion_matrix(tgts, preds, labels=clss)
        plt.figure(figsize=(len(clss), len(clss)))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap="YlGn", cbar=True, cbar_kws={'label': 'Number of the correct predictions'})
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=35)
        ax.xaxis.tick_top()
        plt.yticks(rotation=0)
        plt.xticks(np.arange(len(clss)) + .5, labels=self.classes)
        plt.yticks(np.arange(len(clss)) + .5, labels=self.classes)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.show()

    def plot_confusion_matrix(self, targets, preds, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        cm_mat = torch.stack((targets, preds), dim=1).long().tolist()
        cmt = torch.zeros(len(self.classes), len(self.classes), dtype=torch.int64)
        for p in cm_mat:
            tl, pl = p
            cmt[tl, pl] = cmt[tl, pl] + 1
        log.info(cmt)

        if normalize:
            cmt = cmt.astype('float') / cmt.sum(axis=1)[:, np.newaxis]
            log.info("Normalized confusion matrix")
        else:
            log.info("Confusion matrix, without normalization")

        plt.imshow(cmt, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = ".2f" if normalize else "d"
        thresh = cmt.max() / 2.
        for i, j in itertools.product(range(cmt.shape[0]), range(cmt.shape[1])):
            plt.text(j, i, format(cmt[i, j], fmt), horizontalalignment="center", color="white" if cmt[i, j] > thresh else "black")

        # plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
