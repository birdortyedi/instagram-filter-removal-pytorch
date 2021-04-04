import os
import wandb
import glog as log

from torch.utils import data
from torchvision import transforms
import torchvision.models as models

from modeling.build import build_model, build_discriminators
from modeling.vgg import VGG16FeatLayer
from datasets.transforms import *
from datasets.iffi import IFFIDataset
from losses.consistency import SemanticConsistencyLoss, IDMRFLoss
from losses.adversarial import compute_gradient_penalty
from utils.data_utils import linear_scaling, linear_unscaling


# torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        assert self.opt.DATASET.NAME.lower() in ["iffi"]

        self.model_name = "{}_{}".format(self.opt.MODEL.NAME, self.opt.DATASET.NAME) + \
                          "_{}step_{}bs".format(self.opt.TRAIN.NUM_TOTAL_STEP, self.opt.TRAIN.BATCH_SIZE) + \
                          "_{}lr_{}gpu".format(self.opt.MODEL.IFR.SOLVER.LR, self.opt.SYSTEM.NUM_GPU) + \
                          "_{}run".format(self.opt.WANDB.RUN)

        self.opt.WANDB.LOG_DIR = os.path.join("./logs/", self.model_name)
        self.wandb = wandb
        self.wandb.init(project=self.opt.WANDB.PROJECT_NAME, resume=self.opt.TRAIN.RESUME, notes=self.opt.WANDB.LOG_DIR, config=self.opt, entity=self.opt.WANDB.ENTITY)

        self.transform = Compose([
            ResizeTwoInstances(self.opt.DATASET.SIZE),
            RandomHorizontalFlipTwoInstances(),
            ToTensor(),
        ])

        self.dataset = IFFIDataset(root=self.opt.DATASET.ROOT, transform=self.transform)
        self.image_loader = data.DataLoader(dataset=self.dataset, batch_size=self.opt.TRAIN.BATCH_SIZE, shuffle=self.opt.TRAIN.SHUFFLE, num_workers=self.opt.SYSTEM.NUM_WORKERS)

        self.to_pil = transforms.ToPILImage()

        self.net, self.mlp = build_model(self.opt)
        self.discriminator, self.patch_discriminator = build_discriminators(self.opt)
        self.vgg16 = models.vgg16(pretrained=True).features.eval().cuda()
        self.vgg_feat = VGG16FeatLayer(self.vgg16)

        self.optimizer = torch.optim.Adam(self.net.parameters() if self.mlp is None else list(self.net.parameters()) + list(self.mlp.parameters()),
                                          lr=self.opt.MODEL.IFR.SOLVER.LR, betas=self.opt.MODEL.IFR.SOLVER.BETAS)
        self.optimizer_discriminator = torch.optim.Adam(list(self.discriminator.parameters()) + list(self.patch_discriminator.parameters()),
                                                        lr=self.opt.MODEL.D.SOLVER.LR, betas=self.opt.MODEL.D.SOLVER.BETAS)

        self.num_step = self.opt.TRAIN.START_STEP

        if self.opt.TRAIN.START_STEP != 0 and self.opt.TRAIN.RESUME:  # find start step from checkpoint file name. TODO
            log.info("Checkpoints loading...")
            self.load_checkpoints(self.opt.TRAIN.START_STEP)
        elif self.opt.MODEL.CKPT != "":
            log.info("Checkpoints loading from ckpt file...")
            self.load_checkpoints_from_ckpt(self.opt.MODEL.CKPT)

        self.check_and_use_multi_gpu()

        self.reconstruction_loss = torch.nn.L1Loss().cuda()
        self.semantic_consistency_loss = SemanticConsistencyLoss().cuda()
        self.texture_consistency_loss = IDMRFLoss().cuda()
        self.auxiliary_loss = torch.nn.CrossEntropyLoss().cuda()

    def run(self):
        while self.num_step < self.opt.TRAIN.NUM_TOTAL_STEP:
            self.num_step += 1
            info = " [Step: {}/{} ({}%)] ".format(self.num_step, self.opt.TRAIN.NUM_TOTAL_STEP, 100 * self.num_step / self.opt.TRAIN.NUM_TOTAL_STEP)

            imgs, y_imgs, labels = next(iter(self.image_loader))
            imgs = linear_scaling(imgs.float().cuda())
            y_imgs = y_imgs.float().cuda()
            labels = labels.cuda()

            for _ in range(self.opt.MODEL.D.NUM_CRITICS):
                d_loss = self.train_D(imgs, y_imgs)
            info += "D Loss: {} ".format(d_loss)

            g_loss, output = self.train_G(imgs, y_imgs, labels)
            info += "G Loss: {} ".format(g_loss)

            if self.num_step % self.opt.TRAIN.LOG_INTERVAL == 0:
                log.info(info)

            if self.num_step % self.opt.TRAIN.VISUALIZE_INTERVAL == 0:
                idx = self.opt.WANDB.NUM_ROW
                self.wandb.log({"examples": [
                    self.wandb.Image(self.to_pil(y_imgs[idx].cpu()), caption="original_image"),
                    self.wandb.Image(self.to_pil(linear_unscaling(imgs[idx]).cpu()), caption="filtered_image"),
                    self.wandb.Image(self.to_pil(torch.clamp(output, min=0., max=1.)[idx].cpu()), caption="output")
                ]}, commit=False)
            self.wandb.log({})
            if self.num_step % self.opt.TRAIN.SAVE_INTERVAL == 0 and self.num_step != 0:
                self.do_checkpoint(self.num_step)

    def train_D(self, x, y):
        self.optimizer_discriminator.zero_grad()

        with torch.no_grad():
            vgg_feat = self.vgg16(x)
        output, _ = self.net(x, vgg_feat)

        real_global_validity = self.discriminator(y).mean()
        fake_global_validity = self.discriminator(output.detach()).mean()
        gp_global = compute_gradient_penalty(self.discriminator, output.data, y.data)

        real_patch_validity = self.patch_discriminator(y).mean()
        fake_patch_validity = self.patch_discriminator(output.detach()).mean()
        gp_fake = compute_gradient_penalty(self.patch_discriminator, output.data, y.data)

        real_validity = real_global_validity + real_patch_validity
        fake_validity = fake_global_validity + fake_patch_validity
        gp = gp_global + gp_fake

        d_loss = -real_validity + fake_validity + self.opt.OPTIM.GP * gp
        d_loss.backward()
        self.optimizer_discriminator.step()

        self.wandb.log({"real_global_validity": -real_global_validity.item(),
                        "fake_global_validity": fake_global_validity.item(),
                        "real_patch_validity": -real_patch_validity.item(),
                        "fake_patch_validity": fake_patch_validity.item(),
                        "gp_global": gp_global.item(),
                        "gp_fake": gp_fake.item(),
                        "real_validity": -real_validity.item(),
                        "fake_validity": fake_validity.item(),
                        "gp": gp.item()}, commit=False)
        return d_loss.item()

    def train_G(self, x, y, labels):
        self.optimizer.zero_grad()

        with torch.no_grad():
            vgg_feat = self.vgg16(x)
        output, aux = self.net(x, vgg_feat)
        labels_pred = self.mlp(aux) if self.mlp is not None else None

        recon_loss = self.reconstruction_loss(output, y)

        with torch.no_grad():
            out_vgg_feat = self.vgg_feat(output)
            y_vgg_feat = self.vgg_feat(y)
        sem_const_loss = self.semantic_consistency_loss(out_vgg_feat, y_vgg_feat)
        tex_const_loss = self.texture_consistency_loss(out_vgg_feat, y_vgg_feat)
        adv_global_loss = -self.discriminator(output).mean()
        adv_patch_loss = -self.patch_discriminator(output).mean()

        if self.mlp is None:
            adv_loss = (adv_global_loss + adv_patch_loss)
        else:
            aux_loss = self.auxiliary_loss(labels_pred, labels.squeeze())
            adv_loss = (adv_global_loss + adv_patch_loss) + self.opt.OPTIM.AUX * aux_loss

        g_loss = self.opt.OPTIM.RECON * recon_loss + \
                 self.opt.OPTIM.SEMANTIC * sem_const_loss + \
                 self.opt.OPTIM.TEXTURE * tex_const_loss * \
                 self.opt.OPTIM.ADVERSARIAL * adv_loss
        g_loss.backward()
        self.optimizer.step()

        self.wandb.log({"recon_loss": recon_loss.item(),
                        "sem_const_loss": sem_const_loss.item(),
                        "tex_const_loss": tex_const_loss.item(),
                        "adv_global_loss": adv_global_loss.item(),
                        "adv_patch_loss": adv_patch_loss.item(),
                        "aux_loss": aux_loss.item() if self.mlp is not None else 0,
                        "adv_loss": adv_loss.item()}, commit=False)
        return g_loss.item(), output.detach()

    def check_and_use_multi_gpu(self):
        if torch.cuda.device_count() > 1 and self.opt.SYSTEM.NUM_GPU > 1:
            log.info("Using {} GPUs...".format(torch.cuda.device_count()))
            self.net = torch.nn.DataParallel(self.net).cuda()
            self.mlp = torch.nn.DataParallel(self.mlp).cuda() if self.mlp is not None else self.mlp
            self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
            self.patch_discriminator = torch.nn.DataParallel(self.patch_discriminator).cuda()
        else:
            log.info("GPU ID: {}".format(torch.cuda.current_device()))
            self.net = self.net.cuda()
            self.mlp = self.mlp.cuda() if self.mlp is not None else self.mlp
            self.discriminator = self.discriminator.cuda()
            self.patch_discriminator = self.patch_discriminator.cuda()

    def do_checkpoint(self, num_step):
        if not os.path.exists("./{}/{}".format(self.opt.TRAIN.SAVE_DIR, self.model_name)):
            os.makedirs("./{}/{}".format(self.opt.TRAIN.SAVE_DIR, self.model_name), exist_ok=True)

        if self.mlp is not None:
            checkpoint = {
                'num_step': num_step,
                'ifr': self.net.module.state_dict() if isinstance(self.net, torch.nn.DataParallel) else self.net.state_dict(),
                'mlp': self.mlp.module.state_dict() if isinstance(self.mlp, torch.nn.DataParallel) else self.mlp.state_dict(),
                'D': self.discriminator.module.state_dict() if isinstance(self.discriminator, torch.nn.DataParallel) else self.discriminator.state_dict(),
                'patch_D': self.patch_discriminator.module.state_dict() if isinstance(self.patch_discriminator, torch.nn.DataParallel) else self.patch_discriminator.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'optimizer_D': self.optimizer_discriminator.state_dict()
            }
        else:
            checkpoint = {
                'num_step': num_step,
                'ifr': self.net.module.state_dict() if isinstance(self.net, torch.nn.DataParallel) else self.net.state_dict(),
                'D': self.discriminator.module.state_dict() if isinstance(self.discriminator, torch.nn.DataParallel) else self.discriminator.state_dict(),
                'patch_D': self.patch_discriminator.module.state_dict() if isinstance(self.patch_discriminator, torch.nn.DataParallel) else self.patch_discriminator.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'optimizer_D': self.optimizer_discriminator.state_dict()
            }
        torch.save(checkpoint, "./{}/{}/checkpoint-{}.pth".format(self.opt.TRAIN.SAVE_DIR, self.model_name, num_step))

    def load_checkpoints(self, num_step):
        checkpoints = torch.load("./{}/{}/checkpoint-{}.pth".format(self.opt.TRAIN.SAVE_DIR, self.model_name, num_step))
        self.num_step = checkpoints["num_step"]
        self.net.load_state_dict(checkpoints["ifr"])
        if "mlp" in checkpoints.keys():
            self.mlp.load_state_dict(checkpoints["mlp"])
        self.discriminator.load_state_dict(checkpoints["D"])
        self.patch_discriminator.load_state_dict(checkpoints["patch_D"])

        self.optimizer.load_state_dict(checkpoints["optimizer"])
        self.optimizer_discriminator.load_state_dict(checkpoints["optimizer_D"])
        self.optimizers_to_cuda()

    def load_checkpoints_from_ckpt(self, ckpt_path):
        checkpoints = torch.load(ckpt_path)
        self.num_step = checkpoints["num_step"]
        self.net.load_state_dict(checkpoints["ifr"])
        if "mlp" in checkpoints.keys():
            self.mlp.load_state_dict(checkpoints["mlp"])
        self.discriminator.load_state_dict(checkpoints["D"])
        self.patch_discriminator.load_state_dict(checkpoints["patch_D"])

        self.optimizers_to_cuda()

    def optimizers_to_cuda(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        for state in self.optimizer_discriminator.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
