import glog as log
import argparse
from engine.trainer import Trainer
from engine.tester import Tester
from configs.default import get_cfg_defaults

parser = argparse.ArgumentParser()

parser.add_argument("--base_cfg", default="./wandb/run-20201023_213704-3o2q3c4r/config.yaml", metavar="FILE", help="path to config file")
parser.add_argument("--weights", "-w", default="", type=str, help="weights for IFRNet")
parser.add_argument("--dataset", "-d", default="IFFI", help="dataset names: IFFI")
parser.add_argument("--dataset_dir", default="./datasets/ffhq/images1024x1024", help="dataset directory: './datasets/ffhq/images1024x1024', "
                                                                                     " './datasets/Places/imgs'")
parser.add_argument("--num_step", default=0, help="current step for training")
parser.add_argument("--batch_size", default=8, help="batch size for training")

parser.add_argument("--test", action="store_true")

args = parser.parse_args()

if __name__ == '__main__':
    cfg = get_cfg_defaults()
    # cfg.merge_from_file(args.base_cfg)
    # cfg.MODEL.IS_TRAIN = not args.test
    # cfg.TRAIN.TUNE = args.tune
    # cfg.DATASET.NAME = args.dataset
    # cfg.DATASET.ROOT = args.dataset_dir
    # cfg.TEST.ABLATION = args.ablation
    # cfg.freeze()

    cfg.DATASET.NAME = args.dataset
    cfg.TRAIN.IS_TRAIN = args.test
    cfg.TRAIN.START_STEP = args.num_step
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.MODEL.CKPT = args.weights
    print(cfg)

    if not args.test:
        trainer = Trainer(cfg)
        trainer.run()
    else:
        tester = Tester(cfg)
        tester.eval()
