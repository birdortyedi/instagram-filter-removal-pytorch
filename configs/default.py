from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPU = 2
_C.SYSTEM.NUM_WORKERS = 4

_C.WANDB = CN()
_C.WANDB.PROJECT_NAME = "instagram-filter-removal"
_C.WANDB.ENTITY = "vvgl-ozu"
_C.WANDB.RUN = 12
_C.WANDB.LOG_DIR = ""
_C.WANDB.NUM_ROW = 0

_C.TRAIN = CN()
_C.TRAIN.NUM_TOTAL_STEP = 120000
_C.TRAIN.START_STEP = 0
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.SHUFFLE = True
_C.TRAIN.LOG_INTERVAL = 100
_C.TRAIN.SAVE_INTERVAL = 5000
_C.TRAIN.SAVE_DIR = "./weights"
_C.TRAIN.RESUME = True
_C.TRAIN.VISUALIZE_INTERVAL = 100
_C.TRAIN.TUNE = False

_C.MODEL = CN()
_C.MODEL.NAME = "ifr-no-aux"
_C.MODEL.IS_TRAIN = True
_C.MODEL.NUM_CLASS = 17
_C.MODEL.CKPT = ""

_C.MODEL.IFR = CN()
_C.MODEL.IFR.NAME = "InstaFilterRemovalNetwork"
_C.MODEL.IFR.NUM_CHANNELS = 32
_C.MODEL.IFR.DESTYLER_CHANNELS = 32
_C.MODEL.IFR.SOLVER = CN()
_C.MODEL.IFR.SOLVER.LR = 2e-4
_C.MODEL.IFR.SOLVER.BETAS = (0.5, 0.9)
_C.MODEL.IFR.SOLVER.SCHEDULER = []
_C.MODEL.IFR.SOLVER.DECAY_RATE = 0.

_C.MODEL.D = CN()
_C.MODEL.D.NAME = "1-ChOutputDiscriminator"
_C.MODEL.D.NUM_CHANNELS = 32
_C.MODEL.D.NUM_CRITICS = 5
_C.MODEL.D.SOLVER = CN()
_C.MODEL.D.SOLVER.LR = 1e-3
_C.MODEL.D.SOLVER.BETAS = (0.5, 0.9)
_C.MODEL.D.SOLVER.SCHEDULER = []
_C.MODEL.D.SOLVER.DECAY_RATE = 0.5

_C.OPTIM = CN()
_C.OPTIM.GP = 10
_C.OPTIM.MASK = 1
_C.OPTIM.RECON = 1.4
_C.OPTIM.SEMANTIC = 1e-4
_C.OPTIM.TEXTURE = 1e-3
_C.OPTIM.ADVERSARIAL = 1e-3
_C.OPTIM.AUX = 0.5

_C.DATASET = CN()
_C.DATASET.NAME = "IFFI"  # "IFFI"  # "DIV2K?" #
_C.DATASET.ROOT = "../../Datasets/IFFI-dataset/train"  # "../../Datasets/IFFI-dataset"  # "/media/birdortyedi/e5042b8f-ca5e-4a22-ac68-7e69ff648bc4/IFFI-dataset"
_C.DATASET.TEST_ROOT = "../../Datasets/IFFI-dataset"
_C.DATASET.SIZE = 256
_C.DATASET.CROP_SIZE = 512
_C.DATASET.MEAN = [0.5, 0.5, 0.5]
_C.DATASET.STD = [0.5, 0.5, 0.5]

_C.TEST = CN()
_C.TEST.OUTPUT_DIR = "./outputs"
_C.TEST.ABLATION = False
_C.TEST.WEIGHTS = ""
_C.TEST.BATCH_SIZE = 64
_C.TEST.IMG_ID = 52


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# provide a way to import the defaults as a global singleton:
cfg = _C  # users can `from config import cfg`
