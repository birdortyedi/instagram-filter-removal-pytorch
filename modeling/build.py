from modeling.ifrnet import IFRNet, Discriminator, PatchDiscriminator, MLP
from modeling.benchmark import UNet


def build_model(args):
    if args.MODEL.NAME.lower() == "ifrnet":
        net = IFRNet(base_n_channels=args.MODEL.IFR.NUM_CHANNELS, destyler_n_channels=args.MODEL.IFR.DESTYLER_CHANNELS)
        mlp = MLP(base_n_channels=args.MODEL.IFR.NUM_CHANNELS, num_class=args.MODEL.NUM_CLASS)
    elif args.MODEL.NAME.lower() == "ifr-no-aux":
        net = IFRNet(base_n_channels=args.MODEL.IFR.NUM_CHANNELS, destyler_n_channels=args.MODEL.IFR.DESTYLER_CHANNELS)
        mlp = None
    else:
        raise NotImplementedError
    return net, mlp


def build_discriminators(args):
    return Discriminator(base_n_channels=args.MODEL.D.NUM_CHANNELS), PatchDiscriminator(base_n_channels=args.MODEL.D.NUM_CHANNELS)

