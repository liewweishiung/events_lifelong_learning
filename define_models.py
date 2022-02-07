"""
The code in this file is copied from
https://github.com/GMvandeVen/brain-inspired-replay/blob/master/define_models.py

Adaptations of the code are performed by Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).
"""

import utils
from models.classifier import Classifier
from models.vae import AutoEncoder
from utils import checkattr


##-------------------------------------------------------------------------------------------------------------------##

## Function for defining auto-encoder model
def define_autoencoder(args, config, device, generator=False):
    # -create model
    model = AutoEncoder(
        image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
        # -fc-layers
        fc_layers=args.g_fc_lay if generator and hasattr(args, 'g_fc_lay') else args.fc_lay,
        fc_units=args.g_fc_uni if generator and hasattr(args, 'g_fc_uni') else args.fc_units,
        h_dim=args.g_h_dim if generator and hasattr(args, 'g_h_dim') else args.h_dim,
        fc_drop=0 if generator else args.fc_drop, fc_bn=(args.fc_bn == "yes"), fc_nl=args.fc_nl, excit_buffer=True,
        # -prior
        prior=args.prior if hasattr(args, "prior") else "standard",
        n_modes=args.n_modes if hasattr(args, "prior") else 1,
        per_class=args.per_class if hasattr(args, "prior") else False,
        z_dim=args.g_z_dim if generator and hasattr(args, 'g_z_dim') else args.z_dim,
        # -decoder
        hidden=checkattr(args, 'hidden'),
        recon_loss=args.recon_loss, network_output="none" if checkattr(args, "normalize") else "sigmoid",
        dg_gates=utils.checkattr(args, 'dg_gates'), dg_type=args.dg_type if hasattr(args, 'dg_type') else "task",
        dg_prop=args.dg_prop if hasattr(args, 'dg_prop') else 0.,
        tasks=args.tasks if hasattr(args, 'tasks') else None,
        device=device,
        # -classifier
        classifier=False if generator else True,
        classify_opt=args.classify if hasattr(args, "classify") else "beforeZ",
        # -training-specific components
        lamda_rcl=1. if not hasattr(args, 'rcl') else args.rcl,
        lamda_vl=1. if not hasattr(args, 'vl') else args.vl,
        lamda_pl=(0. if generator else 1.) if not hasattr(args, 'pl') else args.pl,
        habituation_decay_rate=args.habituation_decay_rate,
        top_hab_neurons=args.top_hab_neurons,
    ).to(device)
    return model


##-------------------------------------------------------------------------------------------------------------------##

## Function for defining classifier model
def define_classifier(args, config, device):
    # -create model
    model = Classifier(
        classes=config['classes'],
        # -fc-layers
        fc_layers=args.fc_lay, fc_units=args.fc_units, h_dim=args.h_dim,
        fc_drop=args.fc_drop, fc_bn=True if args.fc_bn == "yes" else False, fc_nl=args.fc_nl, excit_buffer=True,
        # -training-specific components
        hidden=checkattr(args, 'hidden'),
    ).to(device)
    return model


##-------------------------------------------------------------------------------------------------------------------##

## Function for (re-)initializing the parameters of [model]
def init_params(model, args):
    # - reinitialize all parameters according to default initialization
    model.apply(utils.weight_reset)
    # - initialize parameters according to chosen custom initialization (if requested)
    if hasattr(args, 'init_weight') and not args.init_weight == "standard":
        utils.weight_init(model, strategy="xavier_normal")
    if hasattr(args, 'init_bias') and not args.init_bias == "standard":
        utils.bias_init(model, strategy="constant", value=0.01)
    return model

##-------------------------------------------------------------------------------------------------------------------##
