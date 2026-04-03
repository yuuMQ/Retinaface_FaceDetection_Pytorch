def get_config(network):
    configs = {
        "resnet50": cfg_re50,
        "resnet34": cfg_re34,
        "resnet18": cfg_re18,
        'swinv2_cr_tiny':cfg_swinv2_cr_tiny,
        "swinv2_cr_small": cfg_swinv2_cr_small,
        "swinv2_base": cfg_swinv2_base,
        "swinv2_base_22k": cfg_swinv2_base_22k
    }
    return configs.get(network, None)

cfg_re18 = {
    'name': 'resnet18',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 150,
    'milestones': [70, 90],
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 64,
    'out_channel': 128
}
cfg_re34 = {
    'name': 'resnet34',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 32,
    'epochs': 100,
    'milestones': [70, 90],
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 64,
    'out_channel': 128
}

cfg_re50 = {
    'name': 'resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 8,
    'epochs': 100,
    'milestones': [70, 90],
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

# Swinv2 cr tiny
cfg_swinv2_cr_tiny = {
    'name': 'swinv2_cr_tiny_ns_224.sw_in1k',
    'pretrain': True,
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'image_size': 640,
    'batch_size': 8,
    'epochs': 100,
    'milestones': [70, 90],
    'out_channel': 256,
}

# SwinV2 CR Small
cfg_swinv2_cr_small = {
    'name': 'swinv2_cr_small_ns_224.sw_in1k',
    'pretrain': True,

    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,

    'loc_weight': 2.0,

    'image_size': 640,
    'batch_size': 4,
    'epochs': 100,
    'milestones': [70, 90],

    'out_channel': 256,

}

cfg_swinv2_base = {
    'name': 'swinv2_base_window16_256.ms_in1k',
    'pretrain': True,
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'image_size': 640,
    'batch_size': 8,
    'epochs': 100,
    'milestones': [70, 90],
    'out_channel': 256,
}

cfg_swinv2_base_22k = {
    'name': 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k',
    'pretrain': True,
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'image_size': 640,
    'batch_size': 4,
    'epochs': 100,
    'milestones': [70, 90],
    'out_channel': 256,
}

