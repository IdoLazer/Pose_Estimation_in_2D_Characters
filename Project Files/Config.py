from pathlib import Path

config = \
    {
        'dirs':
            {
                'source_dir': str(Path(__file__).resolve().parent) + '\\',
                'comment': 'Testing new dataloader with large dataset'
            },

        'dataset':
            {
                'character': 'Aang2',
                'name': 'samples=50000 angle_range=60',
                'batch_size': 32,
                'samples_num': 50000,
                'angle_range': 60,
            },

        'training':
            {
                'lr': 0.0001,
                'epochs': 50,
                'kernel_sizes': [9, 7, 5, 3],
                'grad_kernel_sizes': [13, 11, 9, 7, 5],
                'supervised_loss': 1.0,
                'lambda': 0.0001,
                'alpha': 0.1,
                'base_model': None,
            },

        'inspection':
            {
                'num_iter_to_print': 30,
            },

        'network':
            {
                'weight_scaling': 0.3,
            },

    }


def serialize():
    conf_str = ""
    for category in config.keys():
        conf_str += category + ":\n  {\n"
        for key in config[category].keys():
            conf_str += "    " + key + ": " + str(config[category][key]) + "\n"
        conf_str += "  }\n"
    return conf_str
