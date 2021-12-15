from pathlib import Path

config = \
    {
        'dirs':
            {
                'source_dir': str(Path(__file__).resolve().parent) + '\\',
                'comment': 'Aang2 25 range with supervised loss'
            },

        'dataset':
            {
                'character': 'Aang2',
                'batch_size': 4,
                'samples_num': 5000,
                'angle_range': 25,
            },

        'training':
            {
                'lr': 0.0001,
                'epochs': 15,
                'kernel_sizes': [27, 19, 13, 9, 7, 5, 3],
                'grad_kernel_sizes': [13, 11, 9, 7, 5],
                'supervised_loss': 0.8,
                'lambda': 0.0001,
                'alpha': 0.1,
                'base_model': None,
            },

        'inspection':
            {
                'num_iter_to_print': 80,
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
