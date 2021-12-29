from pathlib import Path

config = \
    {
        'dirs':
            {
                'source_dir': str(Path(__file__).resolve().parent) + '\\',
                'comment': 'Aang2 60 range refined from 45 with supervised loss + unsupervised loss'
            },

        'dataset':
            {
                'character': 'Aang2',
                'batch_size': 4,
                'samples_num': 10000,
                'angle_range': 60,
            },

        'training':
            {
                'lr': 0.0001,
                'epochs': 40,
                'kernel_sizes': [39, 27, 19, 13, 9, 7, 5, 3],
                'grad_kernel_sizes': [13, 11, 9, 7, 5],
                'supervised_loss': 0.5,
                'lambda': 0.0001,
                'alpha': 0.1,
                'base_model': '15-12-2021 18-49-50 Aang2 45 range with supervised loss ONLY',
            },

        'inspection':
            {
                'num_iter_to_print': 40,
            },

        'network':
            {
                'weight_scaling': 0.8,
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
