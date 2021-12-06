from pathlib import Path

config = \
    {
        'dirs':
            {
                'source_dir': str(Path(__file__).resolve().parent) + '\\',
                'comment': '(Aang2 10 range with parameters)'
            },

        'dataset':
            {
                'character': 'Aang2',
                'batch_size': 4,
                'samples_num': 5000,
                'angle_range': 45,
            },

        'training':
            {
                'lr': 0.001,
                'epochs': 50,
                'kernel_sizes': [55, 39, 27, 19, 13, 9, 7, 5, 3],
                'grad_kernel_sizes': [13, 11, 9, 7, 5],
                'lambda': 0.0001,
                'alpha': 0.1,
                'base_model': None,
            },

        'inspection':
            {
                'num_iter_to_print': 100,
            },

        'network':
            {
                'weight_scaling': 0.7,
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
