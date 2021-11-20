from pathlib import Path

config = \
    {
        'dirs':
            {
                'source_dir': str(Path(__file__).resolve().parent) + '\\',
            },

        'dataset':
            {
                'character': 'Aang2',
                'batch_size': 4,
                'samples_num': 10800,
                'angle_range': 25,
            },

        'training':
            {
                'lr': 0.001,
                'epochs': 30,
                'kernel_sizes': [51, 35, 25, 17, 13, 9, 7, 5, 3],
                'lambda': 0.005,
                'alpha': 0.05,
                'base_model': '17-11-2021 16-25-13 (Aang2 25 range)',
            },

        'inspection':
            {
                'num_iter_to_print': 50,
            },

        'network':
            {
                'weight_scaling': 0.5,
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
