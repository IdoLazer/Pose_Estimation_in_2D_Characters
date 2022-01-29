from pathlib import Path

config = \
    {
        'dirs':
            {
                'source_dir': str(Path(__file__).resolve().parent) + '\\',
                'comment': 'Aang with extra layers'
            },

        'dataset':
            {
                'character': 'Aang2',
                'name': 'samples=100000 angle_range=60',
                'batch_size': 32,
                'samples_num': 100000,
                'angle_range': 60,
            },

        'training':
            {
                'lr': 0.0001,
                'epochs': 30,
                'supervised_loss': 1.0,
                'base_model': None,
            },

        'transformation':
            {
                'noise': 0.2,
                'blur_kernels': [3, 5, 7, 9, 13],
            },

        'inspection':
            {
                'num_iter_to_print': 50,
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
