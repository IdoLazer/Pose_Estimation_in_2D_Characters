from pathlib import Path

config = \
    {
        'dirs':
            {
                'source_dir': str(Path(__file__).resolve().parent) + '\\',
                'comment': 'unsupervised loss and colored layers'
            },

        'dataset':
            {
                'character': 'Aang2',
                'name': 'samples=150000 angle_range=80',
                'batch_size': 32,
                'samples_num': 150000,
                'test_samples_num': 150000,
                'angle_range': 80,
                'scaling_range': [0.85, 1.15],
                'translation_range': [-2, 2],
                'max_layer_swaps': 2,
            },

        'training':
            {
                'lr': 0.0001,
                'epochs': 10,
                'supervised_loss': 0.5,
                'base_model': None,
            },

        'transformation':
            {
                'noise': [0.1, 0.3],
                'blur_kernels': [3, 5, 7],
                'blur_kernels_sigmas': [3, 3, 5],
            },

        'inspection':
            {
                'num_iter_to_print': 100,
            },

        'network':
            {
                'architecture':
                    [
                        {'type': 'conv', 'out_channels': 32, 'activation': 'relu', 'kernel': 6, 'stride': 2},
                        {'type': 'conv', 'out_channels': 64, 'activation': 'relu', 'kernel': 3, 'stride': 1},
                        {'type': 'pooling', 'stride': 2},
                        {'type': 'conv', 'out_channels': 64, 'activation': 'relu', 'kernel': 3, 'stride': 1},
                        {'type': 'pooling', 'stride': 2},
                        {'type': 'conv', 'out_channels': 64, 'activation': 'relu', 'kernel': 3, 'stride': 1},
                        {'type': 'pooling', 'stride': 2},
                        {'type': 'flatten'},
                        {'type': 'fc', 'out_parameters': 512, 'activation': 'relu'},
                        {'type': 'fc', 'out_parameters': 128, 'activation': 'relu'},
                        {'type': 'fc_layers', 'out_parameters': 6, 'activation': 'relu'},
                        {'type': 'fc_layers', 'out_parameters': 6, 'activation': None},
                    ],
                'weight_scaling': 0.3,
            },

    }


def serialize():
    conf_str = ""
    for category in config.keys():
        conf_str += category + ":\n\t{\n"
        for key in config[category].keys():
            if key == 'architecture':
                conf_str += "\t\t[\n"
                for item in config[category][key]:
                    conf_str += f"\t\t\t{item}\n"
                conf_str += "\t\t]\n"
            else:
                conf_str += f"\t\t{key}: {config[category][key]}\n"
        conf_str += "\t}\n"
    return conf_str
