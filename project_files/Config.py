from pathlib import Path

config = \
    {
        'dirs':
            {
                'source_dir': str(Path(__file__).resolve().parent) + '\\',
                'comment': 'Aang hands all poses'
            },

        'dataset':
            {
                'character': 'Goofy',
                'name': 'Aang hands all poses',
                'batch_size': 16,
                'samples_num': 20000 * 0.8,
                'test_samples_num': 1000 * 0.2,
                'angle_range': 80,
                'scaling_range': [0.75, 1/0.75],
                'translation_range': [-2, 2],
                'center_range': [-20, 20],
                'max_layer_swaps': 4,
                'limbs': [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [2, 6], [3, 7], [4, 8], [5, 9], [6, 10], [7, 11],
                          [8, 12], [9, 13]],
            },

        'training':
            {
                'lr': 1e-04,
                'decay': 0,
                'gamma': 0.9,
                'epochs': 15,
                'supervised_loss': 1.0,
                'base_model': None,
            },

        'transformation':
            {
                'noise': [0.2, 0.2],
                'blur_kernels': [7, 7],
                'blur_kernels_sigmas': [5, 5],
            },

        'inspection':
            {
                'num_iter_to_print': 25,
            },

        'network':
            {
                'architecture':
                    [
                        {'type': 'conv', 'out_channels': 16, 'activation': 'relu', 'kernel': 8, 'stride': 2},
                        {'type': 'batch_norm', 'activation': None},
                        {'type': 'conv', 'out_channels': 32, 'activation': 'relu', 'kernel': 6, 'stride': 1},
                        {'type': 'batch_norm', 'activation': None},
                        {'type': 'conv', 'out_channels': 64, 'activation': 'relu', 'kernel': 3, 'stride': 1},
                        {'type': 'pooling', 'stride': 2},
                        {'type': 'batch_norm', 'activation': None},
                        {'type': 'conv', 'out_channels': 64, 'activation': 'relu', 'kernel': 3, 'stride': 1},
                        {'type': 'pooling', 'stride': 2},
                        {'type': 'batch_norm', 'activation': None},
                        {'type': 'flatten'},
                        {'type': 'fc', 'out_parameters': 512, 'activation': 'relu'},
                        {'type': 'fc', 'out_parameters': 128, 'activation': 'relu'},
                        {'type': 'fc_layers', 'out_parameters': 6, 'activation': 'relu'},
                        {'type': 'fc_layers', 'out_parameters': 6, 'activation': None},
                    ],
                'weight_scaling': 0.15,
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
