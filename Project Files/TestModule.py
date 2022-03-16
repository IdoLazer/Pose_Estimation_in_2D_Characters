# import ImageGenerator
# import TorchLearner
from PIL import ImageOps
import TorchLearner
import PIL.Image as Image
import os
import numpy as np
import torch
from Config import config


def test_frames(base_model=None):
    if base_model is None:
        base_model = config['training']['base_model']
    net_path = config['dirs']['source_dir'] + 'Plots\\' + base_model
    net = TorchLearner.create_net(net_path, 1)

    output_path = config['dirs']['source_dir'] + 'Test Outputs\\Images\\' + base_model
    try:
        os.mkdir(output_path)
    except OSError:
        print("Creation of the directory %s failed" % output_path)
    for filename in os.listdir(config['dirs']['source_dir'] + 'Test Inputs\\Images'):
        if filename.endswith(".png"):
            new_im = Image.new("RGBA", (128, 128))
            im = Image.open(config['dirs']['source_dir'] + 'Test Inputs\\Images\\' + filename)
            alpha = ImageOps.invert(im.split()[-1])
            im = Image.composite(new_im, im, alpha)
            im = np.array(im).astype('uint8')
            im = (im - 127.5) / 127.5
            im = torch.tensor(im).double()
            im = im.to(TorchLearner.device)
            im = im.permute(2, 0, 1)
            im = TorchLearner.im_transform(im, local_noise_scale=config['transformation']['noise'][1],
                                           gaussian=TorchLearner.inputs_gaussian_blur[
                                               -1])
            im = im.unsqueeze(0)
            canonical = TorchLearner.create_canonical(net.character, batch=1)
            with torch.no_grad():
                transforms = net(im)
                im = im.permute(0, 2, 3, 1)
                image = (torch.cat([im[i] for i in range(1)], dim=1))
                outputs = TorchLearner.compose_image(transforms, canonical)
                output = torch.cat([outputs[i] for i in range(1)], dim=2).permute(1, 2, 0)
                combined = torch.cat([image, output], dim=1)
                TorchLearner.imsave(combined.cpu(), filename[:-4] + "-combined", output_path)
                TorchLearner.imshow(combined.cpu(), "combined")
        else:
            continue


if __name__ == '__main__':
    test_frames('09-03-2022 19-32-56 checking only hands')
