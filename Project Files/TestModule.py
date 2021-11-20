# import ImageGenerator
# import TorchLearner
from PIL import ImageOps
from TorchLearner import device, imshow, imsave, create_net, normalize_image, create_canonical
import PIL.Image as Image
import os
import numpy as np
import torch
from Config import config


def main():
    path = config['dirs']['source_dir'] + 'Plots\\' + config['training']['base_model']
    net = create_net(path, 1)
    for filename in os.listdir(config['dirs']['source_dir'] + 'Test Inputs\\Images'):
        if filename.endswith(".png"):
            new_im = Image.new("RGBA", (128, 128))
            im = Image.open(config['dirs']['source_dir'] + 'Test Inputs\\Images\\' + filename)
            alpha = ImageOps.invert(im.split()[-1])
            im = Image.composite(new_im, im, alpha)
            im = np.array(im).astype('uint8')
            im = (im - 127.5) / 127.5
            im = torch.tensor(np.array([im], dtype='float64'))
            canonical = create_canonical(net.character, batch=1)
            with torch.no_grad():
                im = im.to(device)
                im = im.permute(0, 3, 1, 2)
                im = normalize_image(im, -1, 1)
                outputs, _, _ = net(torch.cat([im, canonical], dim=2))
                im = im.permute(0, 2, 3, 1)
                image = (torch.cat([im[i] for i in range(1)], dim=1))
                output = (torch.cat([outputs[i] for i in range(1)], dim=2)).permute(1, 2, 0)
                path = 'Test Outputs\\Images'
                imsave(image.cpu(), filename[:-4] + "-input", path)
                imsave(output.cpu(), filename[:-4] + "-output", path)
                imshow(image.cpu(), "input")
                imshow(output.cpu(), "output")
        else:
            continue


if __name__ == '__main__':
    main()
