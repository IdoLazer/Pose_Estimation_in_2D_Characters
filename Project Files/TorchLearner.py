import os
from datetime import datetime

from PIL import ImageOps
import PIL.Image as Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import kornia.filters

import DataModule
import ImageGenerator
import Config
from Config import config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())

inputs_gaussian_blur = [kornia.filters.GaussianBlur2d((i, i), (j, j), 'replicate').to(device)
                        for i, j in
                        zip(config['transformation']['blur_kernels'], config['transformation']['blur_kernels_sigmas'])]
input_gaussian_blur = inputs_gaussian_blur[0]
noise_scale = config['transformation']['noise'][0]


class Net(nn.Module):
    def __init__(self, character, batch):
        super(Net, self).__init__()
        self.num_character_layers = len(character.char_tree_array)
        self.character = character
        self.batch = batch

        # init network layers
        self.layers = []
        curr_im_size = character.image_size
        channels = 4
        parameters = channels * curr_im_size * curr_im_size
        for i, layer_dict in enumerate(config['network']['architecture']):
            layer = None
            if layer_dict['type'] == 'conv':
                layer = nn.Conv2d(channels, layer_dict['out_channels'], layer_dict['kernel'], layer_dict['stride'])
                curr_im_size = ((curr_im_size - layer_dict['kernel']) / layer_dict['stride']) + 1
                channels = layer_dict['out_channels']
                parameters = channels * curr_im_size * curr_im_size
            elif layer_dict['type'] == 'fc':
                layer = nn.Linear(int(parameters), int(layer_dict['out_parameters']))
                parameters = layer_dict['out_parameters']
            elif layer_dict['type'] == 'fc_layers':
                layer = nn.Linear(int(parameters), int(self.num_character_layers * layer_dict['out_parameters']))
                parameters = self.num_character_layers * layer_dict['out_parameters']
            elif layer_dict['type'] == 'pooling':
                curr_im_size //= 2
                parameters = channels * curr_im_size * curr_im_size
                print(f"{parameters=}")

            if layer is not None:
                torch.nn.init.xavier_uniform_(layer.weight, config['network']['weight_scaling'])
                if i == len(config['network']['architecture']) - 1:
                    with torch.no_grad():
                        bias = []
                        for part in character.char_tree_array:
                            part_bias = character.canonical_bias_dict[part]
                            part_bias[2] /= (self.character.image_size / 2)
                            part_bias[5] /= (self.character.image_size / 2)
                            bias += part_bias
                        layer.bias = torch.nn.Parameter(torch.DoubleTensor(bias).
                                                        view(self.num_character_layers * layer_dict['out_parameters']))
                layer_dict['layer'] = layer
                self.add_module(str(i), layer)

            self.layers.append(layer_dict)
            print(f"{curr_im_size=}")
            print(f"{parameters=}")

        # k1, k2, k3, k4 = config['network']['kernels']
        # s1, s2, s3, s4 = config['network']['strides']
        # n_out1 = ((character.image_size - k1) / s1) + 1
        # n_out2 = (((n_out1 - k2) // s2) + 1) // 2
        # n_out3 = (((n_out2 - k3) // s3) + 1) // 2
        # n_out4 = (((n_out3 - k4) // s4) + 1) // 2
        # print(f"{n_out1=}")
        # print(f"{n_out2=}")
        # print(f"{n_out3=}")
        # print(f"{n_out4=}")
        # self.conv1 = nn.Conv2d(4, 32, k1, s1)
        # self.conv2 = nn.Conv2d(32, 64, k2, s2)
        # self.conv3 = nn.Conv2d(64, 64, k3, s3)
        # self.conv4 = nn.Conv2d(64, 64, k4, s4)
        # self.fc1 = nn.Linear(int(64 * n_out4 * n_out4), 512)
        # self.fc2 = nn.Linear(512, 6 * self.num_layers)
        # self.fc3 = nn.Linear(6 * self.num_layers, 6 * self.num_layers)
        # self.fc4 = nn.Linear(6 * self.num_layers, 6 * self.num_layers)
        # torch.nn.init.xavier_uniform_(self.conv1.weight, config['network']['weight_scaling'])
        # torch.nn.init.xavier_uniform_(self.conv2.weight, config['network']['weight_scaling'])
        # torch.nn.init.xavier_uniform_(self.conv3.weight, config['network']['weight_scaling'])
        # torch.nn.init.xavier_uniform_(self.conv4.weight, config['network']['weight_scaling'])
        # torch.nn.init.xavier_uniform_(self.fc1.weight, config['network']['weight_scaling'])
        # torch.nn.init.xavier_uniform_(self.fc2.weight, config['network']['weight_scaling'])
        # torch.nn.init.xavier_uniform_(self.fc3.weight, config['network']['weight_scaling'])
        # torch.nn.init.xavier_uniform_(self.fc4.weight, config['network']['weight_scaling'])

    def localization(self, x):
        for layer_dict in self.layers:
            if layer_dict['type'] in ['conv', 'fc', 'fc_layers']:
                x = layer_dict['layer'](x)
                if layer_dict['activation'] is not None:
                    if layer_dict['activation'] == 'relu':
                        x = F.relu(x)
            elif layer_dict['type'] == 'pooling':
                x = F.max_pool2d(x, (layer_dict['stride'], layer_dict['stride']))
            elif layer_dict['type'] == 'flatten':
                x = torch.flatten(x, start_dim=1)
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        # # x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        # x = torch.flatten(x, start_dim=1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)
        return x

    # Spatial transformer network forward function
    # def stn(self, x, body, head, left_leg, right_leg):
    def stn(self, x):
        affine_transforms = self.localization(x)
        parent_transforms_3x3_matrices = []
        # close_to_eye_matrices = []
        # translation_values = []
        # part_layers = []
        part_transforms = []
        # part_layers_dict = dict()
        affine_matrix_last_row = torch.tensor(x.shape[0] * [0, 0, 1]).view(-1, 1, 3).to(device)
        for i, part in enumerate(self.character.char_tree_array):
            part_transform = affine_transforms[:, i * 6: (i * 6) + 6].view(-1, 2, 3)
            # part_transform[:, 0, 2] /= (self.character.image_size / 2)
            # part_transform[:, 1, 2] /= (self.character.image_size / 2)
            rotation_scaling_matrix = part_transform[:, :, 0: 2]
            # close_to_eye_matrices.append(
            #     (torch.matmul(rotation_scaling_matrix, torch.transpose(rotation_scaling_matrix, 1, 2))).view(-1, 4))
            # translation_values.append(part_transform[:, :, 2])

            # TODO: uncomment if I want to impose a skeleton structure
            part_transform_3x3_matrix = torch.cat([part_transform, affine_matrix_last_row], 1)
            if i != 0:
                parent_transform_3x3_matrix = parent_transforms_3x3_matrices[self.character.parents[i]]
                part_transform_3x3_matrix = torch.matmul(part_transform_3x3_matrix, parent_transform_3x3_matrix)
                part_transform = part_transform_3x3_matrix[:, :2, :]
            parent_transforms_3x3_matrices.append(part_transform_3x3_matrix)
            # part_grid = F.affine_grid(part_transform, x[i + 1].size(), align_corners=False)
            # part_layer = F.grid_sample(x[i + 1], part_grid, align_corners=False, padding_mode='border')
            # part_layers.append(part_layer)
            # part_layers_dict[part] = i
            part_transforms.append(torch.unsqueeze(torch.flatten(part_transform, start_dim=1), dim=0))

        return torch.transpose(torch.cat(part_transforms), 0, 1)

    def forward(self, x):
        # x = list(torch.split(x, int(x.shape[2] / (self.num_character_layers + 1)), dim=2))
        x = self.stn(x)
        return x

    def serialize(self):
        net_str = ""
        for state in self.state_dict().keys():
            net_str += state + str(self.state_dict()[state].shape) + "\n"
        return net_str


def imshow(img, title, cmap=None):
    plt.clf()
    img = img / 2 + 0.5  # unnormalize
    npimg = img.detach().numpy()
    plt.imshow(npimg, cmap=cmap)
    plt.title(title)
    plt.show()


def imsave(img, title, path):
    plt.clf()
    img = img / 2 + 0.5  # unnormalize
    npimg = img.detach().numpy()
    plt.imshow(npimg)
    plt.title(title)
    plt.savefig(path + '\\' + title + '.png')


def train():
    training_conf = config['training']
    if training_conf['base_model'] is None:
        net_path = None
    else:
        net_path = config['dirs']['source_dir'] + 'Plots\\' + training_conf['base_model']

    net = create_net(net_path)
    inspection_path, path = create_folders(net)
    colors = ['blue', 'red', 'orange', 'green', 'pink', 'purple', 'yellow', 'black', 'brown']
    criterion = nn.MSELoss()
    blurred_canonical = create_canonical(ImageGenerator.char,
                                         blur=kornia.filters.GaussianBlur2d((7, 7), (7, 7), 'replicate').to(device),
                                         colored=True)
    canonical = create_canonical(ImageGenerator.char)
    optimizer = optim.Adam(net.parameters(), lr=training_conf['lr'])
    num_iter_to_print = config['inspection']['num_iter_to_print']
    iterations = []
    test_losses = []
    losses_array = []

    new_im = Image.new("RGBA", (128, 128))
    inspection_images = []
    for i in range(5):
        im = Image.open(config['dirs']['source_dir'] + f"Test Inputs\\Images\\Pose{i+1}.png")
        alpha = ImageOps.invert(im.split()[-1])
        im = Image.composite(new_im, im, alpha)
        im = np.array(im).astype('uint8')
        im = (im - 127.5) / 127.5
        im = torch.tensor(im).double()
        im = im.to(device)
        inspection_images.append(im.permute(2, 0, 1))

    torch.autograd.set_detect_anomaly(True)
    gaussian_blur_idx = 0
    for epoch in range(training_conf['epochs']):  # loop over the dataset multiple times
        global input_gaussian_blur, noise_scale
        if int((epoch / training_conf['epochs']) * len(inputs_gaussian_blur)) > gaussian_blur_idx:
            gaussian_blur_idx += 1
            input_gaussian_blur = inputs_gaussian_blur[gaussian_blur_idx]
            save_losses_graphs(colors, iterations, losses_array, path)
            losses_array = []
            iterations = []
            optimizer = optim.Adam(net.parameters(), lr=training_conf['lr'])

        noise_scale = config['transformation']['noise'][0] + \
            (((epoch + 1) / training_conf['epochs'])
                * config['transformation']['noise'][1] - config['transformation']['noise'][0])
        running_loss = []
        trainset = dataset.get_train_dataloader(batch_size=config['dataset']['batch_size'], shuffle=True)

        for i in tqdm(range(np.min(
                [len(trainset),
                 config['dataset']['samples_num'] // config['dataset']['batch_size']]
        ))):

            data = next(iter(trainset))
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # get net output
            transforms = net(inputs)
            inputs = inputs[:, :3, :, :]

            # calc loss
            loss = calc_losses(criterion, labels, transforms, blurred_canonical)
            loss.backward()
            optimizer.step()
            running_loss += [loss.item()]

            if i == 0 or i == len(trainset) - 1:
                with torch.no_grad():
                    save_batch(epoch, i, inputs, transforms, canonical, path)
                    save_inspection_image(inspection_images, canonical, epoch, i, inspection_path, net)
            if i % num_iter_to_print == (
                    config['inspection']['num_iter_to_print']) - 1:
                if len(running_loss) < num_iter_to_print:
                    running_loss = []
                else:
                    append_losses(epoch,
                                  np.min([len(trainset),
                                          config['dataset']['samples_num'] // config['dataset']['batch_size']]),
                                  i, iterations, losses_array,
                                  num_iter_to_print, running_loss)
                    running_loss = []
                    with torch.no_grad():
                        save_inspection_image(inspection_images, canonical, epoch, i, inspection_path, net)

        # if (epoch + 1) % 5 == 0:
        #     save_losses_graphs(colors, iterations, losses_array, path)
        #     losses_array = []
        #     iterations = []
        #     optimizer = optim.Adam(net.parameters(), lr=training_conf['lr'])
        test_loss = test(net, path)
        print("test loss epoch %d = %.9f" % (epoch + 1, test_loss))
        test_losses.append(test_loss)

    save_losses_graphs(colors, iterations, losses_array, path)

    fig = plt.figure()
    plt.title("test losses for %d epochs" % (len(test_losses)))
    plt.plot([i for i in range(len(test_losses))], test_losses)
    plt.savefig(path + '\\Test Losses.png')
    plt.close(fig)
    print('Finished Training')
    torch.save(net.state_dict(), path + '\\aaa_net.pth')
    return net, path


def create_net(net_path, batch=config['dataset']['batch_size']):
    net = Net(ImageGenerator.char, batch).double()
    net.to(device)
    if net_path:
        net.load_state_dict(torch.load(net_path + '\\aaa_net.pth'))
    return net


def create_canonical(character, batch=config['dataset']['batch_size'], blur=None, colored=False):
    canonical, _ = ImageGenerator.generate_layers(character, None, as_tensor=True, transform=False, colored=colored)
    canonical = torch.cat(batch * [canonical]).reshape(
        batch,
        -1,
        character.image_size,
        4).to(device)
    canonical = canonical.permute(0, 3, 1, 2)
    if blur is not None:
        canonical = blur(canonical)
    canonical = normalize_image(canonical, -1, 1)
    return list(torch.split(canonical, int(canonical.shape[2] / len(ImageGenerator.char.char_tree_array)), dim=2))


def create_folders(net):
    current_time = datetime.now()
    path = config['dirs']['source_dir'] + 'Plots\\' + current_time.strftime("%d-%m-%Y %H-%M-%S")
    if config['dirs']['comment'] is not None:
        path += ' ' + config['dirs']['comment']
    inspection_path = path + '\\inspection'
    try:
        os.mkdir(path)
        os.mkdir(inspection_path)
        conf_file = open(path + "\\config.txt", "w")
        conf_file.write(Config.serialize() + "\n" + net.serialize())
        conf_file.close()
    except OSError:
        print("Creation of the directory %s failed" % path)
    return inspection_path, path


def save_losses_graphs(colors, iterations, losses, path):
    fig = plt.figure()
    fig.suptitle("Training Loss")
    plt.plot(iterations, losses, color=colors[0])
    iteration = 0 if len(iterations) == 0 else iterations[-1]
    plt.savefig(path + '\\Training Loss iter %d.png' % (iteration,))
    plt.close(fig)


def save_inspection_image(images, canonical, epoch, i, inspection_path, net):
    # parse inputs and outputs
    inspections = []
    for image in images:
        inspection_input = im_transform(image)
        inspection_input = inspection_input.unsqueeze(0)

        transforms = net(inspection_input)
        inspection_output = compose_image(transforms, canonical)
        inspection_input = inspection_input[:, :3, :, :]
        inspection_output = inspection_output[:, :3, :, :]
        inspection = torch.cat((inspection_input[0].permute(1, 2, 0).cpu(),
                                inspection_output[0].permute(1, 2, 0).cpu()))
        inspections.append(inspection)

    # concat and save all images
    final = torch.cat([inspection for inspection in inspections], dim=1)
    imsave(final, f"{epoch=} {i=}",
           inspection_path)


def normalize_image(image, a, b):
    """

    :param image: a 4d-tensor with shape (batch_num, colors, width, height)
    :param a: new minimum value
    :param b: new maximum value
    :return:
    """
    if image.shape[1] == 4:
        alpha = image[:, 3, :, :].unsqueeze(1)
        alpha = (alpha - torch.min(alpha)) * ((b - a) / torch.max(
            torch.tensor([0.000001, (torch.max(image) - torch.min(image))]))) + a
        image = image[:, :3, :, :]
        image = (image - torch.min(image)) * ((b - a) / torch.max(
            torch.tensor([0.000001, (torch.max(image) - torch.min(image))]))) + a
        image = torch.cat([image, alpha], dim=1)
    else:
        image = (image - torch.min(image)) * ((b - a) / torch.max(
            torch.tensor([0.000001, (torch.max(image) - torch.min(image))]))) + a
    return image


def append_losses(epoch, len_trainset, i, iterations, losses, num_iter_to_print, running_loss):
    losses.append(np.sum(running_loss) / num_iter_to_print)
    iterations.append(epoch * len_trainset + i + 1)
    print(
        '[%d, %5d] loss: %.9f' %
        (epoch + 1, i + 1, np.sum(running_loss) / num_iter_to_print,))


def save_batch(epoch, i, inputs, transforms, canonical, path):
    input_batch = (torch.cat([inputs[i] for i in range(np.min([len(inputs), 4]))],
                             dim=2)).permute(1, 2, 0).cpu()

    outputs = compose_image(transforms, canonical)
    outputs = outputs[:, :3, :, :]
    output_batch = (torch.cat([outputs[i] for i in range(np.min([len(inputs), 4]))],
                              dim=2)).permute(1, 2, 0).cpu()
    imsave(torch.cat([input_batch, output_batch], dim=0),
           "input (up) vs output (down)- epoch %d iteration %d" % (epoch + 1, i,), path)


def compose_image(transforms, canonical):
    part_layers = []
    part_layers_dict = dict()
    affine_transforms = transforms.permute(1, 0, 2)
    for i, part in enumerate(ImageGenerator.char.char_tree_array):
        part_transform = affine_transforms[i].reshape((-1, 2, 3))
        part_layer = canonical[i][:part_transform.shape[0]]
        part_grid = F.affine_grid(part_transform, part_layer.size(), align_corners=False)
        part_layer = F.grid_sample(part_layer, part_grid, align_corners=False, padding_mode='border')
        part_layers.append(part_layer)
        part_layers_dict[part] = i

    stack = None
    for i, part in enumerate(ImageGenerator.char.drawing_order):
        part_layer = part_layers[part_layers_dict[part]]
        if i == 0:
            stack = part_layer
        else:
            # part_alpha = (part_layer[:, -1, :, :]).clamp(0, 1).view(batch_size, 1, ImageGenerator.IMAGE_SIZE,
            #                                                         ImageGenerator.IMAGE_SIZE)
            part_alpha = normalize_image((part_layer[:, -1, :, :]).unsqueeze(1), 0, 1)
            # imshow(part_alpha[0].cpu().view(1, 128, 128).permute(1, 2, 0), '')
            # imshow(stack[0].cpu().view(4, 128, 128).permute(1, 2, 0), '')
            stack = (stack * (1 - part_alpha))  # .clamp(-1)
            # imshow(stack[0].cpu().view(4, 128, 128).permute(1, 2, 0), '')
            # stack_alpha = (stack[:, -1, :, :]).clamp(0, 1)
            # part_layer = (part_layer - 2 * stack_alpha).clamp(-1, 1)
            stack = (stack + (part_layer * part_alpha))  # .clamp(-1)
    stack = normalize_image(stack, -1, 1)
    # imshow(stack[0].cpu().view(4, 128, 128).permute(1, 2, 0), '')
    return stack


def calc_losses(criterion, labels, transforms, canonical):
    supervised_loss = calc_supervised_loss(criterion, labels, transforms)
    unsupervised_loss = calc_unsupervised_loss(criterion, labels, transforms, canonical)
    loss = config['training']['supervised_loss'] * supervised_loss + \
           (1 - config['training']['supervised_loss']) * unsupervised_loss
    return loss


def calc_supervised_loss(criterion, labels, transforms):
    return criterion(labels, transforms)


def calc_unsupervised_loss(criterion, labels, transforms, canonical):
    output = compose_image(transforms, canonical)
    input = compose_image(labels, canonical)
    return criterion(input, output)


def append_inputs_outpus(g_inputs, g_outputs, input_images, output_images):
    image = (torch.cat([g_inputs[i] for i in range(g_inputs.shape[0])], dim=2)).permute(1, 2, 0).cpu()
    output = (torch.cat([g_outputs[i] for i in range(g_inputs.shape[0])], dim=2)).permute(1, 2, 0).cpu()
    input_images.append(image)
    output_images.append(output)


def check_inputs_outputs(g_inputs, g_outputs, iteration):
    image = (torch.cat([g_inputs[i] for i in range(g_inputs.shape[0])], dim=2)).permute(1, 2, 0).cpu()
    output = (torch.cat([g_outputs[i] for i in range(g_inputs.shape[0])], dim=2)).permute(1, 2, 0).cpu()
    imshow(torch.cat([image, output], dim=0),
           "input (up) vs output (down): iteration %d " % (iteration,), cmap='gray')


def test(net, path):
    canonical = create_canonical(net.character,
                                 blur=kornia.filters.GaussianBlur2d((7, 7), (7, 7), 'replicate').to(device),
                                 colored=True)
    with torch.no_grad():
        criterion = nn.MSELoss()
        losses = []
        testset = dataset.get_test_dataloader(batch_size=config['dataset']['batch_size'], shuffle=True)
        for i in tqdm(range(np.min(
                [len(testset),
                 config['dataset']['test_samples_num'] // config['dataset']['batch_size']]
        ))):
            data = next(iter(testset))
            inputs, labels = data[0], data[1]
            # zero the parameter gradients

            # get net output
            transforms = net(inputs)

            # calc loss
            loss = calc_losses(criterion, labels, transforms, canonical)
            losses.append(loss)

        test_loss = np.average(loss.cpu())
        return test_loss


def showcase(net, path):
    canonical = create_canonical(net.character)
    with torch.no_grad():
        data = next(iter(dataset.get_test_dataloader(batch_size=config['dataset']['batch_size'], shuffle=False)))
        images, labels = data[0], data[1]
        transforms = net(images[:4])
        outputs = compose_image(transforms, canonical)
        images = images.permute(0, 2, 3, 1)
        image = (torch.cat([images[i] for i in range(np.min([images.shape[0], 4]))], dim=1))
        output = (torch.cat([outputs[i] for i in range(np.min([images.shape[0], 4]))], dim=2)).permute(1, 2, 0)
        imsave(image.cpu(), "input", path)
        imsave(output.cpu(), "output", path)
        imshow(image.cpu(), "input")
        imshow(output.cpu(), "output")


def main():
    net, path = train()
    showcase(net, path)
    from TestModule import test_frames
    test_frames(path.split('\\')[-1])


def im_transform(im, local_noise_scale=0, gaussian=None):
    if gaussian is None:
        im = input_gaussian_blur(im.unsqueeze(0))[0]
    else:
        im = gaussian(im.unsqueeze(0))[0]
    if local_noise_scale == 0:
        local_noise_scale = noise_scale
    noise = np.random.uniform(-local_noise_scale, local_noise_scale, (3, im.shape[1], im.shape[2]))
    im[:3, :, :] += torch.tensor(noise).to(device)
    im = normalize_image(im, -1, 1)
    return im


if __name__ == '__main__':
    dataset = DataModule.get_dataset(transform=im_transform, device=device)
    main()