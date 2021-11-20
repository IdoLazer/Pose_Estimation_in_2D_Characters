from __future__ import print_function

import os
from datetime import datetime

import kornia.filters
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import ImageGenerator
import Config
from Config import config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())


def get_gaussian_kernel(kernel_size=3, sigma=18, channels=3, padding=0):  # Set these to whatever you want
    # for your gaussian filter

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * np.math.pi * variance)) * \
        torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depth-wise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=padding,
                                padding_mode='replicate')

    gaussian_filter.weight.data = gaussian_kernel.type(torch.cuda.DoubleTensor)
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


def get_sobel():
    a = np.array([[0.125, 0, -0.125], [0.25, 0, -0.25], [0.125, 0, -0.125]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='replicate').to(device)
    conv1.weight = nn.Parameter(torch.from_numpy(a).double().cuda().unsqueeze(0).unsqueeze(0))
    conv1.weight.requires_grad = False

    b = np.array([[0.125, 0.25, 0.125], [0, 0, 0], [-0.125, -0.25, -0.125]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='replicate').to(device)
    conv2.weight = nn.Parameter(torch.from_numpy(b).double().cuda().unsqueeze(0).unsqueeze(0))
    conv2.weight.requires_grad = False
    return conv1, conv2


class Net(nn.Module):
    def __init__(self, character, batch):
        super(Net, self).__init__()
        self.num_layers = len(character.char_tree_array)
        self.batch = batch
        self.conv1 = nn.Conv2d(4, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * (((((((character.image_size - 2) - 2) // 2) - 2) // 2) - 2) // 2) *
                             (((((((character.image_size - 2) - 2) // 2) - 2) // 2) - 2) // 2), 512)
        self.fc2 = nn.Linear(512, 3 * self.num_layers)
        self.fc3 = nn.Linear(3 * self.num_layers, 6 * self.num_layers)
        self.fc4 = nn.Linear(6 * self.num_layers, 6 * self.num_layers)
        torch.nn.init.xavier_uniform_(self.conv1.weight, config['network']['weight_scaling'])
        torch.nn.init.xavier_uniform_(self.conv2.weight, config['network']['weight_scaling'])
        torch.nn.init.xavier_uniform_(self.conv3.weight, config['network']['weight_scaling'])
        torch.nn.init.xavier_uniform_(self.conv4.weight, config['network']['weight_scaling'])
        torch.nn.init.xavier_uniform_(self.fc1.weight, config['network']['weight_scaling'])
        torch.nn.init.xavier_uniform_(self.fc2.weight, config['network']['weight_scaling'])
        torch.nn.init.xavier_uniform_(self.fc3.weight, config['network']['weight_scaling'])
        torch.nn.init.xavier_uniform_(self.fc4.weight, config['network']['weight_scaling'])
        with torch.no_grad():
            bias = []
            for part in character.char_tree_array:
                bias += character.canonical_bias_dict[part]
            self.fc4.bias = torch.nn.Parameter(torch.DoubleTensor(bias).view(6 * self.num_layers))
        self.character = character

    def localization(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    # Spatial transformer network forward function
    # def stn(self, x, body, head, left_leg, right_leg):
    def stn(self, x):
        affine_transforms = self.localization(x[0])
        parent_transforms_3x3_matrices = []
        close_to_eye_matrices = []
        translation_values = []
        part_layers = []
        part_layers_dict = dict()
        affine_matrix_last_row = torch.tensor(self.batch * [0, 0, 1]).view(-1, 1, 3).to(device)
        for i, part in enumerate(self.character.char_tree_array):
            part_transform = affine_transforms[:, i * 6: (i * 6) + 6].view(-1, 2, 3).to(device)
            part_transform[:, 0, 2] /= (self.character.image_size / 2)
            part_transform[:, 1, 2] /= (self.character.image_size / 2)
            rotation_scaling_matrix = part_transform[:, :, 0: 2]
            close_to_eye_matrices.append(
                (torch.matmul(rotation_scaling_matrix, torch.transpose(rotation_scaling_matrix, 1, 2))).view(-1, 4))
            translation_values.append(part_transform[:, :, 2])

            # TODO: uncomment if I want to impose a skeleton structure
            part_transform_3x3_matrix = torch.cat([part_transform, affine_matrix_last_row], 1)
            if i != 0:
                parent_transform_3x3_matrix = parent_transforms_3x3_matrices[self.character.parents[i]]
                part_transform_3x3_matrix = torch.matmul(part_transform_3x3_matrix, parent_transform_3x3_matrix)
                part_transform = part_transform_3x3_matrix[:, :2, :]
            parent_transforms_3x3_matrices.append(part_transform_3x3_matrix)
            part_grid = F.affine_grid(part_transform, x[i + 1].size(), align_corners=False)
            part_layer = F.grid_sample(x[i + 1], part_grid, align_corners=False, padding_mode='border')
            part_layers.append(part_layer)
            part_layers_dict[part] = i

        stack = None
        for i, part in enumerate(self.character.drawing_order):
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
                # imshow(stack[0].cpu().view(4, 128, 128).permute(1, 2, 0), '')
        stack = normalize_image(stack, -1, 1)

        # imshow(stack[0].cpu().view(4, 128, 128).permute(1, 2, 0), '')
        # imshow(x[0][0].cpu().view(4, 128, 128).permute(1, 2, 0), '')
        return stack, close_to_eye_matrices, translation_values

    def forward(self, x):
        x = list(torch.split(x, int(x.shape[2] / (self.num_layers + 1)), dim=2))
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
    canonical = create_canonical(ImageGenerator.char)
    optimizer = optim.Adam(net.parameters(), lr=training_conf['lr'])
    kernel_sizes = training_conf['kernel_sizes']
    num_iter_to_print = config['inspection']['num_iter_to_print']
    kernel_index = 0
    gaussian = kornia.filters.GaussianBlur2d((kernel_sizes[kernel_index], kernel_sizes[kernel_index]),
                                             (18, 18), 'replicate')
    running_losses = [0.0, 0.0, 0.0, 0.0]
    losses_arrays = [[], [], [], []]
    iterations = []
    test_losses = []
    for epoch in range(training_conf['epochs']):  # loop over the dataset multiple times
        grad_gaussian = kornia.filters.GaussianBlur2d((11, 11), (6, 6), 'replicate') if epoch < len(kernel_sizes) // 2 \
            else kornia.filters.GaussianBlur2d((5, 5), (3, 3), 'replicate')

        for i in tqdm(range(len(trainset))):
            if int(float((i + (epoch * len(trainset))) / float(training_conf['epochs']
                                                               * (len(trainset)) / len(kernel_sizes)))) > kernel_index:
                save_losses_graphs(colors, iterations, kernel_sizes[kernel_index], losses_arrays, path)
                kernel_index = int(float((i + (epoch * len(trainset))) / float(training_conf['epochs']
                                                                               * (len(trainset)) / len(kernel_sizes))))
                gaussian = kornia.filters.GaussianBlur2d((kernel_sizes[kernel_index], kernel_sizes[kernel_index]),
                                                         (18, 18), 'replicate')
                running_losses = [0.0, 0.0, 0.0, 0.0]
                losses_arrays = [[], [], [], []]
                iterations = []
            data = trainset[i]
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = normalize_image(inputs, -1, 1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # get net output
            outputs, close_to_eye_matrices, translation_values = net(torch.cat([inputs, canonical], dim=2))

            # separate inputs outputs to colors and alpha
            inputs_alpha = inputs[:, 3, :, :]
            outputs_alpha = outputs[:, 3, :, :]
            inputs = inputs[:, :3, :, :]
            outputs = outputs[:, :3, :, :]

            # calc loss
            alpha_modified = training_conf['alpha']   # + (i / len(trainset))
            # * (i / len(trainset)) * (i / len(trainset)) * (0.01 - training_conf['alpha'])
            losses = calc_losses(alpha_modified, close_to_eye_matrices, criterion, gaussian, grad_gaussian,
                                 inputs, inputs_alpha, kernel_sizes[kernel_index], outputs, outputs_alpha,
                                 translation_values)
            losses[0].backward()
            optimizer.step()

            running_losses[0] += losses[0].item()
            running_losses[1] += losses[1].item()
            running_losses[2] += losses[2].item()
            running_losses[3] += losses[3].item()

            if i == 0 or i == len(trainset) - 1:
                with torch.no_grad():
                    save_batch(epoch, i, inputs, outputs, path)
                    save_inspection_image(canonical, epoch, i, inspection_path, net, kernel_sizes[kernel_index],
                                          gaussian, grad_gaussian)
            if i % num_iter_to_print == (
                    config['inspection']['num_iter_to_print']) - 1:
                append_losses(alpha_modified, epoch, i, iterations, losses_arrays, num_iter_to_print, running_losses)
                running_losses = [0.0, 0.0, 0.0, 0.0]
                with torch.no_grad():
                    save_inspection_image(canonical, epoch, i, inspection_path, net, kernel_sizes[kernel_index],
                                          gaussian, grad_gaussian)

        test_loss = test(net, path)
        print("test loss epoch %d = %.9f" % (epoch + 1, test_loss))
        test_losses.append(test_loss)

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


def create_canonical(character, batch=config['dataset']['batch_size']):
    num_layers = len(character.char_tree_array)
    canonical = ImageGenerator.generate_layers(character, [0] * num_layers, as_tensor=True, transform=False)
    canonical = torch.cat(batch * [canonical]).reshape(
        batch,
        -1,
        character.image_size,
        4).to(device)
    canonical = canonical.permute(0, 3, 1, 2)
    return normalize_image(canonical, -1, 1)


def create_folders(net):
    current_time = datetime.now()
    path = config['dirs']['source_dir'] + 'Plots\\' + current_time.strftime("%d-%m-%Y %H-%M-%S")
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


def save_losses_graphs(colors, iterations, kernel_size, losses, path):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Training Loss with gaussian kernel of size = %d" % kernel_size)
    ax1.plot(iterations, losses[0], color=colors[0])
    ax1.set_title('Final Loss')
    ax1.set(xlabel='iteration', ylabel='loss')
    ax2.plot(iterations, losses[1], color=colors[1])
    ax2.set_title('Gaussian Loss')
    ax2.set(xlabel='iteration', ylabel='loss')
    ax3.plot(iterations, losses[2], color=colors[2])
    ax3.set_title('Depth Loss')
    ax3.set(xlabel='iteration', ylabel='loss')
    ax4.plot(iterations, losses[3], color=colors[3])
    ax4.set_title('Orthogonal Loss')
    ax4.set(xlabel='iteration', ylabel='loss')
    fig.tight_layout()
    iteration = 0 if len(iterations) == 0 else iterations[-1]
    plt.savefig(path + '\\Training Loss iter %d kernel size = %d.png' % (iteration, kernel_size))
    plt.close(fig)


def save_inspection_image(canonical, epoch, i, inspection_path, net, kernel_size, gaussian, grad_gaussian):
    # parse inputs and outputs
    inspection_input = trainset[0][0].to(device)
    inspection_input = inspection_input.permute(0, 3, 1, 2)
    inspection_output = net(torch.cat([inspection_input, canonical], dim=2))[0]
    inspection_input_alpha = inspection_input[:, 3, :, :]
    inspection_output_alpha = inspection_output[:, 3, :, :]
    inspection_input = inspection_input[:, :3, :, :]
    inspection_output = inspection_output[:, :3, :, :]

    # calc gaussian loss image
    inspection_gaussian_loss_image = calc_gaussian_loss(torch.nn.MSELoss(reduction='none'), gaussian,
                                                        inspection_input, inspection_input_alpha, kernel_size,
                                                        inspection_output, inspection_output_alpha)
    inspection_gaussian_loss_image = normalize_image(
        inspection_gaussian_loss_image, -1, torch.min(torch.tensor([torch.max(inspection_gaussian_loss_image), 1])))

    # calc depth loss image
    inspection_depth_loss_image = calc_depth_loss(grad_gaussian, torch.nn.MSELoss(reduction='none'),
                                                  inspection_input, inspection_output)
    inspection_depth_loss_image = kornia.color.grayscale_to_rgb(inspection_depth_loss_image)
    inspection_depth_loss_image = normalize_image(inspection_depth_loss_image, -1,
                                                  torch.min(torch.tensor([torch.max(inspection_depth_loss_image), 1])))

    # concat and save all images
    inspection = torch.cat((inspection_input[0].permute(1, 2, 0).cpu(), inspection_output[0].permute(1, 2, 0).cpu()))
    inspection_losses = torch.cat((inspection_gaussian_loss_image[0].permute(1, 2, 0).cpu(),
                                   inspection_depth_loss_image[0].permute(1, 2, 0).cpu()))
    imsave(torch.cat([inspection, inspection_losses], dim=1), "epoch %d iter %d" % (epoch, i),
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
        alpha = (alpha - torch.min(alpha)) * ((b - a) / (torch.max(alpha) - torch.min(alpha))) + a
        image = image[:, :3, :, :]
        image = (image - torch.min(image)) * ((b - a) / (torch.max(image) - torch.min(image))) + a
        image = torch.cat([image, alpha], dim=1)
    else:
        image = (image - torch.min(image)) * ((b - a) / (torch.max(image) - torch.min(image))) + a
    return image


def append_losses(alpha_modified, epoch, i, iterations, losses, num_iter_to_print, running_losses):
    losses[0].append(running_losses[0] / num_iter_to_print)
    losses[1].append(((1 - config['training']['lambda'])
                      * (1 - alpha_modified)) * running_losses[1] / num_iter_to_print)
    losses[2].append(((1 - config['training']['lambda']) * alpha_modified) * running_losses[2] / num_iter_to_print)
    losses[3].append(config['training']['lambda'] * running_losses[3] / num_iter_to_print)
    iterations.append(epoch * len(trainset) + i + 1)
    print(
        '[%d, %5d] loss: %.9f, gaussian loss: %.9f, depth loss: %.9f, orthogonal loss: %.9f, alpha modified: %.9f' %
        (epoch + 1, i + 1, running_losses[0] / num_iter_to_print,
         ((1 - config['training']['lambda']) * (1 - alpha_modified)) * running_losses[1] / num_iter_to_print,
         ((1 - config['training']['lambda']) * alpha_modified) * running_losses[2] / num_iter_to_print,
         config['training']['lambda'] * running_losses[3] / num_iter_to_print,
         alpha_modified))


def save_batch(epoch, i, inputs, outputs, path):
    input_batch = (torch.cat([inputs[i] for i in range(config['dataset']['batch_size'])],
                             dim=2)).permute(1, 2, 0).cpu()
    output_batch = (torch.cat([outputs[i] for i in range(config['dataset']['batch_size'])],
                              dim=2)).permute(1, 2, 0).cpu()
    imsave(torch.cat([input_batch, output_batch], dim=0),
           "input (up) vs output (down)- epoch %d iteration %d" % (epoch + 1, i,), path)


def calc_losses(alpha_modified, close_to_eye_matrices, criterion, gaussian, grad_gaussian, inputs, inputs_alpha,
                kernel_size, outputs, outputs_alpha, translation_values):
    orthogonal_loss = calc_orthogonal_loss(close_to_eye_matrices, criterion)
    translation_loss = calc_translation_loss(translation_values)
    depth_loss = calc_depth_loss(grad_gaussian, criterion, inputs, outputs)
    gaussian_loss = calc_gaussian_loss(criterion, gaussian, inputs, inputs_alpha, kernel_size, outputs,
                                       outputs_alpha)
    loss = (1 - config['training']['lambda']) * (alpha_modified * depth_loss + (1 - alpha_modified) * gaussian_loss) + \
        config['training']['lambda'] * (translation_loss + orthogonal_loss)
    return loss, gaussian_loss, depth_loss, orthogonal_loss


def calc_gaussian_loss(criterion, gaussian, inputs, inputs_alpha, kernel_size, outputs, outputs_alpha):
    inputs_dichotomized = inputs + ((kernel_size / 7) * inputs_alpha.unsqueeze(1))
    outputs_dichotomized = outputs + ((kernel_size / 7) * outputs_alpha.unsqueeze(1))
    gaussian_inputs = normalize_image(gaussian(inputs_dichotomized), -1, 1)
    gaussian_outputs = normalize_image(gaussian(outputs_dichotomized), -1, 1)
    gaussian_loss = criterion(gaussian_outputs, gaussian_inputs)
    return gaussian_loss


def calc_depth_loss(grad_gaussian, criterion, inputs, outputs):
    inputs_depth_map = normalize_image(grad_gaussian(kornia.filters.canny(inputs)[1]), -1, 1)
    outputs_depth_map = normalize_image(grad_gaussian(kornia.filters.canny(outputs)[1]), -1, 1)
    depth_loss = criterion(inputs_depth_map, outputs_depth_map)
    return depth_loss


def calc_translation_loss(translation_values):
    translation_loss = 0
    ones = 0.8 * torch.ones(config['dataset']['batch_size'] * 2).double().to(device)
    for translation_value in translation_values:
        translation_loss += \
            torch.sum(torch.relu(torch.abs(translation_value.reshape(config['dataset']['batch_size'] * 2)) - ones))
    return translation_loss


def calc_orthogonal_loss(close_to_eye_matrices, criterion):
    orthogonal_loss = 0
    eye = torch.tensor(config['dataset']['batch_size'] * [1, 0, 0, 1]).double().to(device).view(-1, 4)
    for close_to_eye_matrix in close_to_eye_matrices:
        orthogonal_loss += criterion(close_to_eye_matrix, eye)
    return orthogonal_loss


def append_inputs_outpus(g_inputs, g_outputs, input_images, output_images):
    image = (torch.cat([g_inputs[i] for i in range(config['dataset']['batch_size'])], dim=2)).permute(1, 2, 0).cpu()
    output = (torch.cat([g_outputs[i] for i in range(config['dataset']['batch_size'])], dim=2)).permute(1, 2, 0).cpu()
    input_images.append(image)
    output_images.append(output)


def check_inputs_outputs(g_inputs, g_outputs, iteration):
    image = (torch.cat([g_inputs[i] for i in range(config['dataset']['batch_size'])], dim=2)).permute(1, 2, 0).cpu()
    output = (torch.cat([g_outputs[i] for i in range(config['dataset']['batch_size'])], dim=2)).permute(1, 2, 0).cpu()
    imshow(torch.cat([image, output], dim=0),
           "input (up) vs output (down): iteration %d " % (iteration,), cmap='gray')


def test(net, path):
    canonical = create_canonical(net.character)
    with torch.no_grad():
        criterion = nn.MSELoss()
        gaussian = kornia.filters.GaussianBlur2d((3, 3), (18, 18), 'replicate')
        grad_gaussian = kornia.filters.GaussianBlur2d((5, 5), (3, 3), 'replicate')
        losses = []
        for data in testset:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = normalize_image(inputs, -1, 1)
            # zero the parameter gradients

            # get net output
            outputs, close_to_eye_matrices, translation_values = net(torch.cat([inputs, canonical], dim=2))

            # separate inputs outputs to colors and alpha
            inputs_alpha = inputs[:, 3, :, :]
            outputs_alpha = outputs[:, 3, :, :]
            inputs = inputs[:, :3, :, :]
            outputs = outputs[:, :3, :, :]

            # calc loss
            alpha_modified = config['training']['alpha']  # + (i / len(trainset))
            # * (i / len(trainset)) * (i / len(trainset)) * (0.01 - training_conf['alpha'])
            loss, _, _, _ = calc_losses(alpha_modified, close_to_eye_matrices, criterion, gaussian, grad_gaussian,
                                        inputs, inputs_alpha, 3, outputs, outputs_alpha, translation_values)
            losses.append(loss)
        test_loss = np.average(loss.cpu())
        return test_loss


def showcase(net, path):
    canonical = create_canonical(net.character)
    with torch.no_grad():
        data = testset[0]
        images, labels = data[0].to(device), data[1].to(device)
        images = images.permute(0, 3, 1, 2)
        images = normalize_image(images, -1, 1)
        outputs, _, _ = net(torch.cat([images, canonical], dim=2))
        images = images.permute(0, 2, 3, 1)
        image = (torch.cat([images[i] for i in range(config['dataset']['batch_size'])], dim=1))
        output = (torch.cat([outputs[i] for i in range(config['dataset']['batch_size'])], dim=2)).permute(1, 2, 0)
        imsave(image.cpu(), "input", path)
        imsave(output.cpu(), "output", path)
        imshow(image.cpu(), "input")
        imshow(output.cpu(), "output")


def main():
    net, path = train()
    showcase(net, path)


if __name__ == '__main__':
    trainset, testset = ImageGenerator.load_data(batch_size=config['dataset']['batch_size'],
                                                 samples_num=config['dataset']['samples_num'],
                                                 angle_range=config['dataset']['angle_range'])
    main()
