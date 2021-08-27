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
import torchvision.transforms
from tqdm import tqdm

import HeatMethod as heat
import ImageGenerator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())
batch_size = 4
grayscale = torchvision.transforms.Grayscale().to(device)
grayscale.requires_grad_(False)

trainset, testset, char = ImageGenerator.load_data('Cartman', batch_size=4, samples_num=2000, angle_range=15)

# CANONICAL_BIAS_DICT = {  # Here we assume no skeleton structure so layers are not affected by each other
#     'Root' : [0.9961946980917455, 0.08715574274765817, 0.6972459419812653, -0.08715574274765817, 0.9961946980917455, 7.969557584733964],
# 'Upper Left Leg' : [0.9455185755993168, 0.3255681544571567, 5.520648702239894, -0.3255681544571567, 0.9455185755993168, -1.407826683411776],
# 'Lower Left Leg' : [0.984807753012208, -0.17364817766693036, 19.0856654378275, 0.17364817766693036, 0.984807753012208, -23.044201160605365],
# 'Upper Right Leg' : [0.981627183447664, -0.1908089953765448, -4.12768159616796, 0.1908089953765448, 0.981627183447664, -1.215262879010235],
# 'Lower Right Leg' : [0.981627183447664, 0.1908089953765448, -14.949320503350735, -0.1908089953765448, 0.981627183447664, -23.31268900434717],
# 'Chest' : [0.9961946980917455, -0.08715574274765817, -1.9127831856497777, 0.08715574274765817, 0.9961946980917455, 14.86321185581942],
# 'Head' : [0.8910065241883679, -0.45399049973954675, -14.833490715867047, 0.45399049973954675, 0.8910065241883679, 31.608065957937672],
# 'Left Shoulder' : [0.9876883405951378, 0.15643446504023087, 14.58771294208637, -0.15643446504023087, 0.9876883405951378, 25.07908056217115],
# 'Left Arm' : [0.9781476007338057, -0.20791169081775934, 15.608266394707787, 0.20791169081775934, 0.9781476007338057, 16.78760880281413],
# 'Right Shoulder' : [0.8571673007021123, -0.5150380749100542, -23.82036078287659, 0.5150380749100542, 0.8571673007021123, 19.382697353450855],
# 'Right Arm' : [0.6156614753256583, -0.7880107536067219, -27.184794599472447, 0.7880107536067219, 0.6156614753256583, -6.811905611424363],
# }
# CANONICAL_BIAS_DICT = {  # Here we assume a skeleton structure so layers are affected by each other
#     'Root': [0.9961946980917455, 0.08715574274765817, 0.01089446784345727 * (ImageGenerator.IMAGE_SIZE / 2),
#                     -0.08715574274765817, 0.9961946980917455, 0.1245243372614682 * (ImageGenerator.IMAGE_SIZE / 2)],
#     'Upper Left Leg': [0.9702957262759965, 0.24192189559966773, 0.04556411666535376 * (ImageGenerator.IMAGE_SIZE / 2),
#                        -0.24192189559966773, 0.9702957262759965, -0.1401871138782236 * (ImageGenerator.IMAGE_SIZE / 2)],
#     'Lower Left Leg': [0.8746197071393957, -0.48480962024633706, 0.21210420885777245 * (ImageGenerator.IMAGE_SIZE / 2),
#                        0.48480962024633706, 0.8746197071393957, -0.3826461218734856 * (ImageGenerator.IMAGE_SIZE / 2)],
#     'Upper Right Leg': [0.9612616959383189, -0.27563735581699916,
#                         -0.04064390051805627 * (ImageGenerator.IMAGE_SIZE / 2), 0.27563735581699916, 0.9612616959383189,
#                         -0.1416918804154929 * (ImageGenerator.IMAGE_SIZE / 2)],
#     'Lower Right Leg': [0.9271838545667874, 0.374606593415912, -0.1666711763028203 * (ImageGenerator.IMAGE_SIZE / 2),
#                         -0.374606593415912, 0.9271838545667874, -0.3708152128956338 * (ImageGenerator.IMAGE_SIZE / 2)],
#     'Chest': [0.984807753012208, -0.17364817766693033, -0.018992769432320505 * (ImageGenerator.IMAGE_SIZE / 2),
#               0.17364817766693033, 0.984807753012208, 0.10771334798571025 * (ImageGenerator.IMAGE_SIZE / 2)],
#     'Head': [0.9271838545667874, -0.374606593415912, -0.1170645604424725 * (ImageGenerator.IMAGE_SIZE / 2),
#              0.374606593415912, 0.9271838545667874, 0.28974495455212107 * (ImageGenerator.IMAGE_SIZE / 2)],
#     'Left Shoulder': [0.9702957262759965, 0.24192189559966773, 0.20074909227430696 * (ImageGenerator.IMAGE_SIZE / 2),
#                       -0.24192189559966773, 0.9702957262759965, 0.1592910232123637 * (ImageGenerator.IMAGE_SIZE / 2)],
#     'Left Arm': [0.9335804264972017, -0.35836794954530027, 0.1715156531587924 * (ImageGenerator.IMAGE_SIZE / 2),
#                  0.35836794954530027, 0.9335804264972017, -0.18521091719040977 * (ImageGenerator.IMAGE_SIZE / 2)],
#     'Right Shoulder': [0.898794046299167, -0.4383711467890774, -0.24352436589920068 * (ImageGenerator.IMAGE_SIZE / 2),
#                        0.4383711467890774, 0.898794046299167, 0.10722249980014562 * (ImageGenerator.IMAGE_SIZE / 2)],
#     'Right Arm': [0.9335804264972017, -0.35836794954530027, 0.031243210730249958 * (ImageGenerator.IMAGE_SIZE / 2),
#                   0.35836794954530027, 0.9335804264972017, -0.2557931034708817 * (ImageGenerator.IMAGE_SIZE / 2)],
# }

LAMBDA = 0.006
ALPHA = 0.05


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
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=padding,
                                padding_mode='replicate')

    gaussian_filter.weight.data = gaussian_kernel.type(torch.cuda.DoubleTensor)
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter


def get_gradient_image(x, sobel):
    # with torch.no_grad:
    x = grayscale(x)
    conv1, conv2 = sobel
    G_x = conv1(x)
    G_y = conv2(x)

    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2) + 1e-06)
    return G


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
    def __init__(self, character):
        super(Net, self).__init__()
        self.num_layers = len(character.char_tree_array)
        self.conv1 = nn.Conv2d(4, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * (((((((character.image_size - 2) - 2) // 2) - 2) // 2) - 2) // 2) *
                             (((((((character.image_size - 2) - 2) // 2) - 2) // 2) - 2) // 2), 512)
        self.fc2 = nn.Linear(512, 3 * self.num_layers)
        self.fc3 = nn.Linear(3 * self.num_layers, 6 * self.num_layers)
        self.fc4 = nn.Linear(6 * self.num_layers, 6 * self.num_layers)
        torch.nn.init.xavier_uniform_(self.conv1.weight, 0.3)
        torch.nn.init.xavier_uniform_(self.conv2.weight, 0.3)
        torch.nn.init.xavier_uniform_(self.conv3.weight, 0.3)
        torch.nn.init.xavier_uniform_(self.conv4.weight, 0.3)
        torch.nn.init.xavier_uniform_(self.fc1.weight, 0.3)
        torch.nn.init.xavier_uniform_(self.fc2.weight, 0.3)
        torch.nn.init.xavier_uniform_(self.fc3.weight, 0.3)
        torch.nn.init.xavier_uniform_(self.fc4.weight, 0.3)
        with torch.no_grad():
            bias = []
            for part in character.char_tree_array:
                bias += character.canonical_bias_dict[part]
            self.fc4.bias = torch.nn.Parameter(torch.DoubleTensor(bias).view(-1, 6 * self.num_layers))
        self.character = character

    # @staticmethod
    # def rotate(origin, point, angle):
    #     ox, oy = origin
    #     x, y = point
    #     qx = ox + torch.cos(angle) * (x - ox) - torch.sin(angle) * (y - oy)
    #     qy = oy + torch.sin(angle) * (x - ox) + torch.cos(angle) * (y - oy)
    #     return qx, qy
    #
    # @staticmethod
    # def affine(angle, center, dx, dy):
    #     tensor_0 = torch.tensor([0] * batch_size, dtype=torch.float64).to(device)
    #     tensor_1 = torch.tensor([1] * batch_size, dtype=torch.float64).to(device)
    #     cos = torch.cos(angle)
    #     sin = torch.sin(angle)
    #     x, y = center
    #     nx, ny = x + dx, y + dy
    #     a = cos / 1
    #     b = sin / 1
    #     c = x - a * nx - b * ny
    #     d = -sin / 1
    #     e = cos / 1
    #     f = y - d * nx - e * ny
    #     mat = torch.stack([a, b, c, d, e, f, tensor_0, tensor_0, tensor_1]).transpose(0, 1)
    #     return mat.view(-1, 3, 3)
    #
    # def composite_affine(self, absolute_center, inner_rotation, joint_rotation, joint_length):
    #     tensor_0 = torch.tensor([0] * batch_size, dtype=torch.float64).to(device)
    #     t1 = self.affine(inner_rotation - joint_rotation, absolute_center, tensor_0, tensor_0)
    #     t2 = self.affine(tensor_0, absolute_center, tensor_0, -joint_length)
    #     t3 = self.affine(joint_rotation, absolute_center, tensor_0, tensor_0)
    #     return t1 @ t2 @ t3

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
        affine_matrix_last_row = torch.tensor(batch_size * [0, 0, 1]).view(-1, 1, 3).to(device)
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
            part_layer = normalize_image(part_layer, 0, 1)
            part_layers.append(part_layer)
            part_layers_dict[part] = i
        for i, part in enumerate(self.character.drawing_order):
            part_layer = part_layers[part_layers_dict[part]]
            if i == 0:
                stack = part_layer
            else:
                # part_alpha = (part_layer[:, -1, :, :]).clamp(0, 1).view(batch_size, 1, ImageGenerator.IMAGE_SIZE,
                #                                                         ImageGenerator.IMAGE_SIZE)
                part_alpha = 1 - (part_layer[:, -1, :, :]).unsqueeze(1)
                # imshow(part_alpha[0].cpu().view(1, 128, 128).permute(1, 2, 0), '')
                # imshow(stack[0].cpu().view(4, 128, 128).permute(1, 2, 0), '')
                stack = (stack * part_alpha)#.clamp(-1)
                # imshow(stack[0].cpu().view(4, 128, 128).permute(1, 2, 0), '')
                # stack_alpha = (stack[:, -1, :, :]).clamp(0, 1)
                # part_layer = (part_layer - 2 * stack_alpha).clamp(-1, 1)
                stack = (stack + part_layer)#.clamp(-1)
                # imshow(stack[0].cpu().view(4, 128, 128).permute(1, 2, 0), '')
        stack = normalize_image(stack, -1, 1)

        # imshow(stack[0].cpu().view(4, 128, 128).permute(1, 2, 0), '')
        # imshow(x[0][0].cpu().view(4, 128, 128).permute(1, 2, 0), '')
        return stack, close_to_eye_matrices, translation_values

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = list(torch.split(x, int(x.shape[2] / (self.num_layers + 1)), dim=2))
        x = self.stn(x)
        return x


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


def train(net_path=None):
    inspection_path, path = create_folders()
    colors = ['blue', 'red', 'orange', 'green', 'pink', 'purple', 'yellow', 'black', 'brown']
    kernel_sizes = [51, 35, 25, 17, 13, 9, 7, 5, 3]
    net = create_net(net_path)
    criterion = nn.MSELoss()
    canonical = create_canonical(char)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # num_patches = 1
    # patch_size = ImageGenerator.IMAGE_SIZE // num_patches #len(kernel_sizes)
    # sobel = get_sobel()
    # optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.001)
    # grad_gaussian = kornia.filters.GaussianBlur2d((5, 5), (3, 3), 'replicate')

    for epoch in range(len(kernel_sizes)):  # loop over the dataset multiple times
        gaussian = kornia.filters.GaussianBlur2d((kernel_sizes[epoch], kernel_sizes[epoch]), (18, 18), 'replicate')
        grad_gaussian = kornia.filters.GaussianBlur2d((11, 11), (6, 6), 'replicate') if epoch < len(kernel_sizes) // 2 \
            else kornia.filters.GaussianBlur2d((5, 5), (3, 3), 'replicate')
        running_losses = [0.0, 0.0, 0.0, 0.0]
        losses_arrays = [[], [], [], []]
        iterations = []
        num_iter_to_print = int(70 * (0.2 * epoch + 1))
        num_iterations = 7 * len(trainset)
        cur_epoch_num_iterations = int(num_iterations * (0.2 * epoch + 1))

        # gaussian = get_gaussian_kernel(channels=3, kernel_size=kernel_sizes[epoch],
        #                                padding=((kernel_sizes[epoch] - 1) // 2)).to(device)
        # grad_gaussian = get_gaussian_kernel(channels=1, kernel_size=kernel_sizes[epoch],
        #                                padding=((kernel_sizes[epoch] - 1) // 2)).to(device)
        # sobel = kornia.filters.Sobel()

        for i in tqdm(range(cur_epoch_num_iterations)):  # run each epoch one more time than the last one
            data = trainset[i % len(trainset)]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # get net output
            outputs, close_to_eye_matrices, translation_values = net(torch.cat([inputs, canonical], dim=1))

            # separate inputs outputs to colors and alpha
            inputs_alpha = inputs.permute(0, 3, 1, 2)[:, 3, :, :]
            outputs_alpha = outputs[:, 3, :, :]
            inputs = inputs.permute(0, 3, 1, 2)[:, :3, :, :]
            outputs = outputs[:, :3, :, :]

            # calc loss
            alpha_modified = ALPHA + (i / cur_epoch_num_iterations) * (i / cur_epoch_num_iterations) * (i / cur_epoch_num_iterations) * (0.01 - ALPHA)
            losses = calc_losses(alpha_modified, close_to_eye_matrices, criterion, epoch, gaussian, grad_gaussian, inputs, inputs_alpha, kernel_sizes,
                                 outputs, outputs_alpha, translation_values)
            losses[0].backward()
            optimizer.step()

            running_losses[0] += losses[0].item()
            running_losses[1] += losses[1].item()
            running_losses[2] += losses[2].item()
            running_losses[3] += losses[3].item()

            if i == 0 or i == cur_epoch_num_iterations - 1:
                with torch.no_grad():
                    save_batch(epoch, i, inputs, outputs, path)
                    save_inspection_image(canonical, epoch, i, inspection_path, net, kernel_sizes, gaussian, grad_gaussian)
            if i % num_iter_to_print == (
                    num_iter_to_print) - 1:
                append_losses(alpha_modified, epoch, i, iterations, losses_arrays, num_iter_to_print, running_losses)
                running_losses = [0.0, 0.0, 0.0, 0.0]
                with torch.no_grad():
                    save_inspection_image(canonical, epoch, i, inspection_path, net, kernel_sizes, gaussian, grad_gaussian)

        save_losses_graphs(colors, epoch, iterations, kernel_sizes, losses_arrays, path)

    print('Finished Training')
    torch.save(net.state_dict(), path + '\\aaa_net.pth')
    return net, path


def create_net(net_path):
    net = Net(char).double()
    net.to(device)
    if net_path:
        net.load_state_dict(torch.load(net_path + '\\aaa_net.pth'))
    return net


def create_canonical(character):
    num_layers = len(character.char_tree_array)
    canonical = ImageGenerator.generate_layers(character, [0] * num_layers, as_tensor=True, transform=False)
    canonical = torch.cat(batch_size * [canonical]).reshape(
        batch_size,
        -1,
        character.image_size,
        4).to(device)
    return canonical


def create_folders():
    current_time = datetime.now()
    path = ImageGenerator.PATH + 'Plots\\' + current_time.strftime("%d-%m-%Y %H-%M-%S")
    inspection_path = path + '\\inspection'
    try:
        os.mkdir(path)
        os.mkdir(inspection_path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    return inspection_path, path


def save_losses_graphs(colors, epoch, iterations, kernel_sizes, losses, path):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Training Loss epoch %d, gaussian kernel size %d" % (epoch + 1, kernel_sizes[epoch]))
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
    plt.savefig(path + '\\Training Loss epoch %d.png' % (epoch + 1,))
    plt.close(fig)


def save_inspection_image(canonical, epoch, i, inspection_path, net, kernel_sizes, gaussian, grad_gaussian):
    # parse inputs and outputs
    inspection_input = trainset[0][0].to(device)
    inspection_output = net(torch.cat([inspection_input, canonical], dim=1))[0]
    inspection_input_alpha = inspection_input.permute(0, 3, 1, 2)[:, 3, :, :]
    inspection_output_alpha = inspection_output[:, 3, :, :]
    inspection_input = inspection_input.permute(0, 3, 1, 2)[:, :3, :, :]
    inspection_output = inspection_output[:, :3, :, :]

    # calc gaussian loss image
    inspection_gaussian_loss_image = calc_gaussian_loss(torch.nn.MSELoss(reduction='none'), epoch, gaussian,
                                                        inspection_input, inspection_input_alpha, kernel_sizes,
                                                        inspection_output, inspection_output_alpha)
    inspection_gaussian_loss_image = normalize_image(inspection_gaussian_loss_image, -1, torch.min(torch.tensor([torch.max(inspection_gaussian_loss_image), 1])))

    # calc depth loss image
    inspection_depth_loss_image = calc_depth_loss(grad_gaussian, torch.nn.MSELoss(reduction='none'), inspection_input, inspection_output)
    inspection_depth_loss_image = kornia.color.grayscale_to_rgb(inspection_depth_loss_image)
    inspection_depth_loss_image = normalize_image(inspection_depth_loss_image, -1, torch.min(torch.tensor([torch.max(inspection_depth_loss_image), 1])))

    # concat and save all images
    inspection = torch.cat((inspection_input[0].permute(1, 2, 0).cpu(), inspection_output[0].permute(1, 2, 0).cpu()))
    inspection_losses = torch.cat((inspection_gaussian_loss_image[0].permute(1, 2, 0).cpu(), inspection_depth_loss_image[0].permute(1, 2, 0).cpu()))
    imsave(torch.cat([inspection, inspection_losses], dim=1), "epoch %d iter %d" % (epoch, i),
           inspection_path)


def normalize_image(image, a, b):
    image = (image - torch.min(image)) * ((b - a) / (torch.max(image) - torch.min(image))) + a
    return image


def append_losses(alpha_modified, epoch, i, iterations, losses, num_iter_to_print, running_losses):
    losses[0].append(running_losses[0] / num_iter_to_print)
    losses[1].append(((1 - LAMBDA) * (1 - alpha_modified)) * running_losses[1] / num_iter_to_print)
    losses[2].append(((1 - LAMBDA) * alpha_modified) * running_losses[2] / num_iter_to_print)
    losses[3].append(LAMBDA * running_losses[3] / num_iter_to_print)
    iterations.append(i + 1)
    print(
        '[%d, %5d] loss: %.9f, gaussian loss: %.9f, depth loss: %.9f, orthogonal loss: %.9f, alpha modified: %.9f' %
        (epoch + 1, i + 1, running_losses[0] / num_iter_to_print,
         ((1 - LAMBDA) * (1 - alpha_modified)) * running_losses[1] / num_iter_to_print,
         ((1 - LAMBDA) * alpha_modified) * running_losses[2] / num_iter_to_print,
         LAMBDA * running_losses[3] / num_iter_to_print,
         alpha_modified))


def save_batch(epoch, i, inputs, outputs, path):
    input_batch = (torch.cat([inputs[i] for i in range(batch_size)], dim=2)).permute(1, 2, 0).cpu()
    output_batch = (torch.cat([outputs[i] for i in range(batch_size)], dim=2)).permute(1, 2, 0).cpu()
    imsave(torch.cat([input_batch, output_batch], dim=0),
           "input (up) vs output (down)- epoch %d iteration %d" % (epoch + 1, i,), path)


def calc_losses(alpha_modified, close_to_eye_matrices, criterion, epoch, gaussian, grad_gaussian, inputs, inputs_alpha,
                kernel_sizes, outputs, outputs_alpha, translation_values):
    orthogonal_loss = calc_orthogonal_loss(close_to_eye_matrices, criterion)
    translation_loss = calc_translation_loss(translation_values)
    depth_loss = calc_depth_loss(grad_gaussian, criterion, inputs, outputs)
    gaussian_loss = calc_gaussian_loss(criterion, epoch, gaussian, inputs, inputs_alpha, kernel_sizes, outputs,
                                       outputs_alpha)
    loss = (1 - LAMBDA) * (alpha_modified * depth_loss + (1 - alpha_modified) * gaussian_loss) + \
           LAMBDA * (translation_loss + orthogonal_loss)

    # rand_end_x, rand_end_y, rand_start_x, rand_start_y = generate_patch_indices(epoch, kernel_sizes,
    #                                                                             num_patches, patch_size)
    # inputs_depth_map = grad_gaussian(kornia.filters.canny(inputs)[1])
    # outputs_depth_map = grad_gaussian(kornia.filters.canny(outputs)[1])
    # grad_inputs = grad_gaussian(sobel(kornia.rgb_to_grayscale(inputs)))
    # grad_outputs = grad_gaussian(sobel(kornia.rgb_to_grayscale(outputs)))
    # imshow(gaussian_inputs[0].permute(1, 2, 0).cpu() - gaussian_outputs[0].permute(1, 2, 0).cpu(), '')
    # imshow(inputs_depth_map[0, :, rand_start_y: rand_end_y, rand_start_x: rand_end_x].permute(1, 2, 0).cpu() - outputs_depth_map[0, :, rand_start_y: rand_end_y, rand_start_x: rand_end_x].permute(1, 2, 0).cpu(), '', cmap='gray')
    # imshow(gaussian_outputs[0].permute(1, 2, 0).cpu(), '')
    # imshow(outputs_depth_map[0, :, rand_start_y: rand_end_y, rand_start_x: rand_end_x].permute(1, 2, 0).cpu(), '', cmap='gray')
    # loss = grad_loss + \
    # ALPHA = (float(epoch) / len(kernel_sizes))
    # LAMBDA_MODIFIED = LAMBDA #+ (LAMBDA * (0.5 * epoch))

    return loss, gaussian_loss, depth_loss, orthogonal_loss


def calc_gaussian_loss(criterion, epoch, gaussian, inputs, inputs_alpha, kernel_sizes, outputs, outputs_alpha):
    inputs_dichotomized = inputs + ((kernel_sizes[epoch] / 7) * inputs_alpha.unsqueeze(1))
    outputs_dichotomized = outputs + ((kernel_sizes[epoch] / 7) * outputs_alpha.unsqueeze(1))
    gaussian_inputs = normalize_image(gaussian(inputs_dichotomized), -1, 1)
    gaussian_outputs = normalize_image(gaussian(outputs_dichotomized), -1, 1)
    # gaussian_inputs = gaussian_inputs / (gaussian_inputs.norm(dim=1).unsqueeze(1)).clamp(1e-12)
    # gaussian_outputs = gaussian_outputs / (gaussian_outputs.norm(dim=1).unsqueeze(1)).clamp(1e-12)
    gaussian_loss = criterion(gaussian_outputs, gaussian_inputs)
    return gaussian_loss


def calc_depth_loss(grad_gaussian, criterion, inputs, outputs):
    # inputs_edges = kornia.filters.canny(inputs)[1]
    # outputs_edges = kornia.filters.canny(outputs)[1]
    # inputs_depth_map = heat.heat_method(inputs_edges, timestep=0.1, mass=.01, iters_diffusion=50,
    #                                     iters_poisson=25)
    # outputs_depth_map = heat.heat_method(outputs_edges, timestep=0.1, mass=.01, iters_diffusion=50,
    #                                      iters_poisson=25)
    inputs_depth_map = normalize_image(grad_gaussian(kornia.filters.canny(inputs)[1]), -1, 1)
    outputs_depth_map = normalize_image(grad_gaussian(kornia.filters.canny(outputs)[1]), -1, 1)
    depth_loss = criterion(inputs_depth_map, outputs_depth_map)
    return depth_loss


# def generate_patch_indices(epoch, kernel_sizes, num_patches, patch_size):
#     patch_index = int(num_patches * (float(epoch) / float(len(kernel_sizes))))
#     if patch_index > 0:
#         rand_start_x = np.random.randint(0, patch_index * patch_size)
#         rand_end_x = rand_start_x + (num_patches - patch_index) * patch_size
#         rand_start_y = np.random.randint(0, patch_index * patch_size)
#         rand_end_y = rand_start_y + (num_patches - patch_index) * patch_size
#     else:
#         rand_start_x = 0
#         rand_end_x = ImageGenerator.IMAGE_SIZE
#         rand_start_y = 0
#         rand_end_y = ImageGenerator.IMAGE_SIZE
#     return rand_end_x, rand_end_y, rand_start_x, rand_start_y


def calc_translation_loss(translation_values):
    translation_loss = 0
    ones = 0.8 * torch.ones(batch_size * 2).double().to(device)
    for translation_value in translation_values:
        translation_loss += torch.sum(torch.relu(torch.abs(translation_value.reshape(batch_size * 2)) - ones))
    return translation_loss


def calc_orthogonal_loss(close_to_eye_matrices, criterion):
    orthogonal_loss = 0
    eye = torch.tensor(batch_size * [1, 0, 0, 1]).double().to(device).view(-1, 4)
    for close_to_eye_matrix in close_to_eye_matrices:
        orthogonal_loss += criterion(close_to_eye_matrix, eye)
    return orthogonal_loss


def append_inputs_outpus(g_inputs, g_outputs, input_images, output_images):
    image = (torch.cat([g_inputs[i] for i in range(batch_size)], dim=2)).permute(1, 2, 0).cpu()
    output = (torch.cat([g_outputs[i] for i in range(batch_size)], dim=2)).permute(1, 2, 0).cpu()
    input_images.append(image)
    output_images.append(output)


def check_inputs_outputs(g_inputs, g_outputs, iteration):
    dim = g_inputs.shape[2]
    image = (torch.cat([g_inputs[i] for i in range(batch_size)], dim=2)).permute(1, 2, 0).cpu()
    output = (torch.cat([g_outputs[i] for i in range(batch_size)], dim=2)).permute(1, 2, 0).cpu()
    imshow(torch.cat([image, output], dim=0),
           "input (up) vs output (down): iteration %d " % (iteration,), cmap='gray')


def test(net, path):
    canonical = create_canonical(net.character)
    with torch.no_grad():
        data = testset[0]
        images, labels = data[0].to(device), data[1].to(device)
        print(labels)
        outputs, _, _ = net(torch.cat([images, canonical], dim=1))
        images = images.view(-1, net.character.image_size, net.character.image_size, 4)
        image = (torch.cat([images[i] for i in range(batch_size)], dim=1))
        output = (torch.cat([outputs[i] for i in range(batch_size)], dim=2)).permute(1, 2, 0)
        imsave(image.cpu(), "input", path)
        imsave(output.cpu(), "output", path)
        imshow(image.cpu(), "input")
        imshow(output.cpu(), "output")

        # for data in testset:
        #     images, labels = data[0].to(device), data[1].to(device)
        #     outputs = net(images)


def main():
    # path = ImageGenerator.PATH + 'Plots\\' + '11-08-2021 16-56-12 (Good 15 range)'
    net, path = train()
    test(net, path)


if __name__ == '__main__':
    main()
