from __future__ import print_function

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import ImageGenerator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())
batch_size = 4
num_layers = 11

trainset, testset = ImageGenerator.load_data(batch_size)

PATH = 'C:\\School\\Huji\\Lab\\LabProject\\'
CANONICAL_BIAS_DICT = {
    'Lower Torso': [0.848048096156426, -0.5299192642332049, -0.06623990802915061, 0.5299192642332049, 0.848048096156426,
                    0.10600601201955324],
    'Upper Left Leg': [0.9986295347545738, 0.05233595624294379, -0.001158806553435294, -0.05233595624294379,
                       0.9986295347545738, -0.022375972781727708],
    'Lower Left Leg': [0.8290375725550417, -0.5591929034707469, 0.2758348051438573, 0.5591929034707469,
                       0.8290375725550417, -0.3679706692738145],
    'Upper Right Leg': [0.898794046299167, -0.4383711467890774, -0.1455594743327377, 0.4383711467890774,
                        0.898794046299167, -0.0037996949408528533],
    'Lower Right Leg': [0.7431448254773944, -0.6691306063588581, -0.04191544748895007, 0.6691306063588581,
                        0.7431448254773944, -0.4385935274816633],
    'Chest': [0.984807753012208, 0.17364817766693036, 0.09489218227886641, -0.17364817766693036, 0.984807753012208,
              0.204382434413116],
    'Head': [0.754709580222772, 0.6560590289905073, 0.34755260971666235, -0.6560590289905073, 0.754709580222772,
             0.3941816342013059],
    'Left Shoulder': [0.992546151641322, 0.12186934340514749, 0.2294707093902736,
                      -0.12186934340514749, 0.992546151641322, 0.42009272589950425],
    'Left Arm': [0.766044443118978, 0.6427876096865394, 0.3722248189351549, -0.6427876096865394, 0.766044443118978,
                 -0.02028262460152125],
    'Right Shoulder': [0.9993908270190958, -0.03489949670250096, -0.16002611816129286, 0.03489949670250096,
                       0.9993908270190958, 0.3825967914560618],
    'Right Arm': [0.9961946980917455, 0.08715574274765818, -0.2047079583082938, -0.08715574274765818,
                  0.9961946980917455,
                  0.15872754699051406],
}
# CANONICAL_BIAS_DICT = {
# 'Lower Torso' : [0.9396926207859084, 0.3420201433256687, 0.04275251791570859, -0.3420201433256687, 0.9396926207859084, 0.11746157759823855],
# 'Upper Left Leg' : [0.766044443118978, 0.6427876096865393, 0.1110094192940079, -0.6427876096865393, 0.766044443118978, -0.04842634590568417],
# 'Lower Left Leg' : [0.5000000000000001, 0.8660254037844386, -0.06188186632407247, -0.8660254037844386, 0.5000000000000001, -0.494588858990468],
# 'Upper Right Leg' : [0.766044443118978, 0.6427876096865393, -0.03581755270379028, -0.6427876096865393, 0.766044443118978, 0.005014301488951586],
# 'Lower Right Leg' : [0.5000000000000001, 0.8660254037844386, -0.18557087828226543, -0.8660254037844386, 0.5000000000000001, -0.35944383583792305],
# 'Chest' : [0.766044443118978, 0.6427876096865393, 0.11775690438706243, -0.6427876096865393, 0.766044443118978, 0.19853443578833097],
# 'Head' : [0.5000000000000001, 0.8660254037844386, 0.2854393650717916, -0.8660254037844386, 0.5000000000000001, 0.4399400549617346],
# 'Left Shoulder' : [0.5000000000000001, 0.8660254037844386, 0.3948578838933447, -0.8660254037844386, 0.5000000000000001, 0.28372052716864016],
# 'Left Arm' : [0.17364817766693041, 0.984807753012208, 0.4760183872270929, -0.984807753012208, 0.17364817766693041, -0.12074411073009174],
# 'Right Shoulder' : [0.5000000000000001, 0.8660254037844386, 0.08652124269796851, -0.8660254037844386, 0.5000000000000001, 0.3959458866973752],
# 'Right Arm' : [0.17364817766693041, 0.984807753012208, 0.07248901759141649, -0.984807753012208, 0.17364817766693041, 0.12892852389315987],
# }

LAMBDA = 0.02


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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 7, 5)
        self.conv2 = nn.Conv2d(7, 9, 3)
        self.fc1 = nn.Linear(9 * ((((ImageGenerator.IMAGE_SIZE - 4) // 2) - 2) // 2) * (
                (((ImageGenerator.IMAGE_SIZE - 4) // 2) - 2) // 2), 32)
        self.fc2 = nn.Linear(32, 3 * num_layers)
        self.fc3 = nn.Linear(3 * num_layers, 6 * num_layers)
        torch.nn.init.xavier_uniform_(self.conv1.weight, 0.3)
        torch.nn.init.xavier_uniform_(self.conv2.weight, 0.3)
        torch.nn.init.xavier_uniform_(self.fc1.weight, 0.3)
        torch.nn.init.xavier_uniform_(self.fc2.weight, 0.3)
        torch.nn.init.xavier_uniform_(self.fc3.weight, 0.3)
        with torch.no_grad():
            bias = []
            for part in ImageGenerator.DRAWING_ORDER:
                bias += CANONICAL_BIAS_DICT[part]
            self.fc3.bias = torch.nn.Parameter(torch.DoubleTensor(bias).view(-1, 6 * num_layers))

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
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # Spatial transformer network forward function
    # def stn(self, x, body, head, left_leg, right_leg):
    def stn(self, x):
        affine_transforms = self.localization(x[0])
        # print(angles)
        # image_center = torch.tensor([0, 0] * batch_size).view(2, -1).to(device)
        # tensor_0 = torch.tensor([0] * batch_size, dtype=torch.float64).to(device)
        # angles[0] += np.math.radians(20)
        # for i in range(len(angles)):
        #     if ImageGenerator.PARENTS[i] is not None:
        #         angles[i] += angles[ImageGenerator.PARENTS[i]]
        displacements = []
        joint_angles = []
        close_to_eye_matrices = []
        translation_values = []
        for i, part in enumerate(ImageGenerator.DRAWING_ORDER):
            # displacement = ImageGenerator.CHARACTER_DICT[part]['displacement']
            # dx, dy = displacement.x / (ImageGenerator.IMAGE_SIZE / 2), -displacement.y / (ImageGenerator.IMAGE_SIZE / 2)
            # pdx = pdy = 0
            # j_angle = torch.zeros(batch_size).to(device)
            # if i != 0:
            #     pdx, pdy = displacements[ImageGenerator.PARENTS[i]]
            #     j_angle = joint_angles[ImageGenerator.PARENTS[i]]
            # joint_angles.append(angles[i] + j_angle)
            # dx += pdx
            # dy += pdy
            # dx, dy = self.rotate((pdx, pdy),
            #                      (dx, dy),
            #                      j_angle)
            # displacements.append((dx, dy))
            part_transform = affine_transforms[:, i * 6: (i * 6) + 6].view(-1, 2, 3).to(device)
            rotation_scaling_matrix = part_transform[:, :, 0: 2]
            close_to_eye_matrices.append(
                (torch.matmul(rotation_scaling_matrix, torch.transpose(rotation_scaling_matrix, 1, 2))).view(-1, 4))
            translation_values.append(part_transform[:, :, 2])
            # part_transform = torch.DoubleTensor(4 * CANONICAL_BIAS_DICT[part]).view(-1, 2, 3).to(device)
            # part_transform = torch.DoubleTensor(4 * [0.9848, -0.1736, 0.03, 0.1736, 0.9848, -0.05]).view(-1, 2, 3).to(device)
            part_grid = F.affine_grid(part_transform, x[1 + i].size(), align_corners=False)
            part_layer = F.grid_sample(x[1 + i], part_grid, align_corners=False, padding_mode='border')
            if i == 0:
                stack = part_layer
            else:
                part_alpha = (part_layer[:, -1, :, :]).clamp(0, 1).view(batch_size, 1, ImageGenerator.IMAGE_SIZE,
                                                                        ImageGenerator.IMAGE_SIZE)
                # imshow(part_alpha[0].cpu().view(1, 128, 128).permute(1, 2, 0), '')
                # imshow(stack[0].cpu().view(4, 128, 128).permute(1, 2, 0), '')
                stack = (stack - 2 * part_alpha).clamp(-1, 1)
                # imshow(stack[0].cpu().view(4, 128, 128).permute(1, 2, 0), '')
                # stack_alpha = (stack[:, -1, :, :]).clamp(0, 1)
                # part_layer = (part_layer - 2 * stack_alpha).clamp(-1, 1)
                stack = (stack + part_layer + 1).clamp(-1, 1)
        # imshow(stack[0].cpu().view(4, 128, 128).permute(1, 2, 0), '')
        # imshow(x[0][0].cpu().view(4, 128, 128).permute(1, 2, 0), '')
        return stack, close_to_eye_matrices, translation_values

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = torch.split(x, int(x.shape[2] / (num_layers + 1)), dim=2)
        # x, head, left_leg, right_leg, body = torch.split(x, int(x.shape[2] / num_layers), dim=2)
        # x = self.stn(x, body, head, left_leg, right_leg)
        x = self.stn(x)
        return x


def imshow(img, title):
    plt.clf()
    img = img / 2 + 0.5  # unnormalize
    npimg = img.detach().numpy()
    plt.imshow(npimg)
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
    current_time = datetime.now()
    path = PATH + 'Plots\\' + current_time.strftime("%d-%m-%Y %H-%M-%S")
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    colors = ['blue', 'red', 'orange', 'green', 'pink', 'purple', 'yellow', 'black']
    kernel_sizes = [50, 35, 25, 17, 13, 9, 7]
    num_iter_to_print = 250
    net = Net().double()
    net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.5, momentum=0.1)
    origin = ImageGenerator.create_body_hierarchy([0] * num_layers)
    canonical = ImageGenerator.generate_layers(origin, as_tensor=True, transform=False)
    canonical = torch.cat(batch_size * [canonical]).reshape(
        batch_size,
        -1,
        ImageGenerator.IMAGE_SIZE,
        4).to(device)
    for epoch in range(len(kernel_sizes)):  # loop over the dataset multiple times
        gaussian = get_gaussian_kernel(channels=4, kernel_size=kernel_sizes[epoch],
                                       padding=((kernel_sizes[epoch] - 1) // 2)).to(device)
        running_loss = 0.0
        losses = []
        iterations = []
        input_images = []
        output_images = []
        for i in tqdm(range((epoch + 1) * len(trainset))):  # run each epoch one more time than the last one
            # get the inputs; data is a list of [inputs, labels]
            data = trainset[i % len(trainset)]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs, close_to_eye_matrices, translation_values = net(torch.cat([inputs, canonical], dim=1))
            orthogonal_loss = 0
            eye = torch.tensor(batch_size * [1, 0, 0, 1]).double().to(device).view(-1, 4)
            for close_to_eye_matrix in close_to_eye_matrices:
                orthogonal_loss += criterion(close_to_eye_matrix, eye)
            translation_loss = 0
            ones = torch.ones(batch_size * 2).double().to(device)
            for translation_value in translation_values:
                translation_loss += torch.sum(torch.relu(torch.abs(translation_value.reshape(batch_size * 2)) - ones))
            g_inputs = gaussian(inputs.permute(0, 3, 1, 2))
            g_outputs = gaussian(outputs)
            loss = criterion(g_outputs, g_inputs) + LAMBDA * (translation_loss + orthogonal_loss)
            loss.backward()
            optimizer.step()

            # for debugging
            # check_inputs_outputs(g_inputs, g_outputs, i)

            # print statistics
            running_loss += loss.item()
            if i == 0 or i == (epoch + 1) * len(trainset) - 1:
                with torch.no_grad():
                    append_inputs_outpus(g_inputs, g_outputs, input_images, output_images)
            if i % ((epoch + 1) * num_iter_to_print) == (
                    (epoch + 1) * num_iter_to_print) - 1:  # print every 300 * epoch mini-batches
                losses.append(running_loss / ((epoch + 1) * num_iter_to_print))
                iterations.append(i + 1)
                print('[%d, %5d] loss: %.9f' %
                      (epoch + 1, i + 1, running_loss / ((epoch + 1) * num_iter_to_print)))
                running_loss = 0.0
        plt.clf()
        plt.plot(iterations, losses, color=colors[epoch])
        plt.title("Training Loss epoch %d, gaussian kernel size %d" % (epoch + 1, kernel_sizes[epoch]))
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.savefig(path + '\\Training Loss epoch %d.png' % (epoch + 1,))
        imsave(torch.cat([input_images[0], output_images[0]], dim=0),
               "input (up) vs output (down)- epoch %d beginning" % (epoch + 1,), path)
        imsave(torch.cat([input_images[1], output_images[1]], dim=0),
               "input (up) vs output (down)- epoch %d end" % (epoch + 1,), path)
    print('Finished Training')
    torch.save(net.state_dict(), path + '\\aaa_net.pth')
    return path


def append_inputs_outpus(g_inputs, g_outputs, input_images, output_images):
    image = (torch.cat([g_inputs[i] for i in range(batch_size)], dim=2)).permute(1, 2, 0).cpu()
    output = (torch.cat([g_outputs[i] for i in range(batch_size)], dim=2)).permute(1, 2, 0).cpu()
    input_images.append(image)
    output_images.append(output)


def check_inputs_outputs(g_inputs, g_outputs, iteration):
    image = (torch.cat([g_inputs[i] for i in range(batch_size)], dim=2)). \
        view(ImageGenerator.IMAGE_SIZE, ImageGenerator.IMAGE_SIZE * batch_size).cpu()
    output = (torch.cat([g_outputs[i] for i in range(batch_size)], dim=2)). \
        view(ImageGenerator.IMAGE_SIZE, ImageGenerator.IMAGE_SIZE * batch_size).cpu()
    imshow(torch.cat([image, output], dim=0),
           "input (up) vs output (down): iteration %d " % (iteration,))


def test(path):
    net = Net().double()
    net.to(device)
    net.load_state_dict(torch.load(path + '\\aaa_net.pth'))

    origin = ImageGenerator.create_body_hierarchy([0] * num_layers)
    canonical = ImageGenerator.generate_layers(origin, as_tensor=True, transform=False)

    canonical = torch.cat(batch_size * [canonical]).reshape(
        batch_size,
        -1,
        ImageGenerator.IMAGE_SIZE,
        4).to(device)

    with torch.no_grad():
        data = testset[0]
        images, labels = data[0].to(device), data[1].to(device)
        print(labels)
        outputs, _, _ = net(torch.cat([images, canonical], dim=1))
        images = images.view(-1, ImageGenerator.IMAGE_SIZE, ImageGenerator.IMAGE_SIZE, 4)
        image = (torch.cat([images[i] for i in range(batch_size)], dim=1))
        output = (torch.cat([outputs[i] for i in range(batch_size)], dim=2)).permute(1, 2, 0)
        imsave(image.cpu(), "input", path)
        imsave(output.cpu(), "output", path)
        # for data in testset:
        #     images, labels = data[0].to(device), data[1].to(device)
        #     outputs = net(images)


def main():
    # path = PATH + 'Plots\\' + '26-05-2021 17-35-25'
    path = train()
    test(path)


if __name__ == '__main__':
    main()
