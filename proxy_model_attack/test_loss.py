# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from scipy import misc
import numpy as np
import sys
import os

from torch.autograd import gradcheck

####Added
import imageio
####
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from load_data import LoadVisualData, LoadData
from load_data import LoadSourceData
from load_data import LoadTargetData
from network import Generator
import torch.nn.functional as F
from torch.optim import Adam

to_image = transforms.Compose([transforms.ToPILImage()])
dataset_dir = 'images/'
max_iter_range = 20
beta = 2.5
interpolation_method = 'nearest'   ####bilinear,bicubic
learning_rate = 0.01
TEST_SIZE = 20
test_batch_size = 1

# def loss_define(source, attack, output, target, beta):
#     loss = F.mse_loss(source, attack) + beta * F.mse_loss(target, output)
#     return loss


def test_model():

    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    # Creating dataset loaders

    # visual_dataset = LoadVisualData(dataset_dir, 10)
    # visual_loader = DataLoader(dataset=visual_dataset, batch_size=1, shuffle=False, num_workers=0,
    #                            pin_memory=True, drop_last=False)

    # source_dataset = LoadSourceData(dataset_dir, 10)
    # source_loader = DataLoader(dataset=source_dataset, batch_size=1, shuffle=False, num_workers=0,
    #                            pin_memory=True, drop_last=False)

    test_dataset = LoadData(dataset_dir, TEST_SIZE, test=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=1,
                             pin_memory=True, drop_last=False)

    # target_dataset = LoadTargetData(dataset_dir, 1)
    # target_loader = DataLoader(dataset=target_dataset, batch_size=1, shuffle=False, num_workers=0,
    #                            pin_memory=True, drop_last=False)

    # Creating and loading pre-trained model
    generator = Generator().to(device)
    generator = torch.nn.DataParallel(generator)
    generator.load_state_dict(torch.load("models/generator_epoch_42.pth"), strict=True)



    generator.eval()
    # visual_iter = iter(visual_loader)
    test_iter = iter(test_loader)
    # source_iter = iter(source_loader)
    # target_iter = iter(target_loader)
    total_loss_raw = 0.0
    total_loss_image = 0.0
    for i in range(len(test_loader)):
        print("Processing image " + str(i))
        with torch.enable_grad():
            raw_image, _ = next(test_iter)
            raw_image = raw_image.to(device)
            raw_shape = raw_image.shape
            noise = torch.randn(raw_shape).to(device)
            attack_raw = raw_image + noise
            attack_raw = torch.clamp(attack_raw, 0.0, 1.0)
            enhanced = generator(raw_image)
            attack = generator(attack_raw)
            loss_raw = F.mse_loss(raw_image, attack_raw)
            loss_image = F.mse_loss(enhanced, attack)
            print('Image %d, loss_raw: %.4f, loss_image: %.4f' % (i, loss_raw.item(), loss_image.item()))
            total_loss_raw += loss_raw.item()
            total_loss_image += loss_image.item()

    total_loss_raw /= TEST_SIZE
    total_loss_image /= TEST_SIZE

    print('Total_loss_raw: %.4f, Total_loss_image: %.4f' % (total_loss_raw, total_loss_image))


        # for j in range(len(visual_loader)):
        #     print("Processing image " + str(j))
        #
        #     torch.cuda.empty_cache()
        #     raw_image = next(visual_iter)
        #     raw_image = raw_image.to(device)
        #     #raw_size = raw_image.shape
        #     source_image = next(source_iter)
        #     source_image = source_image.to(device)
        #
        #     #initialize attack_raw
        #     attack_raw = raw_image
        #     attack_raw = attack_raw.to(device)
        #     attack_raw.requires_grad = True
        #     optimizer = Adam([attack_raw], learning_rate)
        #     with torch.enable_grad():
        #         for f in range(max_iter_range):
        #             optimizer.zero_grad()
        #             attack_image = model(attack_raw)
        #             output_image = F.interpolate(attack_image, size= (target_size[2], target_size[3]), mode=interpolation_method)
        #             loss1 = F.mse_loss(attack_image, source_image)
        #             loss2 = F.mse_loss(output_image, target_image)
        #             total_loss = loss1 + beta * loss2
        #             total_loss.backward()
        #             optimizer.step()
        #             attack_raw = torch.clamp(attack_raw, 0, 1)
        #             print('Iter_epoch %d, loss1: %.4f, loss2: %.4f' % (f, loss1, loss2))



    # with torch.no_grad():

        # visual_iter = iter(visual_loader)
        # for j in range(len(visual_loader)):
        #
        #     print("Processing image " + str(j))
        #
        #     torch.cuda.empty_cache()
        #     raw_image = next(visual_iter)
        #
        #     if use_gpu == "true":
        #         raw_image = raw_image.to(device, dtype=torch.half)
        #     else:
        #         raw_image = raw_image.to(device)
        #
        #     # Run inference
        #
        #     enhanced = model(raw_image.detach())
        #     enhanced = np.asarray(to_image(torch.squeeze(enhanced.float().detach().cpu())))
        #
        #     # Save the results as .png images
        #
        #     # if orig_model == "true":
        #     #     misc.imsave("results/full-resolution/" + str(j) + "_level_" + str(level) + "_orig.png", enhanced)
        #     # else:
        #     #     misc.imsave("results/full-resolution/" + str(j) + "_level_" + str(level) +
        #     #             "_epoch_" + str(restore_epoch) + ".png", enhanced)
        #
        #     if orig_model == "true":
        #         imageio.imwrite("results/full-resolution/" + str(j) + "_level_" + str(level) + "_orig.png", enhanced)
        #     else:
        #         imageio.imwrite("results/full-resolution/" + str(j) + "_level_" + str(level) +
        #                 "_epoch_" + str(restore_epoch) + ".png", enhanced)


if __name__ == '__main__':
    test_model()
