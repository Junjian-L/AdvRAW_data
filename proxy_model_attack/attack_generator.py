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

from load_data import LoadVisualData
from load_data import LoadSourceData
from load_data import LoadTargetData
from network import Generator
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.utils as vutils

to_image = transforms.Compose([transforms.ToPILImage()])
dataset_dir = 'images/'
max_iter_range = 8000
beta = 1.5
interpolation_method = 'nearest'   ####bilinear,bicubic
learning_rate = 0.05

# def loss_define(source, attack, output, target, beta):
#     loss = F.mse_loss(source, attack) + beta * F.mse_loss(target, output)
#     return loss


def test_model():

    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    # Creating dataset loaders

    visual_dataset = LoadVisualData(dataset_dir, 10)
    visual_loader = DataLoader(dataset=visual_dataset, batch_size=1, shuffle=False, num_workers=0,
                               pin_memory=True, drop_last=False)

    # source_dataset = LoadSourceData(dataset_dir, 10)
    # source_loader = DataLoader(dataset=source_dataset, batch_size=1, shuffle=False, num_workers=0,
    #                            pin_memory=True, drop_last=False)

    target_dataset = LoadTargetData(dataset_dir, 1)
    target_loader = DataLoader(dataset=target_dataset, batch_size=1, shuffle=False, num_workers=0,
                               pin_memory=True, drop_last=False)

    # Creating and loading pre-trained model
    generator = Generator().to(device)
    generator = torch.nn.DataParallel(generator)
    generator.load_state_dict(torch.load("models/generator_epoch_42.pth"), strict=True)



    generator.eval()
    visual_iter = iter(visual_loader)
    target_iter = iter(target_loader)

    for i in range(len(target_loader)):
        target_image = next(target_iter)
        target_image = target_image.to(device)
        target_size = target_image.shape
        print(target_size)
        for j in range(len(visual_loader)):
            print("Processing image " + str(j))

            torch.cuda.empty_cache()
            raw_image = next(visual_iter)
            raw_image = raw_image.to(device)
            initial_raw = raw_image * 1.0
            source_image = generator(raw_image)
            # source_image = next(source_iter)
            # source_image = source_image.to(device)

            #initialize attack_raw
            attack_raw = raw_image
            attack_raw = attack_raw.to(device)
            attack_raw.requires_grad = True
            optimizer = Adam([attack_raw], learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.75)
            with torch.enable_grad():
                for f in range(max_iter_range):
                    optimizer.zero_grad()
                    attack_image = generator(attack_raw)
                    output_image = F.interpolate(attack_image, size= (target_size[2], target_size[3]), mode=interpolation_method)
                    loss_raw = F.mse_loss(attack_raw, initial_raw)
                    loss1 = F.mse_loss(attack_image, source_image)
                    loss2 = F.mse_loss(output_image, target_image)
                    #total_loss = loss1 + beta * loss2
                    total_loss = loss_raw + loss1 + beta * loss2
                    total_loss.backward(retain_graph=True)
                    optimizer.step()

                    if f % 50 == 0:
                        print('Iter_epoch %d, loss_raw: %.4f, loss1: %.4f, loss2: %.4f' % (f, loss_raw.item(), loss1.item(), loss2.item()))

                ###交替优化loss
                # for f in range(max_iter_range):
                #     optimizer.zero_grad()
                #     attack_image = generator(attack_raw)
                #     output_image = F.interpolate(attack_image, size=(target_size[2], target_size[3]),
                #                                  mode=interpolation_method)
                #     loss2 = F.mse_loss(output_image, target_image)
                #     loss2.backward(retain_graph=True)
                #     optimizer.step()
                #
                #     optimizer.zero_grad()
                #     attack_image = generator(attack_raw)
                #     loss1 = F.l1_loss(attack_image, source_image)
                #     loss1.backward(retain_graph=True)
                #     optimizer.step()
                #
                #     if f % 50 == 0:
                #         print('Iter_epoch %d, loss1: %.4f, loss2: %.4f' % (f, loss1.item(), loss2.item()))

                    scheduler.step()
            attack_image = generator(attack_raw)
            output_image = F.interpolate(attack_image, size=(target_size[2], target_size[3]), mode=interpolation_method)
            vutils.save_image(attack_image, 'results/attack/att_{}.png'.format(j), normalize=False)
            vutils.save_image(output_image, 'results/attack/out_{}.png'.format(j), normalize=False)
            #attack_raw = torch.clamp(attack_raw, 0.0, 1.0)
            vutils.save_image(attack_raw, 'results/attack/vis_raw_{}.png'.format(j), normalize=False)


if __name__ == '__main__':
    test_model()
