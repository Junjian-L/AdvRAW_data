from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from network import Generator

import torch
import imageio
import numpy as np
import math
import sys

from load_data import LoadData, LoadVisualData
from msssim import MSSSIM

to_image = transforms.Compose([transforms.ToPILImage()])

np.random.seed(0)
torch.manual_seed(0)
dataset_dir = 'images/'
# Dataset size
TRAIN_SIZE = 46839
TEST_SIZE = 1204

train_batch_size = 8
test_batch_size = 2

learning_rate = 0.0008 # 0.001
num_train_epochs = 50 # 50
def train_model():

    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    train_dataset = LoadData(dataset_dir, TRAIN_SIZE, test=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=1,
                              pin_memory=True, drop_last=True)

    test_dataset = LoadData(dataset_dir, TEST_SIZE, test=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=1,
                             pin_memory=True, drop_last=False)

    visual_dataset = LoadVisualData(dataset_dir, 10)
    visual_loader = DataLoader(dataset=visual_dataset, batch_size=1, shuffle=False, num_workers=0,
                               pin_memory=True, drop_last=False)

    generator = Generator().to(device)
    generator = torch.nn.DataParallel(generator)
    optimizer = Adam(params=generator.parameters(), lr=learning_rate)

    MSE_loss = torch.nn.MSELoss()
    MS_SSIM = MSSSIM()

    for epoch in range(num_train_epochs):

        torch.cuda.empty_cache()

        train_iter = iter(train_loader)
        generator.train()
        for i in range(len(train_loader)):

            optimizer.zero_grad()
            raw, dslr = next(train_iter)

            raw = raw.to(device)
            dslr = dslr.to(device)

            enhanced = generator(raw)

            loss_mse = MSE_loss(enhanced, dslr)
            loss_ssim = MS_SSIM(enhanced, dslr)
            total_loss = loss_mse #+ (1-loss_ssim) * 0.5

            total_loss.backward()
            optimizer.step()

            if i == 0:

                # Save the model that corresponds to the current epoch
                torch.save(generator.state_dict(), "models/generator" + "_epoch_" + str(epoch) + ".pth")

                generator.eval()
                with torch.no_grad():

                    visual_iter = iter(visual_loader)
                    for j in range(len(visual_loader)):
                        torch.cuda.empty_cache()

                        raw_image = next(visual_iter)
                        raw_image = raw_image.to(device)

                        enhanced = generator(raw_image.detach())
                        enhanced = np.asarray(to_image(torch.squeeze(enhanced.detach().cpu())))

                        imageio.imwrite("results/generator_" + str(j) + "_epoch_" +
                                        str(epoch) + ".jpg", enhanced)

            if i % 10 ==0:
                print("Train_results_Epoch %d, batch_idx %d, mse: %.4f, ms-ssim: %.4f" % (epoch, i, loss_mse, loss_ssim))

        loss_mse_eval = 0
        loss_ssim_eval = 0

        generator.eval()
        with torch.no_grad():

            test_iter = iter(test_loader)
            for j in range(len(test_loader)):
                raw, dslr = next(test_iter)
                raw = raw.to(device)
                dslr = dslr.to(device)

                enhanced = generator(raw)

                loss_mse_temp = MSE_loss(enhanced, dslr).item()
                loss_ssim_temp = MS_SSIM(enhanced, dslr).item()

                loss_mse_eval += loss_mse_temp
                loss_ssim_eval += loss_ssim_temp

            loss_mse_eval = loss_mse_eval / TEST_SIZE
            loss_ssim_eval = loss_ssim_eval / TEST_SIZE
            print("Test_results_Epoch %d, mse: %.4f, ms-ssim: %.4f" % (epoch, loss_mse_eval, loss_ssim_eval))



if __name__ == '__main__':
    train_model()