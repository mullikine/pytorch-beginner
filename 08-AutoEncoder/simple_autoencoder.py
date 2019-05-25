__author__ = 'SherlockLiao'

import os

import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./data', transform=img_transform,download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, z):
        x = self.decoder(z)
        return x

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = decoder()
model.load_state_dict(torch.load('./sim_autoencoder.pth'))
model.eval()

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider,Button

blank = np.ones((28,28))
fig,ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
ax.matshow(blank)
z1_ax = plt.axes([0.25,0.1,0.65,0.03])
z2_ax = plt.axes([0.25,0.15,0.65,0.03])
z3_ax = plt.axes([0.25,0.2,0.65,0.03])

z1 = Slider(z1_ax,'z1',0,10,valinit=0,valstep=0.1)
z2 = Slider(z2_ax,'z2',0,10,valinit=0,valstep=0.1)
z3 = Slider(z3_ax,'z3',0,10,valinit=0,valstep=0.1)

def update(val):
    z1_val = z1.val
    z2_val = z2.val
    z3_val = z3.val
    z = torch.tensor([z1_val,z2_val,z3_val])
    out_img = model(z)
    out_img = np.resize(out_img.detach().numpy(),(28,28))
    ax.matshow(out_img)
    fig.canvas.draw_idle()

z1.on_changed(update)
z2.on_changed(update)
z3.on_changed(update)
plt.show()




# model = autoencoder()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(
#     model.parameters(), lr=learning_rate, weight_decay=1e-5)

# for epoch in range(num_epochs):
#     for data in dataloader:
#         img, _ = data
#         img = img.view(img.size(0), -1)
#         img = Variable(img)
#         # ===================forward=====================
#         output = model(img)
#         loss = criterion(output, img)
#         # ===================backward====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # ===================log========================
#     print('epoch [{}/{}], loss:{:.4f}'
#           .format(epoch + 1, num_epochs, loss.data[0]))
#     if epoch % 10 == 0:
#         pic = to_img(output.cpu().data)
#         save_image(pic, './mlp_img/image_{}.png'.format(epoch))

# torch.save(model.state_dict(), './sim_autoencoder.pth')
