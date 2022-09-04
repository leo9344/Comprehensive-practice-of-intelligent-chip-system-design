import torch
from matplotlib import pyplot as plt
import numpy as np

class visualizer(object):
    def __init__(self, num, y_label) -> None:
        self.data = []
        self.x = np.arange(0, num)
        self.labels = []
        self.y_label = y_label

    def load(self, data_path, label):
        data = torch.load(data_path)
        self.data.append(data)
        self.labels.append(label)
        print(label, max(data), min(data))
    def show(self):
        plt.cla()
        plt.clf()

        for i in range(len(self.data)):
            plt.plot(self.x, self.data[i], label = self.labels[i])

        plt.xlabel("Epochs")
        plt.ylabel(self.y_label)
        plt.legend()
        plt.show()

def load_acc(root_path, tag=""):
    acc_visualizer.load(f"./{root_path}/train_acc.txt", f"{tag} Train Acc")
    # acc_visualizer.load(f"./{root_path}/test_acc.txt", f"{tag} Test Acc")

def load_loss(root_path, tag=""):
    loss_visualizer.load(f"./{root_path}/train_loss.txt", f"{tag} Train Loss")
    # loss_visualizer.load(f"./{root_path}/test_loss.txt", f"{tag} Test Loss")

acc_visualizer = visualizer(20, "ACC")
loss_visualizer = visualizer(20, "Loss")

root_path = "./with_aug"
load_acc(root_path, "f32+Aug")
load_loss(root_path, "f32+Aug")

root_path = "./original"
load_acc(root_path,"f32")
load_loss(root_path,"f32")

root_path = "./float16"
load_acc(root_path,"f16")
load_loss(root_path,"f16")

root_path = "./float16+aug"
load_acc(root_path,"f16+Aug")
load_loss(root_path,"f16+Aug")

acc_visualizer.show()
loss_visualizer.show()