import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import numpy as np
import torch.nn as nn
import pandas as pd
#pip install pandas
import cv2
#pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
from torchvision import transforms as T
from collections import defaultdict
import torch.nn.functional as F
from tqdm.notebook import tqdm, tnrange
#pip install ipywidgets
#pip install --upgrade jupyter ipywidgets tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
#python -m pip install scikit-learn  -i https://pypi.tuna.tsinghua.edu.cn/simple
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '5'


epochs = 2000
batch_size = 32
test_batch_size = 32

train_df = pd.read_csv('/home/wyc22/oil_paiting2/oil1.csv')
test_df = pd.read_csv('/home/wyc22/oil_paiting2/oil2.csv')

IMG_SHAPE = (256, 256)


col_names = list(train_df.columns.values)
ing_names = col_names[2:-1]


targets = ing_names
y2 = col_names[-1]


path = '/home/wyc22/oil_paiting2/oil'

from modle0 import foodnet
def main():
    transform_ds = T.Compose([
        T.ToPILImage(),
        T.Resize(IMG_SHAPE),
        T.RandomHorizontalFlip(),
        T.ColorJitter(hue=.05, saturation=.05),
        T.ToTensor()
    ])

    test_transform_ds = T.Compose([
        T.ToPILImage(),
        T.Resize(IMG_SHAPE),
        T.ToTensor()
    ])

    train_dataset = DataWrapper(path, train_df, True, transform_ds)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=batch_size,
                                               pin_memory=True, drop_last=False)

    test_dataset = DataWrapper(path, test_df, True, test_transform_ds)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=test_batch_size, num_workers=8,
                                              pin_memory=True, drop_last=False)

    model = foodnet()
    optimizer_s = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    sch_s = torch.optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=[10, 20, 30, 40], gamma=0.1)
    model.cuda()

    criterion = nn.BCEWithLogitsLoss()
    ###############################################
    best_acc = 0
    f1_scores = defaultdict(list)
    train_results = defaultdict(list)
    loss_dict = defaultdict(list)
    for i in tnrange(epochs, desc='Epochs'):
        print("Epoch ", i+1, 2000)
        lrs = []
        train_loss = []
        model.train()
        for img_data, target, target2 in tqdm(train_loader, desc='Training'):
            img_data, target, target2 = img_data.cuda(), target.cuda(), target2.cuda()

            output_s, output_m = model(img_data)
            #criterion = nn.CrossEntropyLoss()# FWD prop
            #loss_s = criterion(output_s, target2)
            loss_s = F.cross_entropy(output_s, target2)  # 单标签损失函数
            loss_m = criterion(output_m, target)
            loss = loss_s + 10 * loss_m
            train_loss.append(loss_s)
            optimizer_s.zero_grad()  # Zero out any cached gradients
            loss.backward()  # Backward pass
            optimizer_s.step()  # Update the weights

        # lrs.append(get_lr(optimizer_s))
        # ############################## 验证
        sch_s.step()
        model.eval()
        batch_loss_s = []
        batch_loss_m = []
        batch_acc1 = []
        batch_acc3 = []

        loss_val, correct, total = 0, 0, 0
        all_outputs = []
        all_targets = []

        for img_data, target, target2 in tqdm(test_loader, desc='Testing'):
            img_data, target, target2 = img_data.cuda(), target.cuda(), target2.cuda()
            output_s, output_m = model(img_data)

            loss_single = F.cross_entropy(output_s, target2)
            loss_multiple = criterion(output_m, target)

            accuracy_calculator = Accuracy()  # 创建 Accuracy 类的实例
            acc1, acc3 = accuracy_calculator(output_s, target2)
            dict = {"val_loss_m": loss_multiple.detach(), "val_loss_s": loss_single.detach(), "val_acc1": acc1, "val_acc3": acc3}
            batch_loss_s.append(dict["val_loss_s"])
            l_s = torch.stack(batch_loss_s).mean()
            batch_loss_m.append(dict["val_loss_m"])
            l_m = torch.stack(batch_loss_m).mean()
            batch_acc1.append(dict["val_acc1"])
            batch_acc3.append(dict["val_acc3"])
            epoch_acc1 = torch.stack(batch_acc1).mean()
            epoch_acc3 = torch.stack(batch_acc3).mean()

            result = {"val_loss": loss_single.item(), "val_acc1": epoch_acc1.item(),"val_acc3": epoch_acc3.item()}

            ######多标签
            total_batch = (target.size(0) * target.size(1))
            total += total_batch
            output_data = torch.sigmoid(output_m) >= 0.5
            target_data = (target == 1.0)
            for arr1, arr2 in zip(output_data, target_data):
                all_outputs.append(list(arr1.cpu().numpy()))
                all_targets.append(list(arr2.cpu().numpy()))
            c_acc = torch.sum((output_data == target_data.cuda()).to(torch.float)).item()

            correct += c_acc


            ###############
            # F1 Score



        all_outputs = np.array(all_outputs)
        all_targets = np.array(all_targets)
        f1score_macro = f1_score(y_true=all_targets, y_pred=all_outputs, average='macro')
        f1score_micro = f1_score(y_true=all_targets, y_pred=all_outputs, average='micro')

        f1_scores["macro_test"].append(f1score_macro)
        f1_scores["micro_test"].append(f1score_micro)
        loss_dict["sl_loss"].append(l_s.item())
        loss_dict["ml_loss"].append(l_m.item())
        #test_loss_val = loss_val / len(test_loader.dataset)

        torch.save(model.state_dict(), f'/home/wyc22/oil_paiting2/model/epoch_{i+1}_weights.pth')

        if epoch_acc1.item() > best_acc:
            best_acc = epoch_acc1.item()
            torch.save(model.state_dict(), f'/home/wyc22/oil_paiting2/model/best_epoch_{i+1}_weights.pth')

        if i+1 in [10, 100, 500, 1000, 2000]:
            epoch = list(range(1, len(f1_scores["macro_test"]) + 1))

            # 绘制曲线图
            plt.figure(figsize=(8, 6))
            plt.plot(epoch, f1_scores["macro_test"], marker='o', linestyle='-')
            plt.title('macro_test vs. Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('macro_test')
            plt.grid(True)
            # 检查目录是否存在，不存在则创建
            output_dir = '/home/wyc22/oil_paiting2/figure'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # 保存曲线图到指定文件夹
            output_path = os.path.join(output_dir, f"macro_test_{i+1}.png")
            plt.savefig(output_path)

            if i + 1 in [10, 100, 500, 1000, 2000]:
                epoch = list(range(1, len(f1_scores["micro_test"]) + 1))

                # 绘制曲线图
                plt.figure(figsize=(8, 6))
                plt.plot(epoch, f1_scores["micro_test"], marker='o', linestyle='-')
                plt.title('micro_test vs. Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('micro_test')
                plt.grid(True)
                # 检查目录是否存在，不存在则创建
                output_dir = '/home/wyc22/oil_paiting2/figure'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # 保存曲线图到指定文件夹
                output_path = os.path.join(output_dir, f"micro_test_{i + 1}.png")
                plt.savefig(output_path)


        print("==============")
        print("VALIDATION")
        print("单标签精度为top1：", result["val_acc1"])
        print("单标签精度为top3：", result["val_acc3"])
        #print(test_loss_val)


        #print("macro_test: ", f1_scores["macro_test"])
        #print("micro_test: ", f1_scores["micro_test"])

        result["train_loss"] = torch.stack(train_loss).mean().item()
        result["lrs"] = lrs

        print("多标签f1score_micro：", f1score_micro)
        print("多标签f1score_macro：", f1score_macro)
class DataWrapper(data.Dataset):
    ''' Data wrapper for pytorch's data loader function '''
    def __init__(self, path, image_df, resize, transform):
        self.dataset = image_df
        self.resize = resize
        self.transform = transform
        self.path = path

    def __getitem__(self, index):
        c_row = self.dataset.iloc[index]
        target_arr = []
        for item in c_row[targets].values:
            target_arr.append(item)
        target_m = torch.from_numpy(np.array(target_arr)).float()

        target_arr2=[]
        target_arr2.append(int(c_row[y2]))
        #y2_value = c_row[y2]
        #target_arr2 = [0]*4
        #target_arr2[y2_value-1] = y2_value



        image_path=self.path+c_row['path']
        target_s = torch.from_numpy(np.array(c_row[y2]))  #image and target
        #read as rgb image, resize and convert to range 0 to 1
        image = cv2.imread(image_path, 1)
        if self.resize:
            image = self.transform(image)
        else:
            image = image/255.0
        return image, target_m, target_s

    def __len__(self):
        return self.dataset.shape[0]

class Accuracy:
    def __call__(self, out, labels):
        preds=out.argmax(dim=1)
        acc1 = torch.tensor(sum(preds == labels).float().item() / len(preds))
        maxk = max((1, 3))
        y_resize = labels.view(-1, 1)
        _, pred_5 = out.topk(maxk, 1, True, True)
        correct = torch.eq(pred_5, y_resize).sum().float().item()
        acc3 = torch.tensor( correct / len(preds))
        return acc1, acc3




############################################################################################
if __name__ == '__main__':
    # 导入需要的库
    main()

    # 其他代码继续在这里写
    # ...
