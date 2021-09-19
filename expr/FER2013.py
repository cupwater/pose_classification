import os
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import math
import numpy as np
from dataset.Sampler import RandomSampler, BatchSampler
from dataset import transform
from dataset.RandAugment import RandomAugment
import shutil
from torchvision import transforms
import random
import copy
import torch

classes_map = {
    'angry': 0,
    'disgusted': 1,
    'fearful': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprised': 6
}

# mean = [0.578, 0.582, 0.591]
# std = [0.228, 0.228, 0.229]
# mean = [0.392, 0.439, 0.549]  # RAF-DB  BGR
# std = [0.263, 0.265, 0.286]
# mean = [0.506, 0.506, 0.506]  # FER+
# std = [0.255, 0.255, 0.255]
mean = [0.508, 0.508, 0.508]  # FER2013
std = [0.255, 0.255, 0.255]


map_classes = {
    0: 'angry',
    1: 'disgusted',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised'
}


def load_traindata(path):
    imgs, labels = [], []
    for emotion in os.listdir(path):
        label = classes_map[emotion]
        for img in os.listdir(os.path.join(path, emotion)):
            data = cv2.imread(os.path.join(path, emotion, img))
            imgs.append(data)
            labels.append(label)

    del data, img

    return imgs, labels


def load_traindata_land(sup_path):
    imgs, labels, landmarks = [], [], []
    for emotion in os.listdir(sup_path):
        label = classes_map[emotion]
        for img in os.listdir(os.path.join(sup_path, emotion)):
            data = cv2.imread(os.path.join(sup_path, emotion, img))
            landmark_name = img.split('.')[0] + '.txt'
            landmark = np.loadtxt('data/Datawhale/landmark_5/training/' + os.path.join(emotion, landmark_name)).astype(np.int)
            imgs.append(data)
            labels.append(label)
            landmarks.append(landmark)

    del data, img

    return imgs, labels, landmarks


def load_valdata(path):
    imgs, labels = [], []
    for emotion in os.listdir(path):
        label = classes_map[emotion]
        for img in os.listdir(os.path.join(path, emotion)):
            data = cv2.imread(os.path.join(path, emotion, img))
            imgs.append(data)
            labels.append(label)

    del data, img

    return imgs, labels


def load_valdata_land(path):
    imgs, labels, landmarks = [], [], []
    for emotion in os.listdir(path):
        label = classes_map[emotion]
        for img in os.listdir(os.path.join(path, emotion)):
            data = cv2.imread(os.path.join(path, emotion, img))
            landmark_name = img.split('.')[0] + '.txt'
            landmark = np.loadtxt('data/Datawhale/landmark_5/val/' + os.path.join(emotion, landmark_name)).astype(np.int)
            imgs.append(data)
            labels.append(label)
            landmarks.append(landmark)

    del data, img

    return imgs, labels, landmarks


def load_testdata(path):
    imgs, names = [], []
    for img in os.listdir(path):
        data = cv2.imread(os.path.join(path, img))
        # data = Image.open(os.path.join(path, img))
        # data = data.convert('RGB')
        names.append(img)
        imgs.append(data)

    del data, img

    return imgs, names,


def load_testdata_land(path):
    imgs, names, landmarks = [], [], []
    for img in os.listdir(path):
        data = cv2.imread(os.path.join(path, img))
        names.append(img)
        landmark_name = img.split('.')[0] + '.txt'
        landmark = np.loadtxt('data/Datawhale/landmark_5/test/' + landmark_name).astype(np.int)
        imgs.append(data)
        landmarks.append(landmark)

    del data, img

    return imgs, names, landmarks


def train_dataloader(path, batch_size=32, num_workers=4, pin_memory=True):
    imgs, labels = load_traindata(path)
    data = FER2013(imgs, labels, is_train=True)
    # sampler_data = RandomSampler(data, replacement=True, num_samples=batch_size)
    # batch_data = BatchSampler(sampler_data, batch_size=batch_size, drop_last=True)
    # dataloader = DataLoader(data, batch_sampler=batch_data, num_workers=num_workers, pin_memory=pin_memory)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=num_workers, pin_memory=pin_memory)

    # del imgs, labels, data, sampler_data, batch_data

    return dataloader


def train_dataloader_land(path, batch_size=32, num_workers=2, pin_memory=True):
    imgs, labels, landmarks = load_traindata_land(path)
    data = FER2013_land(imgs, labels, landmarks, is_train=True)
    # sampler_data = RandomSampler(data, replacement=True, num_samples=batch_size)
    # batch_data = BatchSampler(sampler_data, batch_size=batch_size, drop_last=True)
    # dataloader = DataLoader(data, batch_sampler=batch_data, num_workers=num_workers, pin_memory=pin_memory)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=num_workers, pin_memory=pin_memory)

    # del imgs, labels, data, sampler_data, batch_data

    return dataloader


def val_dataloader(path, batch_size, num_workers, pin_memory=True):
    imgs, labels = load_valdata(path)
    data = FER2013(imgs, labels, is_train=False)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=num_workers, pin_memory=pin_memory)

    del data, imgs, labels

    return dataloader


def val_dataloader_land(path, batch_size, num_workers, pin_memory=True):
    imgs, labels, landmarks = load_valdata_land(path)
    data = FER2013_land(imgs, labels, landmarks, is_train=False)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                             drop_last=False, num_workers=num_workers, pin_memory=pin_memory)

    del data, imgs, labels

    return data_loader


def test_dataloader(path, batch_size, num_workers, pin_memory=True):
    imgs, names = load_testdata(path)
    data = test_dataset(imgs, names)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=num_workers, pin_memory=pin_memory)

    del data, imgs, names

    return dataloader


def test_dataloader_land(path, batch_size, num_workers, pin_memory=True):
    imgs, names, landmarks = load_testdata_land(path)
    data = test_dataset_land(imgs, names, landmarks)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=num_workers, pin_memory=pin_memory)

    del data, imgs, names

    return dataloader


class FER2013(Dataset):
    def __init__(self, imgs, labels, is_train):
        super(FER2013, self).__init__()
        self.is_train = is_train
        self.imgs = imgs
        self.labels = labels
        size = 112
        if self.is_train:
            self.transforms = transform.Compose([
                transform.Resize((size, size)),
                transform.PadandRandomCrop(border=6, cropsize=(size, size)),
                transform.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                transform.Normalize(mean, std),
                transform.ToTensor(),
                transforms.RandomRotation(15)
                ])
        else:
            self.transforms = transform.Compose([
                transform.Resize((size, size)),
                transform.Normalize(mean, std),
                transform.ToTensor(),
            ])

    def __getitem__(self, idx):
        img, label = self.imgs[idx], self.labels[idx]

        return self.transforms(img), label

    def __len__(self):
        return len(self.imgs)


class FER2013_land(Dataset):
    def __init__(self, imgs, labels, landmarks, is_train):
        super(FER2013_land, self).__init__()
        self.is_train = is_train
        self.imgs = imgs
        self.labels = labels
        self.landmarks = landmarks
        size = 48
        if self.is_train:
            self.transforms = transform.Compose([
                transform.Resize((size, size)),
                transform.PadandRandomCrop(border=6, cropsize=(size, size)),
                transform.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                transform.Normalize(mean, std),
                transform.ToTensor(),
                transforms.RandomRotation(15)
                ])
        else:
            self.transforms = transform.Compose([
                transform.Resize((size, size)),
                transform.Normalize(mean, std),
                transform.ToTensor(),
            ])

    def __getitem__(self, idx):
        img, label, landmark = self.imgs[idx], self.labels[idx], self.landmarks[idx]
        if landmark[0][0] == 0:
            img_crop = copy.copy(img)
        else:
            img_crop = copy.copy(img)
            hight = math.floor(0.08 * img.shape[0])
            wide = math.floor(0.2 * img.shape[1])
            if landmark[0][1] < landmark[1][1]:
                img_crop[:landmark[0][1] - hight, :, :] = 0
                up = landmark[1][1]
            else:
                img_crop[:landmark[1][1] - hight, :, :] = 0
                up = landmark[0][1]
            if landmark[3][1] > landmark[4][1]:
                img_crop[landmark[3][1] + hight:, :, :] = 0
                down = landmark[4][1]
            else:
                img_crop[landmark[4][1] + hight:, :, :] = 0
                down = landmark[3][1]
            img_crop[up + hight:down - hight, :, :] = 0
            img_crop[:, :landmark[0][0] - wide, :] = 0
            img_crop[:, landmark[1][0] + wide:, :] = 0

        return self.transforms(img), label, self.transforms(img_crop)

    def __len__(self):
        return len(self.imgs)


class test_dataset(Dataset):
    def __init__(self, imgs, names):
        super(test_dataset, self).__init__()
        self.imgs = imgs
        self.names = names
        self.transforms = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            # transforms.FiveCrop(38),
            # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),

        ])

    def __getitem__(self, idx):
        img, name = self.imgs[idx], self.names[idx]
        if self.transforms is not None:
            img = self.transforms(Image.fromarray(img))
        return img, name

    def __len__(self):
        return len(self.names)


class test_dataset_land(Dataset):
    def __init__(self, imgs, names, landmarks):
        super(test_dataset_land, self).__init__()
        self.imgs = imgs
        self.names = names
        self.landmarks = landmarks
        self.transforms = transform.Compose([
            transform.Resize((48, 48)),
            transform.Normalize(mean, std),
            transform.ToTensor(),
        ])

    def __getitem__(self, idx):
        img, name, landmark = self.imgs[idx], self.names[idx], self.landmarks[idx]
        img_crop = copy.copy(img)
        if landmark[0][0] != 0:
            hight = math.floor(0.08 * img.shape[0])
            wide = math.floor(0.2 * img.shape[1])
            if landmark[0][1] < landmark[1][1]:
                img_crop[:landmark[0][1] - hight, :, :] = 0
                up = landmark[1][1]
            else:
                img_crop[:landmark[1][1] - hight, :, :] = 0
                up = landmark[0][1]
            if landmark[3][1] > landmark[4][1]:
                img_crop[landmark[3][1] + hight:, :, :] = 0
                down = landmark[4][1]
            else:
                img_crop[landmark[4][1] + hight:, :, :] = 0
                down = landmark[3][1]
            img_crop[up + hight:down - hight, :, :] = 0
            img_crop[:, :landmark[0][0] - wide, :] = 0
            img_crop[:, landmark[1][0] + wide:, :] = 0

        return self.transforms(img), name, self.transforms(img_crop)

    def __len__(self):
        return len(self.names)


def move_image_orig(result, test_data_path, train_data_path, alpha=1/3):
    labels = []
    number_per_emotion = {
        'surprise': 0,
        'fear': 0,
        'disgust': 0,
        'happy': 0,
        'sadness': 0,
        'anger': 0,
        'neutral': 0
    }
    for emotion in os.listdir(train_data_path):
        label = classes_map[emotion]
        for _ in os.listdir(os.path.join(train_data_path, emotion)):
            labels.append(label)
    labels = np.array(labels)
    for i in range(len(classes_map)):
        indices = np.where(labels == i)[0]
        n_imgs = len(indices)
        number_per_emotion[map_classes[i]] = n_imgs
    number_per_emotion = sorted(number_per_emotion.items(), key=lambda e: e[1])  # [('',154),('',987),...
    # org
    # N_max = int(number_per_emotion[6][1])
    # for i in range(len(classes_map)):
    #     N_i = int(number_per_emotion[6-i][1])
    #     emotion_i = classes_map[number_per_emotion[i][0]]
    #
    #     rate = (N_i / N_max) ** alpha
    #     for idex in range(len(list)):
    #         image_name = list[idex][0]
    #         image_label = int(list[idex][1])
    #         random_seed = random.random()
    #         if image_label == emotion_i and random_seed <= rate:
    #             src = os.path.join(test_data_path, image_name)
    #             new_name = str(image_name.split('.')[0]) + str(random.randint(0, 1000)) + '.jpg'
    #             dst = os.path.join(train_data_path, str(map_classes[image_label]), new_name)
    #             shutil.copy(src, dst)

    # my
    pink = classes_map[number_per_emotion[0][0]]
    for idex in range(len(result)):
        image_name = result[idex][0]
        image_label = int(result[idex][1])
        random_seed = random.random()
        if image_label == pink and random_seed <= 0.3:
            src = os.path.join(test_data_path, image_name)
            new_name = str(image_name.split('.')[0]) + str(random.randint(0, 1000)) + '.jpg'
            dst = os.path.join(train_data_path, str(map_classes[image_label]), new_name)
            shutil.copy(src, dst)
    del labels, indices, result


def move_image(result, test_data_path, train_data_path, pink, rate_i):
    for idex in range(len(result)):
        image_name = result[idex][0]
        image_label = int(result[idex][1])
        if float(result[idex][2]) >= 0.95:
            random_seed = random.random()
            if image_label == pink and random_seed <= rate_i:
                src = os.path.join(test_data_path, image_name)
                new_name = str(image_name.split('.')[0]) + str(random.randint(0, 1000)) + '.jpg'
                dst = os.path.join(train_data_path, str(map_classes[image_label]), new_name)

                shutil.copy(src, dst)


def main():
    path = '../data/RAF-DB/train'
    # data_loader_x, data_loader_u = train_data_loader(path=path)
    # imgs_x, labels_x, imgs_u,labels_u = load_data(path,labeled_ratio=0.01)
    # for i,(imgs,labels) in enumerate(data_loader_x):
    #     print(imgs)
    # data_loader_x = iter(data_loader_x)
    # data_loader_u = iter(data_loader_u)
    # for i in range(10):
    #     img_weak_x, img_strong_x, label_x = next(data_loader_x)


if __name__ == '__main__':
    main()
