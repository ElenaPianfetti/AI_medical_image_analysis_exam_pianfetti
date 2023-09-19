import torch.utils.data as data
import os
import PIL
import numpy as np


class Dataset(data.Dataset):
    def __init__(self, df, transforms, path) -> None:
        plates = df['plate'].tolist()
        images = df['image'].tolist()
        self.concentrations = df['concentration'].tolist()
        self.mab_names = df['mab_name'].tolist()
        self.wells = df['well'].tolist()
        self.labels = df['label'].tolist()
        self.images = [os.path.join(path, plate, image) for plate, image in zip(plates, images)]
        self.image_names = ['/'.join(img.split('/')[-2:]) for img in self.images]
        # get transforms
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> dict:
        data = {}
        data['image_name'] = self.image_names[index]
        data['label'] = self.labels[index]
        data['concentration'] = self.concentrations[index]
        data['mab_name'] = self.mab_names[index]
        data['well'] = self.wells[index]

        # load image
        ch1 = PIL.Image.open(self.images[index] + '-ch1.png')
        ch2 = PIL.Image.open(self.images[index] + '-ch2.png')
        ch3 = PIL.Image.open(self.images[index] + '-ch3.png')
        ch4 = PIL.Image.open(self.images[index] + '-ch4.png')
        ch1 = np.array(ch1)
        ch2 = np.array(ch2)
        ch3 = np.array(ch3)
        ch4 = np.array(ch4)

        image = np.stack([ch1, ch2, ch3, ch4], axis=2)
        # cast to float
        image = image.astype(np.float32)
        
        image = self.transforms(image)
        data['image'] = image
        
        return data