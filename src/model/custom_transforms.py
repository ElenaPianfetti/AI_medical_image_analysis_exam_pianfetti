import numpy as np
import torch


class TurnOffTransform:
    def __init__(self, turn_off):
        self.turn_off = turn_off

    def __call__(self, image):
        empty = np.zeros_like(image[0])
        empty = torch.from_numpy(empty)
        image[self.turn_off-1] = empty
        return image
    

class MockTransform:
    def __init__(self):
        pass

    def __call__(self, image):
        return image
    

class SplitPatchTransform:

    def __init__(self, patch_size=512):
        self.patch_size = patch_size

    # Split current image into N patches
    def __call__(self, image):
        img_h, img_w = image.shape[1:]
        n_patches_h = img_h // self.patch_size
        n_patches_w = img_w // self.patch_size
        patches = []
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch = image[:, i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size]
                patches.append(patch)
        return torch.stack(patches)