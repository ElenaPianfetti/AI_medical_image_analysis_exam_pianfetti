# imports
import numpy as np

RGB_MAP = {
            "A488": {'rgb': np.array([42, 255, 31]), 'range': [0, 107]},                # green
            "Cellmask Deep Red": {'rgb': np.array([255, 0, 25]), 'range': [0, 64]},     # red
            "mCherry": {'rgb': np.array([254, 255, 40]), 'range': [0, 191]},            # yellow
            "DAPI": {'rgb': np.array([19, 0, 249]), 'range': [0, 51]},                  # blue
            }
DEFAULT_CHANNELS = ("Cellmask Deep Red", "mCherry", "A488", "DAPI")


def convert_tensor_to_rgb(t, channels=DEFAULT_CHANNELS, vmax=[255, 255, 255, 255], rgb_map=RGB_MAP):
    # https://github.com/recursionpharma/rxrx1-utils/blob/trunk/rxrx/io.py
    colored_channels = []
    for i, channel in enumerate(channels):
        x = (t[:, :, i] / vmax[i]) / \
            ((rgb_map[channel]['range'][1] - rgb_map[channel]['range'][0]) / 255) + \
            rgb_map[channel]['range'][0] / 255
        x = np.where(x > 1., 1., x)
        x_rgb = np.array(
            np.outer(x, rgb_map[channel]['rgb']).reshape(x.shape[0], x.shape[1], 3),
            dtype=int)
        colored_channels.append(x_rgb)
    im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    im = np.where(im > 255, 255, im)
    return im, colored_channels