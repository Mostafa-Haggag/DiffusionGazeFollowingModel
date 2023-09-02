import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import math 
from math import prod

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution):
    # width is 300 and height is 400 and reosolution is 224
    # the x min and ymin and all the rest are values 
    head_box = np.array([x_min / width, y_min / height, x_max / width, y_max / height]) * resolution # scaling it betweee 224in an array array([ 35,  -3, 118,  82])
    head_box = head_box.astype(int)# make it a tyype int in case there is something 
    head_box = np.clip(head_box, 0, resolution - 1) # make sure that there is no negative valeus and everything in range of 0 to resolution -1
    head_channel = np.zeros((resolution, resolution), dtype=np.float32)# create a matrix 224,224
    # head box has the values array([ 35,   0, 118,  82])
    head_channel[head_box[1] : head_box[3], head_box[0] : head_box[2]] = 1
    head_channel = torch.from_numpy(head_channel) # turn into tensor and return it 

    return head_channel
def get_label_map_1(img, pt, sigma, pdf="Gaussian"):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)
    # Check that any part of the gaussian is in-bounds
    ul = [
        pt[0].round().int().item() - 3 * sigma,
        pt[1].round().int().item() - 3 * sigma,
    ]
    br = [
        pt[0].round().int().item() + 3 * sigma + 1,
        pt[1].round().int().item() + 3 * sigma + 1,
    ]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if pdf == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    elif pdf == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma**2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] += g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    img = img / np.max(img)  # normalize heatmap so it has max value of 1

    return to_torch(img)

def get_label_map(img, pt, sigma, pdf="Gaussian"):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if pdf == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
    elif pdf == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma**2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0] : img_y[1], img_x[0] : img_x[1]] += g[g_y[0] : g_y[1], g_x[0] : g_x[1]]
    img = img / np.max(img)  # normalize heatmap so it has max value of 1

    return to_torch(img)

def get_multi_hot_map(gaze_pts, out_res):
    w, h = out_res
    target_map = np.zeros((h, w))
    for p in gaze_pts:
        if p[0] >= 0:
            x, y = map(int, [p[0] * w.float(), p[1] * h.float()])
            x = min(x, w - 1)
            y = min(y, h - 1)
            target_map[y, x] = 1

    return target_map

def get_auc(heatmap, onehot_im, is_im=True):
    if is_im:
        auc_score = roc_auc_score(np.reshape(onehot_im, onehot_im.size), np.reshape(heatmap, heatmap.size))
    else:
        auc_score = roc_auc_score(onehot_im, heatmap)
    return auc_score

def get_ap(label, pred):
    return average_precision_score(label, pred)


def get_heatmap_peak_coords(heatmap):
    idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    pred_y, pred_x = map(float, idx)
    return pred_x, pred_y

def get_l2_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_angular_error(p1, p2):
    norm_p1 = (p1[0] ** 2 + p1[1] ** 2) ** 0.5
    norm_p2 = (p2[0] ** 2 + p2[1] ** 2) ** 0.5
    cos_sim = (p1[0] * p2[0] + p1[1] * p2[1]) / (norm_p2 * norm_p1 + 1e-6)
    cos_sim = np.maximum(np.minimum(cos_sim, 1.0), -1.0)

    return np.arccos(cos_sim) * 180 / np.pi


def get_memory_format(config):
    # set to false 
    if config.Dataset.channels_last:
        return torch.channels_last

    return torch.contiguous_format

def unravel_index(
    indices: torch.LongTensor,
    shape,
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.
    This is a `torch` implementation of `numpy.unravel_index`.
    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).
    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord
def batch_argmax(tensor, batch_dim=1):
    """
    Assumes that dimensions of tensor up to batch_dim are "batch dimensions"
    and returns the indices of the max element of each "batch row".
    More precisely, returns tensor `a` such that, for each index v of tensor.shape[:batch_dim], a[v] is
    the indices of the max element of tensor[v].
    """
    if batch_dim >= len(tensor.shape):
        raise NoArgMaxIndices()
    batch_shape = tensor.shape[:batch_dim]
    non_batch_shape = tensor.shape[batch_dim:]
    flat_non_batch_size = prod(non_batch_shape)
    tensor_with_flat_non_batch_portion = tensor.reshape(*batch_shape, flat_non_batch_size)

    dimension_of_indices = len(non_batch_shape)

    # We now have each batch row flattened in the last dimension of tensor_with_flat_non_batch_portion,
    # so we can invoke its argmax(dim=-1) method. However, that method throws an exception if the tensor
    # is empty. We cover that case first.
    if tensor_with_flat_non_batch_portion.numel() == 0:
        # If empty, either the batch dimensions or the non-batch dimensions are empty
        batch_size = prod(batch_shape)
        if batch_size == 0:  # if batch dimensions are empty
            # return empty tensor of appropriate shape
            batch_of_unraveled_indices = torch.ones(*batch_shape, dimension_of_indices).long()  # 'ones' is irrelevant as it will be empty
        else:  # non-batch dimensions are empty, so argmax indices are undefined
            raise NoArgMaxIndices()
    else:   # We actually have elements to maximize, so we search for them
        indices_of_non_batch_portion = tensor_with_flat_non_batch_portion.argmax(dim=-1)
        batch_of_unraveled_indices = unravel_index(indices_of_non_batch_portion, non_batch_shape)

    if dimension_of_indices == 1:
        # above function makes each unraveled index of a n-D tensor a n-long tensor
        # however indices of 1D tensors are typically represented by scalars, so we squeeze them in this case.
        batch_of_unraveled_indices = batch_of_unraveled_indices.squeeze(dim=-1)
    return batch_of_unraveled_indices
class NoArgMaxIndices(BaseException):

    def __init__(self):
        super(NoArgMaxIndices, self).__init__(
            "no argmax indices: batch_argmax requires non-batch shape to be non-empty")