import torch
from scipy.misc import imresize

def image_to_tensor(img, size=256):
    """
    Converts an image array into a Tensor of fixed size
    :param img: image to transpose
    :return: image tensor size x size
    """
    state = imresize(img,[size,size], 'nearest')

    input_tensor = torch.FloatTensor(state.reshape(-1,3,size,size)/255.0)

    return input_tensor