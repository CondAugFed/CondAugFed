import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, rotate
import random
import cv2

# @torch.no_grad()
def img2patch(x:torch.Tensor, k:int, s:int) -> torch.Tensor:
    b, c, h, w = x.size()
    orig_shape = [b, c, h, w]

    if s*int(h/s) == h:
        padh = 0
        padw = 0
    else:
        nh = s * int(h / s) + k
        nw = s * int(w / s) + k
        padh = max(nh-h, 0)
        padw = max(nw-w, 0)


    padded = F.pad(x, pad=(0, padw, 0, padh), mode='reflect')
    padded_shape = list(padded.shape)
    patches = padded.unfold(2,k,s).unfold(3,k,s)
    unfolded_shape = list(patches.shape)
    # print(unfolded_shape) # [1, 3, 10, 16, 64, 64]
    patches = patches.contiguous().view(b, c, -1, k, k).permute(0,2,1,3,4)
    return patches, orig_shape, padded_shape, unfolded_shape

# @torch.no_grad()
def patch2img(x:torch.Tensor, k:int, s:int, orig_shape:list, padded_shape:list, unfolded_shape:list):
    _, _, nh, nw = padded_shape
    b, c, h, w = orig_shape

    x = x.permute(0,2,1,3,4)  # b, 3, 160, 64, 64
    
    if s*int(h/s) == h:
        padh = 0
        padw = 0
    else:
        padh = max(s * int(h / s) + k- h, 0) 
        padw = max(s * int(w / s) + k- w, 0) 

    unfolded = x.view(unfolded_shape).permute(0, 1, 4, 5, 2, 3).contiguous()
    unfolded = unfolded.view(b, -1, unfolded.size(4)*unfolded.size(5))

    ones = torch.ones_like(unfolded)
    # unfolded = x
    recon = F.fold(unfolded, output_size=(nh,nw), kernel_size=k, stride=s)
    normalize_mat = F.fold(ones, output_size=(nh,nw), kernel_size=k, stride=s)

    recon = torch.mul(recon, 1/normalize_mat)
    recon = recon[:, :, :recon.size(2)-padh, :recon.size(3)-padw]
    return recon

def RGB2YUV(img):
    """
    img.size() = [B, C, H, W]
    """
    r = img[0, :, :] # [B, H, W]
    g = img[1, :, :]
    b = img[2, :, :]

    y = 0.257 * r + 0.504 * g + 0.098 * b + 16
    u = -0.148 * r - 0.291 * g + 0.439 * b + 128
    v = 0.439 * r - 0.368 * g - 0.071 * b + 128

    return torch.stack([y,u,v], dim=0)

def YUV2RGB(img):
    y = img[0, :, :] 
    u = img[1, :, :]
    v = img[2, :, :]

    y -= 16
    u -= 128
    v -= 128

    r = 1.164 * y + 1.596 * v
    g = 1.164 * y - 0.392 * u - 0.813 * v
    b = 1.164 * y + 2.017 * u

    return torch.stack([r,g,b], dim=0)

class ToYUVTensor:
    def __call__(self, pic):
        # to_tensor convert [0 255] to [0 1] by dividing 255.
        pic = to_tensor(pic) * 255
        return RGB2YUV(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToRGBTensor:
    def __call__(self, pic):
        # return self.normalize_scale(YUV2RGB(pic))
        return YUV2RGB(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ReScale:
    def __init__(self, maxi:int):
        self.m = maxi

    def __call__(self, pic):
        if self.m == 1:
            pic = torch.clamp(pic, min=0, max=1)
            return (pic-0.5)*2
        elif self.m == 255:
            pic = torch.clamp(pic, min=0, max=255)
            return (pic/255-0.5)*2

    def __repr__(self):
        return self.__class__.__name__ +'()'

class BGRtoRGB:
    def __call__(self, pic):
        return pic[..., ::-1].copy()
    
    def __repr__(self):
        return self.__class__.__name__

class RandomRotate90:
    def __call__(self, pic):
        angles = [0, 90, 180, 270]
        idx = random.randint(1, 4)-1
        return rotate(pic, angle=angles[idx])

    def __repr__(self):
        return self.__class__.__name__

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std/255 + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)