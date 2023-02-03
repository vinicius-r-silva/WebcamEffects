
import cv2
import torch
import numpy as np
from torchvision import transforms
from RobustVideoMatting.model import MattingNetwork
torch.backends.cudnn.benchmark = True

class rvm:
    def __init__(self, input_res = {'h' : 1080, 'w' : 1920}, output_res = {'h' : 720, 'w' : 1280}, back_img = None):
        self.device = 'cuda'
        self.variant = 'mobilenetv3'
        self.checkpoint = './rvm_mobilenetv3.pth'
        self.precision = torch.float32

        self.model = MattingNetwork(self.variant).eval().to(self.device)
        self.model.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)
        self.model = self.model.eval()

        self.TensorT = transforms.ToTensor()
        self.blurT = transforms.GaussianBlur(kernel_size=(19, 23), sigma=(20, 25))
        self.ScaleT = transforms.Resize((output_res['h'],output_res['w']))

        self.w = input_res['w']
        self.h = input_res['h']
        self.downsample_ratio = min(512 / max(self.h, self.w), 1)

        self.rec = [None] * 4  # Initial recurrent states are None

        if back_img is None:
            self.back_cpu = None
            self.back_gpu = None
            # print('loaded back image')
        else:
            self.back_cpu = self.TensorT(cv2.cvtColor(back_img, cv2.COLOR_BGR2RGB))
            self.back_gpu = self.back_cpu.to(self.device, self.precision, non_blocking=True).unsqueeze(0)
            # print('not loaded back image')

    def setup(self, frame):
        self.h = frame.shape[0]
        self.w = frame.shape[1]
        self.downsample_ratio = min(512 / max(self.h, self.w), 1)


    def blurBackground(self, frame):
        self.src_cpu = self.TensorT(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.src_gpu = self.src_cpu.to(self.device, self.precision, non_blocking=True).unsqueeze(0)
        self.blur_gpu = self.blurT(self.src_gpu)
        
        self.fgr, self.pha, *(self.rec) = self.model(self.src_gpu, *(self.rec), self.downsample_ratio)
        self.com = self.fgr * self.pha + self.blur_gpu * (1 - self.pha)
        self.res_gpu = self.ScaleT(self.com)

        self.blurred = self.res_gpu[0].cpu()
        self.blurred_np = (self.blurred.data.numpy()* 255).astype(np.uint8).transpose((1,2,0))
        # return cv2.cvtColor(self.blurred_np, cv2.COLOR_RGB2BGR)
        return self.blurred_np

    def replaceBackground(self, frame):
        self.src_cpu = self.TensorT(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.src_gpu = self.src_cpu.to(self.device, self.precision, non_blocking=True).unsqueeze(0)
        # self.blur_gpu = self.blurT(self.src_gpu)
        
        self.fgr, self.pha, *(self.rec) = self.model(self.src_gpu, *(self.rec), self.downsample_ratio)
        self.com = self.fgr * self.pha + self.back_gpu * (1 - self.pha)
        self.res_gpu = self.ScaleT(self.com)

        self.blurred = self.res_gpu[0].cpu()
        self.blurred_np = (self.blurred.data.numpy()* 255).astype(np.uint8).transpose((1,2,0))
        # return cv2.cvtColor(self.blurred_np, cv2.COLOR_RGB2BGR)
        return self.blurred_np

    # def __del__(self):
    #     del self.model
    #     del self.TensorT
    #     del self.blurT
    #     del self.ScaleT
    #     del self.back_cpu
    #     del self.back_cpu
    #     del self.src_gpu
    #     del self.blur_gpu
    #     del self.res_gpu
    #     del self.blurred
    #     del self.blurred_np
    #     del self.fgr
    #     del self.pha
    #     del self.rec
    #     del self.com
    #     del self.ScaleT
    #     del self.src_cpu
    #     del self.rec

    #     torch.cuda.empty_cache() 
