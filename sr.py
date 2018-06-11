# SR_PROCESS
import cv2
import torch 
from torch.autograd import Variable
import numpy as np
import torch.backends.cudnn as cudnn
import time
cudnn.fastest = True



class sr_process():

    def __init__(self, model_path):
        super(sr_process, self).__init__()
        self.model_path = model_path
        self.model = torch.load(model_path)
        self.model.cuda()
        self.scale = 3
        self.sr_time = 0
        if 'B' in self.model_path:
            self.tensor_4D = torch.Tensor(1,12,320,180)
        else:
            self.tensor_4D = torch.Tensor(1,3,640,360)

    def compute(self, frame, size):

        
        frame_in = frame.transpose(2,0,1)
        # bicubic
        if 'model_B' in self.model_path:
            self.tensor_4D = torch.Tensor(1,12,size[1]//2,size[0]//2)
            in_bicubic = cv2.resize(frame, (size[0]*self.scale,size[1]*self.scale), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            frame_in = frame_in.reshape(3, size[1]//2, 2, size[0]//2, 2)
            frame_in = frame_in.transpose(0, 2, 4, 1, 3)
            sub_frame = frame_in.reshape(12, size[1]//2, size[0]//2)
            self.tensor_4D[0,:,:,:] = torch.Tensor(sub_frame.astype(float)).mul_(1)
            input = Variable(self.tensor_4D.cuda(), volatile=True)
            output = self.model(input)
            torch.cuda.synchronize()
            output_img = (output.data[0].cpu().numpy()).transpose(1,2,0)
            torch.cuda.synchronize()

            out_ = output_img + in_bicubic
        # bilinear  
        if 'model_A' in self.model_path:
            self.sr_time = 0
            self.tensor_4D = torch.Tensor(1,3,size[1],size[0])
            self.tensor_4D[0,:,:,:] = torch.Tensor(frame_in.astype(float)).mul_(1)
            input = Variable(self.tensor_4D.cuda(), volatile=True)
            t0 = time.time()
            output = self.model(input)
            torch.cuda.synchronize()
            self.sr_time = time.time() - t0
            output_img = (output.data[0].cpu().numpy()).transpose(1,2,0)
            torch.cuda.synchronize()

            out_ = output_img

        out_[out_ > 255] = 255
        out_[out_ < 0] = 0
        out_sr = out_.astype(np.uint8)



        return out_sr, self.sr_time

