'''
Author  : Leixy
Version : 1.0.0 
'''
from tarfile import is_tarfile
import time
from typing import Optional
import cv2
import torch 
import pdb
import argparse
import numpy as np
import os 
from collections import OrderedDict
import torch.nn.functional as F
from thop import profile,clever_format
from model.segnet import SegMattingNet
from collections import OrderedDict
from postprocess import postprocess,cal_optical_flow_tracking,fuse_optical_flow_tracking,threshold_mask

parser = argparse.ArgumentParser(description='human matting')
parser.add_argument('--model', default='./pre_trained/erd_seg_matting/model/ckpt_lastest.pth', help='preTrained model')
parser.add_argument('--without_gpu', action='store_true', default=False, help='use cpu')
parser.add_argument('--input_w',type=int,default=256)
parser.add_argument('--input_h',type=int,default=256)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.set_grad_enabled(False)
INPUT_SIZE_W = 256
INPUT_SIZE_H = 256

#################################
#----------------
if args.without_gpu:
    print("use CPU !")
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print("----------------------------------------------------------")
        print("|       use GPU !      ||   Available GPU number is {} !  |".format(n_gpu))
        print("----------------------------------------------------------")

        device = torch.device('cuda')

#################################
#---------------
def load_model(args):
    print('Loading model from {}...'.format(args.model))
    myModel=SegMattingNet()
    '''
    if args.without_gpu:
        myModel = torch.load(args.model, map_location=lambda storage, loc: storage)
    else:
        myModel = torch.load(args.model)
    '''
    new_dict=OrderedDict()
    ckpt=torch.load(args.model)
    state_dict=ckpt['state_dict']
    for k,v in state_dict.items():
        if "module" in k:
            name=k[7:]
        else:
            name=k
        new_dict[name]=v
    myModel.load_state_dict(new_dict)
    myModel.eval()
    myModel.to(device)
    
    return myModel

def camera_seg(args, net):

    videoCapture = cv2.VideoCapture(0)
    videoCapture.set(3,1280)
    videoCapture.set(4,720)

    disflow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    prev_gray=np.zeros((INPUT_SIZE_H,INPUT_SIZE_W),np.uint8)
    prev_cfd=np.zeros((INPUT_SIZE_H,INPUT_SIZE_W),np.float32)
    is_init=True

    while True:
        # get a frame
        ret, frame = videoCapture.read()
        if ret==None:
            break
        frame = cv2.flip(frame,1)

        # opencv
        origin_h, origin_w, c = frame.shape
        image_resize = cv2.resize(frame, (INPUT_SIZE_W,INPUT_SIZE_H), interpolation=cv2.INTER_LINEAR)
        img=image_resize.copy()
        image_resize = (image_resize - (104., 112., 121.,)) / 255.0
        
        tensor_4D = torch.FloatTensor(1, 3, INPUT_SIZE_H, INPUT_SIZE_W)
        tensor_4D[0,:,:,:] = torch.FloatTensor(image_resize.transpose(2,0,1))
        inputs = tensor_4D.to(device)
        # -----------------------------------------------------------------
       
        seg, alpha = net(inputs)  

        if args.without_gpu:
            alpha_np = alpha[0,0,:,:].data.numpy()
        else:
            alpha_np = alpha[0,0,:,:].cpu().data.numpy()
        
        alpha_np=255 * alpha_np
        cur_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        optflow_map=postprocess(cur_gray,alpha_np,prev_gray,prev_cfd,disflow,is_init)
        
        prev_gray=cur_gray.copy()
        prev_cfd=optflow_map.copy()
        is_init=False
        optflow_map=cv2.GaussianBlur(optflow_map,(3,3),0)
        optflow_map=threshold_mask(optflow_map,thresh_bg=0.2,thresh_fg=0.8)
        
        alpha=np.repeat(optflow_map[:,:,np.newaxis],3,axis=2)
        alpha=cv2.resize(alpha,(origin_w,origin_h),interpolation=cv2.INTER_LINEAR)
        
        bg_img=np.ones_like(frame) * 255
        out=(alpha * frame+(1 - alpha) * bg_img).astype(np.uint8)
        out[out>255]=255

        # show a frame
        cv2.imshow("capture", out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    videoCapture.release()
    cv2.destroyAllWindows()

def main(args):

    INPUT_SIZE_W=args.input_w
    INPUT_SIZE_H=args.input_h
    myModel = load_model(args)

    camera_seg(args, myModel)


if __name__ == "__main__":
    main(args)


