# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 05:02:40 2023

@author: user
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math

def pad_within(x, stride = 2):
  
  x_real = torch.real(x)
  x_imag = torch.imag(x)
  w = x_real.new_zeros(1,stride)
  w[0,0] = 1
  out_real = F.conv_transpose2d(x_real, w.expand(x_real.size(1),1,1,stride), stride=[1,stride], groups = x_real.size(1))
  out_imag = F.conv_transpose2d(x_imag, w.expand(x_imag.size(1),1,1,stride), stride=[1,stride], groups = x_imag.size(1))
  out = out_real + 1j*out_imag
  return out

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).init__()
        self.conv1 = nn.Conv2d(16,64,kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(64,16,kernel_size = 3, padding = 1)
    def forward(self, x):
        shortcut = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + shortcut
        return out
class DualNet_Encoder(nn.Module):
    def __init__(self,input_size,latent_size):
        super(DualNet_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2,16,kernel_size = 7, padding = 3)
        self.conv2 = nn.Conv2d(16,8,kernel_size = 7, padding = 3)
        self.conv3 = nn.Conv2d(8,4,kernel_size = 7, padding = 3)
        self.conv4 = nn.Conv2d(4,2,kernel_size = 7, padding = 3)
        self.dense = nn.Linear(input_size,latent_size)
     
    def forward(self,x):
       
        x_dl = x
        x_dl = x_dl[:,0::2,:,:] + 1j*x_dl[:,1::2,:,:]
        x_dl = torch.fft.fft(x_dl,dim=-2)
        x_dl = torch.fft.ifft(x_dl,dim=-1)
        x_dl = torch.concat((torch.real(x_dl),torch.imag(x_dl)),dim=-3)
        
        out = F.leaky_relu(self.conv1(x_dl))
        out = F.leaky_relu(self.conv2(out))
        out = F.leaky_relu(self.conv3(out))
        out = F.leaky_relu(self.conv4(out))
        out = out.reshape((out.size(0),out.size(-3)*out.size(-2)*out.size(-1)))
        out = self.dense(out)
        return out
class DualNet_Decoder(nn.Module):
    def __init__(self,latent_size,output_size):
        super(DualNet_Decoder, self).__init__()
        self.conv1 = nn.Conv2d(2,16,kernel_size = 7, padding = 3)
        self.conv2 = nn.Conv2d(16,8,kernel_size = 7, padding = 3)
        self.conv3 = nn.Conv2d(8,4,kernel_size = 7, padding = 3)
        self.conv4 = nn.Conv2d(4,2,kernel_size = 7, padding = 3)
        
        self.conv5 = nn.Conv2d(2,16,kernel_size = 7, padding = 3)
        self.conv6 = nn.Conv2d(16,8,kernel_size = 7, padding = 3)
        self.conv7 = nn.Conv2d(8,4,kernel_size = 7, padding = 3)
        self.conv8 = nn.Conv2d(4,1,kernel_size = 7, padding = 3)
        
        self.conv9 = nn.Conv2d(2,16,kernel_size = 7, padding = 3)
        self.conv10 = nn.Conv2d(16,8,kernel_size = 7, padding = 3)
        self.conv11 = nn.Conv2d(8,4,kernel_size = 7, padding = 3)
        self.conv12 = nn.Conv2d(4,2,kernel_size = 7, padding = 3)
        self.dense = nn.Linear(latent_size,output_size)
        
    def forward(self,x,x_ul):
        
        x_ul = x_ul[:,0::2,:,:] + 1j*x_ul[:,1::2,:,:]
        x_ul = torch.fft.fft(x_ul,dim=-2)
        x_ul = torch.fft.ifft(x_ul,dim=-1)
        x_ul_abs = x_ul.abs()
#        x_ul = torch.concat((torch.real(x_ul),torch.imag(x_ul)),dim=-3)
        
        out = self.dense(x)
        out = out.reshape((out.size(0),2,8,int(out.size(-1)/2/8)))
        out = F.tanh(self.conv1(out))
        out = F.tanh(self.conv2(out))
        out = F.tanh(self.conv3(out))
        x_dl_ini_est = F.tanh(self.conv4(out))
        x_ini_est_complex = x_dl_ini_est[:,0::2,:,:] + 1j*x_dl_ini_est[:,1::2,:,:]
        x_dl_ini_abs_est = x_ini_est_complex.abs()
        x_dl_ini_abs_angle = x_ini_est_complex.angle()
        x_dl_ini_abs_cossin = torch.concat((torch.cos(x_dl_ini_abs_angle),torch.sin(x_dl_ini_abs_angle)),dim=-3)
        
        inputs = torch.concat((x_dl_ini_abs_est,x_ul_abs),dim=-3)
        
        out = F.leaky_relu(self.conv5(inputs))
        out = F.leaky_relu(self.conv6(out))
        out = F.leaky_relu(self.conv7(out))
        x_dl_ini_abs_est2 = F.relu(self.conv8(out))
        
        x_dl_ini = x_dl_ini_abs_est2*x_dl_ini_abs_cossin
        
        copy = x_dl_ini
        
        out = F.tanh(self.conv9(x_dl_ini))
        out = F.tanh(self.conv10(out))
        out = F.tanh(self.conv11(out))
        out = F.tanh(self.conv12(out))
        out = out + copy
        
        out = out[:,0::2,:,:] + 1j*out[:,1::2,:,:]
        out = torch.fft.fft(out,dim=-1)
        out = torch.fft.ifft(out, dim = -2)
        out = torch.concat((torch.real(out), torch.imag(out)), dim = -3)
        return out
class DualNet_Decoder2(nn.Module):
    def __init__(self,latent_size,output_size):
        super(DualNet_Decoder2, self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size = 7, padding = 3)
        self.conv2 = nn.Conv2d(16,8,kernel_size = 7, padding = 3)
        self.conv3 = nn.Conv2d(8,4,kernel_size = 7, padding = 3)
        self.conv4 = nn.Conv2d(4,2,kernel_size = 7, padding = 3)
        
        self.conv5 = nn.Conv2d(2,16,kernel_size = 7, padding = 3)
        self.conv6 = nn.Conv2d(16,8,kernel_size = 7, padding = 3)
        self.conv7 = nn.Conv2d(8,4,kernel_size = 7, padding = 3)
        self.conv8 = nn.Conv2d(4,1,kernel_size = 7, padding = 3)
        
        self.conv9 = nn.Conv2d(2,16,kernel_size = 7, padding = 3)
        self.conv10 = nn.Conv2d(16,8,kernel_size = 7, padding = 3)
        self.conv11 = nn.Conv2d(8,4,kernel_size = 7, padding = 3)
        self.conv12 = nn.Conv2d(4,2,kernel_size = 7, padding = 3)
        self.dense = nn.Linear(latent_size,output_size)
        
    def forward(self,x,x_ul):
        
        x_ul = x_ul[:,0::2,:,:] + 1j*x_ul[:,1::2,:,:]
        x_ul = torch.fft.fft(x_ul,dim=-2)
        x_ul = torch.fft.ifft(x_ul,dim=-1)
        x_ul_abs = x_ul.abs()
#        x_ul = torch.concat((torch.real(x_ul),torch.imag(x_ul)),dim=-3)
        
        out = self.dense(x)
        out = out.reshape((out.size(0),2,8,int(out.size(-1)/2/8)))
        inputs = torch.concat((out,x_ul_abs),dim=-3)
        out = F.tanh(self.conv1(inputs))
        out = F.tanh(self.conv2(out))
        out = F.tanh(self.conv3(out))
        out = F.tanh(self.conv4(out))
    
        out = out[:,0::2,:,:] + 1j*out[:,1::2,:,:]
        out = torch.fft.fft(out,dim=-1)
        out = torch.fft.ifft(out, dim = -2)
        out = torch.concat((torch.real(out), torch.imag(out)), dim = -3)
        return out
class CsiNetPro_Encoder_AF(nn.Module):
    def __init__(self,input_size,latent_size):
        super(CsiNetPro_Encoder_AF, self).__init__()
        self.conv1 = nn.Conv2d(2,16,kernel_size = 7, padding = 3)
        self.conv2 = nn.Conv2d(16,8,kernel_size = 7, padding = 3)
        self.conv3 = nn.Conv2d(8,4,kernel_size = 7, padding = 3)
        self.conv4 = nn.Conv2d(4,2,kernel_size = 7, padding = 3)
        self.dense = nn.Linear(input_size,latent_size)
     
    def forward(self,x):
       
        
        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        out = F.leaky_relu(self.conv3(out))
        out = F.leaky_relu(self.conv4(out))
        out = out.reshape((out.size(0),out.size(-3)*out.size(-2)*out.size(-1)))
        out = self.dense(out)
        return out
    
class CsiNetPro_Decoder_AF(nn.Module):
    def __init__(self,latent_size,output_size):
        super(CsiNetPro_Decoder_AF, self).__init__()
        self.conv1 = nn.Conv2d(2,16,kernel_size = 7, padding = 3)
        self.conv2 = nn.Conv2d(16,8,kernel_size = 7, padding = 3)
        self.conv3 = nn.Conv2d(8,4,kernel_size = 7, padding = 3)
        self.conv4 = nn.Conv2d(4,2,kernel_size = 7, padding = 3)
        self.dense = nn.Linear(latent_size,output_size)
        
    def forward(self,x):
        out = self.dense(x)
        out = out.reshape((out.size(0),2,8,int(out.size(-1)/2/8)))
        out = F.tanh(self.conv1(out))
        out = F.tanh(self.conv2(out))
        out = F.tanh(self.conv3(out))
        out = F.tanh(self.conv4(out))
        
        return out

class CsiNetPro_Encoder(nn.Module):
    def __init__(self,input_size,latent_size):
        super(CsiNetPro_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2,16,kernel_size = 7, padding = 3)
        self.conv2 = nn.Conv2d(16,8,kernel_size = 7, padding = 3)
        self.conv3 = nn.Conv2d(8,4,kernel_size = 7, padding = 3)
        self.conv4 = nn.Conv2d(4,2,kernel_size = 7, padding = 3)
        self.dense = nn.Linear(input_size,latent_size)
     
    def forward(self,x):
        x_dl = x
        x_dl = x_dl[:,0::2,:,:] + 1j*x_dl[:,1::2,:,:]
        x_dl = torch.fft.fft(x_dl,dim=-2)
        x_dl = torch.fft.ifft(x_dl,dim=-1)
        x_dl = torch.concat((torch.real(x_dl),torch.imag(x_dl)),dim=-3)
        
        out = F.leaky_relu(self.conv1(x_dl))
        out = F.leaky_relu(self.conv2(out))
        out = F.leaky_relu(self.conv3(out))
        out = F.leaky_relu(self.conv4(out))
        out = out.reshape((out.size(0),out.size(-3)*out.size(-2)*out.size(-1)))
        out = self.dense(out)
        return out
    
class CsiNetPro_Decoder(nn.Module):
    def __init__(self,latent_size,output_size):
        super(CsiNetPro_Decoder, self).__init__()
        self.conv1 = nn.Conv2d(2,16,kernel_size = 7, padding = 3)
        self.conv2 = nn.Conv2d(16,8,kernel_size = 7, padding = 3)
        self.conv3 = nn.Conv2d(8,4,kernel_size = 7, padding = 3)
        self.conv4 = nn.Conv2d(4,2,kernel_size = 7, padding = 3)
        self.dense = nn.Linear(latent_size,output_size)
        
    def forward(self,x):
        out = self.dense(x)
        out = out.reshape((out.size(0),2,8,int(out.size(-1)/2/8)))
        out = F.tanh(self.conv1(out))
        out = F.tanh(self.conv2(out))
        out = F.tanh(self.conv3(out))
        out = F.tanh(self.conv4(out))
        
        out = out[:,0::2,:,:] + 1j*out[:,1::2,:,:]
        out = torch.fft.fft(out,dim=-1)
        out = torch.fft.ifft(out, dim = -2)
        out = torch.concat((torch.real(out), torch.imag(out)), dim = -3)
        return out

class ReEsNet(nn.Module):
    def __init__(self):
        super(ReEsNet, self).__init__()
        self.conv1 = nn.Conv2d(2,16,kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(16,2,kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(2,2,kernel_size = 3, padding = 1)
        
        self.resblock1 = ResBlock()
        self.resblock2 = ResBlock()
        self.resblock3 = ResBlock()
        self.resblock4 = ResBlock()
    def forward(self,x):
        out = self.conv1(x)
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.resblock4(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return out
        

class SIMaskNet(nn.Module):
  def __init__(self):
    super(SIMaskNet, self).__init__()
    self.conv1 = nn.Conv2d(2,16,kernel_size = 9, padding = 4)
    self.conv2 = nn.Conv2d(16,8,kernel_size = 1, padding = 0)
    self.conv3 = nn.Conv2d(8,1,kernel_size = 5, padding = 2)
    self.conv4 = nn.Conv2d(8,2,kernel_size = 5, padding = 2)
    
    self.conv5 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
    self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
    self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
    
    self.conv8 = nn.Conv2d(2,16,kernel_size = 9, padding = 4)
    self.conv9 = nn.Conv2d(16,8,kernel_size = 1, padding = 0)
    self.conv10 = nn.Conv2d(8,2,kernel_size = 5, padding = 2)
  def forward(self,x,scale = 1):
    x_dl, x_side = x
    
    
    
    x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
    x_side = torch.fft.fft(x_side,dim=-2)
    x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,32,x_side.size(-1)))
    # TODO upsampling
    out = F.relu(self.conv5(x_side_abs))
    out = F.relu(self.conv6(out))
    Mask = F.sigmoid(self.conv7(out))
    
    x_dl = x_dl[:,0,:,:] + 1j*x_dl[:,1,:,:]
    x_dl = torch.fft.fft(x_dl,dim=-2)
    x_dl = x_dl.reshape((x_dl.size(0),1,32,x_dl.size(-1)))
    x_dl = pad_within(x_dl,int(4/scale))
    x_dl = torch.fft.ifft(x_dl,dim=-1).reshape((x_dl.size(0),1,32,x_dl.size(-1)))
    x_dl_abs = x_dl.abs()
    
    x_dl_ini = x_dl*Mask
    
    x_dl = torch.concat((torch.real(x_dl_ini),torch.imag(x_dl_ini)),dim=-3)
    
    copy1 = x_dl
    
    out = F.tanh(self.conv1(x_dl))
    out = F.tanh(self.conv2(out))
    out = self.conv4(out)
    
    out = out + copy1
    
    out = out[:,0,:,:] + 1j*out[:,1,:,:]
    out = torch.fft.fft(out,dim=-1).reshape((out.size(0),1,32,out.size(-1)))
    out = torch.fft.ifft(out, dim = -2)
    out = out.reshape((out.size(0),1,32, out.size(-1)))
    out = torch.concat((torch.real(out), torch.imag(out)), dim = -3)

    copy2 = out

    out = F.tanh(self.conv8(out))
    out = F.tanh(self.conv9(out))
    out = self.conv10(out)   
    
    out = out + copy2
    return out, x_dl_abs, x_side_abs, Mask, x_dl_ini


class SIMaskNet_direct_input(nn.Module):
  def __init__(self):
    super(SIMaskNet_direct_input, self).__init__()
    self.conv1 = nn.Conv2d(2,16,kernel_size = 9, padding = 4)
    self.conv2 = nn.Conv2d(16,8,kernel_size = 1, padding = 0)
    self.conv3 = nn.Conv2d(8,1,kernel_size = 5, padding = 2)
    self.conv4 = nn.Conv2d(8,2,kernel_size = 5, padding = 2)
    
    self.conv5 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
    self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
    self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
    
    self.conv8 = nn.Conv2d(2,16,kernel_size = 9, padding = 4)
    self.conv9 = nn.Conv2d(16,8,kernel_size = 1, padding = 0)
    self.conv10 = nn.Conv2d(8,2,kernel_size = 5, padding = 2)
  def forward(self,x,scale = 1):
    x_dl, x_side = x
    x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
    x_side = torch.fft.fft(x_side,dim=-2)
    x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,8,x_side.size(-1)))
    # TODO upsampling
    out = F.relu(self.conv5(x_side_abs))
    out = F.relu(self.conv6(out))
    Mask = F.sigmoid(self.conv7(out))
    
    x_dl = x_dl[:,0,:,:] + 1j*x_dl[:,1,:,:]
    x_dl = torch.fft.fft(x_dl,dim=-2)
    x_dl = x_dl.reshape((x_dl.size(0),1,8,x_dl.size(-1)))
    x_dl = torch.fft.ifft(x_dl,dim=-1).reshape((x_dl.size(0),1,8,x_dl.size(-1)))
    x_dl_abs = x_dl.abs()
    
    x_dl_ini = x_dl*Mask
    
    x_dl = torch.concat((torch.real(x_dl_ini),torch.imag(x_dl_ini)),dim=-3)
    
    copy1 = x_dl
    
    out = F.tanh(self.conv1(x_dl))
    out = F.tanh(self.conv2(out))
    out = self.conv4(out)
    
    out = out + copy1
    
    out = out[:,0,:,:] + 1j*out[:,1,:,:]
    out = torch.fft.fft(out,dim=-1).reshape((out.size(0),1,8,out.size(-1)))
    out = torch.fft.ifft(out, dim = -2)
    out = out.reshape((out.size(0),1,8, out.size(-1)))
    out = torch.concat((torch.real(out), torch.imag(out)), dim = -3)

    copy2 = out

    out = F.tanh(self.conv8(out))
    out = F.tanh(self.conv9(out))
    out = self.conv10(out)
    
    out = out + copy2
    return out, x_dl_abs, x_side_abs, Mask, x_dl_ini

class ResBlock2(nn.Module):
    def __init__(self):
        super(ResBlock2, self).__init__()
        self.conv1 = nn.Conv2d(2,64,kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(64,2,kernel_size = 3, padding = 1)
    def forward(self, x):
        shortcut = x
        out = F.leaky_relu(self.conv1(x))
        out = self.conv2(out)
        out = out + shortcut
        return out
class SIMaskNet2(nn.Module):
  def __init__(self):
    super(SIMaskNet2, self).__init__()
    self.conv1 = nn.Conv2d(2,16,kernel_size = 9, padding = 4)
    self.conv2 = nn.Conv2d(16,8,kernel_size = 1, padding = 0)
    self.conv3 = nn.Conv2d(8,1,kernel_size = 5, padding = 2)
    self.conv4 = nn.Conv2d(8,2,kernel_size = 5, padding = 2)
    
    self.conv5 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
    self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
    self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
    
    self.conv8 = nn.Conv2d(2,16,kernel_size = 9, padding = 4)
    self.conv9 = nn.Conv2d(16,8,kernel_size = 1, padding = 0)
    self.conv10 = nn.Conv2d(8,2,kernel_size = 5, padding = 2)
    
    self.resblock1 = ResBlock2()
    self.resblock2 = ResBlock2()
    self.resblock3 = ResBlock2()
    self.resblock4 = ResBlock2()
    
    self.resblock5 = ResBlock2()
    self.resblock6 = ResBlock2()
    self.resblock7 = ResBlock2()
    self.resblock8 = ResBlock2()
  def forward(self,x,scale = 1):
    x_dl, x_side = x
    
    
    
    x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
    x_side = torch.fft.fft(x_side,dim=-2)
    x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,8,x_side.size(-1)))
    # TODO upsampling
    out = F.relu(self.conv5(x_side_abs))
    out = F.relu(self.conv6(out))
    Mask = F.sigmoid(self.conv7(out))
    
    x_dl = x_dl[:,0,:,:] + 1j*x_dl[:,1,:,:]
    x_dl = torch.fft.fft(x_dl,dim=-2)
    x_dl = x_dl.reshape((x_dl.size(0),1,8,x_dl.size(-1)))
    x_dl = pad_within(x_dl,int(12/scale))
    x_dl = torch.fft.ifft(x_dl,dim=-1).reshape((x_dl.size(0),1,8,x_dl.size(-1)))
    x_dl_abs = x_dl.abs()
    
    x_dl_ini = x_dl*Mask
    
    x_dl = torch.concat((torch.real(x_dl_ini),torch.imag(x_dl_ini)),dim=-3)
    
    copy1 = x_dl
    
    out = self.resblock1(x_dl)
    out = self.resblock2(out)
    out = self.resblock3(out)
    out = self.resblock4(out)
    
    out = out + copy1
    
    out = out[:,0,:,:] + 1j*out[:,1,:,:]
    out = torch.fft.fft(out,dim=-1).reshape((out.size(0),1,8,out.size(-1)))
    out = torch.fft.ifft(out, dim = -2)
    out = out.reshape((out.size(0),1,8, out.size(-1)))
    out = torch.concat((torch.real(out), torch.imag(out)), dim = -3)

    copy2 = out

    out = self.resblock5(out)
    out = self.resblock6(out)
    out = self.resblock7(out)
    out = self.resblock8(out)  
    
    out = out + copy2
    return out, x_dl_abs, x_side_abs, Mask, x_dl_ini



class SIMaskNet_var(nn.Module):
  def __init__(self):
    super(SIMaskNet_var, self).__init__()
    self.conv1 = nn.Conv2d(2,16,kernel_size = 9, padding = 4)
    self.conv2 = nn.Conv2d(16,8,kernel_size = 1, padding = 0)
    self.conv3 = nn.Conv2d(8,1,kernel_size = 5, padding = 2)
    self.conv4 = nn.Conv2d(8,2,kernel_size = 5, padding = 2)
    
    self.conv5 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
    self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
    self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
    
    self.conv8 = nn.Conv2d(2,16,kernel_size = 9, padding = 4)
    self.conv9 = nn.Conv2d(16,8,kernel_size = 1, padding = 0)
    self.conv10 = nn.Conv2d(8,2,kernel_size = 5, padding = 2)
  def forward(self,x,scale = 1):
    x_dl, x_side = x
    
    
    
    x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
    x_side = torch.fft.fft(x_side,dim=-2)
    x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,8,x_side.size(-1)))
    x_side_abs_interp = nn.Upsample(scale_factor=tuple([1,scale]),mode='bilinear')(x_side_abs)
    # TODO upsampling
    out = F.relu(self.conv5(x_side_abs_interp))
    out = F.relu(self.conv6(out))
    Mask = F.sigmoid(self.conv7(out))
    
    x_dl = x_dl[:,0,:,:] + 1j*x_dl[:,1,:,:]
    x_dl = torch.fft.fft(x_dl,dim=-2)
    x_dl = x_dl.reshape((x_dl.size(0),1,8,x_dl.size(-1)))
    x_dl = pad_within(x_dl,int(12))
    x_dl = torch.fft.ifft(x_dl,dim=-1).reshape((x_dl.size(0),1,8,x_dl.size(-1)))
    x_dl_abs = x_dl.abs()
    
    x_dl_ini = x_dl*Mask
    
    x_dl = torch.concat((torch.real(x_dl_ini),torch.imag(x_dl_ini)),dim=-3)
    
    copy1 = x_dl
    
    out = F.tanh(self.conv1(x_dl))
    out = F.tanh(self.conv2(out))
    out = self.conv4(out)
    
    out = out + copy1
    
    out = out[:,0,:,:] + 1j*out[:,1,:,:]
    out = torch.fft.fft(out,dim=-1).reshape((out.size(0),1,8,out.size(-1)))
    out = torch.fft.ifft(out, dim = -2)
    out = out.reshape((out.size(0),1,8, out.size(-1)))
    out = torch.concat((torch.real(out), torch.imag(out)), dim = -3)

    copy2 = out

    out = F.tanh(self.conv8(out))
    out = F.tanh(self.conv9(out))
    out = self.conv10(out)   
    
    out = out + copy2
    return out, x_dl_abs, x_side_abs, Mask, x_dl_ini


class SRCNN(nn.Module):
    def __init__(self) -> None:
        super(SRCNN, self).__init__()
        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )

        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )

        # Rebuild the layer.
        self.reconstruction = nn.Conv2d(32, 2, (5, 5), (1, 1), (2, 2))

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)

def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)



class SRCNN2(nn.Module):
    def __init__(self) -> None:
        super(SRCNN2, self).__init__()
        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )

        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )

        # Rebuild the layer.
        self.reconstruction = nn.Conv2d(32, 2, (5, 5), (1, 1), (2, 2))

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x_complex = x[:,0::2,:,:] + 1j*x[:,1::2,:,:]
        x_complex = torch.fft.ifft(x_complex)
        x = torch.cat((torch.real(x_complex),torch.imag(x_complex)),dim=1)
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)
        x_complex = out[:,0::2,:,:] + 1j*out[:,1::2,:,:]
        x_complex = torch.fft.fft(x_complex)
        out = torch.cat((torch.real(x_complex),torch.imag(x_complex)),dim=1)
        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0)
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)

# Define ISTA-Net-plus Block
# Define ISTA-Net Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 2, 8, 660)

        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_pred = x_backward.view(-1, 660)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]


# Define ISTA-Net
class ISTANet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit):

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]
    

class BasicBlock2(torch.nn.Module):
    def __init__(self):
        super(BasicBlock2, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 2, 8, 660)

        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_pred = x_backward.view(-1, 660)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]


# Define ISTA-Net
class ISTANet2(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANet2, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit):

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]

class BasicBlock_side(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_side, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))
        
        self.weight = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 1, 1)))
        
        self.conv5 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
        self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
        
    def forward(self, x, PhiTPhi, PhiTb, x_side):
        flag_beam = 1
        x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
        if flag_beam== 1:
            x_side = torch.fft.fft(x_side,dim=-2)
        x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,8,x_side.size(-1)))
        out = F.relu(self.conv5(x_side_abs))
        out = F.relu(self.conv6(out))
        Mask = F.sigmoid(self.conv7(out))
        
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 2, 8, 660)
        x_input_complex = x_input[:,0::2,:,:] + 1j*x_input[:,1::2,:,:]
        if flag_beam== 1:
            x_input_complex = torch.fft.fft(x_input_complex,dim=-2)
        
        x_input = torch.concat((torch.real(x_input_complex),torch.imag(x_input_complex)),dim=-3)
        
        
        x_bp_filtered = x_input*Mask
        x_merged = self.weight*x_bp_filtered + (1-self.weight)*x_input
        
        x = F.conv2d(x_merged, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
        
        if flag_beam== 1:
            x_backward = x_backward[:,0::2,:,:] + 1j*x_backward[:,1::2,:,:]
            x_backward = torch.fft.ifft(x_backward,dim=-2)
            x_backward = torch.concat((torch.real(x_backward),torch.imag(x_backward)),dim=-3)
        x_pred = x_backward.view(-1, 660)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1) 
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]


# Define ISTA-Net
class SR_ISTANet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(SR_ISTANet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, ul_csi):

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb, ul_csi)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]
    
class BasicBlock_side2(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_side2, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))
        
        self.weight = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 1, 1)))
        
        self.conv5 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
        self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
        
    def forward(self, x, PhiTPhi, PhiTb, x_side):
        flag_beam = 1
        x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
        if flag_beam== 1:
            x_side = torch.fft.fft(x_side,dim=-2)
        x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,8,x_side.size(-1)))
        out = F.relu(self.conv5(x_side_abs))
        out = F.relu(self.conv6(out))
        Mask = F.sigmoid(self.conv7(out))
        
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 2, 8, 660)
        x_input_complex = x_input[:,0::2,:,:] + 1j*x_input[:,1::2,:,:]
        if flag_beam== 1:
            x_input_complex = torch.fft.fft(x_input_complex,dim=-2)
        
        x_input = torch.concat((torch.real(x_input_complex),torch.imag(x_input_complex)),dim=-3)
        
        
        x_bp_filtered = x_input*Mask
#        x_merged = self.weight*x_bp_filtered + (1-self.weight)*x_input
        
        x = F.conv2d(x_bp_filtered, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
        
        if flag_beam== 1:
            x_backward = x_backward[:,0::2,:,:] + 1j*x_backward[:,1::2,:,:]
            x_backward = torch.fft.ifft(x_backward,dim=-2)
            x_backward = torch.concat((torch.real(x_backward),torch.imag(x_backward)),dim=-3)
        x_pred = x_backward.view(-1, 660)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1) 
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]


# Define ISTA-Net
class SR_ISTANet2(torch.nn.Module):
    def __init__(self, LayerNo):
        super(SR_ISTANet2, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side2())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, ul_csi):

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb, ul_csi)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]
    
    
class BasicBlock_side3(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_side3, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))
        
        self.weight = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 1, 1)))
        
        self.conv5 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
        self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
        
    def forward(self, x, PhiTPhi, PhiTb):
        # x: complex, reshaped
        # PhiTPhi: complex
        # PhiTb: complex
        # x_side: real, original size
        
#        flag_beam = 1
#        x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
#        if flag_beam== 1:
#            x_side = torch.fft.fft(x_side,dim=-2)
#        x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,8,x_side.size(-1)))
#        out = F.relu(self.conv5(x_side_abs))
#        out = F.relu(self.conv6(out))
#        Mask = F.sigmoid(self.conv7(out)) # real, original shape
#        x = x - torch.mm(x, PhiTPhi) # complex, (-1,660)
        x = x - self.lambda_step * torch.mm(x, PhiTPhi) # complex, (-1,660)
        x = x + self.lambda_step * PhiTb # complex, (-1,660)
        
        
        x_input = x.view(-1, 1, 8, 660)
#        if flag_beam== 1:
#            x_input_complex = torch.fft.fft(x_input,dim=-2)
        
        x_input = torch.concat((torch.real(x_input),torch.imag(x_input)),dim=1)
        
        
#        x_bp_filtered = x_input*Mask
#        x_merged = self.weight*x_bp_filtered + (1-self.weight)*x_input
        
        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
        
#        if flag_beam== 1:
        x_backward = x_backward[:,0::2,:,:] + 1j*x_backward[:,1::2,:,:]
#            x_backward = torch.fft.ifft(x_backward,dim=-2)        
        x_pred = x_backward.view(-1, 660)
        

        x = F.conv2d(x_forward, self.conv1_backward, padding=1) 
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]


# Define ISTA-Net
class SR_ISTANet3(torch.nn.Module):
    def __init__(self, LayerNo):
        super(SR_ISTANet3, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side3())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi):
        y =  dl_csi, ul_csi
        x_dealiasing, a, b, c, d = self.srcsinet(y)
        x_dealiasing = torch.fft.ifft(x_dealiasing[:,0::2,:,:] + 1j*x_dealiasing[:,1::2,:,:])
        x_dealiasing = x_dealiasing.view(-1,660)
        
        Phix = Phix.view(-1,2,8,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        
        Phix = Phix.view(-1,1,8,Phix.size(-1))
        Phix = torch.cat((torch.real(Phix),torch.imag(Phix)),1).view(-1,Phix.size(-1))
        
        x = x - x_dealiasing
        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)
        x = x.view(-1,1,8,x.size(-1))
        x = torch.cat((torch.real(x),torch.imag(x)),1)
        x = x.view(-1,660)
        x_final = x

        return [x_final, layers_sym]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class BasicBlock_side4(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_side4, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))
        
        self.weight = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 1, 1)))
        
        self.conv5 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
        self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
        
    def forward(self, x, PhiHPhi, PhiHb, x_side):
        # x: complex, reshaped
        # PhiTPhi: complex
        # PhiTb: complex
        # x_side: real, original size
        
        flag_beam = 1
        x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
        if flag_beam== 1:
            x_side = torch.fft.fft(x_side,dim=-2)
        x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,8,x_side.size(-1)))
        out = F.relu(self.conv5(x_side_abs))
        out = F.relu(self.conv6(out))
        Mask = F.sigmoid(self.conv7(out)) # real, original shape
#        x = x - torch.mm(x, PhiTPhi) # complex, (-1,660)
        x = x - self.lambda_step * torch.mm(x, PhiHPhi) # complex, (-1,660)
        x = x + self.lambda_step * PhiHb # complex, (-1,660)
        
        
        x_input = x.view(-1, 1, 8, 660)
        if flag_beam== 1:
            x_input = torch.fft.fft(x_input,dim=-2)
        
        x_input = torch.concat((torch.real(x_input),torch.imag(x_input)),dim=1)
        
        
        x_bp_filtered = x_input*Mask
        x_merged = self.weight*x_bp_filtered + (1-self.weight)*x_input
        
        x = F.conv2d(x_merged, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
#        print(str(x_backward.size()))
        if flag_beam== 1:
            x_backward = x_backward[:,0::2,:,:] + 1j*x_backward[:,1::2,:,:]
#            print(str(x_backward.size()))
            x_backward = torch.fft.ifft(x_backward,dim=-2)
#        print(str(x_backward.size()))
#        x_pred = x_backward.view(-1, 660)
        x_pred = x_backward.reshape(x_backward.size(0)*1*8, 660)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1) 
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]


# Define ISTA-Net
class SR_ISTANet4(torch.nn.Module):
    def __init__(self, LayerNo):
        super(SR_ISTANet4, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side4())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi):
#        y =  dl_csi, ul_csi
#        x_dealiasing, a, b, c, d = self.srcsinet(y)
#        x_dealiasing = torch.fft.ifft(x_dealiasing[:,0::2,:,:] + 1j*x_dealiasing[:,1::2,:,:])
#        x_dealiasing = x_dealiasing.view(-1,660)
        
        Phix = Phix.view(-1,2,8,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        
        Phix = Phix.view(-1,1,8,Phix.size(-1))
        Phix = torch.cat((torch.real(Phix),torch.imag(Phix)),1).view(-1,Phix.size(-1))
        
#        x = x - x_dealiasing
        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
        x = x.view(-1,1,8,x.size(-1))
        x = torch.cat((torch.real(x),torch.imag(x)),1)
        x = x.view(-1,660)
        x_final = x

        return [x_final, layers_sym]
    
    
    
class BasicBlock_side5(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_side5, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))
        
        self.weight = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 1, 1)))
        
        self.conv5 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
        self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
        
    def forward(self, x, PhiHPhi, PhiHb):
        # x: complex, reshaped
        # PhiTPhi: complex
        # PhiTb: complex
        # x_side: real, original size
        
        
#        x = x - torch.mm(x, PhiTPhi) # complex, (-1,660)
        x = x - self.lambda_step * torch.mm(x, PhiHPhi) # complex, (-1,660)
        x = x + self.lambda_step * PhiHb # complex, (-1,660)
        
        
        x_input = x.view(-1, 1, 8, 660)
        
        x_input = torch.concat((torch.real(x_input),torch.imag(x_input)),dim=1)
        
                
        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
#        print(str(x_backward.size()))
        x_backward = x_backward[:,0::2,:,:] + 1j*x_backward[:,1::2,:,:]
#            print(str(x_backward.size()))
#            x_backward = torch.fft.ifft(x_backward,dim=-2)
#        print(str(x_backward.size()))
        x_pred = x_backward.view(-1, 660)
#        x_pred = x_backward.reshape(x_backward.size(0)*1*8, 660)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1) 
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]


# Define ISTA-Net
class SR_ISTANet5(torch.nn.Module):
    def __init__(self, LayerNo):
        super(SR_ISTANet5, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side5())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi):
#        y =  dl_csi, ul_csi
#        x_dealiasing, a, b, c, d = self.srcsinet(y)
#        x_dealiasing = torch.fft.ifft(x_dealiasing[:,0::2,:,:] + 1j*x_dealiasing[:,1::2,:,:])
#        x_dealiasing = x_dealiasing.view(-1,660)
        
        Phix = Phix.view(-1,2,8,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        
#        Phix = Phix.view(-1,1,8,Phix.size(-1))
#        Phix = torch.cat((torch.real(Phix),torch.imag(Phix)),1).view(-1,Phix.size(-1))
        
#        x = x - x_dealiasing
        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb)
            layers_sym.append(layer_sym)
        x = x.view(-1,1,8,x.size(-1))
        x = torch.cat((torch.real(x),torch.imag(x)),1)
        x = x.view(-1,660)
        x_final = x

        return [x_final, layers_sym]
    
class BasicBlock_side6(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_side6, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))
        
        self.weight = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 1, 1)))
        
        self.conv5 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
        self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
        
    def forward(self, x, PhiHPhi, PhiHb, x_side):
        # x: complex, reshaped
        # PhiTPhi: complex
        # PhiTb: complex
        # x_side: real, original size
        
        flag_beam = 1
        x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
        if flag_beam== 1:
            x_side = torch.fft.fft(x_side,dim=-2)
        x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,8,x_side.size(-1)))
        out = F.relu(self.conv5(x_side_abs))
        out = F.relu(self.conv6(out))
        Mask = F.sigmoid(self.conv7(out)) # real, original shape
        
#        x = x - torch.mm(x, PhiTPhi) # complex, (-1,660)
        x = x - self.lambda_step * torch.mm(x, PhiHPhi) # complex, (-1,660)
        x = x + self.lambda_step * PhiHb # complex, (-1,660)
        
        
        x_input = x.view(-1, 1, 8, 660)
        if flag_beam== 1:
            x_input = torch.fft.fft(x_input,dim=-2)
            
        x_input = torch.concat((torch.real(x_input),torch.imag(x_input)),dim=1)
        
        
        x_bp_filtered = x_input*Mask
        x_merged = self.weight*x_bp_filtered + (1-self.weight)*x_input
        
        x = F.conv2d(x_merged, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
#        print(str(x_backward.size()))
        x_backward = x_backward[:,0::2,:,:] + 1j*x_backward[:,1::2,:,:]
#            print(str(x_backward.size()))
#            x_backward = torch.fft.ifft(x_backward,dim=-2)
#        print(str(x_backward.size()))
        x_pred = x_backward.view(-1, 660)
#        x_pred = x_backward.reshape(x_backward.size(0)*1*8, 660)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1) 
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]


# Define ISTA-Net
class SR_ISTANet6(torch.nn.Module):
    def __init__(self, LayerNo):
        super(SR_ISTANet6, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side6())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi):
#        y =  dl_csi, ul_csi
#        x_dealiasing, a, b, c, d = self.srcsinet(y)
#        x_dealiasing = torch.fft.ifft(x_dealiasing[:,0::2,:,:] + 1j*x_dealiasing[:,1::2,:,:])
#        x_dealiasing = x_dealiasing.view(-1,660)
        
        Phix = Phix.view(-1,2,8,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        
#        Phix = Phix.view(-1,1,8,Phix.size(-1))
#        Phix = torch.cat((torch.real(Phix),torch.imag(Phix)),1).view(-1,Phix.size(-1))
        
#        x = x - x_dealiasing
        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
        x = x.view(-1,1,8,x.size(-1))
        x = torch.cat((torch.real(x),torch.imag(x)),1)
        x = x.view(-1,660)
        x_final = x

        return [x_final, layers_sym]
    
    
class BasicBlock_side7(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_side7, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))
        
        self.weight = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 1, 1)))
        
        self.conv5 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
        self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
        
        self.conv1 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv2 = nn.Conv2d(8,1,kernel_size = [3,5], padding = [1,2])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,10))
        self.fc1 = nn.Linear(4*66,1)
        
    def forward(self, x, PhiHPhi, PhiHb, x_side):
        # x: complex, reshaped
        # PhiTPhi: complex
        # PhiTb: complex
        # x_side: real, original size
        
        flag_beam = 1
        x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
        if flag_beam== 1:
            x_side = torch.fft.fft(x_side,dim=-2)
        x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,8,x_side.size(-1)))
        out = F.relu(self.conv5(x_side_abs))
        out = F.relu(self.conv6(out))
        Mask = F.sigmoid(self.conv7(out)) # real, original shape
        
        
        xx = F.relu(self.conv1(x_side_abs))
        xx = F.relu(self.conv2(xx))
        xx = self.maxpool1(xx).view(-1,4*66)
        weight = F.sigmoid(self.fc1(xx)).view(-1,1,1,1)
#        x = x - torch.mm(x, PhiTPhi) # complex, (-1,660)
        x = x - self.lambda_step * torch.mm(x, PhiHPhi) # complex, (-1,660)
        x = x + self.lambda_step * PhiHb # complex, (-1,660)
        
        
        x_input = x.view(-1, 1, 8, 660)
        if flag_beam== 1:
            x_input = torch.fft.fft(x_input,dim=-2)
            
        x_input = torch.concat((torch.real(x_input),torch.imag(x_input)),dim=1)
        
        
        x_bp_filtered = x_input*Mask
        x_merged = weight*x_bp_filtered + (1-weight)*x_input
        
        x = F.conv2d(x_merged, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
#        print(str(x_backward.size()))
        x_backward = x_backward[:,0::2,:,:] + 1j*x_backward[:,1::2,:,:]
#            print(str(x_backward.size()))
#            x_backward = torch.fft.ifft(x_backward,dim=-2)
#        print(str(x_backward.size()))
        x_pred = x_backward.view(-1, 660)
#        x_pred = x_backward.reshape(x_backward.size(0)*1*8, 660)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1) 
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]


# Define ISTA-Net
class SR_ISTANet7(torch.nn.Module):
    def __init__(self, LayerNo):
        super(SR_ISTANet7, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side7())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi):
#        y =  dl_csi, ul_csi
#        x_dealiasing, a, b, c, d = self.srcsinet(y)
#        x_dealiasing = torch.fft.ifft(x_dealiasing[:,0::2,:,:] + 1j*x_dealiasing[:,1::2,:,:])
#        x_dealiasing = x_dealiasing.view(-1,660)
        
        Phix = Phix.view(-1,2,8,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        
#        Phix = Phix.view(-1,1,8,Phix.size(-1))
#        Phix = torch.cat((torch.real(Phix),torch.imag(Phix)),1).view(-1,Phix.size(-1))
        
#        x = x - x_dealiasing
        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
        x = x.view(-1,1,8,x.size(-1))
        x = torch.cat((torch.real(x),torch.imag(x)),1)
        x = x.view(-1,660)
        x_final = x

        return [x_final, layers_sym]
    
    
    
    

class BasicBlock_side8(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_side8, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))
        
        self.weight = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 1, 1)))
        
        self.conv5 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
        self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
        
        self.conv1 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv2 = nn.Conv2d(8,1,kernel_size = [3,5], padding = [1,2])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,10))
        self.fc1 = nn.Linear(4*66,1)
        
    def forward(self, x, PhiHPhi, PhiHb, x_side):
        # x: complex, reshaped
        # PhiTPhi: complex
        # PhiTb: complex
        # x_side: real, original size
        
        flag_beam = 1
        x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
        if flag_beam== 1:
            x_side = torch.fft.fft(x_side,dim=-2)
        x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,8,x_side.size(-1)))
        out = F.relu(self.conv5(x_side_abs))
        out = F.relu(self.conv6(out))
        Mask = F.sigmoid(self.conv7(out)) # real, original shape
        
        
#        xx = F.relu(self.conv1(x_side_abs))
#        xx = F.relu(self.conv2(xx))
#        xx = self.maxpool1(xx).view(-1,4*66)
#        weight = F.sigmoid(self.fc1(xx)).view(-1,1,1,1)
#        x = x - torch.mm(x, PhiTPhi) # complex, (-1,660)
        x = x - self.lambda_step * torch.mm(x, PhiHPhi) # complex, (-1,660)
        x = x + self.lambda_step * PhiHb # complex, (-1,660)
        
        
        x_input = x.view(-1, 1, 8, 660)
        if flag_beam== 1:
            x_input = torch.fft.fft(x_input,dim=-2)
            
        x_input = torch.concat((torch.real(x_input),torch.imag(x_input)),dim=1)
        
        
        x_bp_filtered = x_input*Mask
        x_merged = x_bp_filtered
        
        x = F.conv2d(x_merged, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
#        print(str(x_backward.size()))
        x_backward = x_backward[:,0::2,:,:] + 1j*x_backward[:,1::2,:,:]
#            print(str(x_backward.size()))
#            x_backward = torch.fft.ifft(x_backward,dim=-2)
#        print(str(x_backward.size()))
        x_pred = x_backward.view(-1, 660)
#        x_pred = x_backward.reshape(x_backward.size(0)*1*8, 660)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1) 
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]


# Define ISTA-Net
class SR_ISTANet8(torch.nn.Module):
    def __init__(self, LayerNo):
        super(SR_ISTANet8, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side8())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi):
#        y =  dl_csi, ul_csi
#        x_dealiasing, a, b, c, d = self.srcsinet(y)
#        x_dealiasing = torch.fft.ifft(x_dealiasing[:,0::2,:,:] + 1j*x_dealiasing[:,1::2,:,:])
#        x_dealiasing = x_dealiasing.view(-1,660)
        
        Phix = Phix.view(-1,2,8,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        
#        Phix = Phix.view(-1,1,8,Phix.size(-1))
#        Phix = torch.cat((torch.real(Phix),torch.imag(Phix)),1).view(-1,Phix.size(-1))
        
#        x = x - x_dealiasing
        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
        x = x.view(-1,1,8,x.size(-1))
        x = torch.cat((torch.real(x),torch.imag(x)),1)
        x = x.view(-1,660)
        x_final = x

        return [x_final, layers_sym]
    
    
class BasicBlock_side9(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_side9, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))
        
        self.weight = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 1, 1)))
        
        self.conv5 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
        self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
        
        self.conv1 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv2 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
        self.conv3 = nn.Conv2d(4,1,kernel_size = [3,5], padding = [1,2])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,10))
        self.fc1 = nn.Linear(4*66,1)
        
    def forward(self, x, PhiHPhi, PhiHb, x_side):
        # x: complex, reshaped
        # PhiTPhi: complex
        # PhiTb: complex
        # x_side: real, original size
        
        flag_beam = 1
        x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
        if flag_beam== 1:
            x_side = torch.fft.fft(x_side,dim=-2)
        x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,8,x_side.size(-1)))
        out = F.relu(self.conv5(x_side_abs))
        out = F.relu(self.conv6(out))
        Mask = F.sigmoid(self.conv7(out)) # real, original shape
        
        
        xx = F.relu(self.conv1(x_side_abs))
        xx = F.relu(self.conv2(xx))
        xx = F.relu(self.conv3(xx))
#        xx = self.maxpool1(xx).view(-1,4*66)
        weight = F.sigmoid(xx)
#        x = x - torch.mm(x, PhiTPhi) # complex, (-1,660)
        x = x - self.lambda_step * torch.mm(x, PhiHPhi) # complex, (-1,660)
        x = x + self.lambda_step * PhiHb # complex, (-1,660)
        
        
        x_input = x.view(-1, 1, 8, 660)
        if flag_beam== 1:
            x_input = torch.fft.fft(x_input,dim=-2)
            
        x_input = torch.concat((torch.real(x_input),torch.imag(x_input)),dim=1)
        
        
        x_bp_filtered = x_input*Mask 
        x_merged = weight*x_bp_filtered + (1-weight)*x_input
        
        x = F.conv2d(x_merged, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
#        print(str(x_backward.size()))
        x_backward = x_backward[:,0::2,:,:] + 1j*x_backward[:,1::2,:,:]
#            print(str(x_backward.size()))
#            x_backward = torch.fft.ifft(x_backward,dim=-2)
#        print(str(x_backward.size()))
        x_pred = x_backward.view(-1, 660)
#        x_pred = x_backward.reshape(x_backward.size(0)*1*8, 660)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1) 
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]


# Define ISTA-Net
class SR_ISTANet9(torch.nn.Module):
    def __init__(self, LayerNo):
        super(SR_ISTANet9, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side9())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi):
#        y =  dl_csi, ul_csi
#        x_dealiasing, a, b, c, d = self.srcsinet(y)
#        x_dealiasing = torch.fft.ifft(x_dealiasing[:,0::2,:,:] + 1j*x_dealiasing[:,1::2,:,:])
#        x_dealiasing = x_dealiasing.view(-1,660)
        
        Phix = Phix.view(-1,2,8,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        
#        Phix = Phix.view(-1,1,8,Phix.size(-1))
#        Phix = torch.cat((torch.real(Phix),torch.imag(Phix)),1).view(-1,Phix.size(-1))
        
#        x = x - x_dealiasing
        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
        x = x.view(-1,1,8,x.size(-1))
        x = torch.cat((torch.real(x),torch.imag(x)),1)
        x = x.view(-1,660)
        x_final = x

        return [x_final, layers_sym]
    
class BasicBlock_side10(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_side10, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))
        
        self.weight = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 1, 1)))
        
        self.conv5 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
        self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
        
        self.conv1 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv2 = nn.Conv2d(8,1,kernel_size = [3,5], padding = [1,2])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,10))
        self.fc1 = nn.Linear(4*66,1)
        
    def forward(self, x, PhiHPhi, PhiHb, x_side):
        # x: complex, reshaped
        # PhiTPhi: complex
        # PhiTb: complex
        # x_side: real, original size
        
        flag_beam = 1
        x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
        if flag_beam== 1:
            x_side = torch.fft.fft(x_side,dim=-2)
        x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,8,x_side.size(-1)))
        out = F.relu(self.conv5(x_side_abs))
        out = F.relu(self.conv6(out))
        Mask = F.sigmoid(self.conv7(out)) # real, original shape
        
        
        xx = F.relu(self.conv1(x_side_abs))
        xx = F.relu(self.conv2(xx))
        xx = self.maxpool1(xx).view(-1,4*66)
        weight = F.sigmoid(self.fc1(xx)).view(-1,1,1,1)
#        x = x - torch.mm(x, PhiTPhi) # complex, (-1,660)
        x = x - self.lambda_step * torch.mm(x, PhiHPhi) # complex, (-1,660)
        x = x + self.lambda_step * PhiHb # complex, (-1,660)
        
        
        x_input = x.view(-1, 1, 8, 660)
        if flag_beam== 1:
            x_input = torch.fft.fft(x_input,dim=-2)
            
        x_input = torch.concat((torch.real(x_input),torch.imag(x_input)),dim=1)
        
        x_input_ini = x.view(-1, 1, 8, 660)
        x_input_ini = torch.concat((torch.real(x_input_ini),torch.imag(x_input_ini)),dim=1)
#        
        
        x_bp_filtered = x_input*Mask
        x_merged = weight*x_bp_filtered + (1-weight)*x_input
        
        x = F.conv2d(x_merged, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
#        print(str(x_backward.size()))
        x_backward = x_backward[:,0::2,:,:] + 1j*x_backward[:,1::2,:,:]
#            print(str(x_backward.size()))
#            x_backward = torch.fft.ifft(x_backward,dim=-2)
#        print(str(x_backward.size()))
        x_pred = x_backward.view(-1, 660)
#        x_pred = x_backward.reshape(x_backward.size(0)*1*8, 660)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1) 
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input_ini

        return [x_pred, symloss]


# Define ISTA-Net
class SR_ISTANet10(torch.nn.Module):
    def __init__(self, LayerNo):
        super(SR_ISTANet10, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side10())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi):
#        y =  dl_csi, ul_csi
#        x_dealiasing, a, b, c, d = self.srcsinet(y)
#        x_dealiasing = torch.fft.ifft(x_dealiasing[:,0::2,:,:] + 1j*x_dealiasing[:,1::2,:,:])
#        x_dealiasing = x_dealiasing.view(-1,660)
        
        Phix = Phix.view(-1,2,8,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        
#        Phix = Phix.view(-1,1,8,Phix.size(-1))
#        Phix = torch.cat((torch.real(Phix),torch.imag(Phix)),1).view(-1,Phix.size(-1))
        
#        x = x - x_dealiasing
        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
        x = x.view(-1,1,8,x.size(-1))
        x = torch.cat((torch.real(x),torch.imag(x)),1)
        x = x.view(-1,660)
        x_final = x

        return [x_final, layers_sym]
    
class BasicBlock_side11(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_side11, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))
        
        self.weight = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 1, 1)))
        
        self.conv5 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
        self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
        
        self.conv1 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv2 = nn.Conv2d(8,1,kernel_size = [3,5], padding = [1,2])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,10))
        self.fc1 = nn.Linear(4*66,1)
        
    def forward(self, x, PhiHPhi, PhiHb, x_side):
        # x: complex, reshaped
        # PhiTPhi: complex
        # PhiTb: complex
        # x_side: real, original size
        
        flag_beam = 1
        x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
        if flag_beam== 1:
            x_side = torch.fft.fft(x_side,dim=-2)
        x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,8,x_side.size(-1)))
        out = F.relu(self.conv5(x_side_abs))
        out = F.relu(self.conv6(out))
        Mask = F.sigmoid(self.conv7(out)) # real, original shape
        
        
        xx = F.relu(self.conv1(x_side_abs))
        xx = F.relu(self.conv2(xx))
        xx = self.maxpool1(xx).view(-1,4*66)
        weight = F.sigmoid(self.fc1(xx)).view(-1,1,1,1)
#        x = x - torch.mm(x, PhiTPhi) # complex, (-1,660)
        x = x - self.lambda_step * torch.mm(x, PhiHPhi) # complex, (-1,660)
        x = x + self.lambda_step * PhiHb # complex, (-1,660)
        
        
        x_input = x.view(-1, 1, 8, 660)
        if flag_beam== 1:
            x_input = torch.fft.fft(x_input,dim=-2)
            
        x_input = torch.concat((torch.real(x_input),torch.imag(x_input)),dim=1)
        
#        x_input_ini = x.view(-1, 1, 8, 660)
#        x_input_ini = torch.concat((torch.real(x_input_ini),torch.imag(x_input_ini)),dim=1)
#        
        
        x_bp_filtered = x_input*Mask
        x_merged = weight*x_bp_filtered + (1-weight)*x_input
        
        x = F.conv2d(x_merged, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
#        print(str(x_backward.size()))
        x_backward = x_backward[:,0::2,:,:] + 1j*x_backward[:,1::2,:,:]
#            print(str(x_backward.size()))
#            x_backward = torch.fft.ifft(x_backward,dim=-2)
#        print(str(x_backward.size()))
        x_pred = x_backward.view(-1, 660)
#        x_pred = x_backward.reshape(x_backward.size(0)*1*8, 660)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1) 
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_merged

        return [x_pred, symloss]


# Define ISTA-Net
class SR_ISTANet11(torch.nn.Module):
    def __init__(self, LayerNo):
        super(SR_ISTANet11, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side11())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi):
#        y =  dl_csi, ul_csi
#        x_dealiasing, a, b, c, d = self.srcsinet(y)
#        x_dealiasing = torch.fft.ifft(x_dealiasing[:,0::2,:,:] + 1j*x_dealiasing[:,1::2,:,:])
#        x_dealiasing = x_dealiasing.view(-1,660)
        
        Phix = Phix.view(-1,2,8,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        
#        Phix = Phix.view(-1,1,8,Phix.size(-1))
#        Phix = torch.cat((torch.real(Phix),torch.imag(Phix)),1).view(-1,Phix.size(-1))
        
#        x = x - x_dealiasing
        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
        x = x.view(-1,1,8,x.size(-1))
        x = torch.cat((torch.real(x),torch.imag(x)),1)
        x = x.view(-1,660)
        x_final = x

        return [x_final, layers_sym]
    
    
class SR_ISTANet12(torch.nn.Module):
    def __init__(self, LayerNo):
        super(SR_ISTANet12, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side10())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi):
#        y =  dl_csi, ul_csi
#        x_dealiasing, a, b, c, d = self.srcsinet(y)
#        x_dealiasing = torch.fft.ifft(x_dealiasing[:,0::2,:,:] + 1j*x_dealiasing[:,1::2,:,:])
#        x_dealiasing = x_dealiasing.view(-1,660)
        
        Phix = Phix.view(-1,2,8,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        
#        Phix = Phix.view(-1,1,8,Phix.size(-1))
#        Phix = torch.cat((torch.real(Phix),torch.imag(Phix)),1).view(-1,Phix.size(-1))
        
#        x = x - x_dealiasing
        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
        x = x.view(-1,1,8,x.size(-1))
        x = torch.cat((torch.real(x),torch.imag(x)),1)
        x = x.view(-1,660)
        x_final = x

        return [x_final, layers_sym]
    
class BasicBlock_side12(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_side11, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 2, 3, 7)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 16, 3, 5)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 16, 3, 7)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 16, 3, 5)))
        
        self.weight = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 1, 1)))
        
        self.conv5 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
        self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
        
        self.conv1 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv2 = nn.Conv2d(8,1,kernel_size = [3,5], padding = [1,2])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,10))
        self.fc1 = nn.Linear(4*66,1)
        
    def forward(self, x, PhiHPhi, PhiHb, x_side):
        # x: complex, reshaped
        # PhiTPhi: complex
        # PhiTb: complex
        # x_side: real, original size
        
        flag_beam = 1
        x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
        if flag_beam== 1:
            x_side = torch.fft.fft(x_side,dim=-2)
        x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,8,x_side.size(-1)))
        out = F.relu(self.conv5(x_side_abs))
        out = F.relu(self.conv6(out))
        Mask = F.sigmoid(self.conv7(out)) # real, original shape
        
        
        xx = F.relu(self.conv1(x_side_abs))
        xx = F.relu(self.conv2(xx))
        xx = self.maxpool1(xx).view(-1,4*66)
        weight = F.sigmoid(self.fc1(xx)).view(-1,1,1,1)
#        x = x - torch.mm(x, PhiTPhi) # complex, (-1,660)
        x = x - self.lambda_step * torch.mm(x, PhiHPhi) # complex, (-1,660)
        x = x + self.lambda_step * PhiHb # complex, (-1,660)
        
        
        x_input = x.view(-1, 1, 8, 660)
        if flag_beam== 1:
            x_input = torch.fft.fft(x_input,dim=-2)
            
        x_input = torch.concat((torch.real(x_input),torch.imag(x_input)),dim=1)
        
#        x_input_ini = x.view(-1, 1, 8, 660)
#        x_input_ini = torch.concat((torch.real(x_input_ini),torch.imag(x_input_ini)),dim=1)
#        
        
        x_bp_filtered = x_input*Mask
        x_merged = weight*x_bp_filtered + (1-weight)*x_input
        
        x = F.conv2d(x_merged, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
#        print(str(x_backward.size()))
        x_backward = x_backward[:,0::2,:,:] + 1j*x_backward[:,1::2,:,:]
#            print(str(x_backward.size()))
#            x_backward = torch.fft.ifft(x_backward,dim=-2)
#        print(str(x_backward.size()))
        x_pred = x_backward.view(-1, 660)
#        x_pred = x_backward.reshape(x_backward.size(0)*1*8, 660)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1) 
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_merged

        return [x_pred, symloss]
    
    
    
#class BasicBlock_side_UL(torch.nn.Module):
#    def __init__(self):
#        super(BasicBlock_side_UL, self).__init__()
#
#        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
#        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
#
#        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
#        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
#        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
#        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))
#        
#        self.weight = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 1, 1)))
#        
#        self.conv5 = nn.Conv2d(3,8,kernel_size = [3,7], padding = [1,3])
#        self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
#        self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
#        
#        self.conv1 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
#        self.conv2 = nn.Conv2d(8,1,kernel_size = [3,5], padding = [1,2])
#        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,4))
#        self.fc1 = nn.Linear(16*13,1)
#        
#    def forward(self, x, PhiHPhi, PhiHb, x_side):
#        # x: complex, reshaped
#        # PhiTPhi: complex
#        # PhiTb: complex
#        # x_side: real, original size
#        N_RB = x_side.size(-1)
#        flag_beam = 1
#        x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
#        if flag_beam== 1:
#            x_side = torch.fft.fft(x_side,dim=-2)
#        x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,32,x_side.size(-1)))
#       
#        
#        
#        xx = F.relu(self.conv1(x_side_abs))
#        xx = F.relu(self.conv2(xx))
#        xx = self.maxpool1(xx).view(-1,16*13)
#        weight = F.sigmoid(self.fc1(xx)).view(-1,1,1,1)
##        x = x - torch.mm(x, PhiTPhi) # complex, (-1,660)
#        x = x - self.lambda_step * torch.mm(x, PhiHPhi) # complex, (-1,660)
#        x = x + self.lambda_step * PhiHb # complex, (-1,660)
#        
#        
#        x_input = x.view(-1, 1, 32, 52)
#        if flag_beam== 1:
#            x_input = torch.fft.fft(x_input,dim=-2)
#            
#        x_input = torch.concat((torch.real(x_input),torch.imag(x_input)),dim=1)
#        x_inter = torch.concat((x_input,x_side_abs),dim=1)
#        
#        out = F.leaky_relu(self.conv5(x_inter))
#        out = F.leaky_relu(self.conv6(out))
#        x_bp_filtered = self.conv7(out) # real, original shape
##        x_input_ini = x.view(-1, 1, 8, 660)
##        x_input_ini = torch.concat((torch.real(x_input_ini),torch.imag(x_input_ini)),dim=1)
##        
#        x_merged = weight*x_bp_filtered + (1-weight)*x_input
#        
#        x = F.conv2d(x_merged, self.conv1_forward, padding=1)
#        x = F.relu(x)
#        x_forward = F.conv2d(x, self.conv2_forward, padding=1)
#
#        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
#        x = F.conv2d(x, self.conv1_backward, padding=1)
#        x = F.relu(x)
#        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
##        print(str(x_backward.size()))
#        x_backward = x_backward[:,0::2,:,:] + 1j*x_backward[:,1::2,:,:]
##            print(str(x_backward.size()))
##            x_backward = torch.fft.ifft(x_backward,dim=-2)
##        print(str(x_backward.size()))
#        x_pred = x_backward.view(-1, 52)
##        x_pred = x_backward.reshape(x_backward.size(0)*1*8, 660)
#
#        x = F.conv2d(x_forward, self.conv1_backward, padding=1) 
#        x = F.relu(x)
#        x_est = F.conv2d(x, self.conv2_backward, padding=1)
#        symloss = x_est - x_merged
#
#        return [x_pred, symloss]
class BasicBlock_side_UL(torch.nn.Module):
    def __init__(self, N_ant, N_RB):
        super(BasicBlock_side_UL, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))
        
        self.weight = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 1, 1)))
        
        self.conv5 = nn.Conv2d(3,8,kernel_size = [3,7], padding = [1,3])
        self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
        self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
        
        self.conv1 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv2 = nn.Conv2d(8,1,kernel_size = [3,5], padding = [1,2])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,4))
        self.fc1 = nn.Linear(N_ant*N_RB//4//2,1)
        
    def forward(self, x, PhiHPhi, PhiHb, x_side):
        # x: complex, reshaped
        # PhiTPhi: complex
        # PhiTb: complex
        # x_side: real, original size
        N_RB = x_side.size(-1)
        flag_beam = 1
        x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
        if flag_beam== 1:
            x_side = torch.fft.fft(x_side,dim=-2)
        x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,32,x_side.size(-1)))
       
        
        
        xx = F.relu(self.conv1(x_side_abs))
        xx = F.relu(self.conv2(xx))
        xx = self.maxpool1(xx).view(-1,x_side.size(-2)*x_side.size(-1)//4//2)
        weight = F.sigmoid(self.fc1(xx)).view(-1,1,1,1)
#        x = x - torch.mm(x, PhiTPhi) # complex, (-1,660)
        x = x - self.lambda_step * torch.mm(x, PhiHPhi) # complex, (-1,660)
        x = x + self.lambda_step * PhiHb # complex, (-1,660)
        
        
        x_input = x.view(-1, 1, 32, N_RB)
        if flag_beam== 1:
            x_input = torch.fft.fft(x_input,dim=-2)
            
        x_input = torch.concat((torch.real(x_input),torch.imag(x_input)),dim=1)
        x_inter = torch.concat((x_input,x_side_abs),dim=1)
        
        out = F.leaky_relu(self.conv5(x_inter))
        out = F.leaky_relu(self.conv6(out))
        x_bp_filtered = self.conv7(out) # real, original shape
#        x_input_ini = x.view(-1, 1, 8, 660)
#        x_input_ini = torch.concat((torch.real(x_input_ini),torch.imag(x_input_ini)),dim=1)
#        
        x_merged = weight*x_bp_filtered + (1-weight)*x_input
        
        x = F.conv2d(x_merged, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
#        print(str(x_backward.size()))
        x_backward = x_backward[:,0::2,:,:] + 1j*x_backward[:,1::2,:,:]
#            print(str(x_backward.size()))
#            x_backward = torch.fft.ifft(x_backward,dim=-2)
#        print(str(x_backward.size()))
        x_pred = x_backward.view(-1, N_RB)
#        x_pred = x_backward.reshape(x_backward.size(0)*1*8, 660)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1) 
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_merged

        return [x_pred, symloss]

class BasicBlock_side_UL_2D(torch.nn.Module):
    def __init__(self, N_ant, N_RB):
        super(BasicBlock_side_UL_2D, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 2, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(2, 32, 3, 3)))
        
        self.weight = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 1, 1)))
        
        self.conv5 = nn.Conv2d(3,8,kernel_size = [3,7], padding = [1,3])
        self.conv6 = nn.Conv2d(8,4,kernel_size = [3,5], padding = [1,2])
        self.conv7 = nn.Conv2d(4,1,kernel_size = 3, padding = 1)
        
        self.conv1 = nn.Conv2d(1,8,kernel_size = [3,7], padding = [1,3])
        self.conv2 = nn.Conv2d(8,1,kernel_size = [3,5], padding = [1,2])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,4))
        self.fc1 = nn.Linear(N_ant*N_RB//4//2,1)
    def forward(self, x, PhiHPhi, PhiHb, x_side):
        # x: complex, reshaped
        # PhiTPhi: complex
        # PhiTb: complex
        # x_side: real, original size
        N_RB = x_side.size(-1)
        flag_beam = 1
        x_side = x_side[:,0,:,:] + 1j*x_side[:,1,:,:]
        if flag_beam== 1:
            x_side = torch.reshape(x_side,[x_side.size(0),4,8,N_RB])
            x_side = torch.fft.fft(torch.fft.fft(x_side,dim=-2),dim=-3)
            x_side = torch.reshape(x_side,[x_side.size(0),32,N_RB])
        x_side_abs = torch.fft.ifft(x_side, dim=-1).abs().reshape((x_side.size(0),1,32,x_side.size(-1)))
       
        
        
        xx = F.relu(self.conv1(x_side_abs))
        xx = F.relu(self.conv2(xx))
        xx = self.maxpool1(xx).view(-1,x_side.size(-2)*x_side.size(-1)//4//2)
        weight = F.sigmoid(self.fc1(xx)).view(-1,1,1,1)
#        x = x - torch.mm(x, PhiTPhi) # complex, (-1,660)
        x = x - self.lambda_step * torch.mm(x, PhiHPhi) # complex, (-1,660)
        x = x + self.lambda_step * PhiHb # complex, (-1,660)
        
        
        x_input = x.view(-1, 1, 32, N_RB)
        if flag_beam== 1:
            x_input = torch.reshape(x_input,[x_input.size(0),4,8,N_RB])
            x_input = torch.fft.fft(torch.fft.fft(x_input,dim=-2),dim=-3)
            x_input = torch.reshape(x_input,[x_input.size(0),1,32,N_RB])
            
        x_input = torch.concat((torch.real(x_input),torch.imag(x_input)),dim=1)
        x_inter = torch.concat((x_input,x_side_abs),dim=1)
        
        out = F.leaky_relu(self.conv5(x_inter))
        out = F.leaky_relu(self.conv6(out))
        x_bp_filtered = self.conv7(out) # real, original shape
#        x_input_ini = x.view(-1, 1, 8, 660)
#        x_input_ini = torch.concat((torch.real(x_input_ini),torch.imag(x_input_ini)),dim=1)
#        
        x_merged = weight*x_bp_filtered + (1-weight)*x_input
        
        x = F.conv2d(x_merged, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
#        print(str(x_backward.size()))
        x_backward = x_backward[:,0::2,:,:] + 1j*x_backward[:,1::2,:,:]
#            print(str(x_backward.size()))
#            x_backward = torch.fft.ifft(x_backward,dim=-2)
#        print(str(x_backward.size()))
        x_pred = x_backward.view(-1, N_RB)
#        x_pred = x_backward.reshape(x_backward.size(0)*1*8, 660)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1) 
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_merged

        return [x_pred, symloss]




class ResidualRefiner(torch.nn.Module):
    def __init__(self):
        super(ResidualRefiner, self).__init__()
        self.conv1 = nn.Conv2d(2,16,kernel_size = [9,9], padding = [4,4])
        self.conv2 = nn.Conv2d(16,8,kernel_size = [5,5], padding = [2,2])
        self.conv3 = nn.Conv2d(8,2,kernel_size = [3,3], padding = [1,1])
    def forward(self, diff):
        output = nn.Tanh()(self.conv1(diff))
        output = nn.Tanh()(self.conv2(output))
        output = nn.Tanh()(self.conv3(output))
        return output
    
class ResidualRefiner_ex(torch.nn.Module):
    def __init__(self):
        super(ResidualRefiner_ex, self).__init__()
        self.conv1 = nn.Conv2d(4,16,kernel_size = [9,9], padding = [4,4])
        self.conv2 = nn.Conv2d(16,8,kernel_size = [5,5], padding = [2,2])
        self.conv3 = nn.Conv2d(8,2,kernel_size = [3,3], padding = [1,1])
    def forward(self, diff, out):
        diff = torch.concat((diff,out),dim=1)
        output = nn.Tanh()(self.conv1(diff))
        output = nn.Tanh()(self.conv2(output))
        output = nn.Tanh()(self.conv3(output))
        return output
# Define ISTA-Net
class UL_ISTANet(torch.nn.Module):
    def __init__(self, LayerNo, N_ant, N_RB):
        super(UL_ISTANet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side_UL(N_ant, N_RB))
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi):
#        y =  dl_csi, ul_csi
#        x_dealiasing, a, b, c, d = self.srcsinet(y)
#        x_dealiasing = torch.fft.ifft(x_dealiasing[:,0::2,:,:] + 1j*x_dealiasing[:,1::2,:,:])
#        x_dealiasing = x_dealiasing.view(-1,660)
        N_RB = ul_csi.size(-1)
        Phix = Phix.view(-1,2,32,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        
#        Phix = Phix.view(-1,1,8,Phix.size(-1))
#        Phix = torch.cat((torch.real(Phix),torch.imag(Phix)),1).view(-1,Phix.size(-1))
        
#        x = x - x_dealiasing
        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
        x = x.view(-1,1,32,x.size(-1))
        x = torch.cat((torch.real(x),torch.imag(x)),1)
        x = x.view(-1,N_RB)
        x_final = x

        return [x_final, layers_sym]

class UL_ISTANet_2D(torch.nn.Module):
    def __init__(self, LayerNo, N_ant, N_RB):
        super(UL_ISTANet_2D, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side_UL_2D(N_ant, N_RB))
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi):
#        y =  dl_csi, ul_csi
#        x_dealiasing, a, b, c, d = self.srcsinet(y)
#        x_dealiasing = torch.fft.ifft(x_dealiasing[:,0::2,:,:] + 1j*x_dealiasing[:,1::2,:,:])
#        x_dealiasing = x_dealiasing.view(-1,660)
        N_RB = ul_csi.size(-1)
        Phix = Phix.view(-1,2,32,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        
#        Phix = Phix.view(-1,1,8,Phix.size(-1))
#        Phix = torch.cat((torch.real(Phix),torch.imag(Phix)),1).view(-1,Phix.size(-1))
        
#        x = x - x_dealiasing
        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
        x = x.view(-1,1,32,x.size(-1))

        x = torch.reshape(x,[x.size(0),4,8,N_RB])###
        x = torch.fft.ifft(torch.fft.ifft(x,dim=-2),dim=-3)####
        x = torch.reshape(x,[x.size(0),1,32,N_RB])  ####
        
        x = torch.cat((torch.real(x),torch.imag(x)),1)
        x = x.view(-1,N_RB)
        x_final = x

        return [x_final, layers_sym]
    
class UL_ISTANet_R(torch.nn.Module):
    def __init__(self, LayerNo):
        super(UL_ISTANet_R, self).__init__()
        onelayer = []
        onelayer_refiner = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side_UL())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)
        self.freq_domain_refiner = ResidualRefiner()
        for i in range(LayerNo):
            onelayer_refiner.append(ResidualRefiner())
        self.fcs_refiner = nn.ModuleList(onelayer_refiner)
    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi, Q):
        
        Phix = Phix.view(-1,2,32,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
            # batch x 2 x 32 x 52
            
            out = torch.mm(x,torch.transpose(Q,0,1))
            out = out.view(-1,1,32,52)
            tmp_real = torch.real(out)
            tmp_imag = torch.imag(out)
            out = torch.cat((tmp_real,tmp_imag),1)
            
            out_copy = torch.clone(out)
            out_copy[:,:,:,0::4] = dl_csi
            diff = out - out_copy
            x_diff = self.fcs_refiner[i](diff)
            x_diff = x_diff[:,0,:,:] + 1j*x_diff[:,1,:,:]
            x_diff = x_diff.view(-1,52)
            
            x = x + x_diff
            
            x = x.view(-1,52)
            x = torch.mm(x,torch.conj(Q))
            
            
            
        x = x.view(-1,1,32,x.size(-1))
        x = torch.cat((torch.real(x),torch.imag(x)),1)
        x = x.view(-1,52)
        x_final = x

        return [x_final, layers_sym]

class UL_ISTANet_R2(torch.nn.Module):
    def __init__(self, LayerNo):
        super(UL_ISTANet_R2, self).__init__()
        onelayer = []
        onelayer_refiner = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side_UL())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)
        self.freq_domain_refiner = ResidualRefiner()
        for i in range(LayerNo):
            onelayer_refiner.append(ResidualRefiner())
        self.fcs_refiner = nn.ModuleList(onelayer_refiner)
    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi, Q):
        
        Phix = Phix.view(-1,2,32,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
            # batch x 2 x 32 x 52
            
            out = torch.mm(x,torch.transpose(Q,0,1))
            out = out.view(-1,1,32,52)
            tmp_real = torch.real(out)
            tmp_imag = torch.imag(out)
            out = torch.cat((tmp_real,tmp_imag),1)
            
            out_copy = torch.clone(out)
            out_copy[:,:,:,0::4] = dl_csi[:,:,:,0::4]
            diff = out - out_copy
            x_diff = self.fcs_refiner[i](diff)
            
            x = out - x_diff
            x = x[:,0,:,:]+1j*x[:,1,:,:]
            x = x.view(-1,52)
            x = torch.mm(x,torch.conj(Q))
            
            
            
        x = x.view(-1,1,32,x.size(-1))
        x = torch.cat((torch.real(x),torch.imag(x)),1)
        x = x.view(-1,52)
        x_final = x

        return [x_final, layers_sym]
    
    
class UL_ISTANet_R3(torch.nn.Module):
    def __init__(self, LayerNo):
        super(UL_ISTANet_R3, self).__init__()
        onelayer = []
        onelayer_refiner = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side_UL())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)
        for i in range(LayerNo):
            onelayer_refiner.append(ResidualRefiner_ex())
        self.fcs_refiner = nn.ModuleList(onelayer_refiner)
    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi, Q):
        
        Phix = Phix.view(-1,2,32,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
            # batch x 2 x 32 x 52
            
            out = torch.mm(x,torch.transpose(Q,0,1))
            out = out.view(-1,1,32,52)
            tmp_real = torch.real(out)
            tmp_imag = torch.imag(out)
            out = torch.cat((tmp_real,tmp_imag),1)
            
            out_copy = torch.clone(out)
            out_copy[:,:,:,0::4] = dl_csi[:,:,:,0::4]
            diff = out - out_copy
            x_diff = self.fcs_refiner[i](diff, out)
            
            x = out - x_diff
            x = x[:,0,:,:] + 1j*x[:,1,:,:]
            x = x.view(-1,52)
            x = torch.mm(x,torch.conj(Q))
            
            
            
        x = x.view(-1,1,32,x.size(-1))
        x = torch.cat((torch.real(x),torch.imag(x)),1)
        x = x.view(-1,52)
        x_final = x

        return [x_final, layers_sym]
    

class UL_ISTANet_R4(torch.nn.Module):
    def __init__(self, LayerNo):
        super(UL_ISTANet_R4, self).__init__()
        onelayer = []
        onelayer_refiner = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side_UL())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)
        for i in range(LayerNo):
            onelayer_refiner.append(ResidualRefiner_ex())
        self.fcs_refiner = nn.ModuleList(onelayer_refiner)
    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi, Q):
        
        Phix = Phix.view(-1,2,32,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss
        layers_out = []

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
            # batch x 2 x 32 x 52
            
            out = torch.mm(x,torch.transpose(Q,0,1))
            out = out.view(-1,1,32,52)
            tmp_real = torch.real(out)
            tmp_imag = torch.imag(out)
            out = torch.cat((tmp_real,tmp_imag),1)
            
            
            
            out_copy = torch.clone(out)
            out_copy[:,:,:,0::4] = dl_csi[:,:,:,0::4]
            diff = out - out_copy
#            x_diff = self.fcs_refiner[i](diff, out)
            
            x = out - diff
#            x = out
            layers_out.append(x)
            
            x = x[:,0,:,:] + 1j*x[:,1,:,:]
            x = x.view(-1,52)
            x = torch.mm(x,torch.conj(Q))

        return [layers_out, layers_sym]
    
    
class UL_ISTANet_R5(torch.nn.Module):
    def __init__(self, LayerNo):
        super(UL_ISTANet_R5, self).__init__()
        onelayer = []
        onelayer_refiner = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side_UL())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)
        for i in range(LayerNo):
            onelayer_refiner.append(ResidualRefiner_ex())
        self.fcs_refiner = nn.ModuleList(onelayer_refiner)
    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi, Q):
        
        Phix = Phix.view(-1,2,32,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss
        layers_out = []

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
            # batch x 2 x 32 x 52
            
            out = torch.mm(x,torch.transpose(Q,0,1))
            out = out.view(-1,1,32,52)
            tmp_real = torch.real(out)
            tmp_imag = torch.imag(out)
            out = torch.cat((tmp_real,tmp_imag),1)
            
            
            
#            out_copy = torch.clone(out)
#            out_copy[:,:,:,0::4] = dl_csi[:,:,:,0::4]
            diff = out[:,:,:,0::4] - dl_csi[:,:,:,0::4]
            diff = nn.Upsample(scale_factor=tuple([1,4]),mode='bilinear')(diff)
#            x_diff = self.fcs_refiner[i](diff, out)
            
            x = out - diff
#            x = out
            layers_out.append(x)
            
            x = x[:,0,:,:] + 1j*x[:,1,:,:]
            x = x.view(-1,52)
            x = torch.mm(x,torch.conj(Q))

        return [layers_out, layers_sym]
    
class UL_ISTANet_R6(torch.nn.Module):
    def __init__(self, LayerNo):
        super(UL_ISTANet_R6, self).__init__()
        onelayer = []
        onelayer_refiner = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side_UL())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)
        for i in range(LayerNo):
            onelayer_refiner.append(ResidualRefiner_ex())
        self.fcs_refiner = nn.ModuleList(onelayer_refiner)
    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi, Q):
        
        Phix = Phix.view(-1,2,32,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss
        layers_out = []

        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
            # batch x 2 x 32 x 52
            
            out = torch.mm(x,torch.transpose(Q,0,1))
            out = out.view(-1,1,32,52)
            tmp_real = torch.real(out)
            tmp_imag = torch.imag(out)
            out = torch.cat((tmp_real,tmp_imag),1)
            
            
            
            out_copy = torch.clone(out)
            out_copy[:,:,:,0::4] = dl_csi[:,:,:,0::4]
            diff = out - out_copy
            x_diff = self.fcs_refiner[i](diff, out) + diff
            
            x = out - x_diff
#            x = out
            layers_out.append(x)
            
            x = x[:,0,:,:] + 1j*x[:,1,:,:]
            x = x.view(-1,52)
            x = torch.mm(x,torch.conj(Q))

        return [layers_out, layers_sym]
    
class UL_ISTANet_R7(torch.nn.Module):
    def __init__(self, LayerNo):
        super(UL_ISTANet_R7, self).__init__()
        onelayer = []
        onelayer_refiner = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side_UL())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)
        self.freq_domain_refiner = ResidualRefiner()
        for i in range(LayerNo):
            onelayer_refiner.append(ResidualRefiner())
        self.fcs_refiner = nn.ModuleList(onelayer_refiner)
    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi, Q):
        
        Phix = Phix.view(-1,2,32,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss
        layers_out = []
        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
            # batch x 2 x 32 x 52
            out = torch.mm(x,torch.transpose(Q,0,1))
            out = out.view(-1,1,32,52)
            tmp_real = torch.real(out)
            tmp_imag = torch.imag(out)
            out = torch.cat((tmp_real,tmp_imag),1)
            if i == self.LayerNo-1:
                
                out_copy = torch.clone(out)
                out_copy[:,:,:,0::4] = dl_csi[:,:,:,0::4]
                diff = out - out_copy
                x_diff = self.fcs_refiner[i](diff)
                
                x = out - x_diff
            else:
                x = out
#            x = out
            layers_out.append(x)
            
            x = x[:,0,:,:] + 1j*x[:,1,:,:]
            x = x.view(-1,52)
            x = torch.mm(x,torch.conj(Q))

        return [layers_out, layers_sym]
    
class UL_ISTANet_R8(torch.nn.Module):
    def __init__(self, LayerNo):
        super(UL_ISTANet_R8, self).__init__()
        onelayer = []
        onelayer_refiner = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock_side_UL())
        self.srcsinet = SIMaskNet()
        self.fcs = nn.ModuleList(onelayer)
        self.freq_domain_refiner = ResidualRefiner()
        for i in range(LayerNo):
            onelayer_refiner.append(ResidualRefiner())
        self.fcs_refiner = nn.ModuleList(onelayer_refiner)
    def forward(self, Phix, Phi, Qinit, dl_csi, ul_csi, Q):
        
        Phix = Phix.view(-1,2,32,Phix.size(-1))
        Phix = Phix[:,0::2,:,:] + 1j*Phix[:,1::2,:,:]
        Phix = Phix.view(-1,Phix.size(-1))
        
        PhiHPhi = torch.mm(torch.transpose(torch.conj(Phi), 0, 1), Phi)
        PhiHb = torch.mm(Phix, torch.conj(Phi))
        # Phix: freqeuncy domain dl csi
        # PhiHb: LS solution to delay 
        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
                
        layers_sym = []   # for computing symmetric loss
        layers_out = []
        for i in range(self.LayerNo):
            # x: complex, reshaped
            # PhiTPhi: complex
            # PhiTb: complex
            # x_side: real, original size
            [x, layer_sym] = self.fcs[i](x, PhiHPhi, PhiHb, ul_csi)
            layers_sym.append(layer_sym)
            # batch x 2 x 32 x 52
            out = torch.mm(x,torch.transpose(Q,0,1))
            out = out.view(-1,1,32,52)
            tmp_real = torch.real(out)
            tmp_imag = torch.imag(out)
            out = torch.cat((tmp_real,tmp_imag),1)
            if i == self.LayerNo-1:
                
                out_copy = torch.clone(out)
                out_copy[:,:,:,0::4] = dl_csi[:,:,:,0::4]
                diff = out - out_copy
                x_diff = self.fcs_refiner[i](diff)
                
                x = out - x_diff
            else:
                x = out
#            x = out
            layers_out.append(x)
            
            x = x[:,0,:,:] + 1j*x[:,1,:,:]
            x = x.view(-1,52)
            x = torch.mm(x,torch.conj(Q))

        return [layers_out, layers_sym]