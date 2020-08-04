# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 17:52:42 2018
@author: simon
"""
import numpy as np
import random
import io as inout
import nibabel as nib
import os
join = os.path.join
import datetime
import sys
from PIL import Image
from scipy import ndimage
from matplotlib import pyplot as plt
import glob
from functools import partial
from skimage import io
import datetime
import time
import json
from tqdm import tqdm

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, Flatten, Dense, Reshape, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.utils import multi_gpu_model, to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint


def crop(im,xc,yc,cs):
    """ 
    xc,yc = crop centre
    cs = crop size
    """

    res = np.zeros(cs,dtype=im.dtype)
    
    xs,ys = im.shape
    x0,x1 = int(np.max([0.,int(xc)-int(cs[0]/2)])), int(np.min([int(xc)+int(cs[0]/2),xs]))
    y0,y1 = int(np.max([0.,int(yc)-int(cs[1]/2)])), int(np.min([int(yc)+int(cs[1]/2),ys]))

    im = im[x0:x1,y0:y1]

    px,py = 0,0
    padx,pady = [0,0],[0,0]
    if im.shape[0]<cs[0]:
        px = int((cs[0]-im.shape[0])/2.)
        # Check for odd sized images dimensions
        if im.shape[0]+px*2 == cs[0]-1:
            padx = [px+1,px]
        else:
            padx = [px,px]
    if im.shape[1]<cs[1]:
        py = int((cs[1]-im.shape[0])/2.)
        # Check for odd sized images dimensions
        if im.shape[1]+py*2 == cs[1]-1:
            pady = [py+1,py]
        else:
            pady = [py,py]

    im = np.pad(im,(padx,pady),'constant', constant_values=(0))

    return im

class TubeNet(object):

    def __init__(self,size=512):
        self.cropSize = size
        self.volume = None
        
    def encode_block(self,input_layer,n,input_shape=None,no_pool=False,index=0,add_skips=True):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        if input_shape is not None:
            conv = Conv3D(n, 3, activation='relu', padding='same',kernel_initializer=init,input_shape=input_shape,name='conv_encode{}_{}_{}'.format(index,n,1))(input_layer)
        else:
            conv = Conv3D(n, 3, activation='relu', padding='same',kernel_initializer=init,name='conv_encode{}_{}_{}'.format(index,n,1))  (input_layer)

        dr = tfa.layers.InstanceNormalization(axis=1,center=True,scale=True,beta_initializer="random_uniform",gamma_initializer="random_uniform")(conv)            
        conv = Conv3D(n, 3, activation='relu', padding='same',kernel_initializer=init,name='conv_encode{}_{}_{}'.format(index,n,2))(dr)
        dr = tfa.layers.InstanceNormalization(axis=1,center=True,scale=True,beta_initializer="random_uniform",gamma_initializer="random_uniform")(conv)
        
        if no_pool:
            return dr,conv
        pool = MaxPooling3D(pool_size=(2, 2, 2))(dr)
        return pool,conv
        
    def decode_block(self,input_layer,n,conv_layer=None,index=0,add_skips=True,level=0):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        if True: #level>0:
            convTr_layer = Conv3DTranspose(n, (2, 2, 2), strides=(2, 2, 2), padding='same')(input_layer)
            if add_skips:
                up = concatenate([convTr_layer, conv_layer], axis=4)
            else:
                up = convTr_layer
        else:
            up = input_layer
        conv = Conv3D(n, 3, activation='relu', padding='same',kernel_initializer=init,name='conv_decode{}_{}_{}'.format(index,n,1))(up)
        dr = tfa.layers.InstanceNormalization(axis=1,center=True,scale=True,beta_initializer="random_uniform",gamma_initializer="random_uniform")(conv)
        conv = Conv3D(n, 3, activation='relu', padding='same',kernel_initializer=init,name='conv_decode{}_{}_{}'.format(index,n,2))(dr)
        dr = tfa.layers.InstanceNormalization(axis=1,center=True,scale=True,beta_initializer="random_uniform",gamma_initializer="random_uniform")(conv)
        return dr  
        
    def create_encoder(self,size=512,nchannel=1,max_blocks=15,nblock=None,input_layer=None,ns=None):    
    
        # Encoding
        
        conv_layers = []
        for i in range(nblock):
            no_pool = i==(nblock-1) # Don't use pooling on last encoding layer
            if i==0:
                input_shape = input_layer.shape[1:]
                out,conv = self.encode_block(input_layer,ns[i],input_shape=input_shape,no_pool=no_pool,index=i+1)
            else:
                out,conv = self.encode_block(out,ns[i],no_pool=no_pool,index=i+1)
            conv_layers.append(conv)  
        return conv_layers,out
        
    def create_decoder(self,nblock,conv_layers,out,ns,add_skips=True):
        
        # Decoding 
        
        for i in range(nblock-1): 
            out = self.decode_block(out,ns[nblock-i-2],conv_layer=conv_layers[nblock-i-2],index=i+1,add_skips=add_skips,level=i)
        return out
    
    def create_model(self,size=512,nchannel=1,nclasses=2,max_blocks=15,gfv=False,add_skips=True,nblock=None):

        # Input layer
        input_shape = (size,size,size,nchannel)
        input_layer = Input(shape=input_shape)
        
        if nblock is None:
            nblockLrg = int(np.log(size) / np.log(2))
            nblock = np.min([max_blocks,nblockLrg])
            
        ns = np.asarray([np.power(2,x+1) for x in range(2,nblock+2)])
        ns = np.clip(ns,0,512)

        # Encoding
        conv_layers,out = self.create_encoder(size=size,nchannel=nchannel,max_blocks=max_blocks,nblock=nblock,input_layer=input_layer,ns=ns)
           
        # Global feature vector
        # http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Huang_Range_Scaling_Global_U-Net_for_Perceptual_Image_Enhancement_on_Mobile_ECCVW_2018_paper.pdf
        #if gfv:
        #    _,h,w,d,n = conv._shape_tuple()
        #    ap = AveragePooling3D(pool_size=(h,w,d))(conv)
        #    fl = Flatten()(ap)
        #    out = Dense(n,kernel_initializer=RandomNormal(stddev=0.02))(fl)
        #    out = Reshape((1,1,1,n))(out)
        #    out = Lambda(keras.backend.tile, arguments={'n':(1,h,w,1)})(out)
         
        # Decoding
        out = self.create_decoder(nblock,conv_layers,out,ns,add_skips=add_skips)
  
        # Final layers
        conv = Conv3D(nclasses, 1, activation='sigmoid')(out)
    
        return Model(inputs=[input_layer], outputs=[conv])  
       
        
class TrainingDataGenerator(tf.keras.utils.Sequence):

    def __init__(self,xpath,ypath,batch_size,subvol_size=512,n_classes=2,matched=True,n_example=None,n_batches_per_epoch=10):
        self.xpath = xpath
        self.ypath = ypath
        self.batch_size = batch_size
        self.dims = (subvol_size,subvol_size,subvol_size)
        self.nchannels = 1
        self.n_classes = n_classes
        self.matched_files = matched
        self.n_example = n_example
        self.n_batches_per_epoch = n_batches_per_epoch
        self.x_files = []
        self.y_files = []
        
        x_files,y_files,sim_data = [],[],[]
        for xp in xpath:
            new_x = self.parse_directory(xp)
            x_files.extend(new_x)
            if xp in ypath:
                sim_data.extend([True]*len(new_x))
            else:
                sim_data.extend([False]*len(new_x))
        for yp in ypath:
            new_y = self.parse_directory(yp)
            y_files.extend(new_y)
            
        nx = len(x_files)
        
        y_matched = ['' for i in range(nx)]
        for i,x_file in enumerate(x_files):
            if not sim_data[i]:
                x_basename = os.path.basename(x_file) 
                x_filename, ext = os.path.splitext(x_basename)
                for y_file in y_files:
                    y_basename = os.path.basename(y_file) 
                    y_filename, ext = os.path.splitext(y_basename)
                    if self.matched_files:
                        if x_filename+'_labels'==y_filename:
                            y_matched[i] = y_file
                    else:
                        y_matched[i] = y_file
            else:
                y_matched[i] = x_file

        x_files = [x_files[i] for i,f in enumerate(y_matched) if f!='']
        y_files = [y_matched[i] for i,f in enumerate(y_matched) if f!='']
        sim_data = [sim_data[i] for i,f in enumerate(y_matched) if f!='']
        
        print('Loading data')
        self.xim = []
        self.yim = []
        self.x_tr = []
        self.y_tr = []
        self.label_inds = []
        self.sim_data = []
        valid = []
        count = 0
        for xf,yf in zip(x_files,y_files):
            im, tr = self.read_image(xf)
            # Check that the data is big enough to provide a subvolume (TODO: padding...)
            if np.any(im.shape<self.dims):
                print('Could not add data set with dimensions {} ({})'.format(im.shape,xf))
                valid.append(False)
            else:
                print('Loaded file: {}'.format(xf))
                valid.append(True)
                yim, ytr = self.read_image(yf)
                if xf==yf:
                    self.sim_data.append(True)
                    im[im>0] = 1
                    im = im.astype('float32').squeeze()
                else:
                    self.sim_data.append(False)
                    
                yim[yim>0] = 1
                yim = yim.astype('float32').squeeze()

                if len(yim.shape)!=3:
                    import pdb
                    pdb.set_trace()
                    
                self.yim.append(yim)
                self.xim.append(im)
                self.x_tr.append(tr)
                self.y_tr.append(ytr)

                sx,sy,sz = np.where(self.yim[-1]>0)
                self.label_inds.append([sx,sy,sz])
                count += 1
                self.x_files.append(xf)
                self.y_files.append(yf)
                if self.n_example is not None and count>=n_example:
                    break
        self.sim_data = np.asarray(self.sim_data)
        
    def parse_directory(self,path):
        res = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".nii"):
                    res.append(join(root,file))
        return res
        
    def read_image(self,path): 
        try: 
            img = nib.load(path)
        except Exception as e:
            print(e)
            import pdb
            pdb.set_trace()
        tr = img.affine
        im = np.array(img.dataobj)
        #im = im*255./np.max(im)
        return im, tr
        
    def __len__(self):
        return self.n_batches_per_epoch # int(np.floor(len(self.list_IDs) / self.batch_size))        

    def __getitem__(self,index,export=False,simulated=None,def_index=None):

        # Select a simulated dataset
        if def_index is not None:
            finds = [def_index]
        elif simulated is not None and np.any(self.sim_data==simulated):
            if simulated:
                finds = np.where(self.sim_data)[0]
            else:
                finds = np.where([not x for x in self.sim_data])[0]
        # Randomly select simulated or real data
        elif np.any(self.sim_data) and np.any(~self.sim_data):
            if np.random.uniform()>0.5: # Sim
                finds = np.where(self.sim_data)[0]
            else: # Real
                finds = np.where([not x for x in self.sim_data])[0]
        else:
            finds = np.arange(len(self.xim))
        
        # Randomly select dataset
        fileInd = np.random.choice(finds)
        print('Selected file {} (simulated={})'.format(fileInd,self.sim_data[fileInd]))
        xim = self.xim[fileInd]
        yim = self.yim[fileInd]
        sx,sy,sz = self.label_inds[fileInd]
        sim = self.sim_data[fileInd]

        # Random selection of central point (with filled mask)
        if True:
            idx = int(np.random.uniform(0,len(sx)-1)) # np.random.choice(np.arange(len(sx)))
            cx,cy,cz = sx[idx],sy[idx],sz[idx]
            #print('Centre:{}, value:{}'.format([cx,cy,cz],yim[cx,cy,cz]))
        else:
            cx,cy,cz = np.random.choice(np.arange(0,self.dims[0]-1)), np.random.choice(np.arange(0,self.dims[1]-1)), np.random.choice(np.arange(0,self.dims[2]-1))
            
        X = self.subsample(xim,[cx,cy,cz])
        Y = self.subsample(yim,[cx,cy,cz])
            
        # Augmentations
        
        # Add background
        if sim:
            import cv2
            src_path = '/home/simon/Downloads/2101.jpg'
            img = cv2.imread(src_path)
            H,W = img.shape[:2]
            cx,cy = np.random.uniform(self.dims[0]/2,H-1-self.dims[0]/2),np.random.uniform(self.dims[1]/2,W-1-self.dims[2]/2)
            rx,ry = [int(cx-self.dims[0]/2),int(cx+self.dims[0]/2)], \
                    [int(cy-self.dims[1]/2),int(cy+self.dims[1]/2)]
                    
            if rx[0]<0:
                rem = np.abs(rx[0])
                rx[0],rx[1] = 0,rx[1]+rem
            if rx[1]>=xim.shape[0]:
                rem = np.abs(rx[1]-xim.shape[0])
                rx[0] -= rem
                rx[1] = xim.shape[0]
                
            if ry[0]<0:
                rem = np.abs(ry[0])
                ry[0],ry[1] = 0,ry[1]+rem
            if ry[1]>=xim.shape[1]:
                rem = np.abs(ry[1]-xim.shape[1])
                ry[0] -= rem
                ry[1] = xim.shape[1]                  
                
            img = img[:,:,0]
            img = img[rx[0]:rx[1],ry[0]:ry[1]]
            
            # Blur vessels
            sigma_rnd = np.random.uniform(0,10)
            X = ndimage.gaussian_filter(X,sigma=sigma_rnd)
             
            # Virtual camera object
            from vcam import vcam,meshGen
            c1 = vcam(H=self.dims[0],W=self.dims[1])
            # Surface object
            plane = meshGen(self.dims[0],self.dims[1])
            # Warping plane             
            plane.Z += 20*np.exp(-0.5*((plane.X*1.0/plane.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
            pts3d = plane.getPlane()
            pts2d = c1.project(pts3d)
            map_x,map_y = c1.getMaps(pts2d)
             
            img = cv2.remap(img,map_x,map_y,interpolation=cv2.INTER_LINEAR)
            img = img / 255
            for z in range(X.shape[2]):
                zIm = X[:,:,z]
                #zIm[zIm==0] = img[zIm==0]
                zIm += img
                X[:,:,z] = zIm
            if np.max(X)!=0:
                X = X/np.max(X)
        
            #Add noise
            mx = np.max(X)
            rnd_sd = np.clip(np.random.normal(mx*0.1,0.5),0,mx)
            X = X + np.random.normal(0,rnd_sd,X.shape)
            X = np.clip(X,0,mx+10)
        
        #Rotate
        nr = int(np.random.uniform(1,4))
        k = np.random.uniform(0,4,nr).astype('int')
        axes = np.asarray([[0,1],[0,2],[1,2]])
        axes_rnd = axes[np.random.choice(np.linspace(0,len(axes)-1,len(axes)),nr).astype('int')]
        for i in range(nr):
            X = np.rot90(X,k=k[i],axes=axes_rnd[i])
            Y = np.rot90(Y,k=k[i],axes=axes_rnd[i])
            
        # flip
        nr = int(np.random.uniform(1,4))
        axis = np.random.choice([0,1,2],nr)
        for i in range(nr):
            X = np.flip(X,axis=axis[i])
            Y = np.flip(Y,axis=axis[i])
            
        if export:
            import pdb
            pdb.set_trace()
        
        # Reshape and normalise      
        try:
            X = self.preprocess_X(X)   
            Y = self.preprocess_Y(Y)
        except Exception as e:
            print(e)
            import pdb
            pdb.set_trace()
        
        return (X,Y)
        
    def subsample(self,im,centre):
    
        # Pixel range
        cx,cy,cz = centre
        rx,ry,rz = [int(cx-self.dims[0]/2),int(cx+self.dims[0]/2)], \
                   [int(cy-self.dims[1]/2),int(cy+self.dims[1]/2)], \
                   [int(cz-self.dims[2]/2),int(cz+self.dims[2]/2)]            
        
        if rx[0]<0:
            rem = np.abs(rx[0])
            rx[0],rx[1] = 0,rx[1]+rem
        if rx[1]>=im.shape[0]:
            rem = np.abs(rx[1]-im.shape[0])
            rx[0] -= rem
            rx[1] = im.shape[0]
            
        if ry[0]<0:
            rem = np.abs(ry[0])
            ry[0],ry[1] = 0,ry[1]+rem
        if ry[1]>=im.shape[1]:
            rem = np.abs(ry[1]-im.shape[1])
            ry[0] -= rem
            ry[1] = im.shape[1]
            
        if rz[0]<0:
            rem = np.abs(rz[0])
            rz[0],rz[1] = 0,rz[1]+rem
        if rz[1]>=im.shape[2]:
            rem = np.abs(rz[1]-im.shape[2])
            rz[0] -= rem
            rz[1] = im.shape[2]
            
        if rx[1]-rx[0]!=self.dims[0] or ry[1]-ry[0]!=self.dims[1] or rz[1]-rz[0]!=self.dims[2]:
            import pdb
            pdb.set_trace()
            
        return im[rx[0]:rx[1],ry[0]:ry[1],rz[0]:rz[1]]
        
    def preprocess_X(self,X):
        X = X.astype('float')
        X = np.reshape(X,(1,*self.dims,1))
        X = X / np.max(X)
        return X
        
    def preprocess_Y(self,Y):
        Y = Y.astype('float')
        n_gt_0 = np.sum(Y)
        if n_gt_0>0:
            Y = Y*(self.n_classes-1) / np.max(Y)
        Y = tf.keras.utils.to_categorical(Y, num_classes=self.n_classes)
        Y = np.reshape(Y,(1,*self.dims,self.n_classes))
        return Y      
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        #return int(np.floor(len(self.list_IDs) / self.batch_size))
        return self.batch_size
                        
class EvaluationDataGenerator(TrainingDataGenerator):

    def __init__(self,xpath,subvol_size=512,n_classes=2,matched=True,n_example=None,n_batches_per_epoch=10):
        self.xpath = xpath
        self.dims = (subvol_size,subvol_size,subvol_size)
        self.nchannels = 1
        self.n_classes = n_classes
        self.matched_files = matched
        self.n_example = n_example
        self.x_files = []
        self.x_tr = []

        x_files = []
        for xp in xpath:
            new_x = self.parse_directory(xp)
            x_files.extend(new_x)
            
        nx = len(x_files)
                
        print('Loading data')
        self.xim = []
        valid = []
        count = 0
        for xf in x_files:
            im, tr = self.read_image(xf)
            # Check that the data is big enough to provide a subvolume (TODO: padding...)
            if np.any(im.shape<self.dims):
                print('Could not add data set with dimensions {} ({})'.format(im.shape,xf))
                valid.append(False)
            else:
                print('Loaded file: {}'.format(xf))
                valid.append(True)
                self.xim.append(im)
                self.x_files.append(xf)
                self.x_tr.append(tr)
                if self.n_example is not None and count>=n_example:
                    break

    def __getitem__(self,index):

        fileInd = index
        print('Selected file {}'.format(fileInd))
        xim = self.xim[fileInd]

        # Random selection of central point (with filled mask)
        cx,cy,cz = int(self.dims[0]/2),int(self.dims[1]/2),int(self.dims[2]/2)  
        X = self.subsample(xim,[cx,cy,cz])
        
        # Reshape and normalise      
        X = self.preprocess_X(X) 
          
        return X     
        
class PredictionDataGenerator(EvaluationDataGenerator):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.overlap = 0.25
        self.edge_pad = 10 # pixels
       
    def __getitem__(self,index):

        fileInd = index
        print('Selected file {}'.format(fileInd))
        xim = self.xim[fileInd]

        # Regular gridding of central points
        def centroid_grid(dim,imdim,overlap):
        
            dim2 = int(dim/2)
            p = (1-overlap) * dim2
            x0 = np.arange(dim2,imdim-1,(1-2*overlap)*dim2)
            return x0.astype('int')

            #x0 = np.asarray([int((1-2*overlap)*x) for x in range(dim,imdim-1,dim)])
            #x0,x1 = int(i-p[0]*dim2),int(i+p[0]*dim2[0])
            
            #x0 = np.asarray([int((1-2*overlap)*x) for x in range(0,imdim-1,dim)])
            #x1 = x0 + dim
            #if x1[-1]>imdim-1:
            #    x0[-1] = imdim - dim - 1
            #    x1 = x0 + dim
            #elif x1[-1]<imdim-1:
            #    x0 = np.append(x0,imdim-dim-1)
            #    x1 = x0 + dim
            #return np.mean([x0,x1],axis=0).astype('int')

        cx = centroid_grid(self.dims[0],xim.shape[0],self.overlap)    
        cy = centroid_grid(self.dims[1],xim.shape[1],self.overlap)   
        cz = centroid_grid(self.dims[2],xim.shape[2],self.overlap) 
        
        mg = np.meshgrid(cx,cy,cz)  
        nvol = np.product(mg[0].shape[0:3])

        Xs = []
        for i in cx:
            for j in cy:
                for k in cz:
                    X = self.subsample(xim,[i,j,k])
                    # Reshape and normalise      
                    X = self.preprocess_X(X) 
                    Xs.append(X)
          
        return Xs,[cx,cy,cz],xim.shape
        
    def reconstruct(self,Xpred,centroid,shape):
        recon = np.zeros(shape[0:3],dtype=Xpred[0].dtype)
        count = 0
        dims = self.dims
        nvol = len(centroid[0])*len(centroid[1])*len(centroid[2])
        
        print('Reconstructing {} subvolumes'.format(nvol))
        
        dim2 = np.asarray([int(x/2) for x in self.dims])
        p = (1-self.overlap) * dim2
                
        with tqdm(total=nvol, file=sys.stdout) as pbar:
            for cx,i in enumerate(centroid[0]):
                for cy,j in enumerate(centroid[1]):
                    for cz,k in enumerate(centroid[2]):
                    
                        if cx==0:
                            x0 = int(i-dim2[0])
                            xp0 = 0
                        else:
                            x0 = int(i-dim2[0]+dim2[0]*self.overlap)
                            xp0 = int(dim2[0]*self.overlap)
                        if cx==len(centroid[0])-1:
                            x1 = int(i+dim2[0])
                            xp1 = dims[0]
                        else:
                            x1 = int(i+dim2[0]-self.overlap*dim2[0])
                            xp1 = int(dims[0]-(dim2[0]*self.overlap))
                        if x1>shape[0]:
                            dif = x1-shape[0]
                            x1 = shape[0]
                            xp1 -= dif
                        if x0<0:
                            dif = np.abs(x0)
                            x0 = 0
                            xp0 += dif
                            
                        if cy==0:
                            y0 = int(j-dim2[1])
                            yp0 = 0
                        else:
                            y0 = int(j-dim2[1]+dim2[1]*self.overlap)
                            yp0 = int(dim2[1]*self.overlap)
                        if cy==len(centroid[1])-1:
                            y1 = int(j+dim2[1])
                            yp1 = dims[1]
                        else:
                            y1 = int(j+dim2[1]-self.overlap*dim2[1])
                            yp1 = int(dims[1]-(dim2[1]*self.overlap))
                        if y1>shape[1]:
                            dif = y1-shape[1]
                            y1 = shape[1]
                            yp1 -= dif
                        if y0<0:
                            dif = np.abs(y0)
                            y0 = 0
                            yp0 += dif

                        if cz==0:
                            z0 = int(k-dim2[2])
                            zp0 = 0
                        else:
                            z0 = int(k-dim2[2]+dim2[2]*self.overlap)
                            zp0 = int(dim2[1]*self.overlap)
                        if cz==len(centroid[2])-1:
                            z1 = int(k+dim2[2])
                            zp1 = dims[2]
                        else:
                            z1 = int(k+dim2[2]-self.overlap*dim2[2])
                            zp1 = int(dims[2]-(dim2[2]*self.overlap))
                        if z1>shape[2]:
                            dif = z1-shape[2]
                            z1 = shape[2]
                            zp1 -= dif
                        if z0<0:
                            dif = np.abs(z0)
                            z0 = 0
                            zp0 += dif

                        try:
                            recon[x0:x1,y0:y1,z0:z1] = Xpred[count][xp0:xp1,yp0:yp1,zp0:zp1]
                        except Exception as e:
                            print(e)
                            import pdb
                            pdb.set_trace()
                        count += 1
                        pbar.update(1)
        return recon
 
class SaveModelCallback(tf.keras.callbacks.Callback):

    def __init__(self,generator,validation=None,path=None):
        super().__init__()
        self.data_generator = generator
        self.validation_generator = validation
        self.path = path

    def on_epoch_end(self, epoch, logs={}):
    
        if epoch%10!=0:
            return

        if not os.path.exists(join(self.path,'saved_models')):
            os.makedirs(join(self.path,'saved_models'))

        model_path_w = join(self.path,'saved_models','{}_weights_epoch_{}.hdf5'.format(date_time, epoch))
        self.model.save_weights(model_path_w)
        model_path_m = join(self.path,'saved_models','{}_model_epoch_{}.json'.format(date_time, epoch))
        self.model.save_weights(model_path_m)
        json_string = self.model.to_json()
        with open(model_path_m, 'w') as outfile:
            json.dump(json_string, outfile)
        print('Model saved in saved_models')
        self.saveImages(epoch)
        
    def get_most_recent_saved_model(self):
        wdir = join(self.path,'saved_models')
        wfiles = [f for f in os.listdir(wdir) if os.path.isfile(join(wdir,f)) and join(wdir,f).endswith('.hdf5')]
        mxepoch = 0
        res = ''
        for wfile in wfiles:
            spl = wfile.split('.')
            spl2 = int(spl[0].split('_')[-1])
            if spl2>mxepoch:
                mxepoch = spl2
                res = wfile
        return join(wdir,res),mxepoch
        
    def saveImages(self, epoch, num_saved_images=1):
        directory = join(self.path,'images')
        if not os.path.exists(directory):
            os.makedirs(directory)

        for i in range(num_saved_images):
            X, _ = self.data_generator.__getitem__(None,simulated=False)
            pred = self.model.predict(X)
            pred = np.argmax(pred,axis=-1).squeeze()
            image = np.hstack((X.squeeze(), pred))
            img = nib.Nifti1Image(np.clip(image,-1,1),np.eye(4))
            filename = os.path.join(directory,'epoch{}_sample{}.nii'.format(epoch,i))
            nib.save(img,filename)
            
        if self.validation_generator is not None:
            directory = join(self.path,'validation_images')
            if not os.path.exists(directory):
                os.makedirs(directory)
            for i in range(len(self.validation_generator.x_files)):
                X = self.validation_generator.__getitem__(i)
                pred = self.model.predict(X)
                pred = np.argmax(pred,axis=-1).squeeze()
                image = np.hstack((X.squeeze(), pred))
                img = nib.Nifti1Image(np.clip(image,-1,1),np.eye(4))
                filename = os.path.join(directory,'epoch{}_validation{}.nii'.format(epoch,i))
                nib.save(img,filename)
                print('Saved validation image: {}'.format(filename))
        
class MetricDisplayCallback(tf.keras.callbacks.Callback):

    def __init__(self,log_dir=None):
        super().__init__()
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs={}):
        with self.file_writer.as_default():
            for k,v in zip(logs.keys(),logs.values()):
                if k=='recall':
                    name = 'Recall (fraction of positives that were correctly predicted)'
                elif k=='precision':
                    name = 'Precision (of all predicted positives, ratio that were correct)'
                else:
                    name = k
                tf.summary.scalar(name, v, step=epoch)
        
class ImageDisplayCallback(tf.keras.callbacks.Callback):

    def __init__(self,generator,validation=None,log_dir=None,index=0):
        super().__init__()
        self.log_dir = log_dir
        self.x = None
        self.y = None
        self.pred = None
        self.data_generator = generator
        self.validation_generator = validation
        self.index = index
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs={}, to_buffer=True):

        self.x, self.y = self.data_generator.__getitem__(self.index,simulated=True)
        self.pred = self.model.predict(self.x)
        self.x_real, self.y_real = self.data_generator.__getitem__(self.index,simulated=False)
        self.pred_real = self.model.predict(self.x_real)
        
        sz = self.x.shape
 
        # Recon prediction
        
        ind = int(sz[1]/2.)

        im = self.x[0,ind,:,:,0].squeeze()        
        pred_im = np.argmax(self.pred[0,ind,:,:,:],axis=-1)
        pred_imTrue = np.argmax(self.y[0,ind,:,:,:],axis=-1)
        
        im_real = self.x_real[0,ind,:,:,0].squeeze()        
        pred_im_real = np.argmax(self.pred_real[0,ind,:,:,:],axis=-1)
        pred_imTrue_real = np.argmax(self.y_real[0,ind,:,:,:],axis=-1)
        
        # Plot
        if self.validation_generator is not None:
            nval = len(self.validation_generator.x_files)
        else:
            nval = 0
        columns = 3
        rows = 2 + nval

        fsz = 5
        fig = plt.figure(figsize=(fsz*columns,fsz*rows))
        
        i = 1
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(im)
        ax.title.set_text('Image')
        plt.axis("off")

        i = 2
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(pred_im)
        ax.title.set_text('Predicted labels')
        plt.axis("off")

        i = 3
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(pred_imTrue)
        ax.title.set_text('Labels')
        plt.axis("off")
        
        i = 4
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(im_real)
        ax.title.set_text('Image')
        plt.axis("off")

        i = 5
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(pred_im_real)
        ax.title.set_text('Predicted labels')
        plt.axis("off")

        i = 6
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(pred_imTrue_real)
        ax.title.set_text('Labels')
        plt.axis("off")
        
        if self.validation_generator is not None:
        # Validation
            for j in range(nval):
                x_val = self.validation_generator.__getitem__(j)
                pred_val = self.model.predict(x_val)
                im_val = x_val[0,ind,:,:,0].squeeze()        
                pred_val = np.argmax(pred_val[0,ind,:,:,:],axis=-1)
                
                i += 1
                ax = fig.add_subplot(rows, columns, i)
                plt.imshow(im_val)
                name = os.path.basename(self.validation_generator.x_files[j])
                ax.title.set_text(name)
                plt.axis("off")

                i += 1
                ax = fig.add_subplot(rows, columns, i)
                plt.imshow(pred_val)
                ax.title.set_text('Predicted labels')
                plt.axis("off")
                
                i += 1 # Skip ground truth plot
        
        if to_buffer:
            buf = inout.BytesIO()
            plt.savefig(buf,format='png')
            #plt.savefig('output.png')
            plt.close(fig)
            buf.seek(0)
            image = tf.image.decode_png(buf.getvalue(),channels=4) # #buf.getvalue()
            image = tf.expand_dims(image,0)
            buf.close()
 
            with self.file_writer.as_default():
                tf.summary.image("Images...",image,step=epoch)
        else:
            plt.show()
        
"""Create custom loss"""

cce = CategoricalCrossentropy()
def weighted_crossentropy_Nat(y_true, y_pred, weights=None):
    weights = (1,10)
    weight_mask = y_true[...,0] * weights[0] + y_true[...,1] * weights[1]
    return cce(y_true, y_pred,) * weight_mask
    
def _categorical_crossentropy(target, output, from_logits=False, axis=-1):
    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.math.reduce_sum(output, axis, True)
        # manual computation of crossentropy
        _epsilon = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - tf.math.reduce_sum(target * tf.math.log(output), axis)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,logits=output)
    
    
def weighted_categorical_crossentropy_with_fpr(y_true,y_pred,axis=1,from_logits=False,classes=2, threshold=0.01):
    L = _categorical_crossentropy(target=y_true,output=y_pred,axis=axis,from_logits=from_logits)
    _epsilon = K.epsilon()
    y_true_p = K.argmax(y_true, axis=axis)
    y_pred_bin = K.cast(K.greater_equal(y_pred, threshold), K.dtype(y_true)) if from_logits else K.argmax(y_pred, axis=axis)
    y_pred_probs = y_preds if from_logits else K.max(y_pred, axis=axis)
    for c in range(classes):
        c_true = K.cast(K.equal(y_true_p, c), K.dtype(y_pred))
        w = 1. / (K.sum(c_true) + _epsilon)
        C = K.sum(L * c_true * w) if c == 0 else C + K.sum(L * c_true * w)

        # Calc. FP Rate Correction
        c_false_p = K.cast(K.not_equal(y_true_p, c), K.dtype(y_pred)) * K.cast(K.equal(y_pred_bin, c), K.dtype(y_pred)) # Calculate false predictions
        gamma = 0.5 + (K.sum(K.abs((c_false_p * y_pred_probs) - 0.5)) / (K.sum(c_false_p) + _epsilon)) # Calculate Gamme
        wc = w * gamma # gamma / |Y+|
        C = C + K.sum(L * c_false_p * wc) # Add FP Correction
        
    return C    

"""Create custom metrics"""

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true[...,1] * y_pred[...,1], 0, 1)))
    true_positives0 = K.sum(K.round(K.clip(y_true[...,1] * y_pred[...,1], 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred[...,1], 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true[...,1] * y_pred[...,1], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[...,1], 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def train(resume=False,path=None,resume_date_time=None):

    if resume:
        date_time = resume_date_time

    size = 64   
    log_dir = './logs'
    batch_size = 50
    n_classes = 2
    nepoch = 5000
    n_example = None
    
    tubenet = TubeNet(size=size)

    # TRAINING DATA GENERATOR-----
    xPath,yPath = [],[]

    # Real data
    if True:
        cpath = r'/mnt/data/data/tubenet_data/train/real'
        xPath.append(join(cpath,'X'))
        yPath.append(join(cpath,'Y'))

    # Sims
    if True:
        #cpath = r'/home/simon/Desktop/Share/sims'
        cpath = r'/mnt/data/data/tubenet_data/train/simulated'
        xPath.append(join(cpath))
        yPath.append(join(cpath))

    training_generator = TrainingDataGenerator(xPath,yPath,batch_size,subvol_size=size,n_classes=n_classes,n_example=n_example,n_batches_per_epoch=10)

    # VALIDATION DATA GENERATOR-----
    # Sims
    evPath = []
    if True:
        cpath = r'/mnt/data/data/tubenet_data/evaluate'
        evPath.append(join(cpath))
    validation_generator = EvaluationDataGenerator(evPath,subvol_size=size,n_classes=n_classes,n_example=None)

    model = tubenet.create_model(size=size,nchannel=1,nclasses=n_classes)
    # tell model to run on 2 gpus
    #model_gpu = multi_gpu_model(model, gpus=2)
    model_gpu = model
    model_gpu.compile(optimizer=Adam(lr=1e-3), loss=weighted_crossentropy_Nat, metrics=['accuracy',precision,recall]) # loss= weighted_categorical_crossentropy_with_fpr
    
    # CALLBACKS---
    if os.path.exists(log_dir):
        import shutil
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)
    imlog_dir = join(log_dir,"image")# + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Define the basic TensorBoard callback.
    tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False)
    imageCallback = ImageDisplayCallback(training_generator,validation_generator,log_dir=imlog_dir)
    metriccallback = MetricDisplayCallback(log_dir=log_dir)
    #checkpoint = ModelCheckpoint("best_model.hdf5", monitor='accuracy', verbose=1, save_best_only=True, mode='auto', period=1)
    savemodelCallback = SaveModelCallback(training_generator,validation=validation_generator,path=path)
    
    if resume:
        wfile,initial_epoch = savemodelCallback.get_most_recent_saved_model()
        model_gpu.load_weights(wfile)
    else:
        initial_epoch = 0

    try:
        model_gpu.fit(training_generator,
                        epochs=nepoch,
                        #use_multiprocessing=False,
                        #workers=0,
                        callbacks=[metriccallback,imageCallback,savemodelCallback],#,checkpoint],
                        verbose=1,
                        initial_epoch=initial_epoch)
    except KeyboardInterrupt:
        pass
        
def predict(path=None):

    size = 64
    tubenet = TubeNet(size=size)
    model = tubenet.create_model(size=size,nchannel=1,nclasses=2)
    
    # DATA GENERATOR-----
    # Sims
    evPath = []
    if True:
        path = r'/mnt/data/data/tubenet_data/evaluate'
        evPath.append(join(path))
    generator = EvaluationDataGenerator(evPath,subvol_size=size,n_classes=2,n_example=None)
    
    wdir = join(path,'prediction','model')
    wfiles = [f for f in os.listdir(wdir) if os.path.isfile(join(wdir,f)) and join(wdir,f).endswith('.hdf5')]
    if len(wfiles)!=1:
        print('Invalid number of hdf5 model files in {}'.format(wdir))
        return
        
    print('Loading weights: {}'.format(wfiles[0]))
    model.load_weights(join(wdir,wfiles[0]))

    directory = join(path,'prediction','images')
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    for i in range(len(generator.x_files)):
        X = generator.__getitem__(i)
        fname = os.path.basename(generator.x_files[i])
        pred = model.predict(X)
        pred = np.argmax(pred,axis=-1).squeeze()
        img = nib.Nifti1Image(np.clip(pred,-1,1),np.eye(4))
        filename = join(directory,'{}_predicted.nii'.format(fname))
        nib.save(img,filename)
        print('Saved validation image: {}'.format(filename))
        
def predict_tile(path=None):

    size = 64
    tubenet = TubeNet(size=size)
    model = tubenet.create_model(size=size,nchannel=1,nclasses=2)
    
    # DATA GENERATOR-----
    # Sims
    evPath = []
    if True:
        cpath = r'/mnt/data/data/tubenet_data/evaluate2'
        evPath.append(join(cpath))
    generator = PredictionDataGenerator(evPath,subvol_size=size,n_classes=2,n_example=None)
    
    wdir = join(path,'prediction','model')
    wfiles = [f for f in os.listdir(wdir) if os.path.isfile(join(wdir,f)) and join(wdir,f).endswith('.hdf5')]
    if len(wfiles)!=1:
        print('Invalid number of hdf5 model files in {}'.format(wdir))
        return
        
    print('Loading weights: {}'.format(wfiles[0]))
    model.load_weights(join(wdir,wfiles[0]))

    directory = join(path,'prediction','images')
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    import pdb
    pdb.set_trace()
    for i in range(len(generator.x_files)):
        Xs,centroids,shape = generator.__getitem__(i)
        Xpred = []
        for X in Xs:
            fname = os.path.basename(generator.x_files[i])
            pred = model.predict(X)
            pred = np.argmax(pred,axis=-1).squeeze()
            Xpred.append(pred)
            
        pred = generator.reconstruct(Xpred,centroids,shape)
   
        img = nib.Nifti1Image(np.clip(pred,-1,1).astype('int16'),generator.x_tr[i])
        filename = join(directory,'{}_predicted.nii'.format(fname))
        nib.save(img,filename)
        print('Saved validation image: {}'.format(filename))

if __name__=="__main__":

    resume_date_time = '20200708-133635' # None
    if resume_date_time is None:
        date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        resume = False
    else:
        date_time = resume_date_time
        resume = True
        
    opath = '/home/simon/Desktop/Share/tubenet_unet'
    if not os.path.exists(opath):
        os.makedirs(opath)
    datepath = join(opath,date_time)
    
    #train(path=datepath,resume=resume,resume_date_time=resume_date_time)
    predict_tile(path=opath)
