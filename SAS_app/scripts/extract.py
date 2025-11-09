from pathlib import Path
from .viewer import seismicViewer
from .attri import attrComp
from .mask import extMask

import numpy as np
from numpngw import write_png
from tqdm.auto import tqdm

from pandas import DataFrame
from torchvision import io

from warnings import filterwarnings
filterwarnings('ignore')

class ExtractPatches(object):
    
    def __init__(self, imgs_path:Path,  attribute_type:str='enve', threshold=1, kernel=None, noise='gaussian'):
                
        self.img_paths = Path(imgs_path)
        self.attribute_type = attribute_type
        self.threshold = threshold
        self.kernel = kernel
        self.noise = noise
    
    def __call__(self, preview, idx=None):
        return self.extractPatches(preview=preview, idx=idx)
    
    def get_image_paths(self):
        from os import listdir, path
        image_paths = [path.join(self.img_paths,'images', i) for i in listdir(self.img_paths/'images')]
        image_names = [i[:-4] for i in listdir(self.img_paths/'images')]
        
        mask_names = image_names
        
        data = DataFrame()
        data['images'] = image_paths
        
        return data, mask_names

    def read_images(self):
    
        image_df, mask_names = self.get_image_paths()
        
        images = list()
        for idx in image_df.images:
            image = io.read_image(str(idx), mode=io.ImageReadMode.GRAY).permute(1, 2, 0).numpy()
            images.append(image)
        
        return images, mask_names
    

    def run_attribute(self):
        
        images, mask_names = self.read_images()
        
        masks = list()
        attris = list()
        ori_images = list()
        noise_reds = list()
        
        for img in images:
            ori_image, noise_red, attr = attrComp(data=img,
                                                        attri_type=self.attribute_type,
                                                        kernel=self.kernel,
                                                        noise=self.noise)
            
            mask = extMask(cube=attr, threshold=self.threshold)
            
            masks.append(mask)
            attris.append(attr)
            ori_images.append(ori_image)
            noise_reds.append(noise_red)

        return ori_images, noise_reds, attris, masks, mask_names
    
    def extractPatches(self, preview, idx=None):
        
        ROOT = self.img_paths
        images, noise_reds, attris, tmasks, mask_names = self.run_attribute()
        
        if preview:

            assert idx != None, 'idx must be parsed'
            for _, i in enumerate(images[:idx]):
                
                seismicViewer(i, mask_names[_])
        else:
            try:

                masks = 'masks'

                if Path(ROOT/masks).exists():
                    print(f'{masks} directory already exists! Overwriting....')
                else:
                    print(f'Creating {masks} directory...')
                    Path(ROOT/masks).mkdir()
                    
            except:
                raise Exception('Error creating directories!')

            print('='*60)
            print('     Masks extraction started!...')
            print()
            
            for mask, maskname in tqdm(zip(tmasks, mask_names)):

                shape = (mask.shape[0], mask.shape[1]*mask.shape[2])
                
                write_png(f'{str(ROOT)}/{masks}/{maskname}'+'.png', 
                          mask.reshape(shape).astype(np.uint8), bitdepth=1)          
            
            print('     Masks extraction completed...!')
            print('='*60)
