from ipywidgets import widgets, interact
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .mask import extMask
from .attri import attrComp

from warnings import filterwarnings
filterwarnings('ignore')

def seismicViewer(cube, cube_name):
    '''
    
    helps to preview seismic attribute results,  before extracting the mask patches 
    '''
    
                                      
    image=cube

    attributes = ['sweetness', 'infreq', 'reflin', 'rms', 'timegain', 'enve', 'fder', 'sder',
                  'gradmag', 'inphase', 'cosphase', 'ampcontrast', 'ampacc', 'inband', 'domfreq',
                  'resamp', 'apolar', 'resfreq', 'resphase']

    cmap = ['PuOr_r', 'gray', 'cubehelix', 'jet', 'plasma', 'inferno', 'seismic_r', 'gist_rainbow', 'Accent']

    noise = ['gaussian', 'median', 'convolution']
    
    # Interactive plotting
    attri_type = widgets.Dropdown(description='Attribute', options=attributes)

    noise = widgets.Dropdown(description='Noise Reduction', options=noise)
    
    kernel = widgets.Dropdown(description='Kernel', #kernel may change the shape of the resulting mask, best to leave at None :)
                              options=[None, (10, 9, 1),(1, 1, 3),
                                        (3, 3, 1), (1, 1, 1)])
    
    
    @interact   
    def f(attri_type=attri_type, kernel=kernel, noise=noise):  

        ori_image, noise_red, attr = attrComp(data=image, 
                                                attri_type=attri_type,
                                                kernel=kernel,
                                                noise=noise)
        
        cmap_button = widgets.Dropdown(description='Colormap', options=np.unique(cmap))

        vmin = widgets.FloatSlider(value=np.amin(attr), min=np.amin(attr), 
                             max=np.amax(attr))
    
        vmax = widgets.FloatSlider(value=np.amax(attr), min=np.amin(attr), 
                             max=np.amax(attr))
        
        fthreshold = widgets.FloatSlider(description='Threshold', value=np.amax(attr), 
                                  min=np.amin(attr), max=np.amax(attr))
        
        @interact
        def m(vmin=vmin, vmax=vmax, cmap=cmap_button, fthreshold=fthreshold):

            mask = extMask(cube=attr, threshold=fthreshold)

            print(f'Image shape = {ori_image.shape}')
            print(f'Mask shape = {mask.shape}')
            print(f'Attribute shape = {attr.shape}')  
            
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30, 10))        
            
            fig.suptitle(f'Horizon Tracking of 2D Seismic - {cube_name}', size=60)

            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="2.5%", pad=0.1)
            ax1.set_title('Original Seismic Image', size=20)
            im1=ax1.imshow(ori_image, cmap='RdBu')
            plt.colorbar(im1, cax=cax)
            
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="2.5%", pad=0.1)
            ax2.set_title(f'Denoised Seismic Image\n{noise.upper()}', size=20)
            im2=ax2.imshow(noise_red.squeeze(), cmap='gray')
            plt.colorbar(im2, cax=cax)
            
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="2.5%", pad=0.1)
            ax3.set_title(f'Seismic Attribute\n{attri_type.upper()}', size=20)
            im3 = ax3.imshow(attr.squeeze(), cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bicubic')
            plt.colorbar(im3, cax=cax)
            
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes("right", size="2.5%", pad=0.1)
            ax4.set_title(f'Mask\nThresh Value={fthreshold:.2f}', size=20)
            im4 = ax4.imshow(mask.squeeze(), cmap='gray')
            plt.colorbar(im4, cax=cax)

            plt.show()
            print('*'*240)