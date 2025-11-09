import sys
from warnings import filterwarnings
# from bruges.attribute import similarity
from plotly import express as px
import numpy as np
import segyio
import dash_vtk
import pyvista as pv
from warnings import filterwarnings
filterwarnings('ignore')

def extMask(cube, threshold):

    '''
    Extract salt/horizon mask from an attribute cube
    '''
    # #apply PCA to reduce dimensionality
    # from sklearn.decomposition import KernelPCA
    # cube = KernelPCA(n_components=2, kernel='rbf').fit_transform(cube.squeeze())
    geobody = np.where(cube < threshold, 255, 0).astype('int32') #depends on slice type

    return geobody

def attributes(data, attri_type:str, kernel:tuple, noise:str):
    
    '''
    
    This module helps to apply noise reduction algorithm on 2D seismic before computing seismic attributes
    for creating seismic masks
    '''
    sys.path.append('./attributes')
    
    from attributes.CompleTrace import ComplexAttributes
    from attributes.SignalProcess import SignalProcess
    from attributes.NoiseReduction import NoiseReduction

    def noise_reduction(darray, noise):

        #apply noise reduction algo
        n = NoiseReduction()
        narray, _ = NoiseReduction.create_array(n, darray, kernel=None, preview=None)
        # narray = narray.rechunk('auto')

        if noise == 'gaussian':
            nresult = NoiseReduction.gaussian(n, narray, preview=None)

        if noise == 'median':
            nresult = NoiseReduction.median(n, narray, preview=None)

        if noise == 'convolution':
            nresult = NoiseReduction.convolution(n, narray, preview=None)
        
        nresult = nresult

        return nresult
    
    def makeDask(darray, kernel, attri_type, noise):
        
        #denoised seismic array
        narray = noise_reduction(darray, noise)
        
        #make dask array for attribute computation
        if attri_type == 'rms' or attri_type == 'reflin' or attri_type == 'timegain' or attri_type == 'fder'\
            or attri_type == 'sder' or attri_type == 'gradmag':
            
            x = SignalProcess()
            darray, chunks_init = SignalProcess.create_array(x, narray, kernel, preview=None)
                        
        if attri_type == 'sweetness' or attri_type == 'infreq' or attri_type == 'enve' or attri_type == 'inphase'\
            or attri_type == 'cosphase' or attri_type == 'ampcontrast' or attri_type == 'ampacc' or \
            attri_type == 'inband' or attri_type == 'domfreq' or attri_type == 'apolar' or attri_type == 'resamp'\
            or attri_type == 'resfreq' or attri_type == 'resphase':

            x = ComplexAttributes()
            darray, chunks_init = ComplexAttributes.create_array(x, narray, kernel, preview=None)
        
        darray = darray
            
        return (x, darray, narray)
    
    def compute(x, darray, attri_type):
        
        '''
        Computes the seismic attribute
        '''

        if attri_type == 'reflin':
            result = SignalProcess.reflection_intensity(x, darray, preview=None)

        if attri_type == 'enve':
            result = ComplexAttributes.envelope(x, darray, preview=None)   

        if attri_type == 'sweetness':
            result = ComplexAttributes.sweetness(x, darray, preview=None)
        
        if attri_type == 'infreq':
            result = ComplexAttributes.instantaneous_frequency(x, darray, preview=None)
        
        if attri_type == 'fder':
            result = SignalProcess.first_derivative(x, darray, axis=-1, preview=None)
      
        if attri_type == 'sder':
            result = SignalProcess.second_derivative(x, darray, axis=-1, preview=None)
        
        if attri_type == 'rms':
            result = SignalProcess.rms(x, darray, kernel=(1, 1, 9), preview=None)

        if attri_type == 'timegain':
            result = SignalProcess.time_gain(x, darray, preview=None)
        
        if attri_type == 'gradmag':
            result = SignalProcess.gradient_magnitude(x, darray, sigmas=(1,1,1), preview=None)
        
        if attri_type == 'hist':
            result = SignalProcess.histogram_equalization(x, darray)

        if attri_type == 'tracegain':
            result = SignalProcess.trace_agc(x, darray)

        if attri_type == 'inphase':
            result = ComplexAttributes.instantaneous_phase(x, darray, preview=None)   

        if attri_type == 'cosphase':
            result = ComplexAttributes.cosine_instantaneous_phase(x, darray, preview=None)   

        if attri_type == 'ampcontrast':
            result = ComplexAttributes.relative_amplitude_change(x, darray, preview=None)

        if attri_type == 'ampacc':
            result = ComplexAttributes.amplitude_acceleration(x, darray, preview=None)
        
        if attri_type == 'inband':
            result = ComplexAttributes.instantaneous_bandwidth(x, darray, preview=None)

        if attri_type == 'domfreq':
            result = ComplexAttributes.dominant_frequency(x, darray, sample_rate=4, preview=None)

        if attri_type == 'apolar':
            result = ComplexAttributes.apparent_polarity(x, darray, preview=None)

        if attri_type == 'resamp':
            result = ComplexAttributes.response_amplitude(x, darray, preview=None)

        if attri_type == 'resfreq':
            result = ComplexAttributes.response_frequency(x, darray, sample_rate=4, preview=None)

        if attri_type == 'resphase':
            result = ComplexAttributes.response_phase(x, darray, preview=None)
        
        return result

        
    '''
    Main Program
    
    '''
    ori_image = data.copy()
    darray = data

    if attri_type != 'coherence':
        #apply attribute
        x, darray, noise_red = makeDask(darray, kernel=kernel, 
                                        attri_type=attri_type, noise=noise)
        # darray = darray.rechunk('auto')
        result = compute(x, darray, attri_type=attri_type)

        #extract mask
        attr = result #convert dask array attribute to numpy array
    
        #return result
        return ori_image, noise_red, attr
    
    # elif attri_type == 'coherence':
    #     narray = noise_reduction(darray, noise)
    #     semblance = similarity(narray, duration=0.036, dt=0.004)

    #     return ori_image, narray, semblance

def parse_seismic(segyfile):

    try:
        #read file
        if segyfile.endswith('.npy'):
            volume = np.load(segyfile) 

        elif segyfile.endswith('.segy'):
            volume = segy2numpy(segyfile)

        #create grid points
        num_points_z, num_points_y, num_points_x = volume.shape

        grid_points = []

        for k in range(num_points_z):
            for j in range(num_points_y):
                for i in range(num_points_x):
                    x = i#*10000  
                    y = j#*10000  
                    z = k#*10000  
                    grid_points.extend([x, y, z])

        # convert volume to vtiSeismicImageData  
        seis_vti = pv.wrap(volume)
        seis_vti_vol =  dash_vtk.ImageData(
          dimensions=volume.shape,
          origin=[0, 0, 0],
          spacing=[1, 1, 1],
            children=[
                dash_vtk.PointData([
                    dash_vtk.DataArray(
                        registration="setScalars",
                        values=seis_vti['values'],
                    )
                ])
            ],
        ),

        return seis_vti_vol, grid_points, volume
    
    except Exception:

        raise FileError('Error reading seismic volume!')

def segy2numpy(filename: str) -> np.array:
    with segyio.open(filename) as segyfile:
        return segyio.tools.cube(segyfile)


def numpy2segy(array, filename='maskvolume.segy'):
    return segyio.tools.from_array(filename, array)

def plot(image, cmap, title):
    fig = px.imshow(image.squeeze(), color_continuous_scale=cmap,
                    title=title)
    fig.update_layout(coloraxis_showscale=True)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    # fig.update_traces(interpolation='bilinear')
    
    return fig
class FileError(Exception):
    def __init__(self, msg):
        self.msg = msg
        super().__init__(self.msg)