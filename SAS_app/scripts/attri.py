import sys
from warnings import filterwarnings
filterwarnings('ignore')

def attrComp(data, attri_type:str, kernel:tuple, noise:str):
    
    '''
    
    This module helps to apply noise reduction algorithm on 2D seismic before computing seismic attributes
    for creating seismic masks
    '''
    sys.path.append('./attributes')
    
    from attributes.CompleTrace import ComplexAttributes
    from attributes.SignalProcess import SignalProcess
    from attributes.NoiseReduction import NoiseReduction
    
    def makeDask(darray, kernel, attri_type, noise):

        def noise_reduction(darray, noise):

            #apply noise reduction algo
            n = NoiseReduction()
            narray, _ = NoiseReduction.create_array(n, darray, kernel=None, preview=None)
            narray = narray.T.rechunk('auto')

            if noise == 'gaussian':
                nresult = NoiseReduction.gaussian(n, narray, preview=None)
                nresult = nresult.T

            if noise == 'median':
                nresult = NoiseReduction.median(n, narray, preview=None)
                nresult = nresult.T

            if noise == 'convolution':
                nresult = NoiseReduction.convolution(n, narray, preview=None)
                nresult = nresult.T

            return nresult
        
        #denoised seismic array
        narray = noise_reduction(darray, noise)
        
        #make dask array for attribute computation
        if attri_type == 'rms' or attri_type == 'reflin' or attri_type == 'timegain' or attri_type == 'fder'\
            or attri_type == 'sder' or attri_type == 'gradmag':
            
            x = SignalProcess()
            darray, chunks_init = SignalProcess.create_array(x, narray, kernel, preview=None)
            darray = darray.T
                
            return (x, darray, narray)
        
        if attri_type == 'sweetness' or attri_type == 'infreq' or attri_type == 'enve' or attri_type == 'inphase'\
            or attri_type == 'cosphase' or attri_type == 'ampcontrast' or attri_type == 'ampacc' or \
            attri_type == 'inband' or attri_type == 'domfreq' or attri_type == 'apolar' or attri_type == 'resamp'\
            or attri_type == 'resfreq' or attri_type == 'resphase':

            x = ComplexAttributes()
            darray, chunks_init = ComplexAttributes.create_array(x, narray, kernel, preview=None)
            darray = darray.T
            
            return (x, darray, narray)
    
    def compute(x, darray, attri_type):
        
        '''
        Computes the seismic attribute
        '''

        if attri_type == 'reflin':
            result = SignalProcess.reflection_intensity(x, darray, preview=None)
            return result

        if attri_type == 'enve':
            result = ComplexAttributes.envelope(x, darray, preview=None)
            return result
        
        if attri_type == 'sweetness':
            result = ComplexAttributes.sweetness(x, darray, preview=None)
            return result
        
        if attri_type == 'infreq':
            result = ComplexAttributes.instantaneous_frequency(x, darray, preview=None)
            return result
        
        if attri_type == 'fder':
            result = SignalProcess.first_derivative(x, darray, axis=-1, preview=None)
            return result
      
        if attri_type == 'sder':
            result = SignalProcess.second_derivative(x, darray, axis=-1, preview=None)
            return result
        
        if attri_type == 'rms':
            result = SignalProcess.rms(x, darray, kernel=(1, 1, 9), preview=None)
            return result

        if attri_type == 'timegain':
            result = SignalProcess.time_gain(x, darray, preview=None)
            return result
        
        if attri_type == 'gradmag':
            result = SignalProcess.gradient_magnitude(x, darray, sigmas=(1,1,1), preview=None)
            return result
        
        if attri_type == 'inphase':
            result = ComplexAttributes.instantaneous_phase(x, darray, preview=None)   
            return result

        if attri_type == 'cosphase':
            result = ComplexAttributes.cosine_instantaneous_phase(x, darray, preview=None)   
            return result

        if attri_type == 'ampcontrast':
            result = ComplexAttributes.relative_amplitude_change(x, darray, preview=None)
            return result

        if attri_type == 'ampacc':
            result = ComplexAttributes.amplitude_acceleration(x, darray, preview=None)
            return result
        
        if attri_type == 'inband':
            result = ComplexAttributes.instantaneous_bandwidth(x, darray, preview=None)
            return result

        if attri_type == 'domfreq':
            result = ComplexAttributes.dominant_frequency(x, darray, sample_rate=4, preview=None)
            return result

        if attri_type == 'apolar':
            result = ComplexAttributes.apparent_polarity(x, darray, preview=None)
            return result

        if attri_type == 'resamp':
            result = ComplexAttributes.response_amplitude(x, darray, preview=None)
            return result

        if attri_type == 'resfreq':
            result = ComplexAttributes.response_frequency(x, darray, sample_rate=4, preview=None)
            return result

        if attri_type == 'resphase':
            result = ComplexAttributes.response_phase(x, darray, preview=None)
            return result

        
    '''
    Main Program
    
    '''
    ori_image = data.copy()
    darray = data

    #apply attribute
    x, darray, noise_red = makeDask(darray, kernel=kernel, 
                                    attri_type=attri_type, noise=noise)
    darray = darray.rechunk('auto')
    result = compute(x, darray, attri_type=attri_type)

    #extract mask
    attr = result.T #convert dask array attribute to numpy array
    
    #return result
    return ori_image, noise_red, attr