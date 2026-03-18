import sys
from warnings import filterwarnings
filterwarnings('ignore')


def attrComp(data, attri_type: str, kernel: tuple, noise: str):
    '''
    Apply noise reduction on a 2D seismic image then compute a seismic
    attribute for use in salt/horizon mask generation.

    Parameters
    ----------
    data       : np.ndarray, shape (H, W, 1)
    attri_type : str, attribute key (e.g. 'enve', 'domfreq', 'rms')
    kernel     : tuple or None, operator kernel size
    noise      : str, one of 'gaussian' | 'median' | 'convolution'

    Returns
    -------
    ori_image : np.ndarray (H, W, 1)  — original input
    noise_red : np.ndarray (H, W, 1)  — denoised image
    attr      : np.ndarray (H, W, 1)  — computed attribute
    '''

    sys.path.append('./attributes')

    from attributes.CompleTrace import ComplexAttributes
    from attributes.SignalProcess import SignalProcess
    from attributes.NoiseReduction import NoiseReduction

    # ── noise reduction ───────────────────────────────────────────────────────

    def noise_reduction(darray, noise):
        '''
        Apply denoising directly via scipy so shape is always preserved.
        NoiseReduction's dask ghost+trim path is a no-op on single-chunk
        arrays, which balloons (1,H,W) → (3,H+2,W+2).
        '''
        import numpy as np
        from scipy import ndimage as ndi
        import dask.array as da

        arr = darray.compute() if hasattr(darray, 'compute') else np.array(darray)

        filters = {
            'gaussian':   lambda a: ndi.gaussian_filter(a, sigma=(1, 1, 1)),
            'median':     lambda a: ndi.median_filter(a,   size=(3, 3, 3)),
            'convolution':lambda a: ndi.uniform_filter(a,  size=(3, 3, 3)),
        }
        result = filters.get(noise, lambda a: a.copy())(arr)
        return da.from_array(result, chunks=result.shape)

    # ── attribute dispatch ────────────────────────────────────────────────────

    # Maps attri_type → (processor_class, method_name, extra_kwargs)
    SIGNAL_ATTRS = {
        'rms':    (SignalProcess, 'rms',                  {'kernel': (1, 1, 9)}),
        'reflin': (SignalProcess, 'reflection_intensity', {}),
        'fder':   (SignalProcess, 'first_derivative',     {'axis': -1}),
        'sder':   (SignalProcess, 'second_derivative',    {'axis': -1}),
        'timegain':(SignalProcess,'time_gain',            {}),
        'gradmag':(SignalProcess, 'gradient_magnitude',   {'sigmas': (1, 1, 1)}),
    }

    COMPLEX_ATTRS = {
        'enve':        (ComplexAttributes, 'envelope',                  {}),
        'inphase':     (ComplexAttributes, 'instantaneous_phase',       {}),
        'cosphase':    (ComplexAttributes, 'cosine_instantaneous_phase',{}),
        'infreq':      (ComplexAttributes, 'instantaneous_frequency',   {}),
        'inband':      (ComplexAttributes, 'instantaneous_bandwidth',   {}),
        'domfreq':     (ComplexAttributes, 'dominant_frequency',        {'sample_rate': 4}),
        'sweetness':   (ComplexAttributes, 'sweetness',                 {}),
        'ampcontrast': (ComplexAttributes, 'relative_amplitude_change', {}),
        'ampacc':      (ComplexAttributes, 'amplitude_acceleration',    {}),
        'apolar':      (ComplexAttributes, 'apparent_polarity',         {}),
        'resamp':      (ComplexAttributes, 'response_amplitude',        {}),
        'resfreq':     (ComplexAttributes, 'response_frequency',        {'sample_rate': 4}),
        'resphase':    (ComplexAttributes, 'response_phase',            {}),
    }

    ALL_ATTRS = {**SIGNAL_ATTRS, **COMPLEX_ATTRS}

    if attri_type not in ALL_ATTRS:
        raise ValueError(f"Unknown attribute type '{attri_type}'. "
                         f"Valid options: {sorted(ALL_ATTRS)}")

    def makeDask(darray, kernel, attri_type, noise):
        narray = noise_reduction(darray, noise)
        cls, _, _ = ALL_ATTRS[attri_type]
        return cls(), narray, narray

    def compute(x, darray, attri_type):
        _, method_name, kwargs = ALL_ATTRS[attri_type]
        method = getattr(x, method_name)
        return method(darray, preview=None, **kwargs)

    # ── main pipeline ─────────────────────────────────────────────────────────

    import numpy as np

    ori_image = data.copy()

    # (H, W, 1) → (1, H, W) for the attribute pipeline
    darray = np.squeeze(data)
    darray = darray[np.newaxis, ...]

    x, darray, noise_red = makeDask(darray, kernel=kernel,
                                    attri_type=attri_type, noise=noise)
    darray = darray.rechunk('auto')
    result  = compute(x, darray, attri_type=attri_type)

    noise_red = noise_red.compute() if hasattr(noise_red, 'compute') else noise_red

    # (1, H, W) → (H, W, 1) to match input convention
    attr      = np.moveaxis(result.compute(), 0, -1)
    noise_red = np.moveaxis(noise_red,        0, -1)

    # # Normalise attr to [0, 1] so vmax is always 1 regardless of attribute type
    # attr_min, attr_max = attr.min(), attr.max()
    # if attr_max > attr_min:                      # avoid divide-by-zero on flat arrays
    #     attr = (attr - attr_min) / (attr_max - attr_min)

    return ori_image, noise_red, attr