from ipywidgets import widgets, HBox, VBox
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .mask import extMask, auto_threshold
from .attri import attrComp

from warnings import filterwarnings
filterwarnings('ignore')


def seismicViewer(cube, cube_name):
    '''
    Interactive seismic attribute viewer.

    Row 1 (compute controls): Attribute | Kernel | Noise Reduction
    Row 2 (display controls): vmin | vmax | Colormap | Threshold
    '''

    image = cube

    attributes = ['sweetness', 'infreq', 'reflin', 'rms', 'timegain', 'enve', 'fder', 'sder',
                  'gradmag', 'inphase', 'cosphase', 'ampcontrast', 'ampacc', 'inband', 'domfreq',
                  'resamp', 'apolar', 'resfreq', 'resphase']

    cmaps     = ['PuOr_r', 'gray', 'cubehelix', 'jet', 'plasma', 'inferno',
                 'seismic_r', 'gist_rainbow', 'Accent']
    noise_opts = ['gaussian', 'median', 'convolution']

    _w = {'description_width': 'initial'}
    _sl = widgets.Layout(width='260px')
    _dd = widgets.Layout(width='200px')

    # ── Row 1: compute controls ───────────────────────────────────────────────
    attri_type = widgets.Dropdown(description='Attribute',       options=attributes,   style=_w, layout=_dd)
    kernel     = widgets.Dropdown(description='Kernel',          style=_w, layout=_dd,
                                  options=[None, (1,1,1), (1,1,3), (3,3,1), (10,9,1)])
    noise      = widgets.Dropdown(description='Noise Reduction', options=noise_opts,   style=_w, layout=_dd)

    row1 = HBox([attri_type, kernel, noise])

    # ── Row 2: display controls (built after first compute) ───────────────────
    cmap_btn   = widgets.Dropdown(description='Colormap',  options=np.unique(cmaps), style=_w, layout=_dd)
    vmin_sl    = widgets.FloatSlider(description='vmin',      step=0.01, style=_w, layout=_sl)
    vmax_sl    = widgets.FloatSlider(description='vmax',      step=0.01, style=_w, layout=_sl)
    thresh_sl  = widgets.FloatSlider(description='Threshold', step=0.01, style=_w, layout=_sl)

    row2 = HBox([vmin_sl, vmax_sl, cmap_btn, thresh_sl])

    plot_out = widgets.Output()

    # ── plotting function ─────────────────────────────────────────────────────
    def draw(ori_image, noise_red, attr):
        mask = extMask(cube=attr, threshold=thresh_sl.value)

        print(f'Image shape     = {ori_image.shape}')
        print(f'Attribute shape = {attr.shape}')
        print(f'Mask shape      = {mask.shape}')

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30, 10))
        fig.suptitle(f'Horizon Tracking of 2D Seismic - {cube_name}', size=40)

        for ax, data, cm, title, interp in [
            (ax1, ori_image.squeeze(),   'RdBu',        'Original Seismic Image',          'bicubic'),
            (ax2, np.squeeze(noise_red), 'gray',        f'Denoised\n{noise.value.upper()}', 'bicubic'),
            (ax3, attr.squeeze(),        cmap_btn.value, f'Attribute\n{attri_type.value.upper()}', 'bicubic'),
            (ax4, mask.squeeze(),        'gray',         f'Mask  (t={thresh_sl.value:.3f})', 'nearest'),
        ]:
            div = make_axes_locatable(ax)
            cax = div.append_axes('right', size='2.5%', pad=0.1)
            kw  = dict(vmin=vmin_sl.value, vmax=vmax_sl.value) if ax is ax3 else {}
            im  = ax.imshow(data, cmap=cm, interpolation=interp, **kw)
            ax.set_title(title, size=20)
            plt.colorbar(im, cax=cax)

        plt.tight_layout()
        plt.show()
        print('─' * 120)

    # cache computed results so display controls don't re-run the pipeline
    _cache = {}

    def replot(change=None):
        with plot_out:
            plot_out.clear_output(wait=True)
            draw(_cache['ori'], _cache['noise_red'], _cache['attr'])

    def recompute(change=None):
        with plot_out:
            plot_out.clear_output(wait=True)
            print('Computing...')

        ori_image, noise_red, attr = attrComp(
            data=image,
            attri_type=attri_type.value,
            kernel=kernel.value,
            noise=noise.value,
        )
        _cache.update({'ori': ori_image, 'noise_red': noise_red, 'attr': attr})

        # auto-select optimal threshold, then set all slider bounds safely.
        # Must set min/max in the right order to avoid TraitError (min > max).
        amin, amax  = float(np.amin(attr)), float(np.amax(attr))
        auto_result = auto_threshold(attr, method='all')
        best_thresh = float(auto_result['best'])

        for sl, val in [(vmin_sl, amin), (vmax_sl, amax), (thresh_sl, best_thresh)]:
            sl.unobserve_all()
            if amin < sl.max:
                sl.min, sl.max = amin, amax   # expand: set min first (safe)
            else:
                sl.max, sl.min = amax, amin   # shrink: set max first (safe)
            sl.value = val
            sl.observe(replot, names='value')

        # show all methods so the user knows the starting point
        rec = auto_result['recommended'].upper()
        print(f"Auto threshold → {rec} = {best_thresh:.3f}")
        print(#f"
              # otsu={auto_result['otsu']:.3f}  #yen={auto_result['yen']:.3f}  "
        #       f"li={auto_result['li']:.3f}  isodata={auto_result['isodata']:.3f}  "
              f"mec={auto_result['mec']:.3f}"  #gmm={auto_result['gmm']:.3f}  "
            #   f"triangle={auto_result['triangle']:.3f}"
        )

        replot()

    # ── wire observers ────────────────────────────────────────────────────────
    for w in [attri_type, kernel, noise]:
        w.observe(recompute, names='value')

    cmap_btn.observe(replot, names='value')

    # ── display layout ────────────────────────────────────────────────────────
    display(VBox([row1, row2, plot_out]))

    # initial compute
    recompute()