from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')


def extMask(cube, threshold):
    '''
    Extract salt/horizon mask from a normalised attribute cube.
    Values below threshold → salt (255), above → sediment (0).
    '''
    geobody = np.where(cube < threshold, 255, 0).astype('int32')
    return geobody


def kMeans(attri, nclusters=2):
    '''
    Cluster attribute array into salt / non-salt via Mini-Batch K-Means.
    '''
    kmeans = MiniBatchKMeans(n_clusters=nclusters, random_state=0).fit_predict(
        attri.reshape((-1, 1))
    )
    return kmeans.reshape(*attri.shape)


def auto_threshold(attr, method='all'):
    '''
    Automatically estimate the best threshold to separate salt from sediments
    in a normalised [0, 1] seismic attribute array.

    Parameters
    ----------
    attr   : np.ndarray, normalised attribute (any shape, values in [0, 1])
    method : str, one of:
                'otsu'      - maximises inter-class variance
                'yen'       - maximises entropic correlation (good for textured salt)
                'li'        - minimises cross-entropy (robust when histogram is unimodal)
                'isodata'   - iterative midpoint of class means (Ridler-Calvard)
                'mec'       - Minimum Error (Kittler-Illingworth): Bayes-optimal
                              assuming Gaussian classes; analytical, no fitting needed
                'gmm'       - Gaussian Mixture Model intersection
                'triangle'  - geometric method, best when salt is a small fraction
                'all'       - compute all seven, return dict with recommendation

    Returns
    -------
    threshold : float  (or dict when method='all')

    Usage
    -----
    # Quick single method
    t = auto_threshold(attr, method='li')
    mask = extMask(attr, threshold=t)

    # Let it decide
    results = auto_threshold(attr, method='all')
    print(results)
    mask = extMask(attr, threshold=results['best'])
    '''

    from skimage.filters import (threshold_otsu, threshold_yen,
                                 threshold_li, threshold_isodata)
    from scipy.stats import norm

    flat = attr.flatten().astype(np.float64)

    # ── individual methods ────────────────────────────────────────────────────

    def _otsu(flat):
        '''Maximises between-class variance. Best for clearly bimodal histograms.'''
        return float(threshold_otsu(flat))

    def _yen(flat):
        '''
        Yen's maximum correlation criterion.
        Tends to select a higher threshold than Otsu — useful when the salt
        body has high-amplitude, texturally complex reflections.
        '''
        return float(threshold_yen(flat))

    def _li(flat):
        '''
        Li's minimum cross-entropy.
        Robust when the histogram is not clearly bimodal; iteratively
        minimises the cross-entropy between original and binarised image.
        '''
        return float(threshold_li(flat))

    def _isodata(flat):
        '''
        Isodata / Ridler-Calvard iterative threshold.
        Starts at the midpoint and updates to the mean of the two class means
        until convergence. Fast and parameter-free.
        '''
        return float(threshold_isodata(flat))

    def _mec(flat):
        '''
        Minimum Error Criterion (Kittler & Illingworth, 1986).
        Analytically minimises the Bayes classification error assuming each
        class follows a Gaussian distribution. Closed-form — no EM fitting.
        Sweeps candidate thresholds and picks the one that minimises:
            J(t) = 1 + 2*(w0*log(s0) + w1*log(s1))
                     - 2*(w0*log(w0) + w1*log(w1))
        where w, mu, s are the weight, mean, std of each class at threshold t.
        '''
        counts, edges = np.histogram(flat, bins=256, range=(0.0, 1.0))
        centers = (edges[:-1] + edges[1:]) / 2
        total   = counts.sum()

        best_j = np.inf
        best_t = float(centers[len(centers)//2])

        cumsum  = np.cumsum(counts)
        cumval  = np.cumsum(counts * centers)
        cumval2 = np.cumsum(counts * centers**2)

        for i in range(1, len(counts) - 1):
            w0 = cumsum[i] / total
            w1 = 1.0 - w0
            if w0 <= 0 or w1 <= 0:
                continue

            mu0 = cumval[i] / cumsum[i]
            mu1 = (cumval[-1] - cumval[i]) / (total - cumsum[i])

            var0 = cumval2[i] / cumsum[i] - mu0**2
            var1 = ((cumval2[-1] - cumval2[i]) / (total - cumsum[i])) - mu1**2

            if var0 <= 0 or var1 <= 0:
                continue

            s0, s1 = np.sqrt(var0), np.sqrt(var1)
            J = (1 + 2*(w0*np.log(s0) + w1*np.log(s1))
                   - 2*(w0*np.log(w0)  + w1*np.log(w1)))

            if J < best_j:
                best_j = J
                best_t = float(centers[i])

        return best_t

    def _gmm(flat):
        '''
        2-component GMM: finds the intersection of the two fitted Gaussians.
        More flexible than MEC when classes are clearly non-Gaussian.
        '''
        gm    = GaussianMixture(n_components=2, random_state=0).fit(flat.reshape(-1, 1))
        order = np.argsort(gm.means_.ravel())
        means   = gm.means_.ravel()[order]
        stds    = np.sqrt(gm.covariances_.ravel())[order]
        weights = gm.weights_[order]

        xs = np.linspace(0, 1, 10000)
        p0 = weights[0] * norm.pdf(xs, means[0], stds[0])
        p1 = weights[1] * norm.pdf(xs, means[1], stds[1])
        crossings = np.where(np.diff(np.sign(p0 - p1)))[0]
        if len(crossings) == 0:
            return float(np.mean(means))
        mid  = (means[0] + means[1]) / 2
        best = crossings[np.argmin(np.abs(xs[crossings] - mid))]
        return float(xs[best])

    def _triangle(flat):
        '''
        Triangle (Zack) method: max perpendicular distance from histogram peak
        to the far tail. Best when salt occupies a small fraction of the image.
        '''
        counts, edges = np.histogram(flat, bins=256)
        centers  = (edges[:-1] + edges[1:]) / 2
        peak_idx = np.argmax(counts)

        if (len(counts) - peak_idx - 1) >= peak_idx:
            xs, ys = centers[peak_idx:], counts[peak_idx:]
            x1, y1 = centers[peak_idx], counts[peak_idx]
            x2, y2 = centers[-1],       counts[-1]
        else:
            xs, ys = centers[:peak_idx+1], counts[:peak_idx+1]
            x1, y1 = centers[0],        counts[0]
            x2, y2 = centers[peak_idx], counts[peak_idx]

        dx, dy   = x2 - x1, y2 - y1
        line_len = np.sqrt(dx**2 + dy**2)
        if line_len == 0:
            return float(x1)
        distances = np.abs(dy*xs - dx*ys + x2*y1 - y2*x1) / line_len
        return float(xs[np.argmax(distances)])

    # ── recommendation logic ──────────────────────────────────────────────────

    def _recommend(results, flat):
        '''
        Pick the threshold closest to the histogram valley between the two modes.
        If no clear valley exists (unimodal), prefer Li as the most robust
        single-class method.
        '''
        from scipy.signal import argrelmin
        counts, edges = np.histogram(flat, bins=256)
        centers = (edges[:-1] + edges[1:]) / 2
        valleys = argrelmin(counts, order=5)[0]

        if len(valleys) > 0:
            valley_x = centers[valleys[np.argmin(counts[valleys])]]
            return min(results, key=lambda m: abs(results[m] - valley_x))
        return 'li'   # unimodal → Li is most robust

    # ── dispatch ──────────────────────────────────────────────────────────────

    methods = {
        # 'otsu':    _otsu,
        # 'yen':     _yen,
        # 'li':      _li,
        # 'isodata': _isodata,
        'mec':     _mec,
        # 'gmm':     _gmm,
        # 'triangle':_triangle,
    }

    if method == 'all':
        results = {name: fn(flat) for name, fn in methods.items()}
        best_method        = _recommend(results, flat)
        results['recommended'] = best_method
        results['best']        = results[best_method]
        return results
    elif method in methods:
        return methods[method](flat)
    else:
        raise ValueError(
            f"method must be one of {list(methods.keys()) + ['all']}. Got: {method!r}"
        )