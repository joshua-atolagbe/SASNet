# Create weak labels

from scripts.extract import ExtractPatches
from pathlib import Path

if __name__ == "__main__":
    path = Path(r"data")

    # Create an ExtractPatches object and parse the data dir.
    # Set preview to True to determine the optimal parameters
    # for creating seismic masks before extraction.

    extract = ExtractPatches(path)

    extract(preview=True, idx=1)

    #=-===================================================================================
    # After previewing different parameters, choose the optimal ones
    # for extracting seismic mask patches.
    #=-===================================================================================
    # Next, create an ExtractPatches object with the optimal parameters
    # from the preview option above.
    # For salt mask, the optimal parameters are {'attri_type':'resfreq', 'threshold':0.8}.
    # Other parameters are left constant.
    # Then call the ExtractPatches object and set preview to False to extract seismic mask patches 

    extract = ExtractPatches(path, attribute_type='resfreq', threshold=0.8) 

    extract(preview=False)