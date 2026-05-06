python3 -c 
import tifffile
with tifffile.TiffFile('/mnt/d/photoelastimetry/cp/31_14g.tiff') as tif:
    print('axes:', tif.series[0].axes if tif.series else None)
    print('shape:', tif.asarray().shape)

