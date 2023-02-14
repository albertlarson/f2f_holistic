import glob
import re
import numpy as np
import xarray as xr
from shapely.geometry import mapping
import geopandas
import rioxarray
import sys
import os, sys

logfile = '../data/GLDAS/clipped/yukon/_log.txt'
globfile = '../data/GLDAS/raw/*.nc4'
shapefile = '/work/albertl_uri_edu/f2f_holistic/data/shapes/yukon/3331.shp'

def clip2(raster,basin):
        rast = xr.open_dataset(raster,decode_coords="all")
        rast.rio.write_crs(4326,inplace=True)
        rast.rio.set_spatial_dims(x_dim="lon",y_dim="lat")
        r_clip = rast.rio.clip(basin.geometry.apply(mapping),basin.crs)
        y = re.findall('\d{6}',raster)
    #     plt.imshow(np.where(r_clip[0]<0,np.nan,r_clip[0]))
        r_clip.to_netcdf(f"../data/GLDAS/clipped/yukon/{y[0]}_clipped.nc")
        return r_clip
        

x = glob.glob(globfile)
print(len(x))
basin = geopandas.read_file(shapefile)

for idx,i in enumerate(x):
    clip2(i,basin)
    # if idx == 0:
    #     break