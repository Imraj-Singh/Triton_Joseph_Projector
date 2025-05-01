import array_api_compat.cupy as xp

import parallelproj
from array_api_compat import to_device
import array_api_compat.numpy as np
import matplotlib.pyplot as plt

if "numpy" in xp.__name__:
    dev = "cpu"
elif "cupy" in xp.__name__:
    dev = xp.cuda.Device(0)
elif "torch" in xp.__name__:
    dev = "cuda"

# things we keep constant, radius, num_sides, 

sl_phantom = np.swapaxes(np.load('3D_shepp_logan.npy'), 0, 1)
print(f" The size of the Shepp Logan phantom is {sl_phantom.shape}")

image_xy = sl_phantom.shape[0]

list_slices = [4, 8, 16, 32, 64]
list_lor_endpoints_per_side = [20, 25, 30, 35, 40]
voxels_sizes = (.75, 3.0, .75)
num_lors = []
num_voxels = []

for num_slices in list_slices:
    for num_lor_endpoints_per_side in list_lor_endpoints_per_side:
        
        num_rings = num_slices//2
        max_ring_position = (voxels_sizes[1] * num_rings) - (voxels_sizes[1] / 2)
        scanner = parallelproj.RegularPolygonPETScannerGeometry(
            xp,
            dev,
            radius=210.0, # NOT REALISTIC AT ALL BUT JUST FOR TESTING
            num_sides=12,
            num_lor_endpoints_per_side=num_lor_endpoints_per_side,
            lor_spacing=108/num_lor_endpoints_per_side,
            ring_positions=xp.linspace(-max_ring_position, max_ring_position, num_rings),
            symmetry_axis=1, 
        )

        lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
            scanner,
            radial_trim=1,
            max_ring_difference=list_slices[0],
            sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
        )

        proj = parallelproj.RegularPolygonPETProjector(
            lor_desc, img_shape=(image_xy, num_slices, image_xy), voxel_size=voxels_sizes,
        )
        
        xstart, xend = proj._lor_descriptor.get_lor_coordinates(views=proj._views)
        print(f" Number of LORs is {xstart.shape[0]*xstart.shape[1]*xstart.shape[2]}")
        print(f" Number of Voxels is {image_xy*image_xy*num_slices}")
        num_lors.append(xstart.shape[0]*xstart.shape[1]*xstart.shape[2])
        num_voxels.append(image_xy*image_xy*num_slices)

        if num_slices == sl_phantom.shape[1]:
            x = xp.array(sl_phantom)
        else:
            x = xp.array(sl_phantom[:, (sl_phantom.shape[1]-num_slices)//2:-(sl_phantom.shape[1]-num_slices)//2])

        x_fwd = parallelproj.joseph3d_fwd(
                        xstart, xend, x, proj._img_origin, proj._voxel_size
                    )
                    
        x_back = parallelproj.joseph3d_back(
                        xstart, xend, proj._img_shape, proj._img_origin, proj._voxel_size, x_fwd,
                    )

        dict_data = {}

        dict_data['xstart'] = xstart.get()
        dict_data['xend'] = xend.get()
        dict_data['img_origin'] = proj._img_origin.get()
        dict_data['voxel_size'] = proj._voxel_size.get()
        dict_data['img_shape'] = proj._img_shape
        dict_data['x'] = x.get()
        dict_data['meas'] = x_fwd.get()

        np.save(f"slices_{num_slices}_lors_{num_lor_endpoints_per_side}.npy", dict_data)