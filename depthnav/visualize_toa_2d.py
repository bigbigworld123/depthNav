import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse

def visualize_toa_slice(npz_file_path, slice_z=None):
    """
    Generates a 2D visualization of a Time-of-Arrival (ToA) map
    from a specific Z-axis slice of the geodesic data.

    Args:
        npz_file_path (str): The path to the .npz file containing the geodesic data.
        slice_z (int, optional): The index of the Z-axis slice to visualize.
                                 If None, the middle slice is chosen automatically.
    """
    try:
        data = np.load(npz_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{npz_file_path}' was not found.")
        print("Please ensure you have run 'generate_training_envs.py' first.")
        return

    # Extract data arrays
    costs = data['costs']
    gradients = data['gradients']
    occupancy = data['occupancy']
    bb_std = data['bb_std']
    target_ijk = data['target']
    grid_resolution = data['grid_resolution']

    # --- Select the 2D slice ---
    if slice_z is None:
        # Default to the slice where the target is located
        slice_z = target_ijk[2]
        print(f"No Z-slice specified. Visualizing slice at Z-index = {slice_z} (target height).")

    if not (0 <= slice_z < costs.shape[2]):
        print(f"Error: slice_z index {slice_z} is out of bounds for Z-axis of size {costs.shape[2]}.")
        return

    costs_slice = costs[:, :, slice_z]
    gradients_slice = gradients[:, :, slice_z, 0:2] # Only need X and Y components for 2D plot
    occupancy_slice = occupancy[:, :, slice_z]

    # Mask out the costs and gradients inside obstacles for better visualization
    costs_slice[occupancy_slice == 1] = np.nan

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate the extent for the plot using the bounding box and resolution
    # Note: imshow extent is (left, right, bottom, top)
    x_coords = np.arange(bb_std[0, 0], bb_std[1, 0] + grid_resolution, grid_resolution)
    y_coords = np.arange(bb_std[0, 1], bb_std[1, 1] + grid_resolution, grid_resolution)
    extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

    # 1. Plot the ToA heatmap
    im = ax.imshow(costs_slice.T, cmap='viridis_r', origin='lower', extent=extent, interpolation='bilinear')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Time of Arrival (s)')

    # 2. Overlay the obstacle mask
    # We create a custom colormap for the occupancy: 0 is transparent, 1 is white
    cmap_occ = colors.ListedColormap(['none', 'white'])
    ax.imshow(occupancy_slice.T, cmap=cmap_occ, origin='lower', extent=extent, interpolation='none', alpha=0.8)

    # 3. Overlay the gradient field (quiver plot)
    # We need to downsample the arrows to avoid a cluttered plot
    skip = 4
    x, y = np.meshgrid(x_coords, y_coords, indexing='ij')
    ax.quiver(x[::skip, ::skip], y[::skip, ::skip],
              gradients_slice[::skip, ::skip, 0],
              gradients_slice[::skip, ::skip, 1],
              color='white',
              scale=30) # Adjust scale for arrow size

    # 4. Plot the target point
    target_xy_coords = bb_std[0, 0:2] + target_ijk[0:2] * grid_resolution
    ax.plot(target_xy_coords[0], target_xy_coords[1], 'r*', markersize=15, label='Target')

    ax.set_title(f'ToA Heatmap and Gradient Field (Z-slice: {slice_z})')
    ax.set_xlabel('X coordinate (m)')
    ax.set_ylabel('Y coordinate (m)')
    ax.legend()
    ax.axis('equal')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize a 2D slice of the ToA map from geodesic data.")
    parser.add_argument(
        '--file',
        type=str,
        default='datasets/depthnav_dataset/geodesics/level_1/ring_walls_small/ring_walls_small_0.npz',
        help='Path to the .npz file containing the geodesic data.'
    )
    parser.add_argument(
        '--slice_z',
        type=int,
        default=None,
        help='The index of the Z-axis slice to visualize. Defaults to the target height.'
    )
    args = parser.parse_args()

    visualize_toa_slice(args.file, args.slice_z)