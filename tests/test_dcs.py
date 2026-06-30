import h5py
import matplotlib.pyplot as plt
from matplotlib import animation

def save_trajectory_gif(file_path, trajectory_id='0', output_name='tests/humanoid_walk.gif', fps=30):
    print(f"Opening {file_path}...")
    
    with h5py.File(file_path, 'r') as f:
        # Access the observations for the specified trajectory
        # Shape: (1000, 64, 64, 3)
        images = f[f"{trajectory_id}/obs"][:]
        
    print(f"Processing {len(images)} frames...")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis('off')
    
    # Initialize the plot with the first frame
    im = ax.imshow(images[0])
    
    def update(frame_idx):
        im.set_data(images[frame_idx])
        return [im]

    # Create the animation object
    # interval is delay between frames in milliseconds
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(images), 
        interval=1000/fps, 
        blit=True
    )

    print(f"Saving GIF to {output_name}... this may take a moment.")
    # Use 'pillow' writer for GIF generation
    anim.save(output_name, writer='pillow', fps=fps)
    plt.close()
    print("Done!")

if __name__ == "__main__":
    # Path to your specific dataset
    HDF5_PATH = 'data/dcs/datasets/humanoid-walk-test.hdf5'
    
    try:
        save_trajectory_gif(HDF5_PATH)
    except Exception as e:
        print(f"An error occurred: {e}")