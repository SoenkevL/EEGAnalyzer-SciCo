import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_trajectory(data, elev=30, azim=45):
    """
    Plots a 3D trajectory with color-coded points based on their index,
    allowing interactive rotation of the view.


    Args:
    data (np.ndarray): A NumPy array of shape (N, 3) representing the 3D trajectory.
    elev (float): Elevation angle in degrees.
    azim (float): Azimuth angle in degrees.
    """
    # Create a figure and a 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    # Extract x, y, and z coordinates from the data
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]


    # Create a color map based on the index of each point
    num_points = data.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, num_points))


    # Plot the 3D trace with color-coded points
    scatter = ax.scatter(x, y, z, c=colors, marker='o', s=10)


    # Add a color bar to show the mapping between color and index
    cbar = fig.colorbar(scatter)
    cbar.set_label('Index')


    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    # Set a title for the plot
    ax.set_title('3D Trajectory with Color-Coded Points')


    # Set the view angle
    ax.view_init(elev=elev, azim=azim)


    # Enable interactive rotation (if needed, but can conflict with view_init)
    ax.mouse_init()


    # Show the plot
    plt.show()


if __name__ == '__main__':
    # Generate some sample 3D data (replace with your actual data)
    signal = np.sin(2*np.pi*np.linspace(0, 10, 200))
    embed = [signal[:-2], signal[1:-1], signal[2:]]
    embed = np.array(embed).transpose()

    # Plot the 3D trajectory
    plot_3d_trajectory(embed, elev=30, azim=45)
