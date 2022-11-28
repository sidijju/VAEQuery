import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils.helpers import makedir

def visualize_behavior(world, ax, dim, observations):

    ax.autoscale(False)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    rec_dim = dim/world.size / 2
    for i in range(world.size):
        pos_i = i * rec_dim
        for j in range(world.size):
            pos_j = j * rec_dim
            rec = Rectangle((pos_i, pos_j), rec_dim, rec_dim, facecolor='none', alpha=0.5,
                            edgecolor='k')
            ax.add_patch(rec)

    # visualise behaviour, current position, goal
    plot_obs = observations * rec_dim + 0.5 * rec_dim
    plot_task = world.task * rec_dim + 0.5 * rec_dim
    ax.plot(plot_obs[:, 0], plot_obs[:, 1], 'b-')
    ax.plot(plot_obs[-1, 0], plot_obs[-1, 1], 'b.')
    ax.plot(plot_task[0], plot_task[1], 'kx')