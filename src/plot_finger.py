from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.

# X1 = np.linspace(-R, R)
# Y1 = np.linspace(-R, R)
# X1, Y1 = np.meshgrid(X1, Y1)
# Z1 = np.sqrt(R**2 - X1**2 - Y1 ** 2) + D # np.sqrt(X ** 2 + Y ** 2)
#
#
# Z2 = np.linspace(0, D)
# X2 = np.linspace(-R, R)
# X2, Z2 = np.meshgrid(X2, Z2)
#
# Y2 = np.sqrt(R ** 2 - X2 ** 2)
#

# R = 18
distance = 40
radius = 10
pts = 30

# Semi-sphere
r = np.linspace(0, radius, pts)
p = np.linspace(0, 2 * np.pi, pts)
R1, P1 = np.meshgrid(r, p)
Z1 = distance + radius * np.sin(np.arccos(R1 / radius))
# * np.sin(P) # ((R**2 - 1)**2)
# Express the mesh in the cartesian system.
X1, Y1 = R1 * np.cos(P1), R1 * np.sin(P1)

surf = ax.plot_surface(X1, Y1, Z1, linewidth=0, antialiased=True, color=(0.6, 0.9, 1, 0.5))

r = radius * np.ones(pts).astype(np.float32)
p = np.linspace(0, 2 * np.pi, pts)
R2, P2 = np.meshgrid(r, p)
Z2 = R2 * 0 + np.linspace(0, distance, pts)
# Express the mesh in the cartesian system.
X2, Y2 = R2 * np.cos(P2), R2 * np.sin(P2)

pts2 = 10
width = pts2*2
height = pts2*2
dop = 10

X_ = np.linspace(-width / 2, width / 2, pts2).astype(np.float32)
print(X_)
Y_ = np.linspace(-height / 2, height / 2, pts2).astype(np.float32)
Z_ = np.ones(width).astype(np.float32)
X_, Y_, Z_ = np.meshgrid(X_, Y_, Z_)
print(X_.dtype, Y_.dtype, Z_.dtype)

cx, cy, alpha = 0.00001, 0.00001, dop

print(radius * alpha / distance)
center_reg = np.zeros(np.shape(X_))
center_reg[np.square(X_ - cx) + np.square(Y_ - cy) < radius * alpha / distance] = 1.0

print(np.max(center_reg), np.min(center_reg))
# disc /= disc
# disc /= 1 - disc

ZZ1 = (1 - center_reg) * np.sqrt(np.square(radius * alpha) / np.square((X_ - cx) + np.square(Y_ - cy)))
ZZ2 = 1.0 # (1 - disc) * np.sqrt(np.square(radius * alpha) / np.square((X_ - cx) + np.square(Y_ - cy)) + np.square(Z - d))


Z = ZZ1 + ZZ2
X = ((X_ - cx) / alpha) * Z
Y = ((Y_ - cy) / alpha) * Z

U = X - X_
V = Y - Y_
W = Z - Z_

ax.quiver(X_, Y_, Z_, U, V, W, color=(1, 0.5, 0, 0.25), arrow_length_ratio=0.01, linewidths=0.5)
ax.scatter(X, Y, Z)
ax.scatter(X_, Y_, Z_, color=(0, 1.0, 0))
# X = np.concatenate([X2, X1], axis=0)
# Y = np.concatenate([Y2, Y1], axis=0)
# Z = np.concatenate([Z2, Z1], axis=0)

surf = ax.plot_surface(X2, Y2, Z2, linewidth=0, antialiased=True, color=(0.6, 0.9, 1, 0.5))
set_axes_equal(ax)
# cmap=cm.coolwarm,

# Customize the z axis.
# ax.set_zlim(-5, 5)
# ax.set_ylim(-5, 5)
# ax.set_xlim(-5, 5)
# l = 50
# ax.auto_scale_xyz([-l / 2, l / 2], [-l / 2, l / 2], [0, l], [0, 0.015])
# ax.set_aspect(1)

# max_range = distance + radius
# mid_x = 0  # (X.max()+X.min()) * 0.5
# mid_y = 0  # (Y.max()+Y.min()) * 0.5
# mid_z = (distance + radius) / 2 # (Z.max() + Z.min()) * 0.5
# ax.set_xlim(mid_x - max_range, mid_x + max_range)
# ax.set_ylim(mid_y - max_range, mid_y + max_range)
# ax.set_zlim(mid_z - max_range, mid_z + max_range)

# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('figure.jpg', dpi=150)
plt.show()
