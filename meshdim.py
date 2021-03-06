import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.optimize as opt

traj_name = "conv.npy"
X = np.load('../trajs/' + traj_name)
X = X[:,0].reshape(-1,1)
#X = X[-1000:]


def create_mesh(data, d, initial_mesh=None):
    """ Creates a mesh from the given data using balls of size d
    Args:
        data: np.array, the data you want to create a mesh for
        d: float, the radius for the ball used to determine membership in the mesh
    Returns:
        mesh: list, all the points from data that made it into the mesh
        weights: how many points from the original set are represented by the corresponding point in the mesh
    """
    if initial_mesh is None:
        initial_mesh = []

    mesh = initial_mesh
    weights = []
    in_mesh = np.zeros(data.shape[0], dtype=np.bool)

    for i, x in enumerate(data):
        if in_mesh[i]:
            continue
        else:
            in_criteria = np.linalg.norm(x - data, axis=1, ord=1) < d
            in_mesh = np.logical_or(in_mesh, in_criteria)
            mesh.append(x)
            weights.append(np.sum(in_criteria))

    return mesh, weights


start = time.time()

d = 1e-3 # initial d, requires hand tuning for now
mesh_sizes = []
d_vals = []
while True:
    mesh = create_mesh(X,d)
    mesh_sizes.append(len(mesh))
    d_vals.append(d)

    if mesh_sizes[-1] == 1:
        break

    d = d*2

print(time.time() - start)

# %%
# Now lets plot the results to do a smell test
for i, m in enumerate(mesh_sizes):
    if m < X.shape[0]:
        lin_begin = i
        break

plt.plot(X)
plt.title("Trajectory " + traj_name)
plt.legend(['x', 'y', 'z', 'r'])
plt.show()

xdata = np.array(d_vals[lin_begin:])
ydata = np.array(mesh_sizes[lin_begin:])

plt.plot(xdata, ydata, 'kx--')
#plt.plot(d_vals, mesh_sizes, 'kx--')
plt.title("Mesh sizes")
plt.xlabel('log(d)')
plt.ylabel('log(Points in mesh)')
plt.yscale('log')
plt.xscale('log')
plt.gca().xaxis.grid(True, which='both')  # minor grid on too
plt.gca().yaxis.grid(True, which='both')  # minor grid on too

plt.show()


# %%
def f(x, m, b):
    return m * x + b


popt, pcov = opt.curve_fit(f, np.log10(xdata), np.log10(ydata))

plt.plot(np.log10(xdata), np.log10(ydata), 'kx')
plt.plot(np.log10(xdata), f(np.log10(xdata), *popt), 'r--')
plt.title("Log Log fit " + traj_name)
plt.legend(['Data', 'fit: m*x + b,  m=%5.3f, b=%5.3f' % tuple(popt)])
plt.xlabel('log(d)')
plt.ylabel('log(Points in mesh)')
plt.gca().xaxis.grid(True)  # minor grid on too
plt.gca().yaxis.grid(True)  # minor grid on too
plt.show()
