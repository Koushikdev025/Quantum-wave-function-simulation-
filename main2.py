import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyvista as pv

# --- Simulation Parameters ---
N = 64  # grid size
L = 2.0  # physical size of the box
x = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

# --- Momentum space grids ---
kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2

# --- Initial Wave Packet ---
x0, y0, z0 = -0.5, 0.0, 0.0
kx0, ky0, kz0 = 10.0, 0.0, 0.0
sigma = 0.2
envelope = np.exp(-((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * sigma**2))
phase = np.exp(1j * (kx0 * X + ky0 * Y + kz0 * Z))
Psi = envelope * phase
Psi /= np.sqrt(np.sum(np.abs(Psi)**2) * dx**3)  # normalize

# --- Potential (Choose One) ---
# Gaussian well:
# V0 = -50
# well_width = 0.2
# V = V0 * np.exp(-((X)**2 + (Y)**2 + (Z)**2) / (2 * well_width**2))

# Option A: 3D Barrier
# V = np.zeros_like(X)
# V[np.abs(X) < 0.1] = 100

# Option B: Double Slit
# barrier_width = 0.05
# gap = 0.2
# V = np.zeros_like(X)
# V[(np.abs(X) < barrier_width) & (np.abs(Y) > gap)] = 100

# Option C: Harmonic Potential
V = 50 * (X**2 + Y**2 + Z**2)

# --- Time Evolution Setup ---
dt = 0.001
steps = 300
V_half = np.exp(-1j * V * dt / 2)
T_full = np.exp(-1j * K2 * dt / 2)

# --- Evolution and Observables ---


def evolve(Psi):
    Psi = V_half * Psi
    Psi_k = np.fft.fftn(Psi)
    Psi_k *= T_full
    Psi = np.fft.ifftn(Psi_k)
    Psi = V_half * Psi
    Psi /= np.sqrt(np.sum(np.abs(Psi)**2) * dx**3)
    return Psi


def get_observables(Psi):
    prob = np.abs(Psi)**2
    norm = np.sum(prob) * dx**3
    x_avg = np.sum(X * prob) * dx**3
    y_avg = np.sum(Y * prob) * dx**3
    z_avg = np.sum(Z * prob) * dx**3
    x2_avg = np.sum(X**2 * prob) * dx**3
    y2_avg = np.sum(Y**2 * prob) * dx**3
    z2_avg = np.sum(Z**2 * prob) * dx**3
    dx_val = np.sqrt(x2_avg - x_avg**2)
    dy_val = np.sqrt(y2_avg - y_avg**2)
    dz_val = np.sqrt(z2_avg - z_avg**2)

    Psi_k = np.fft.fftshift(np.fft.fftn(Psi))
    prob_k = np.abs(Psi_k)**2
    prob_k /= np.sum(prob_k) * dx**3
    px_avg = np.sum(KX * prob_k) * dx**3
    py_avg = np.sum(KY * prob_k) * dx**3
    pz_avg = np.sum(KZ * prob_k) * dx**3
    px2_avg = np.sum(KX**2 * prob_k) * dx**3
    py2_avg = np.sum(KY**2 * prob_k) * dx**3
    pz2_avg = np.sum(KZ**2 * prob_k) * dx**3
    dpx = np.sqrt(px2_avg - px_avg**2)
    dpy = np.sqrt(py2_avg - py_avg**2)
    dpz = np.sqrt(pz2_avg - pz_avg**2)

    return {
        "P": norm,
        "x_avg": x_avg, "y_avg": y_avg, "z_avg": z_avg,
        "dx": dx_val, "dy": dy_val, "dz": dz_val,
        "px_avg": px_avg, "py_avg": py_avg, "pz_avg": pz_avg,
        "dpx": dpx, "dpy": dpy, "dpz": dpz
    }


# --- 2D Slice Plotting ---
fig, ax = plt.subplots(figsize=(6, 5))
z_slice = N // 2


def update(frame):
    global Psi
    Psi = evolve(Psi)
    obs = get_observables(Psi)
    prob = np.abs(Psi[:, :, z_slice])**2
    ax.clear()
    img = ax.imshow(prob.T, origin='lower', extent=(-L/2, L/2, -L/2, L/2),
                    cmap='plasma', vmin=0, vmax=np.max(prob)*1.2)
    ax.set_title(f"t = {frame*dt:.3f} s (z=0)")
    print(
        f"t = {frame*dt:.3f} | ⟨x⟩={obs['x_avg']:.3f}, Δx={obs['dx']:.3f}, ⟨px⟩={obs['px_avg']:.3f}, Δpx={obs['dpx']:.3f}, P={obs['P']:.5f}")
    return img,


ani = FuncAnimation(fig, update, frames=steps, interval=30, blit=False)
plt.tight_layout()
plt.show()

# --- Save Animation ---
ani.save("wave_packet_3d.mp4", writer="ffmpeg", fps=30)
# ani.save("wave_packet_3d.gif", writer="pillow", fps=30)

# --- Optional 3D PyVista View ---


def show_3d_volume(Psi):
    prob = np.abs(Psi)**2
    grid = pv.UniformGrid()
    grid.dimensions = np.array(prob.shape) + 1
    grid.origin = (-L/2, -L/2, -L/2)
    grid.spacing = (dx, dx, dx)
    grid.cell_data["prob_density"] = prob.flatten(order="F")
    plotter = pv.Plotter()
    opacity = [0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0]
    plotter.add_volume(grid, scalars="prob_density",
                       opacity=opacity, cmap="viridis")
    plotter.show(title="3D Probability Density")

# Uncomment to launch 3D plot after sim:
# show_3d_volume(Psi)
