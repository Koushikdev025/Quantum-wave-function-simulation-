import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from IPython.display import clear_output

# Set up the grid with higher resolution for smoother animations
x = np.linspace(-1.0, 1.0, 300)  # Increased resolution
y = np.linspace(-1.0, 1.0, 300)
X, Y = np.meshgrid(x, y)

# Crystal lattice parameters
lattice_start = 0.0  # x-coordinate where lattice begins
lattice_spacing = 0.2
lattice_points = []

print("Initializing quantum simulation...")
print(f"Grid resolution: {len(x)}x{len(y)} points")

# Generate lattice points
for lx in np.arange(lattice_start, 1.0, lattice_spacing):
    for ly in np.arange(-1.0, 1.0, lattice_spacing):
        lattice_points.append((lx, ly))

print(f"Created {len(lattice_points)} lattice points")

# Parameters for wave packet
k0 = 15.0      # Wave number
sigma = 0.15   # Width of the wave packet
v = 2.0        # Velocity of propagation
hbar = 1.0     # Reduced Planck constant (set to 1 for simplification)
mass = 0.5     # Particle mass
scattering_strength = 0.4  # Strength of scattering

print("\nSimulation parameters:")
print(f"Wave number (k0): {k0}")
print(f"Wave packet width (sigma): {sigma}")
print(f"Propagation velocity: {v}")
print(f"Particle mass: {mass}")
print(f"Scattering strength: {scattering_strength}")

# Terminal-based progress tracking


class TerminalOutput:
    def __init__(self):
        self.last_update = time.time()
        self.update_interval = 0.5  # seconds

    def update(self, message, force=False):
        current_time = time.time()
        if force or (current_time - self.last_update) > self.update_interval:
            clear_output(wait=True)
            print(message)
            self.last_update = current_time
            return True
        return False


terminal = TerminalOutput()


def calculate_momentum_wavefunction(psi_x, dx):
    """
    Calculate momentum space wavefunction using Fourier transform
    """
    psi_p = np.fft.fftshift(np.fft.fft2(psi_x)) * dx * dx
    return psi_p


def calculate_energy(psi_x, dx):
    """
    Calculate the energy of the wavefunction
    E = -hbar^2/2m * ∇²ψ + V*ψ
    """
    # Calculate Laplacian using finite difference method
    laplacian = np.zeros_like(psi_x)
    for i in range(1, psi_x.shape[0]-1):
        for j in range(1, psi_x.shape[1]-1):
            laplacian[i, j] = (psi_x[i+1, j] + psi_x[i-1, j] +
                               psi_x[i, j+1] + psi_x[i, j-1] - 4*psi_x[i, j]) / (dx*dx)

    # Calculate kinetic energy term: -hbar^2/2m * ∇²ψ
    kinetic = -hbar*hbar/(2*mass) * laplacian

    # For this example, assume zero potential energy
    energy = kinetic
    return energy


def gaussian_wave_packet(x0, y0, kx0, ky0):
    """
    Create a Gaussian wave packet centered at (x0, y0) with momentum (kx0, ky0)
    ψ(x,y) = A * exp(-((x-x0)² + (y-y0)²)/(4*σ²)) * exp(i*(kx0*x + ky0*y))
    """
    terminal.update(
        f"Calculating wave packet at position ({x0:.3f}, {y0:.3f}) with momentum ({kx0:.3f}, {ky0:.3f})")

    # Spatial Gaussian envelope
    r2 = (X - x0)**2 + (Y - y0)**2
    spatial_part = np.exp(-r2 / (4 * sigma**2))

    # Phase factor (plane wave)
    phase = kx0 * X + ky0 * Y
    phase_factor = np.exp(1j * phase)

    # Combine to form wave packet
    psi = spatial_part * phase_factor

    # Normalize
    norm = np.sqrt(np.sum(np.abs(psi)**2) * (x[1]-x[0])**2)
    psi = psi / norm

    # Calculate and display wave packet properties
    position_expectation = np.sum(np.abs(psi)**2 * X) * (x[1]-x[0])**2
    terminal.update(
        f"Wave packet position expectation value: {position_expectation:.5f}")

    return psi


def potential_function(scattering_centers):
    """
    Calculate the potential energy function from the scattering centers
    V(x,y) = sum_i V0 * exp(-((x-xi)² + (y-yi)²)/a²)
    """
    terminal.update("Calculating potential energy landscape...")

    potential = np.zeros_like(X)
    potential_strength = 5.0  # Strength of potential
    potential_width = 0.05   # Width of potential wells

    # Add gaussian potentials for each lattice point
    for i, (lx, ly) in enumerate(scattering_centers):
        if i % 5 == 0:  # Update progress every 5 points
            terminal.update(
                f"Adding potential at point {i+1}/{len(scattering_centers)}: ({lx:.2f}, {ly:.2f})")

        r2 = (X - lx)**2 + (Y - ly)**2
        potential += potential_strength * np.exp(-r2 / (potential_width**2))

    max_potential = np.max(potential)
    terminal.update(
        f"Potential calculation complete. Maximum potential value: {max_potential:.3f}")

    return potential


def evolve_wavefunction(psi, dt, potential):
    """
    Evolve the wavefunction using the split-operator method
    """
    dx = x[1] - x[0]

    # Half step in position space (apply potential)
    psi = psi * np.exp(-1j * potential * dt / (2 * hbar))

    # Step in momentum space (apply kinetic energy)
    psi_k = np.fft.fft2(psi)
    kx = 2 * np.pi * np.fft.fftfreq(len(x), d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(len(y), d=dx)
    KX, KY = np.meshgrid(kx, ky)
    K_squared = KX**2 + KY**2
    psi_k = psi_k * np.exp(-1j * hbar * K_squared * dt / (2 * mass))
    psi = np.fft.ifft2(psi_k)

    # Half step in position space (apply potential)
    psi = psi * np.exp(-1j * potential * dt / (2 * hbar))

    # Calculate some observables
    probability = np.abs(psi)**2
    norm = np.sum(probability) * dx * dx
    max_prob = np.max(probability)

    return psi, norm, max_prob


def scattered_wave(x_src, y_src, t):
    """
    Create circular scattered waves from a point source
    """
    r = np.sqrt((X - x_src)**2 + (Y - y_src)**2)
    # Circular wave with phase based on distance and time
    phase = k0 * (r - v * t)
    # Amplitude decreases with distance and has a Gaussian envelope in time
    amplitude = np.exp(-r / 0.5) * np.sin(phase) * np.exp(-(t - r/v)**2 / 0.01)
    return amplitude


def wave_function(t):
    """
    Calculate the wave function at time t using a simplified model
    """
    # Initial position of the wave packet
    x0 = -0.7 + v * t
    y0 = 0.0

    terminal.update(
        f"Calculating wave function at t = {t:.5f} s\nWave packet center: ({x0:.3f}, {y0:.3f})")

    # Initial Gaussian wave packet
    psi = gaussian_wave_packet(x0, y0, k0, 0)

    # Track scattering interactions
    interaction_count = 0

    # Add scattering waves from lattice points when wave reaches lattice
    if x0 >= lattice_start - 4*sigma:
        for lx, ly in lattice_points:
            # Calculate distance from wave packet center to lattice point
            dist = np.sqrt((x0 - lx)**2 + (y0 - ly)**2)
            # Only scatter if wave has reached the lattice point
            if dist < 4*sigma:
                interaction_time = max(0, t - dist/v)
                psi += scattering_strength * \
                    scattered_wave(lx, ly, interaction_time)
                interaction_count += 1

    # Calculate probability density
    probability = np.abs(psi)**2

    # Calculate observables
    total_probability = np.sum(probability) * (x[1]-x[0])**2
    max_probability = np.max(probability)

    terminal.update(f"Wave function calculation complete for t = {t:.5f} s\n" +
                    f"Interacting with {interaction_count} lattice points\n" +
                    f"Total probability: {total_probability:.5f}\n" +
                    f"Max probability density: {max_probability:.5f}")

    return probability

# Alternative method: Time-dependent Schrödinger equation simulation


def simulate_schrodinger(total_time, dt, save_interval=10):
    """
    Simulate the full time-dependent Schrödinger equation
    """
    terminal.update(
        "Starting full Schrödinger equation simulation...", force=True)

    # Initialize wave packet
    psi_init = gaussian_wave_packet(-0.7, 0.0, k0, 0)

    # Calculate potential
    potential = potential_function(lattice_points)

    # Time evolution
    psi = psi_init
    n_steps = int(total_time / dt)
    save_steps = max(1, int(n_steps / save_interval))

    terminal.update(f"Running {n_steps} time steps with dt = {dt:.6f}\n" +
                    f"Saving {save_interval} frames", force=True)

    # Storage for results
    results = []
    times = []
    norms = []
    max_probs = []

    start_time = time.time()

    for step in range(n_steps):
        # Evolve the wavefunction
        psi, norm, max_prob = evolve_wavefunction(psi, dt, potential)

        # Save results at certain intervals
        if step % save_steps == 0:
            current_time = step * dt
            probability = np.abs(psi)**2
            results.append(probability)
            times.append(current_time)
            norms.append(norm)
            max_probs.append(max_prob)

            # Calculate elapsed and estimated time
            elapsed = time.time() - start_time
            estimated_total = elapsed * n_steps / (step+1)
            remaining = estimated_total - elapsed

            # Print progress
            terminal.update(f"Step {step+1}/{n_steps} ({(step+1)/n_steps*100:.1f}%)\n" +
                            f"Simulation time: {current_time:.5f} s\n" +
                            f"Norm of wavefunction: {norm:.6f}\n" +
                            f"Max probability: {max_prob:.6f}\n" +
                            f"Elapsed time: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")

    terminal.update(f"Simulation complete! Total runtime: {time.time() - start_time:.1f}s\n" +
                    f"Processed {n_steps} time steps and saved {len(results)} frames", force=True)

    return results, times, potential, norms, max_probs


# Create the plot with higher DPI for smoother appearance
plt.rcParams['figure.dpi'] = 120  # Higher DPI for smoother appearance
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95)

# Function to update the plot for each frame


def update(frame):
    ax.clear()

    # Calculate time based on frame with smaller increments for smoother animation
    t = frame * 0.003  # Smaller time step for smoother animation

    # Calculate wave function
    probability = wave_function(t)

    # Plot the probability density with higher interpolation
    im = ax.imshow(probability,
                   extent=[-1, 1, -1, 1],
                   origin='lower',
                   cmap='plasma',
                   vmin=0,
                   vmax=2,
                   interpolation='gaussian')  # Smoother interpolation

    # Plot the lattice points
    lattice_x = [lx for lx, ly in lattice_points]
    lattice_y = [ly for lx, ly in lattice_points]
    ax.scatter(lattice_x, lattice_y, color='white', s=5, alpha=0.7)

    # Add time label
    ax.set_title(f"t = {t:.5f} s", fontsize=12)

    # Add axis labels
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)

    # Add colorbar for probability density
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Probability Density', fontsize=10)

    # Print frame information to terminal
    terminal.update(f"Rendering frame {frame+1}: t = {t:.5f} s")

    return [im]


print("\nPreparing animation...")

# Create animation with more frames and faster FPS for smoother appearance
frames = 150  # More frames
interval = 50  # Faster frame rate (milliseconds)

ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

# Function to save high-quality animation


def save_high_quality_animation(filename='quantum_scattering.mp4', dpi=200):
    print(f"Saving high-quality animation to {filename}...")
    ani.save(filename, writer='ffmpeg', fps=30, dpi=dpi, bitrate=5000)
    print(f"Animation saved to {filename}")

# Function to display specific frames with detailed analysis


def show_specific_frame(t):
    print(f"\nAnalyzing time t = {t:.5f} s...")

    plt.figure(figsize=(12, 9))

    # Main plot - probability density
    plt.subplot(2, 2, 1)
    probability = wave_function(t)
    im = plt.imshow(probability, extent=[-1, 1, -1, 1], origin='lower',
                    cmap='plasma', vmin=0, vmax=2, interpolation='gaussian')

    # Plot the lattice points
    lattice_x = [lx for lx, ly in lattice_points]
    lattice_y = [ly for lx, ly in lattice_points]
    plt.scatter(lattice_x, lattice_y, color='white', s=5, alpha=0.7)

    plt.colorbar(im, label='Probability Density')
    plt.title(f"Probability Density at t = {t:.5f} s")
    plt.xlabel('x')
    plt.ylabel('y')

    # Potential energy plot
    plt.subplot(2, 2, 2)
    potential = potential_function(lattice_points)
    plt.imshow(potential, extent=[-1, 1, -1, 1], origin='lower',
               cmap='viridis', interpolation='gaussian')
    plt.colorbar(label='Potential Energy')
    plt.title("Potential Energy Landscape")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(lattice_x, lattice_y, color='white', s=5, alpha=0.7)

    # Cross-section of probability along y=0
    plt.subplot(2, 2, 3)
    mid_y = len(y) // 2
    plt.plot(x, probability[mid_y, :], 'r-', linewidth=2)
    plt.title("Probability Density Cross-section (y=0)")
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.grid(True)

    # Calculate total probability in different regions
    left_region = np.sum(probability[:, :len(x)//3]) * (x[1]-x[0])**2
    middle_region = np.sum(
        probability[:, len(x)//3:2*len(x)//3]) * (x[1]-x[0])**2
    right_region = np.sum(probability[:, 2*len(x)//3:]) * (x[1]-x[0])**2

    # Cross-section of potential along y=0 with region analysis
    plt.subplot(2, 2, 4)
    plt.plot(x, potential[mid_y, :], 'b-', linewidth=2)
    plt.title("Analysis of Probability Distribution")
    plt.xlabel('x')
    plt.ylabel('Potential')
    plt.grid(True)

    # Add region analysis as text
    plt.figtext(0.15, 0.01, f"Left region: {left_region:.3f}", fontsize=10)
    plt.figtext(0.4, 0.01, f"Middle region: {middle_region:.3f}", fontsize=10)
    plt.figtext(0.7, 0.01, f"Right region: {right_region:.3f}", fontsize=10)
    plt.figtext(0.5, 0.04, f"Total probability: {left_region + middle_region + right_region:.3f}",
                ha='center', fontsize=10, bbox=dict(facecolor='yellow', alpha=0.2))

    plt.tight_layout()
    print("Analysis complete - displaying results")
    plt.show()

    # Print detailed analysis to terminal
    print(f"\nDetailed Analysis at t = {t:.5f} s:")
    print(
        f"Total probability: {left_region + middle_region + right_region:.6f}")
    print(
        f"Left region probability (-1 to -0.33): {left_region:.6f} ({left_region*100:.2f}%)")
    print(
        f"Middle region probability (-0.33 to 0.33): {middle_region:.6f} ({middle_region*100:.2f}%)")
    print(
        f"Right region probability (0.33 to 1): {right_region:.6f} ({right_region*100:.2f}%)")
    print(f"Maximum probability density: {np.max(probability):.6f}")
    print(
        f"Position of maximum density: ({x[np.unravel_index(np.argmax(probability), probability.shape)[1]]:.3f}, {y[np.unravel_index(np.argmax(probability), probability.shape)[0]]:.3f})")

# Run the full Schrödinger simulation with detailed terminal output


def run_full_simulation():
    dt = 0.0001  # Time step
    total_time = 0.05  # Total simulation time

    print("\nRunning full Schrödinger equation simulation...")
    results, times, potential, norms, max_probs = simulate_schrodinger(
        total_time, dt, save_interval=10)

    # Plot results at different time steps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    print("Rendering final visualization...")

    for i, (probability, t) in enumerate(zip(results[:6], times[:6])):
        ax = axes[i]
        im = ax.imshow(probability, extent=[-1, 1, -1, 1], origin='lower',
                       cmap='plasma', vmin=0, vmax=np.max(results[0])*1.2,
                       interpolation='gaussian')

        # Plot the lattice points
        lattice_x = [lx for lx, ly in lattice_points]
        lattice_y = [ly for lx, ly in lattice_points]
        ax.scatter(lattice_x, lattice_y, color='white', s=3, alpha=0.7)

        ax.set_title(f"t = {t:.5f} s")
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()

    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Probability Density')

    # Plot norm conservation and max probability
    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 8))

    ax2[0].plot(times, norms, 'b-o')
    ax2[0].set_title('Norm of Wavefunction (should be conserved)')
    ax2[0].set_xlabel('Time (s)')
    ax2[0].set_ylabel('Norm')
    ax2[0].grid(True)

    ax2[1].plot(times, max_probs, 'r-o')
    ax2[1].set_title('Maximum Probability Density')
    ax2[1].set_xlabel('Time (s)')
    ax2[1].set_ylabel('Max Probability')
    ax2[1].grid(True)

    plt.tight_layout()

    print("Simulation visualization complete")
    plt.show()

    return results, times


print("\nReady to run simulation and visualization!")
print("Available commands:")
print("- plt.show() - Display the animation")
print("- show_specific_frame(0.01) - Analyze a specific time")
print("- run_full_simulation() - Run detailed simulation")
print("- save_high_quality_animation() - Save animation to file")

# To display the animation
plt.show()

# Uncomment these to see specific analysis:
# show_specific_frame(0.0)    # Initial wave packet
# show_specific_frame(0.01)   # Wave approaching lattice
# show_specific_frame(0.02)   # Interaction with lattice
# run_full_simulation()       # Run and visualize full simulation
# save_high_quality_animation()  # Save high-quality animation
