import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- USER SPEED CONTROL ---
time_factor = 1.0  # 0.5 is half-speed (slower), 2.0 is double-speed (faster), 1.0 is normal

# --- Simulation Parameters ---
N_dm = 200
N_gas = 200
G = 1.0
dt = 0.01
steps = 300
softening = 0.1

def initialize_particles(case):
    np.random.seed(42)
    N = N_dm + N_gas
    mass_dm = 2.0
    mass_gas = 1.0
    masses = np.hstack([np.ones(N_dm) * mass_dm, np.ones(N_gas) * mass_gas])

    r = np.sqrt(np.random.rand(N)) * 10
    theta = 2 * np.pi * np.random.rand(N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    positions = np.vstack((x, y)).T

    velocities = np.zeros((N, 2))
    if case == "high":
        v_circ = np.sqrt(G * np.sum(masses) / (r + softening))
        velocities[:, 0] = -v_circ * np.sin(theta)
        velocities[:, 1] = v_circ * np.cos(theta)
    elif case == "low":
        velocities *= 0
    elif case == "asym":
        v_circ = np.sqrt(G * np.sum(masses) / (r + softening))
        velocities[:, 0] = -0.5 * v_circ * np.sin(theta) + np.random.randn(N) * 0.4
        velocities[:, 1] = 0.5 * v_circ * np.cos(theta) + np.random.randn(N) * 0.4

    indices_dm = np.arange(N_dm)
    indices_gas = np.arange(N_dm, N)
    return positions, velocities, masses, indices_dm, indices_gas

def compute_forces(positions, masses):
    N = len(masses)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = (diff ** 2).sum(axis=2) + softening ** 2
    inv_dist3 = np.where(dist_sq > 0, 1.0 / (dist_sq * np.sqrt(dist_sq)), 0.0)
    np.fill_diagonal(inv_dist3, 0.0)
    F = -G * (masses[:, np.newaxis] * masses[np.newaxis, :])[:, :, np.newaxis] * diff * inv_dist3[:, :, np.newaxis]
    accel = F.sum(axis=1) / masses[:, None]
    return accel

def compute_energies(positions, velocities, masses):
    KE = 0.5 * masses * (velocities ** 2).sum(axis=1)
    total_KE = KE.sum()
    N = len(masses)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = (diff ** 2).sum(axis=2) + softening ** 2
    # Fix: Use upper triangular mask to avoid double counting
    mask = np.triu(np.ones((N, N)), k=1)
    inv_dist = np.where(mask, 1.0 / np.sqrt(dist_sq), 0.0)
    PE = -G * (masses[:, None] * masses[None, :]) * inv_dist
    total_PE = PE.sum()
    total_E = total_KE + total_PE
    return total_KE, total_PE, total_E

def compute_total_angular_momentum(positions, velocities, masses):
    Lz = masses * (positions[:, 0] * velocities[:, 1] - positions[:, 1] * velocities[:, 0])
    return np.sum(Lz)

def evolve(positions, velocities, masses):
    accel = compute_forces(positions, masses)
    velocities += accel * dt
    positions += velocities * dt
    return positions, velocities

def run_simulation(case, steps=300):
    print(f"Running simulation with case: {case}")
    positions, velocities, masses, indices_dm, indices_gas = initialize_particles(case)
    pos_history = [positions.copy()]
    KE_list, PE_list, E_list, L_list = [], [], [], []
    
    # Store initial values
    KE, PE, E = compute_energies(positions, velocities, masses)
    L = compute_total_angular_momentum(positions, velocities, masses)
    KE_list.append(KE)
    PE_list.append(PE)
    E_list.append(E)
    L_list.append(L)
    
    for step in range(steps):
        if step % 50 == 0:
            print(f"Step {step}/{steps}")
        positions, velocities = evolve(positions, velocities, masses)
        pos_history.append(positions.copy())
        KE, PE, E = compute_energies(positions, velocities, masses)
        L = compute_total_angular_momentum(positions, velocities, masses)
        KE_list.append(KE)
        PE_list.append(PE)
        E_list.append(E)
        L_list.append(L)
    
    return np.array(pos_history), np.array(KE_list), np.array(PE_list), np.array(E_list), np.array(L_list), indices_dm, indices_gas

def animate_galaxy_evolution(pos_history, indices_dm, indices_gas, interval=30):
    fig, ax = plt.subplots(figsize=(8, 8))
    scat_dm = ax.scatter([], [], s=8, color='blue', alpha=0.7, label="Dark Matter")
    scat_gas = ax.scatter([], [], s=8, color='orange', alpha=0.7, label="Gas")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.legend()
    ax.set_title("Galaxy Formation Simulation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    def update(frame):
        ax.set_title(f"Galaxy Formation Simulation")
        scat_dm.set_offsets(pos_history[frame][indices_dm])
        scat_gas.set_offsets(pos_history[frame][indices_gas])
        return scat_dm, scat_gas

    anim = FuncAnimation(fig, update,
                        frames=len(pos_history),
                        interval=interval,
                        blit=True)
    plt.show()
    # Don't close the figure immediately - let matplotlib handle it
    return anim

def plot_energy_and_L(KE_list, PE_list, E_list, L_list):
    """Plot energy components and angular momentum evolution"""
    time_steps = np.arange(len(E_list)) * dt
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Energy plot
    axs[0, 0].plot(time_steps, E_list, label='Total Energy', color='magenta', linewidth=2)
    axs[0, 0].plot(time_steps, KE_list, label='Kinetic Energy', color='red', alpha=0.7)
    axs[0, 0].plot(time_steps, PE_list, label='Potential Energy', color='blue', alpha=0.7)
    axs[0, 0].set_ylabel("Energy")
    axs[0, 0].set_title("Energy Evolution")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # Angular momentum
    axs[0, 1].plot(time_steps, L_list, label='Total Angular Momentum', color='green', linewidth=2)
    axs[0, 1].set_ylabel("Angular Momentum")
    axs[0, 1].set_title("Angular Momentum Evolution")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # Energy conservation check
    E_initial = E_list[0]
    energy_drift = (E_list - E_initial) / abs(E_initial) * 100
    axs[1, 0].plot(time_steps, energy_drift, color='purple')
    axs[1, 0].set_ylabel("Energy Drift (%)")
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_title("Energy Conservation Check")
    axs[1, 0].grid(True, alpha=0.3)
    
    # Angular momentum conservation check
    L_initial = L_list[0]
    if abs(L_initial) > 1e-10:  # Avoid division by zero
        L_drift = (L_list - L_initial) / abs(L_initial) * 100
    else:
        L_drift = L_list - L_initial
    axs[1, 1].plot(time_steps, L_drift, color='brown')
    axs[1, 1].set_ylabel("L Drift (% or absolute)")
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_title("Angular Momentum Conservation Check")
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_structure(positions, masses, indices_dm, indices_gas):
    """Analyze the final structure of the galaxy"""
    dm_positions = positions[indices_dm]
    gas_positions = positions[indices_gas]
    
    # Calculate center of mass
    total_mass = masses.sum()
    com = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass
    
    # Calculate radial distances from center of mass
    dm_radii = np.sqrt(np.sum((dm_positions - com)**2, axis=1))
    gas_radii = np.sqrt(np.sum((gas_positions - com)**2, axis=1))
    
    print(f"\nFinal Structure Analysis:")
    print(f"Center of mass: ({com[0]:.3f}, {com[1]:.3f})")
    print(f"Dark matter - Mean radius: {dm_radii.mean():.3f}, Std: {dm_radii.std():.3f}")
    print(f"Gas - Mean radius: {gas_radii.mean():.3f}, Std: {gas_radii.std():.3f}")
    
    # Plot radial distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(dm_radii, bins=30, alpha=0.7, label='Dark Matter', color='blue', density=True)
    ax.hist(gas_radii, bins=30, alpha=0.7, label='Gas', color='orange', density=True)
    ax.set_xlabel('Radius from Center of Mass')
    ax.set_ylabel('Density')
    ax.set_title('Final Radial Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

# --- MAIN RUN SCRIPT ---
if __name__ == "__main__":
    case = "high"  # high, low, asym
    print(f"Starting galaxy simulation...")
    
    # Run simulation
    pos_history, KE, PE, E, L, indices_dm, indices_gas = run_simulation(case, steps=300)
    
    # Animation with adjusted speed
    frame_interval = max(1, int(30 / time_factor)) if time_factor > 0 else 30
    
    print("Starting animation...")
    anim = animate_galaxy_evolution(pos_history, indices_dm, indices_gas, interval=frame_interval)
    
    print("Plotting energy and angular momentum...")
    plot_energy_and_L(KE, PE, E, L)
    
    print("Analyzing final structure...")
    analyze_structure(pos_history[-1], 
                     np.hstack([np.ones(N_dm) * 2.0, np.ones(N_gas) * 1.0]), 
                     indices_dm, indices_gas)
    
    print("Simulation complete!")



