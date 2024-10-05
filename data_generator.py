import jax.numpy as jnp
import matplotlib.pyplot as plt


# Constants for the pendulum system
g = 9.81  # Gravity (m/s^2)
l = 1.0   # Length of the pendulum (m)
b = 0.1   # Damping coefficient
m = 1.0   # Mass of the pendulum (kg)

# ODE Function Implementation: The function returns F(y)
def pendulum_ode(y):
    """
    Returns F(y) where y = [theta, omega] and F(y) = [omega, -b/ml*omega - g/l*sin(theta)]
    """
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(b / (m * l)) * omega - (g / l) * jnp.sin(theta)
    return jnp.array([dtheta_dt, domega_dt])


# Euler method to solve the ODE
def solve_pendulum_euler(y0, t, dt, ode_func):
    """
    Solve the pendulum ODE using the Euler method.
    """
    y = jnp.zeros((len(t), len(y0)))  # Initialize the array to store solutions
    y = y.at[0].set(y0)  # Set initial condition

    for i in range(1, len(t)):
        y = y.at[i].set(y[i - 1] + dt * ode_func(y[i - 1]))

    return y


# Runge-Kutta method (4th order) to solve the ODE
def solve_pendulum_rk4(y0, t, dt, ode_func):
    """
    Solve the pendulum ODE using the 4th order Runge-Kutta method.
    """
    y = jnp.zeros((len(t), len(y0)))  # Initialize the array to store solutions
    y = y.at[0].set(y0)  # Set initial condition

    for i in range(1, len(t)):
        k1 = ode_func(y[i - 1])
        k2 = ode_func(y[i - 1] + 0.5 * dt * k1)
        k3 = ode_func(y[i - 1] + 0.5 * dt * k2)
        k4 = ode_func(y[i - 1] + dt * k3)
        y = y.at[i].set(y[i - 1] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4))

    return y


# Function to subsample the data to simulate limited number of samples
def subsample_data(t, y, num_samples):
    """
    Subsample the data to a specified number of samples.
    """
    indices = jnp.linspace(0, len(t) - 1, num=num_samples, dtype=int)
    return t[indices], y[indices]


# Function for Train-Test split (80% training, 20% testing)
def train_test_split(t, y):
    """
    Split the data into 80% training and 20% testing.
    """
    split_index = int(0.8 * len(t))
    t_train, y_train = t[:split_index], y[:split_index]
    t_test, y_test = t[split_index:], y[split_index:]
    return t_train, y_train, t_test, y_test


# Function to plot the solution
def plot_solution(t, y_euler, y_rk4, title="Pendulum ODE Solution"):
    """
    Plot the solutions obtained by Euler and Runge-Kutta methods.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot theta (angle) over time
    plt.plot(t, y_euler[:, 0], label="Euler Method", linestyle="--", color="blue")
    plt.plot(t, y_rk4[:, 0], label="Runge-Kutta Method (RK4)", linestyle="-", color="green")
    
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (theta) [radians]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Environment Setup: Parameters for solving the ODE
    dt = 0.01  # Time step (Δt)
    t = jnp.arange(0, 20, dt)  # Time array from 0 to 20 seconds
    y0 = jnp.array([jnp.pi / 4, 0.0])  # Initial condition [theta(0) = pi/4, omega(0) = 0]

    # Solve ODE using Euler and Runge-Kutta methods
    y_euler = solve_pendulum_euler(y0, t, dt, pendulum_ode)
    y_rk4 = solve_pendulum_rk4(y0, t, dt, pendulum_ode)

    # Subsample the data to 10 samples
    t_subsampled, y_euler_subsampled = subsample_data(t, y_euler, 10)
    _, y_rk4_subsampled = subsample_data(t, y_rk4, 10)

    # Perform Train-Test split
    t_train, y_train_euler, t_test, y_test_euler = train_test_split(t_subsampled, y_euler_subsampled)
    _, y_train_rk4, _, y_test_rk4 = train_test_split(t_subsampled, y_rk4_subsampled)

    # Plot the solutions
    plot_solution(t_subsampled, y_euler_subsampled, y_rk4_subsampled, title="Subsampled Pendulum ODE Solution")
    
    # Print Train-Test split for Euler method
    print("Training Data (Euler Method):", t_train, y_train_euler)
    print("Test Data (Euler Method):", t_test, y_test_euler)

    # Print Train-Test split for Runge-Kutta method
    print("Training Data (Runge-Kutta Method):", t_train, y_train_rk4)
    print("Test Data (Runge-Kutta Method):", t_test, y_test_rk4)
