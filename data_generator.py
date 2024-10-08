import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Global constants for the pendulum system
g = 9.81  # Gravity (m/s^2)
l = 1.0   # Length of the pendulum (m)
b = 0.1   # Damping coefficient
m = 1.0   # Mass of the pendulum (kg)

# ODE Function Implementation: The function returns F(y)
def pendulum_ode(y):
    """
    Returns F(y) where y = [theta, omega] and F(y) = [omega, -b/ml*omega - g/l*sin(theta)]
    Uses global variables for b, m, l, g.
    """
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(b / (m * l)) * omega - (g / l) * jnp.sin(theta)
    return jnp.array([dtheta_dt, domega_dt])


# Euler method to solve the ODE
def euler_method(F, y0, t, dt):
    """
    Solve the pendulum ODE using the Euler method.
    Uses global variables for b, m, l, g.
    """
    y = jnp.zeros((t.size, len(y0)))  # Initialize the array to store solutions
    y = y.at[0].set(y0)  # Set initial condition

    for i in range(1, t.size):
        y = y.at[i].set(y[i - 1] + dt * F(y[i - 1]))

    return y


# Runge-Kutta method (4th order) to solve the ODE
def runge_kutta_method(F, y0, t, dt):
    """
    Solve the pendulum ODE using the 4th order Runge-Kutta method.
    Uses global variables for b, m, l, g.
    """
    y = jnp.zeros((t.size, len(y0)))  # Initialize the array to store solutions
    y = y.at[0].set(y0)  # Set initial condition

    for i in range(1, t.size):
        k1 = F(y[i - 1])
        k2 = F(y[i - 1] + 0.5 * dt * k1)
        k3 = F(y[i - 1] + 0.5 * dt * k2)
        k4 = F(y[i - 1] + dt * k3)
        y = y.at[i].set(y[i - 1] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    return y


def plot_comparison(t_euler, theta_euler, t_rk, theta_rk, title="Pendulum Motion: Euler vs Runge-Kutta"):
    plt.figure(figsize=(10, 6))
    
    # Plot the Euler method solution
    plt.plot(t_euler, theta_euler, label="Euler Method", linestyle='--', marker='o', color='r')
    
    # Plot the Runge-Kutta method solution
    plt.plot(t_rk, theta_rk, label="Runge-Kutta Method", linestyle='-', marker='x', color='b')
    
    # Labels and title
    plt.xlabel("Time")
    plt.ylabel("Theta (Angle)")
    plt.title(title)
    
    # Show legend and grid
    plt.legend()
    plt.grid(True)
    
    # Display the plot
    plt.show()

# Generate and Split Data
def gen_data(t, y, step=None):
    """Generate test and train data from the solution of the numerical method."""
    indices = jnp.arange(t.size, step=step)
    # Shuffle the indices before train-test split using JAX
    key = jax.random.PRNGKey(0)

    shuffled_indices = jax.random.permutation(key, indices)
    t_sliced, y_sliced = (
        t[shuffled_indices],
        y[shuffled_indices],
    )

    split_index = int(0.8 * len(t_sliced))
    t_train, y_train = t_sliced[:split_index], y_sliced[:split_index, 0]
    t_test, y_test = t_sliced[split_index:], y_sliced[split_index:, 0]

    return t_train, y_train, t_test, y_test

# Main execution
if __name__ == "__main__":
    # Global variables (b, m, l, g) are already defined above
    
    dt = 0.01  # Time step
    t_span = (0.0, 20.0)  # Time range

    # Initial conditions: theta = pi/4 (45 degrees), omega = 0
    y0 = jnp.array([jnp.pi / 4, 0.0])

    # Create time array
    t = jnp.arange(t_span[0], t_span[1], dt)

    # Solve using Euler method
    y_euler = euler_method(pendulum_ode, y0, t, dt)
    
    # Solve using Runge-Kutta method
    y_rk = runge_kutta_method(pendulum_ode, y0, t, dt)

    # Extract the theta (angle) values from the solutions
    theta_euler = y_euler[:, 0]  # Euler method theta values
    theta_rk = y_rk[:, 0]        # Runge-Kutta method theta values

    # Plot both solutions for comparison
    plot_comparison(t, theta_euler, t, theta_rk, title="Pendulum Motion: Euler vs Runge-Kutta")
