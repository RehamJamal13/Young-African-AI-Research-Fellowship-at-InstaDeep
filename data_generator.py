import jax.numpy as jnp
import matplotlib.pyplot as plt

# Step 1: ODE Function Implementation the mathmatical model of a damped pendulum
def pendulum_ode(y, b, m, l, g):
    theta, omega = y   # θ is the angular displacement omega is angular velocity
    dtheta_dt = omega
    domega_dt = -(b / (m * l)) * omega - (g / l) * jnp.sin(theta)
    return jnp.array([dtheta_dt, domega_dt])

# Step 2: Euler Method y(t+Δt)=y(t)+Δt⋅f(y(t),t)
"""Where:
y(t) is the current value of y at time t.
Δt is the time step (a small increment of time).
f(y(t),t) is the derivative of y at time t, which gives the slope of the function at that point.
y(t+Δt) is the value of y at the next time step."""
def euler_step(f, y, t, dt, *args): #*ars is the parameters for the ODE function(b,𝑚,l,g)
    return y + dt * f(y, *args)

# Step 2: Runge-Kutta Method (RK4)
def runge_kutta_step(f, y, t, dt, *args):
    k1 = f(y, *args)
    k2 = f(y + 0.5 * dt * k1, *args)
    k3 = f(y + 0.5 * dt * k2, *args)
    k4 = f(y + dt * k3, *args)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Step 3: Solve the ODE
def solve_pendulum(f, method, t_span, y0, dt, *args):
    t_values = jnp.arange(t_span[0], t_span[1], dt)
    y_values = jnp.zeros((t_values.size, y0.size))
    y_values = y_values.at[0].set(y0)

    for i in range(1, t_values.size):
        y_values = y_values.at[i].set(method(f, y_values[i - 1], t_values[i - 1], dt, *args))

    return t_values, y_values

# Step 4: Plot the solution
import matplotlib.pyplot as plt

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

# Step 5: Generate and Split Data
def gen_data(t, y):
    """Generate test and train data from the solution of the numerical method."""
    t_sliced, y_sliced = (
        t[jnp.arange(t.size, step=200)],
        y[jnp.arange(t.size, step=200)],
    )
    split_index = int(0.8 * len(t_sliced))
    t_train, y_train = t_sliced[:split_index], y_sliced[:split_index, 0]
    t_test, y_test = t_sliced[split_index:], y_sliced[split_index:, 0]
    return t_train, y_train, t_test, y_test
if __name__ == "__main__":
    # Parameters for the pendulum
    b = 0.1  # damping coefficient
    m = 1.0  # mass of the pendulum bob
    l = 1.0  # length of the pendulum
    g = 9.81  # acceleration due to gravity
    dt = 0.01  # time step
    t_span = (0.0, 20.0)  # time range

    # Initial conditions: theta = pi/4 (45 degrees), omega = 0
    y0 = jnp.array([jnp.pi / 4, 0.0])

    # Solve using Euler method
    t_euler, y_euler = solve_pendulum(pendulum_ode, euler_step, t_span, y0, dt, b, m, l, g)
    
    # Solve using Runge-Kutta method
    t_rk, y_rk = solve_pendulum(pendulum_ode, runge_kutta_step, t_span, y0, dt, b, m, l, g)

    # Extract the theta (angle) values from the solutions
    theta_euler = y_euler[:, 0]  # Euler method theta values
    theta_rk = y_rk[:, 0]        # Runge-Kutta method theta values

    # Plot both solutions for comparison
    plot_comparison(t_euler, theta_euler, t_rk, theta_rk, title="Pendulum Motion: Euler vs Runge-Kutta")
