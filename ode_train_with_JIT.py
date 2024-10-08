import jax
import jax.numpy as jnp
from jax import lax
import time
from data_generator import gen_data, runge_kutta_method, pendulum_ode, euler_method  # Importing from your module

# Constants
dt = 0.01
b = 0.3
m = 1.0
l = 1.0
g = 9.81

# Euler method using `lax.scan`
@jax.jit
def euler_method_scan(y0, t, dt=dt):
    """Solves ODE using Euler's method with JAX lax.scan."""
    def step_fn(carry, _):
        y = carry
        y_next = y + dt * pendulum_ode(y)  # Assuming pendulum_ode is imported from data_generator
        return y_next, y_next

    y0 = jnp.array(y0)
    _, ys = lax.scan(step_fn, y0, t)
    ys = jnp.vstack([y0, ys])  # Adding the initial condition at the beginning
    return ys

# Runge-Kutta method using `lax.scan`
@jax.jit
def runge_kutta_method_scan(y0, t, dt=dt):
    """Solves ODE using the 4th-order Runge-Kutta method with JAX lax.scan."""
    def step_fn(carry, _):
        y = carry
        k1 = pendulum_ode(y)  # Assuming pendulum_ode is imported from data_generator
        k2 = pendulum_ode(y + 0.5 * dt * k1)
        k3 = pendulum_ode(y + 0.5 * dt * k2)
        k4 = pendulum_ode(y + dt * k3)
        y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y_next, y_next

    y0 = jnp.array(y0)
    _, ys = lax.scan(step_fn, y0, t)
    ys = jnp.vstack([y0, ys])  # Adding the initial condition at the beginning
    return ys

# Main function to call the solvers and compare execution time
# Time array
t_span = (0, 20)
t = jnp.arange(t_span[0], t_span[1], dt)

# Initial condition: y0 = [theta, omega]
y0 = jnp.array([2 * jnp.pi / 3, 0.0])

# Measure execution time for Euler method (Python loop) from data_generator
start_time = time.time()
euler_solution = euler_method(pendulum_ode, y0, t, dt)  # Imported euler_method
euler_time = time.time() - start_time
print(f"Execution time for Euler method (Python loop): {euler_time:.4f} seconds")

# Measure execution time for Runge-Kutta method (Python loop) from data_generator
start_time = time.time()
runge_kutta_solution = runge_kutta_method(pendulum_ode, y0, t, dt)  # Imported runge_kutta_method
runge_kutta_time = time.time() - start_time
print(f"Execution time for Runge-Kutta method (Python loop): {runge_kutta_time:.4f} seconds")

# Measure execution time for Euler method using `lax.scan` + JIT
start_time = time.time()
euler_scan_solution = euler_method_scan(y0, t, dt)
euler_scan_time = time.time() - start_time
print(f"Execution time for Euler method (lax.scan + JIT): {euler_scan_time:.4f} seconds")

# Measure execution time for Runge-Kutta method using `lax.scan` + JIT
start_time = time.time()
runge_kutta_scan_solution = runge_kutta_method_scan(y0, t, dt)
runge_kutta_scan_time = time.time() - start_time
print(f"Execution time for Runge-Kutta method (lax.scan + JIT): {runge_kutta_scan_time:.4f} seconds")
