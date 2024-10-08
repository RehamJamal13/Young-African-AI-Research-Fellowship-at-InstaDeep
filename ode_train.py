import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_generator import gen_data, runge_kutta_method, pendulum_ode
from train import MLP

# Create TrainState class with optimizer
class TrainState(train_state.TrainState):
    pass

# Function to create the training state
def create_train_state(model, init_key, learning_rate, input_shape):
    params_key, init_key = jax.random.split(init_key)
    dummy_input = jnp.ones(input_shape)
    params = model.init(params_key, dummy_input)['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# ODE loss function based on pendulum dynamics and initial conditions
def ode_loss(params, apply_fn, t, y0, ode_params=(0.3, 1.0, 1.0, 9.81)):
    """
    Loss function combining ODE residuals and initial condition errors.
    
    :param params: Model parameters
    :param apply_fn: Model application function
    :param t: Time array
    :param y0: Initial conditions (theta0, omega0)
    :param ode_params: Parameters (b, m, l, g) for the pendulum dynamics
    """
    b, m, l, g = ode_params
    theta0, omega0 = y0

    # Predict theta using the model
    theta = apply_fn({'params': params}, t.reshape(-1, 1)).flatten()

    # Function to compute dθ/dt for a single time point
    def single_time_derivative(t_val):
        return apply_fn({'params': params}, t_val.reshape(-1, 1)).flatten()[0]

    # Compute dθ/dt and d²θ/dt² using automatic differentiation
    dtheta_dt = jax.vmap(jax.grad(single_time_derivative))(t)
    d2theta_dt2 = jax.vmap(jax.grad(jax.grad(single_time_derivative)))(t)

    # ODE residual: should be zero for correct solutions
    residual = d2theta_dt2 + b * dtheta_dt + (g / l) * jnp.sin(theta)

    # Initial condition loss: Compare model's initial state with actual y0
    theta_pred_0 = theta[0]
    omega_pred_0 = dtheta_dt[0]
    initial_loss = (theta_pred_0 - theta0) ** 2 + (omega_pred_0 - omega0) ** 2

    # Minimize the MSE of the residuals and the initial condition loss
    residual_loss = jnp.mean(residual ** 2)
    total_loss = residual_loss + initial_loss

    return total_loss

# Function to perform a single ODE training step
@jax.jit
def ode_train_step(state, t, y0):
    def loss_fn(params):
        return ode_loss(params, state.apply_fn, t, y0)
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

# Training loop for ODE loss
def train_with_ode_loss(model, t, y0, learning_rate, epochs, batch_size):
    key = jax.random.PRNGKey(0)
    state = create_train_state(model, key, learning_rate, input_shape=(1,))
    metrics_history = {'ode_loss': []}

    num_batches = len(t) // batch_size

    for epoch in tqdm(range(epochs), desc="ODE Training Progress"):
        key, subkey = jax.random.split(key)
        permutation = jax.random.permutation(subkey, len(t))
        t_shuffled = t[permutation]

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            t_batch = t_shuffled[start_idx:end_idx]
            state, loss = ode_train_step(state, t_batch, y0)

        metrics_history['ode_loss'].append(loss)

        if epoch % 250 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - ODE Loss: {loss:.4f}")

    return state, metrics_history

# Plot ODE loss history
def plot_ode_loss(metrics_history):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_history['ode_loss'], label='ODE Loss', color='b', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('ODE Loss')
    plt.title('ODE Loss Over Training Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

# Main execution block
if __name__ == "__main__":
    # Parameters for the pendulum ODE
    y0 = jnp.array([2 * jnp.pi / 3, 0.0])  # Initial conditions (theta0, omega0)
    t_span = (0, 20)
    dt = 0.01

    # Generate time data (instead of depending on generated theta)
    t = jnp.arange(t_span[0], t_span[1], dt)

    # Define the MLP model architecture from `train.py`
    key = jax.random.PRNGKey(0)
    model = MLP([16, 16, 16])

    # Hyperparameters
    learning_rate = 1e-3
    epochs = 1000  # Adjust number of epochs as needed
    batch_size = 8

    # Train the model using ODE loss and initial conditions
    state, ode_metrics_history = train_with_ode_loss(model, t, y0, learning_rate, epochs, batch_size)

    # Plot the ODE loss over time
    plot_ode_loss(ode_metrics_history)
####################