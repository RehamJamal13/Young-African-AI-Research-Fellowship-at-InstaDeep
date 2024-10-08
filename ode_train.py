import jax
import jax.numpy as jnp

from train import MLP
from data_generator import gen_data

def ode_loss(params, apply_fn, batch, ode_params=(0.3, 1.0, 1.0, 9.81)):
    t, _ = batch  # Unpack time values
    b, m, l, g = ode_params
    
    # Apply the neural network to get predicted theta and omega (angular velocity)
    y_pred = apply_fn({'params': params}, t)
    theta_pred, omega_pred = y_pred[:, 0], y_pred[:, 1]
    
    # Compute first and second derivatives of theta
    theta_dot = jax.grad(lambda theta: theta)(t)  # d(theta)/dt
    theta_ddot = jax.grad(theta_dot)(t)           # d^2(theta)/dt^2
    
    # Calculate ODE residual
    residual = theta_ddot + (b/(m*l)) * omega_pred + (g/l) * jnp.sin(theta_pred)
    
    # ODE loss is the mean of squared residuals
    loss = jnp.mean(residual ** 2)
    return loss



@jax.jit
def ode_train_step(state, batch):
    """A single training step using the ODE loss."""
    def loss_fn(params):
        return ode_loss(params, state.apply_fn, batch)
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)  # Get loss and gradients
    new_state = state.apply_gradients(grads=grads)  # Update model parameters
    return new_state, loss



import optax
from flax.training import train_state

def create_train_state(rng, model, learning_rate):
    """Creates initial training state."""
    params = model.init(rng, jnp.ones([1, 1]))  # Initialize model parameters
    tx = optax.adam(learning_rate)              # Use Adam optimizer
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def train_model(state, batch, epochs):
    """Training loop for the neural network using ODE loss."""
    metrics_history = []

    for epoch in range(epochs):
        state, loss = ode_train_step(state, batch)  # Run a training step
        metrics_history.append(loss)  # Log loss for each epoch

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return state, metrics_history
import matplotlib.pyplot as plt

def plot_loss_history(metrics_history):
    plt.plot(metrics_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss over Epochs")
    plt.show()


if __name__ == "__main__":
    y0 = jnp.array([2 * jnp.pi / 3, 0.0])  # Initial conditions: theta and omega
    t_span = (0, 20)
    dt = 0.01
    b = 0.3
    m = 1.0
    l = 1.0
    g = 9.81
    ode_params = (b, m, l, g)

    # Generate data using your ODE solver (data generation part needs to be completed)
    data = gen_data(y0, t_span, dt, ode_params)
    batch = (data['t'], data['theta'])  # Time and theta (simplified for illustration)

    # Initialize model and training
    key = jax.random.PRNGKey(0)
    model = MLP([16, 16, 16])
    learning_rate = 1e-3
    state = create_train_state(key, model, learning_rate)

    # Train the model
    epochs = 100_000
    state, ode_metrics_history = train_model(state, batch, epochs)

    # Plot loss history
    plot_loss_history(ode_metrics_history)
