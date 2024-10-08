import jax
import jax.numpy as jnp
from jax import grad, vmap
from flax import linen as nn
from flax.training import train_state
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt


# Define MLP model
class MLP(nn.Module):
    features: list

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]
        self.output_layer = nn.Dense(1)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        return self.output_layer(x)


# Create TrainState to handle model state and optimizer
class TrainState(train_state.TrainState):
    pass


# Function to create a training state with optimizer
def create_train_state(model, init_key, learning_rate, input_shape):
    params_key, init_key = jax.random.split(init_key)
    dummy_input = jnp.ones(input_shape)
    params = model.init(params_key, dummy_input)['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# ODE loss function to enforce ODE residuals and initial condition
def ode_loss(params, apply_fn, t, initial_condition, ode_params):
    b, m, l, g = ode_params

    # Predict theta using the model
    theta_pred = apply_fn({'params': params}, t.reshape(-1, 1)).flatten()

    # Function to compute dθ/dt for a single time point
    def single_time_derivative(t_val):
        return apply_fn({'params': params}, t_val.reshape(-1, 1)).flatten()[0]

    # Compute dθ/dt and d²θ/dt² using automatic differentiation
    dtheta_dt = vmap(grad(single_time_derivative))(t)
    d2theta_dt2 = vmap(grad(grad(single_time_derivative)))(t)

    # ODE residual: d²θ/dt² + b*dθ/dt + (g/l) * sin(θ)
    ode_residual = d2theta_dt2 + b * dtheta_dt + (g / l) * jnp.sin(theta_pred)

    # ODE loss (should be zero ideally)
    ode_loss_value = jnp.mean(ode_residual ** 2)

    # Initial condition loss: enforce θ(0)
    initial_pred = apply_fn({'params': params}, jnp.array([0.0]).reshape(-1, 1)).flatten()
    initial_loss = jnp.mean((initial_pred - initial_condition) ** 2)

    # Total loss: ODE loss + initial condition loss
    total_loss = ode_loss_value + initial_loss
    return total_loss


# A single ODE-based training step
@jax.jit
def ode_train_step(state, t, initial_condition, ode_params):
    def loss_fn(params):
        return ode_loss(params, state.apply_fn, t, initial_condition, ode_params)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


# Function to train the model using the ODE-based loss
def train_model_ode_loss(model, t, initial_condition, ode_params, learning_rate, epochs, batch_size):
    key = jax.random.PRNGKey(0)
    state = create_train_state(model, key, learning_rate, input_shape=(1,))

    loss_history = []

    for epoch in tqdm(range(epochs), desc="ODE Training Progress"):
        # Shuffle the time points
        key, subkey = jax.random.split(key)
        permutation = jax.random.permutation(subkey, len(t))
        t_shuffled = t[permutation]

        # Batch training
        for i in range(0, len(t), batch_size):
            t_batch = t_shuffled[i:i + batch_size]

            # Perform a single training step
            state, loss = ode_train_step(state, t_batch, initial_condition, ode_params)

        # Track the loss
        loss_history.append(loss)

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return state, loss_history


# Function to plot the loss history
def plot_loss_history(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='ODE Loss', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('ODE Loss Over Training Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()


# Runge-Kutta solver using jax.lax.scan for numerical integration
import jax.lax as lax

def runge_kutta_step(F, y, t, dt):
    k1 = F(y)
    k2 = F(y + 0.5 * dt * k1)
    k3 = F(y + 0.5 * dt * k2)
    k4 = F(y + dt * k3)
    y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_next, None

def runge_kutta_method_scan(F, y0, t, dt):
    def step_fn(y, t):
        return runge_kutta_step(F, y, t, dt)

    _, ys = lax.scan(step_fn, y0, t)
    return ys


# Main execution block
if __name__ == "__main__":
    # Pendulum ODE parameters
    b = 0.1  # damping coefficient
    m = 1.0  # mass
    l = 1.0  # length
    g = 9.81  # gravity
    ode_params = (b, m, l, g)

    # Time span and initial conditions
    t_span = (0, 10)
    dt = 0.01
    t = jnp.arange(t_span[0], t_span[1], dt)
    initial_condition = jnp.array([2 * jnp.pi / 3])  # theta(0)

    # Define MLP model
    model = MLP([64, 64, 64])

    # Training hyperparameters
    learning_rate = 1e-3
    epochs = 1000
    batch_size = 32

    # Train the model using ODE-based loss
    state, loss_history = train_model_ode_loss(model, t, initial_condition, ode_params, learning_rate, epochs, batch_size)

    # Plot the loss history
    plot_loss_history(loss_history)
