import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm
import flax.linen as nn
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from data_generator import gen_data, runge_kutta_method, pendulum_ode

# Define the MLP model
class MLP(nn.Module):
    features: list

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]
        self.output_layer = nn.Dense(1)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        x = self.output_layer(x)
        return x

# Extend TrainState to include optimizer
class TrainState(train_state.TrainState):
    pass

# Function to create the training state
def create_train_state(model, init_key, learning_rate, input_shape):
    params_key, init_key = jax.random.split(init_key)
    dummy_input = jnp.ones(input_shape)
    params = model.init(params_key, dummy_input)['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# ODE loss function
def ode_loss(params, apply_fn, batch, ode_params):
    t, _ = batch
    b, m, l, g = ode_params

    theta = apply_fn({'params': params}, t.reshape(-1, 1)).flatten()

    def single_time_derivative(t_val):
        return apply_fn({'params': params}, t_val.reshape(-1, 1)).flatten()[0]

    dtheta_dt = jax.vmap(jax.grad(single_time_derivative))(t)
    d2theta_dt2 = jax.vmap(jax.grad(jax.grad(single_time_derivative)))(t)

    residual = d2theta_dt2 + b * dtheta_dt + (g / l) * jnp.sin(theta)
    loss = jnp.mean(residual ** 2)
    return loss

@jax.jit
def ode_train_step(state, batch, ode_params):
    def loss_fn(params):
        return ode_loss(params, state.apply_fn, batch, ode_params)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

# Train the model using ODE loss
def train_with_ode_loss(model, data, learning_rate, epochs, batch_size, ode_params):
    key = jax.random.PRNGKey(0)
    t_train, _, _, _ = data
    state = create_train_state(model, key, learning_rate, input_shape=(1,))
    metrics_history = {'ode_loss': []}

    num_batches = len(t_train) // batch_size

    for epoch in tqdm(range(epochs), desc="ODE Training Progress"):
        key, subkey = jax.random.split(key)
        permutation = jax.random.permutation(subkey, len(t_train))
        t_train = t_train[permutation]

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch = (t_train[start_idx:end_idx], None)
            state, loss = ode_train_step(state, batch, ode_params)

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
    plt.title('ODE Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function with Hydra for configuration
@hydra.main(config_name="config")
def main(cfg: DictConfig):
    # Load data
    y0 = jnp.array([2 * jnp.pi / 3, 0.0])
    t_span = (0, 20)
    dt = 0.01

    # Generate data from ODE solution
    t = jnp.arange(t_span[0], t_span[1], dt)
    y_rk = runge_kutta_method(pendulum_ode, y0, t, dt)
    data = gen_data(t, y_rk)

    # Initialize and train the model with ODE loss
    model = MLP(cfg.model.features)
    state, metrics_history = train_with_ode_loss(
        model=model,
        data=data,
        learning_rate=cfg.training.learning_rate,
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        ode_params=(cfg.ode.b, cfg.ode.m, cfg.ode.l, cfg.ode.g)
    )

    # Plot ODE loss
    plot_ode_loss(metrics_history)


if __name__ == "__main__":
    main()
