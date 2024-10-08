import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_generator import gen_data, runge_kutta_method, pendulum_ode


# Define MLP model
class MLP(nn.Module):
    """Multilayer Perceptron model."""
    features: list

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]
        self.output_layer = nn.Dense(1)

    def __call__(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        x = self.output_layer(x)
        return x


# Extend TrainState to include metrics
class TrainState(train_state.TrainState):
    metrics: dict


# Create a training state with the optimizer
def create_train_state(model, init_key, learning_rate, input_shape):
    params_key, init_key = jax.random.split(init_key)
    dummy_input = jnp.ones(input_shape)
    params = model.init(params_key, dummy_input)['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, metrics={})


# Mean Squared Error (MSE) loss function
def mse_loss(params, apply_fn, batch):
    predictions = apply_fn({'params': params}, batch['inputs'])
    loss = jnp.mean((predictions - batch['targets']) ** 2)
    return loss


# R^2 score function
def r2_score(params, apply_fn, batch):
    predictions = apply_fn({'params': params}, batch['inputs'])
    ss_res = jnp.sum((batch['targets'] - predictions) ** 2)
    ss_tot = jnp.sum((batch['targets'] - jnp.mean(batch['targets'])) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


# Compute metrics (MSE and R²)
@jax.jit
def compute_metrics(state, batch):
    loss = mse_loss(state.params, state.apply_fn, batch)
    r2 = r2_score(state.params, state.apply_fn, batch)
    return {'mse_loss': loss, 'r2_score': r2}


# Training step
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        return mse_loss(params, state.apply_fn, batch)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    metrics = compute_metrics(state, batch)
    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics


# Validation step
@jax.jit
def val_step(state, batch):
    metrics = compute_metrics(state, batch)
    return metrics


# Train function to train the model on the data using a specified learning rate
def train(model, data, learning_rate, epochs, batch_size):
    key = jax.random.PRNGKey(0)
    t_train_data, y_train_data, _, _ = data

    # Split the training set into training and validation (80/20 split)
    split_index = int(0.8 * len(y_train_data))
    t_train, y_train = t_train_data[:split_index], y_train_data[:split_index]
    t_val, y_val = t_train_data[split_index:], y_train_data[split_index:]

    # Create the initial training state
    state = create_train_state(model, key, learning_rate, input_shape=(1,))
    metrics_history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': []}

    num_batches = len(t_train) // batch_size

    # Training loop
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        # Shuffle the dataset at the start of each epoch
        permutation = jax.random.permutation(key, len(t_train))
        t_train = t_train[permutation]
        y_train = y_train[permutation]

        # Iterate through mini-batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch = {
                'inputs': t_train[start_idx:end_idx].reshape(-1, 1),
                'targets': y_train[start_idx:end_idx].reshape(-1, 1)
            }
            state, train_metrics = train_step(state, batch)

        # Validation after every epoch
        val_batch = {
            'inputs': t_val.reshape(-1, 1),
            'targets': y_val.reshape(-1, 1)
        }
        val_metrics = val_step(state, val_batch)

        # Store metrics
        metrics_history['train_loss'].append(train_metrics['mse_loss'])
        metrics_history['val_loss'].append(val_metrics['mse_loss'])
        metrics_history['train_r2'].append(train_metrics['r2_score'])
        metrics_history['val_r2'].append(val_metrics['r2_score'])

        # Print metrics every 10,000 epochs or on the final epoch
        if epoch % 10000 == 0 or epoch == epochs - 1:
            print(f"\nEpoch {epoch + 1}/{epochs} - Train Loss: {train_metrics['mse_loss']:.4f}, "
                  f"Val Loss: {val_metrics['mse_loss']:.4f}, Train R2: {train_metrics['r2_score']:.4f}, "
                  f"Val R2: {val_metrics['r2_score']:.4f}")

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_history['train_loss'], label='Train Loss')
    plt.plot(metrics_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    return state, metrics_history


# Testing function to evaluate the model on the test data
def test(state, t_test, y_test):
    test_batch = {
        'inputs': t_test.reshape(-1, 1),
        'targets': y_test.reshape(-1, 1)
    }
    test_metrics = compute_metrics(state, test_batch)
    print(f"Test Loss: {test_metrics['mse_loss']:.4f}, Test R2: {test_metrics['r2_score']:.4f}")


# Main function
if __name__ == "__main__":
    # Parameters for the pendulum ODE
    y0 = jnp.array([2 * jnp.pi / 3, 0.0])  # Example initial conditions
    t_span = (0, 20)  # Time span
    dt = 0.01  # Time step
    b = 0.3  # Example damping coefficient
    m = 1.0  # Mass
    l = 1.0  # Length
    g = 9.81  # Gravity

    # Generate the time array (t)
    t = jnp.arange(t_span[0], t_span[1], dt)

    # Generate data from the ODE solution (pass t as a parameter)
    y_rk = runge_kutta_method(pendulum_ode, y0, t, dt)
    data = gen_data(t, y_rk)
    key = jax.random.PRNGKey(0)
    model = MLP([32, 32, 32, 16])
    learning_rate = 1e-4
    epochs = 100_000
    batch_size = 32

    # Train the model
    state, metrics_history = train(model, data, learning_rate, epochs, batch_size)

    # Test the model on the test set
    t_train, y_train, t_test, y_test = data
    test(state, t_test, y_test)