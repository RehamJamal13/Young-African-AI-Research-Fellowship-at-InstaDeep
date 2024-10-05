import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training import train_state

from data_generator import gen_data


class MLP(nn.Module):
    """Multilayer Perceptron model."""

    pass  # TODO: Complete this class


class TrainState(train_state.TrainState):
    metrics: dict


def create_train_state(model, init_key, learning_rate, input_shape):

    pass  # TODO: Complete this function


def mse_loss(params, apply_fn, batch):
    pass  # TODO: Complete this function


@jax.jit
def compute_metrics(state, batch):
    pass  # TODO: Complete this function


@jax.jit
def train_step(state, batch):
    pass  # TODO: Complete this function


@jax.jit
def val_step(state, batch):
    pass  # TODO: Complete this function


# TODO: Write a train function to train a given model on a given data using a specified learning rate.


if __name__ == "__main__":
    y0 = jnp.array([2 * jnp.pi / 3, 0.0])
    t_span = (0, 20)
    dt = 0.01
    b = 0.3
    m = 1.0
    l = 1.0
    g = 9.81
    t, y = None  #  Output from the KR method function.
    data = gen_data(t, y)

    key = jax.random.PRNGKey(0)
    model = MLP([16, 16, 16])
    learning_rate = 1e-3
    epochs = 100_000

    state, metrics_history = None  # The output from the train function.
    # TODO: Add plotting functionality
