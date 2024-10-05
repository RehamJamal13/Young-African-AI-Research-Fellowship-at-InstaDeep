import jax
import jax.numpy as jnp

from train import MLP


def ode_loss(params, apply_fn, batch, ode_params=(0.3, 1.0, 1.0, 9.81)):

    t, _ = batch
    b, m, l, g = ode_params
    pass  # TODO: Complete this function


@jax.jit
def ode_train_step(state, batch):
    """A train step using the ode_loss."""
    pass  # TODO: Complete this function


# TODO: Write a function for training the model using the ODE loss.


if __name__ == "__main__":
    y0 = jnp.array([2 * jnp.pi / 3, 0.0])
    t_span = (0, 20)
    dt = 0.01
    b = 0.3
    m = 1.0
    l = 1.0
    g = 9.81
    data = None  #  Output from the KR method function.
    key = jax.random.PRNGKey(0)
    model = MLP([16, 16, 16])
    learning_rate = 1e-3
    epochs = 100_000
    state, ode_metrics_history = # Output from the ode training function.

    # TODO: Add plotting functionality
