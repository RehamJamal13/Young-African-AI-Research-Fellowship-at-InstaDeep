## Problem Description

In this use case, we will model the motion of a damped pendulum using neural networks. Given limited observations of the pendulum's angle at different times, the goal is to train a neural network to predict the angle (theta) as a function of time.

The pendulum's motion is described by the following ordinary differential equation (ODE):

$$
\frac{d^2\theta}{dt^2} + \frac{b}{m} \frac{d\theta}{dt} + \frac{g}{l}\sin(\theta) = 0
$$

where:
* `b`: damping coefficient
* `m`: mass of the pendulum bob
* `l`: length of the pendulum
* `g`: acceleration due to gravity

We will use Python with [JAX](https://jax.readthedocs.io) for numerical computations and [Flax](https://flax.readthedocs.io) for neural network modeling.

## Data Generation

We will generate a dataset for training using numerical methods:

* **Solve the ODE:** Use the [Euler method ](https://en.wikipedia.org/wiki/Euler_method) and [Runge–Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) with a time step of $\Delta t = 0.01$ over a time range of $[0, 20]$ to obtain a numerical solution.
* **Subsample:**  Reduce the number of samples to 10 to simulate a scenario with limited number of samples.
* **Train-Test Split:** Split the data into an 80% training set and a 20% test set.

The ODE can be rewritten as a system of two first-order ODEs:

* $\frac{d\omega}{dt} = - \frac{b}{ml} \omega - \frac{g}{l}\sin(\theta)$
* $\frac{d\theta}{dt} = \omega$

where $\omega = \frac{d\theta}{dt}$ is the angular velocity.

This can be expressed in vector form as:

$ \frac{dy}{dt} = F(y)$ 

where $y = \begin{bmatrix} \theta \\ \omega \end{bmatrix}$ and $F(y) = \begin{bmatrix} \omega  \\ - \frac{b}{ml} \omega - \frac{g}{l}\sin(\theta)\\  \end{bmatrix}$.

0. **Environment Setup:** Create a conda environment and a `requirements.txt` file, pinning each package to the appropriate version.

1. **ODE Function Implementation:** Write a function that takes `y` and returns `F(y)`.

The `solve_pendulum_euler` function implements the Euler method.

2. **Runge-Kutta Implementation:** Implement the functions for solving the ODE using the Euler method and the 4th order Runge–Kutta method.

3. **Plotting:** Write a function to plot the solution obtained the solvers.

## Model and Training

4. **Model and Training Setup:**  
   * Define a neural network model using Flax with [specify architecture: layers, neurons, activations].
   * Implement training using [specify optimizer, learning rate, epochs].

5. **Loss and Visualization:** 
   * Plot the training and test loss curves and save them to a dedicated directory.
   * Evaluate the model's performance on the test set using Mean Squared Error (MSE).

## Using the ODE as a Loss

We can leverage the ODE and initial conditions to create a loss function without relying on generated data.

The loss function will consist of three terms:

<!-- * **ODE Residual:** Minimize the squared residual of the ODE: $(\frac{d^2MLP(t)}{dt^2} + \frac{b}{ml} \frac{dMLP(t)}{dt} + \frac{g}{l}\sin(MLP(t)))^2$.
* **Initial Condition (Angle):** Minimize the squared error between the predicted initial angle and the true initial angle: $(MLP(t=0) - 2\pi/3)^2$.
* **Initial Condition (Angular Velocity):** Minimize the squared error between the predicted initial angular velocity and the true initial angular velocity: $(\frac{dMLP(t)}{dt}(0))^2$. -->

6. **`ode_loss` Implementation:** Complete the `ode_loss` function to calculate the total loss describing the ODE and the Initial conditions. Use `jax.grad` for automatic differentiation and `jax.vmap` for auto-vectorization.

7. **Training and Comparison:** Train the model using the `ode_loss` and compare the results with the data-driven approach.

8. **Hydra Configuration:** Use Hydra to make the scripts configurable instead of using hard-coded values.

9. **JAX `scan` function**   Use `scan` function from `jax.lax` instead of python for loop for the numerical solvers, do the implementations in separate functions and compare performance.

10. **JIT Compilation:** Use `jax.jit` to compile functions for faster execution whenever possible and compare execution time between the JIT-ed and non JIT-ed versions.

