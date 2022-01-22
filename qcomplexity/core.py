# Core functionality for the QComplexity library

import jax.numpy as jnp
import jax
import numpy as np

##################################################

X = jnp.array([[0, 1],
              [1, 0]])

Y = jnp.array([[0, -1j],
              [1j, 0]])

Z = jnp.array([[1, 0],
              [0, -1]])

gates = jnp.array([Z, Y, X])

#################################################
# Standard functions for implementing unitary parametrizations

def euler_fn(vec):
    """Implements the Euler angle unitary parametrization"""
    return jax.scipy.linalg.expm(1j * vec[0] * Z) @ jax.scipy.linalg.expm(1j * vec[1] * Y) @ jax.scipy.linalg.expm(1j * vec[2] * X)

def ham_fn(vec):
    """Implements the exponentiated Hamiltonian unitary transformation"""
    return jax.scipy.linalg.expm(1j * jnp.sum(vec[:,jnp.newaxis,jnp.newaxis] * gates, axis=0)) 

#################################################

def numerical_metric(inertia, u_fn):
    """Uses a numerical procedure to compute the metric tensor. This is going to have to be fixed eventually, when we decide 
    to support higher-dimensional systems."""

    unitary_fn = u_fn
    gradient = jax.jacobian(unitary_fn, holomorphic=True)

    def fn(coordinates):
        # Computes differentials
        val = unitary_fn(coordinates)
        diff = gradient(coordinates)
        new_val = jnp.conj(jnp.transpose(val))
        
        # Prepares the vectors for use in final computation
        g = 1j * jnp.einsum("ijk,jl->kil", diff, new_val)
        final_values = jnp.einsum("ijk,lkj->li", g, gates)
        
        # Prepares the metric
        met = jnp.zeros((3, 3))
        for i in range(len(gates)):
            for j in range(len(gates)):
                if i != j:
                    met = met + inertia[i][j] * jnp.outer(final_values[i], final_values[j])
                else:
                    met = met + 0.5 * inertia[i][j] * jnp.outer(final_values[i], final_values[j])
            
        return 0.5 * met
    return fn


def numerical_christoffel_symbols(metric):
    """Uses a numerical procedure to compute the Christoffel symbols corresponding to a metric"""

    gradient = jax.jacobian(metric, holomorphic=True)

    def fn(vec):
        g_val = metric(vec)
        g_inv = jnp.linalg.inv(g_val) # Inverse metric tensor
        g_grad = gradient(vec) # Computes the partial derivatives of the metric tensor matrix elements
        
        nr = g_grad + jnp.swapaxes(g_grad, 1, 2) - jnp.swapaxes(jnp.swapaxes(g_grad, 0, 1), 1, 2)
        symbol_mat = jnp.einsum('ij,jkl->ikl', g_inv, nr)
        return (0.5 * symbol_mat)
    return fn


def diff_fn(christoffel_symbols):
    """Defines the differential function for use in the ODE solver, for a given set of Christoffel symbols"""
    def fn(y1, y2):
        return jnp.einsum('j,ijk,k', y2, christoffel_symbols(y1), y2)
    return fn


def length(metric, vals, d_vals, step):
    """Computes the length of a geodesic"""
    s = 0
    for y_val, dy_val in zip(vals, d_vals):
        s += jnp.sqrt( jnp.inner(dy_val, metric(y_val) @ dy_val)) * step
    return s

##########################################################

def solve_geodesic_ivp(diff_fn, steps, x_init=None):
    """Solves for the endpoint of an IVP"""
    def fn(v_init):
        x = x_init if x_init is not None else jnp.array([0.0, 0.0, 0.0], dtype=complex)
        v = v_init
        for s in np.linspace(0, 1, steps):
            a = -diff_fn(x, v)
            x = x + v * (1/steps) + (0.5 * (1/steps) ** 2) * a
            v = v + a * (1/steps)
        return x
    return fn


def solve_geodesic_path(diff_fn, steps, x_init=None):
    """Solves for the path of a geodesic"""
    def fn(v_init):
        x = x_init if x_init is not None else jnp.array([0.0, 0.0, 0.0], dtype=complex)
        v = v_init

        x_l = [x]
        v_l = [v]
        for s in np.linspace(0, 1, steps):
            a = -diff_fn(x, v)
            x = x + v * (1/steps) + (0.5 * (1/steps) ** 2) * a
            v = v + a * (1/steps)

            x_l.append(x)
            v_l.append(v)
        return x_l, v_l
    return fn 


def endpoint_cost_fn(solve_ivp, fin):
    """Cost function based on error of geodesic endpoint"""
    def fn(v):  
        return float(np.linalg.norm(solve_ivp(v) - fin))
    return fn