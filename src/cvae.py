
from flax import linen as nn
from jraph import GraphsTuple, GraphConvolution
from typing import Any, Callable, Sequence
import anndata
import flax
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pandas as pd
import pickle
import squidpy as sq
import sys


"""
Transform matrix columns to z-scores.
"""
def zscore(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def neighborhood_size(adata: anndata.AnnData, radius):
    sq.gr.spatial_neighbors(adata, radius=radius, coord_type="generic")
    A = adata.obsp["spatial_connectivities"]
    ncells = adata.X.shape[0]
    return A.nnz / ncells


def calibrate_neighborhood_radius(adata: anndata.AnnData, target_size: int):
    x = np.array(adata.obsm["spatial"][:,0], dtype=np.float32)
    y = np.array(adata.obsm["spatial"][:,1], dtype=np.float32)
    max_max_radius = np.sqrt((np.max(x) - np.min(x))**2 + (np.max(y) - np.min(y))**2)
    min_radius = max_max_radius / 10000

    # Find a max_radius that isn't too large that it's O(n^)
    step = max_max_radius / 200
    max_radius = min_radius
    while neighborhood_size(adata, max_radius) < target_size and max_radius < max_max_radius:
        max_radius += step

    size = np.inf
    eps = 0.2
    while np.abs(target_size - size) > eps:
        radius = (max_radius + min_radius) / 2
        size = neighborhood_size(adata, radius)
        if size < target_size:
            min_radius = radius
        else:
            max_radius = radius

    return radius


"""
Basic graph convolution layer.
"""
class GCLayer(nn.Module):
    hidden_dim: int
    activation: Callable = lambda x: x

    @nn.compact
    def __call__(self, G):
        gc = jraph.GraphConvolution(
            update_node_fn=lambda nodes: self.activation(nn.Dense(self.hidden_dim)(nodes)))
                # self.activation(nn.Dense(self.hidden_dim)(nodes)))
        return gc(G)


class CVAEEncoder(nn.Module):
    training: bool
    z_dim: int
    hidden_dim: int = 20
    nlayers: int = 2

    @nn.compact
    def __call__(self, G):
        # neighborhood representation
        h = G
        for i in range(self.nlayers):
            h = GCLayer(self.hidden_dim, activation=lambda x: nn.softmax(nn.relu(x)))(h)
            # h = h._replace(nodes=nn.BatchNorm(use_running_average=not self.training)(h.nodes))
        h = h.nodes

        # combine that with node labels
        h = jnp.concatenate([h, G.nodes], axis=-1)

        μ = nn.Dense(self.z_dim)(h)
        logσ2 = nn.Dense(self.z_dim)(h)

        return (μ, logσ2)


class CVAEDecoder(nn.Module):
    training: bool
    expr_dim: int
    hidden_dim: int = 40
    nlayers: int = 2

    @nn.compact
    def __call__(self, z):
        h = z
        for i in range(self.nlayers-1):
            h = nn.relu(nn.Dense(self.hidden_dim)(h))
        h = nn.Dense(self.expr_dim)(h)
        return h


class CVAE(nn.Module):
    training: bool
    z_dim: int
    expr_dim: int
    σbound: float = 1e-4

    @nn.compact
    def __call__(self, key, G):
        μ, logσ2 = CVAEEncoder(training=self.training, z_dim=self.z_dim)(G)

        σ = self.σbound + jnp.exp(0.5 * logσ2)
        z = μ + σ * jax.random.normal(key, μ.shape, dtype=jnp.float32)

        Xsample = CVAEDecoder(training=self.training, expr_dim=self.expr_dim)(z)
        return (Xsample, μ, logσ2)


def elbo(X, Xsample, μ, logσ2):
    ll = -jnp.sum(jnp.square(X - Xsample))
    σ2 = jnp.exp(logσ2)
    kl = 0.5 * jnp.sum(jnp.square(μ) + σ2 - logσ2 - 1.0)
    return ll - kl


@jax.partial(jax.jit, static_argnums=(0,))
def train_step(z_dim, optimizer, model_state, key, G, X):
    def loss_fn(params):
        vars = {"params": params, **model_state}

        (Xsample, μ, logσ2), new_model_state = \
            CVAE(z_dim=z_dim, expr_dim=X.shape[1], training=True).apply(
                vars, key, G, mutable=["batch_stats"])

        neg_elbo = -elbo(X, Xsample, μ, logσ2)

        return neg_elbo, new_model_state

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (neg_elbo, new_model_state), grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)

    return optimizer, new_model_state, -neg_elbo


def fit_cvae(
        adata: anndata.AnnData,
        labels,
        output_filename,
        nepochs: int=5000,
        avg_neighborhood_size: int=10,
        seed: int=0,
        z_dim: int=10):

    ncells, ngenes = adata.shape
    radius = calibrate_neighborhood_radius(adata, avg_neighborhood_size)
    sq.gr.spatial_neighbors(adata, radius=radius, coord_type="generic")

    A = adata.obsp["spatial_connectivities"].tocoo()
    # A = (A + A.transpose()).tocoo() # symmetrize

    nedges = A.getnnz()

    X = jnp.array(
        zscore(adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()),
        dtype=jnp.float32)

    labels = jax.nn.one_hot(labels, np.max(labels)+1)

    # labeled neighborhood graph
    G = GraphsTuple(
        n_node=jnp.array([ncells], dtype=jnp.int32),
        n_edge=jnp.array([nedges], dtype=jnp.int32),
        senders=jnp.array(A.row, dtype=jnp.int32),
        receivers=jnp.array(A.col, dtype=jnp.int32),
        nodes=labels,
        edges=None,
        globals=None)

    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)

    vars = CVAE(training=True, z_dim=z_dim, expr_dim=ngenes).init(
        init_key, key, G)

    model_state, params = vars.pop("params")

    optimizer = flax.optim.Adam(1e-2).create(params)
    optimizer = jax.device_put(optimizer)

    for epoch in range(nepochs):
        key, train_key = jax.random.split(key)

        optimizer, model_state, elbo = train_step(
            z_dim, optimizer, model_state, train_key, G, X)

        if epoch % 100 == 0:
            print(f"epoch: {epoch}, elbo: {elbo}")

    vars = {"params": optimizer.target, **model_state}

    if output_filename is not None:
        with open(output_filename, "wb") as output:
            pickle.dump(
                {
                    "ngenes": ngenes,
                    "z_dim": z_dim,
                    "vars": vars
                },
                output, pickle.HIGHEST_PROTOCOL)


def sample_cvae(
        adata: anndata.AnnData, labels, params_filename, output_filename,
        avg_neighborhood_size: int=10, seed: int=0):
    with open(params_filename, "rb") as input:
        modeldata = jax.device_put(pickle.load(input))

    vars = modeldata["vars"]
    z_dim = modeldata["z_dim"]
    ngenes = modeldata["ngenes"]

    ncells, _ = adata.shape
    radius = calibrate_neighborhood_radius(adata, avg_neighborhood_size)
    sq.gr.spatial_neighbors(adata, radius=radius, coord_type="generic")

    A = adata.obsp["spatial_connectivities"].tocoo()
    # A = (A + A.transpose()).tocoo() # symmetrize
    nedges = A.getnnz()

    labels = jax.nn.one_hot(labels, np.max(labels)+1)

    # labeled neighborhood graph
    G = GraphsTuple(
        n_node=jnp.array([ncells], dtype=jnp.int32),
        n_edge=jnp.array([nedges], dtype=jnp.int32),
        senders=jnp.array(A.row, dtype=jnp.int32),
        receivers=jnp.array(A.col, dtype=jnp.int32),
        nodes=labels,
        edges=None,
        globals=None)

    key = jax.random.PRNGKey(seed)
    X, μ, logσ2 = CVAE(z_dim=z_dim, expr_dim=ngenes, training=False).apply(vars, key, G)

    adata = anndata.AnnData(
        X=np.array(X, dtype=np.float32),
        obs=adata.obs,
        obsm=adata.obsm)

    adata.write_h5ad(output_filename)
