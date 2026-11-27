"""Normalized MLPs and stacked ensembles shared by off-policy learners.

`Dense` is Linear → optional dropout → LayerNorm → activation (TD-MPC2's NormedLinear),
or a plain Linear when `norm=False`. An ensemble stores every member's weights with a
leading `e` axis and evaluates all members with one batched contraction.
"""

from typing import Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

Activation = Literal["mish", "relu", "simnorm", "none"]


def mish(x: Array) -> Array:
    return x * jnp.tanh(jax.nn.softplus(x))


def simnorm(x: Array, group: int) -> Array:
    """Softmax within consecutive groups: a product of simplices (TD-MPC2's latent)."""
    shape = x.shape
    return jax.nn.softmax(x.reshape(*shape[:-1], shape[-1] // group, group), -1).reshape(shape)


def _member(parameter: Array, x: Array, ensemble: bool) -> Array:
    """Broadcast `[e?, out]` parameters against `[e?, ..., out]` activations."""
    if not ensemble:
        return parameter
    return parameter.reshape(parameter.shape[0], *(1,) * (x.ndim - 2), parameter.shape[-1])


class Dense(eqx.Module):
    weight: Array  # [e?, out, in]
    bias: Array  # [e?, out]
    gain: Array | None
    offset: Array | None
    activation: Activation = eqx.field(static=True)
    dropout: float = eqx.field(static=True)
    group: int = eqx.field(static=True)

    def __init__(self, key: Array, size_in: int, size_out: int, *, ensemble: int | None = None,
                 norm: bool = True, activation: Activation = "mish", dropout: float = 0.0,
                 zero: bool = False, group: int = 8):
        lead = () if ensemble is None else (ensemble,)
        # Matches torch trunc_normal_(std=0.02): its ±2 bounds are absolute, i.e. 100σ.
        self.weight = jnp.zeros((*lead, size_out, size_in)) if zero else \
            0.02 * jax.random.normal(key, (*lead, size_out, size_in))
        self.bias = jnp.zeros((*lead, size_out))
        self.gain = jnp.ones((*lead, size_out)) if norm else None
        self.offset = jnp.zeros((*lead, size_out)) if norm else None
        self.activation = activation
        self.dropout = dropout
        self.group = group

    @property
    def ensemble(self) -> bool:
        return self.weight.ndim == 3

    def __call__(self, x: Array, key: Array | None = None) -> Array:
        if self.ensemble:
            x = einx.dot("e ... i, e o i -> e ... o", x, self.weight)  # pyright: ignore[reportPrivateImportUsage]
        else:
            x = einx.dot("... i, o i -> ... o", x, self.weight)  # pyright: ignore[reportPrivateImportUsage]
        x = x + _member(self.bias, x, self.ensemble)
        if self.dropout and key is not None:
            keep = jax.random.bernoulli(key, 1 - self.dropout, x.shape)
            x = jnp.where(keep, x / (1 - self.dropout), 0.0)
        if self.gain is not None and self.offset is not None:
            mean = jnp.mean(x, -1, keepdims=True)
            variance = jnp.var(x, -1, keepdims=True)
            x = (x - mean) * jax.lax.rsqrt(variance + 1e-5)
            x = x * _member(self.gain, x, self.ensemble) + _member(self.offset, x, self.ensemble)
        if self.activation == "mish":
            return mish(x)
        if self.activation == "relu":
            return jax.nn.relu(x)
        if self.activation == "simnorm":
            return simnorm(x, self.group)
        return x


class MLP(eqx.Module):
    layers: tuple[Dense, ...]

    def __init__(self, key: Array, size_in: int, hidden: tuple[int, ...], size_out: int, *,
                 output: Activation | None = None, ensemble: int | None = None, dropout: float = 0.0,
                 zero: bool = False, hidden_activation: Activation = "mish", group: int = 8):
        """`output=None` ends in a plain Linear; otherwise a normalized layer with that activation.
        Dropout applies to the first layer only; `zero` zero-initializes the final weights."""
        sizes = (size_in, *hidden, size_out)
        keys = jax.random.split(key, len(sizes) - 1)
        layers = [Dense(keys[i], sizes[i], sizes[i + 1], ensemble=ensemble, activation=hidden_activation,
                        dropout=dropout if i == 0 else 0.0, group=group) for i in range(len(hidden))]
        layers.append(Dense(keys[-1], sizes[-2], sizes[-1], ensemble=ensemble, norm=output is not None,
                            activation=output or "none", zero=zero, group=group))
        self.layers = tuple(layers)

    @property
    def members(self) -> int | None:
        weight = self.layers[0].weight
        return weight.shape[0] if weight.ndim == 3 else None

    def __call__(self, x: Array, key: Array | None = None) -> Array:
        """Ensembles broadcast a shared `[..., in]` input to `[e, ..., out]`."""
        if self.members is not None:
            x = jnp.broadcast_to(x, (self.members, *x.shape))
        for index, layer in enumerate(self.layers):
            x = layer(x, key if index == 0 else None)
        return x

    def select(self, members: Array) -> "MLP":
        """Sub-ensemble of the given member indices (computes only those members)."""
        return jax.tree.map(lambda leaf: leaf[members], self)
