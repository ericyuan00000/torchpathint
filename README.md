# torchpathint

Parallel adaptive quadrature for definite path integrals on PyTorch.

`torchpathint` evaluates definite integrals of the form

```
I = ∫_{t_init}^{t_final} f(t) dt
```

where `f` is a vector-valued function `f: ℝ → ℝ^D` (e.g. a neural network
along a path, an action integrand, a reward signal). The integrator is
*autograd-transparent*: it never breaks the computation graph, so you can
freely backpropagate through `I` to optimize `f` or the integration bounds.

This package is the successor to
[`torchpathdiffeq`](https://github.com/khegazy/torchpathdiffeq), rebuilt
around Gauss–Kronrod adaptive quadrature instead of Runge–Kutta ODE
integration.

## Status

Pre-alpha. API and internals are unstable. Not yet on PyPI.

## Installation (from source)

```bash
git clone <repo-url>
cd torchpathint
pip install -e .
```

Requires Python 3.10+ and PyTorch.
