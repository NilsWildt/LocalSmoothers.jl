# LocalSmoothers.jl

![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![CI](https://github.com/NilsWildt/LocalSmoothers.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/NilsWildt/LocalSmoothers.jl/actions/workflows/CI.yml)

A small, dependency-light collection of **local smoothers** for nonparametric
regression, see [ACE](https://github.com/NilsWildt/AlternatingConditionalExpectations.jl).

## Smoothers

| Type | Description |
|---|---|
| `LAS` / `LASb` | Local average smoothing (moving average), plain and boundary-corrected |
| `LLSS` / `LLSSb` | Local linear smoothing, plain and boundary-corrected |
| `FRSS` | Multiscale smoothing: repeated local smoothing + leave-one-out CV over candidate scales |
| `Kernelsmooth` | Regularized kernel interpolation (RBF regression) |
| `NWKernelsmooth` | Nadaraya–Watson kernel smoothing |

Kernel types (`Gaussian`, `Imq`, `Mq`, `Polynomial`, `Linear`, `Epanechnikov`,
`Wendland`) come from
[SimpleKernelRegression.jl](https://github.com/NilsWildt/SimpleKernelregression.jl)
and are re-exported here.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/NilsWildt/SimpleKernelregression.jl")
Pkg.add(url = "https://github.com/NilsWildt/LocalSmoothers.jl")
```

## Usage

```julia
using LocalSmoothers

x = collect(range(0, 1; length = 200))
y = sin.(4x) .+ 0.1 .* randn(200)

ŷ = do_smoothing(x, y, LASb(10))            # boundary-corrected local average
ŷ = do_smoothing(x, y, NWKernelsmooth(Gaussian(0.2)))
```

`do_smoothing` accepts any `AbstractVector{<:Real}` (including `SubArray`
views). The in-place `do_smoothing!(out, x, y, smoother)` allocates nothing for
the local-average and local-linear smoothers.

## Development

```bash
git clone https://github.com/NilsWildt/LocalSmoothers.jl
cd LocalSmoothers.jl
julia --project=. -e 'import Pkg; Pkg.test()'
```

## License

MIT — see [LICENSE.md](LICENSE.md).
