"""
    LocalSmoothers

A small, dependency-light collection of **local smoothers** for nonparametric
regression, designed to be fast, type-stable and allocation-friendly.

# Smoothers

- [`LAS`](@ref): local average smoothing (moving average with optional window)
- [`LASb`](@ref): local average smoothing with boundary adjustment
- [`LLSS`](@ref): local linear smoothing
- [`LLSSb`](@ref): local linear smoothing with boundary adjustment
- [`FRSS`](@ref): multiscale smoothing via repeated local smoothing and
  leave-one-out cross-validation over a set of candidate scales
- [`Kernelsmooth`](@ref): regularized kernel interpolation (RBF regression)
- [`NWKernelsmooth`](@ref): Nadaraya-Watson kernel smoothing

Kernel types are provided by
[SimpleKernelRegression](https://github.com/NilsWildt/SimpleKernelregression.jl)
(`Gaussian`, `Imq`, `Mq`, `Polynomial`, `Linear`, `Epanechnikov`, `Wendland`)
and are re-exported here for convenience.

All `do_smoothing` methods accept generic `AbstractVector{<:Real}` inputs
(including `SubArray` views), so they work seamlessly inside tight loops that
pass views.
"""
module LocalSmoothers

using DispatchDoctor
using Statistics
using PrecompileTools

import SimpleKernelRegression
import SimpleKernelRegression:
    SKernel, Gaussian, Imq, Mq, Polynomial, Linear, Epanechnikov, Wendland,
    evalKmatrix, evalKernel, get_kernel_interpolant

export Smoother, LAS, LASb, LLSS, LLSSb, FRSS, Kernelsmooth, NWKernelsmooth
export do_smoothing, do_smoothing!, loocv, validate_inputs
# kernels (re-exported from SimpleKernelRegression for ergonomics)
export SKernel, Gaussian, Imq, Mq, Polynomial, Linear, Epanechnikov, Wendland
export evalKmatrix, evalKernel, get_kernel_interpolant

include("smoothers.jl")
include("frss.jl")
include("kernelsmoothing.jl")
include("chain.jl")

@compile_workload begin
    x = collect(range(0.0, 1.0, length=100))
    y = sin.(4 .* x) .+ 0.1 .* randn(100)
    for sm in (LAS(10), LASb(10), LLSS(10), LLSSb(10))
        do_smoothing(x, y, sm)
    end
    do_smoothing(x, y, FRSS([0.05, 0.1, 0.5], 0.2, 0.2))
    do_smoothing(x, y, NWKernelsmooth(Gaussian(0.2)))
    do_smoothing(x, y, Kernelsmooth(Gaussian(0.2)))
end

end # module
