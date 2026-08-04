# Kernel-based smoothers built on SimpleKernelRegression.
#
# Both smoothers are parametric in the kernel type `K <: SKernel` so the kernel
# field is concrete (type-stable under `@stable`).

"""
    Kernelsmooth(kernel::SKernel, reg::Real = 1e-6)

Regularized kernel interpolation smoother: solves `(K + reg·I) α = y` for the
kernel matrix `K` and evaluates the interpolant on the input points.

# Example
```julia
sm = Kernelsmooth(Gaussian(0.5), 1e-6)
ys = do_smoothing(x, y, sm)
```
"""
mutable struct Kernelsmooth{K<:SKernel} <: Smoother
    reg::Float64
    smoothk::K
end
Kernelsmooth(smoothk::SKernel, reg::Real = 1.0e-6) = Kernelsmooth{typeof(smoothk)}(Float64(reg), smoothk)

Base.String(ks::Kernelsmooth) = try
    string("Smoothed_Mercer_", String(ks.smoothk), "($(ks.smoothk.σ))")
catch
    string("Smoothed_Mercer_", String(ks.smoothk))
end
Base.show(io::IO, ks::Kernelsmooth) = print(io, "Kernelsmooth(", ks.smoothk, ", reg=", ks.reg, ")")

function do_smoothing(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, smoother::Kernelsmooth)
    length(x) == length(y) || throw(DimensionMismatch("x and y must have the same length"))
    Nx = length(x)
    xm = reshape(float.(x), Nx, 1)
    ym = reshape(float.(y), Nx, 1)
    xeval = reshape(collect(range(x[1], x[end], length=Nx)), Nx, 1)
    interp = get_kernel_interpolant(xm, ym, smoother.smoothk; reg=smoother.reg)
    return vec(interp(xeval))
end

function do_smoothing!(out::AbstractVector{<:Real}, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, smoother::Kernelsmooth)
    length(out) == length(x) || throw(DimensionMismatch("out must have the same length as x"))
    out .= do_smoothing(x, y, smoother)
    return out
end

"""
    NWKernelsmooth(kernel::SKernel)

Nadaraya-Watson kernel smoother: `ŷ(xᵢ) = Σⱼ K(xᵢ,xⱼ)yⱼ / Σⱼ K(xᵢ,xⱼ)` computed
via the (symmetrized, mean-normalized) kernel matrix.
"""
mutable struct NWKernelsmooth{K<:SKernel} <: Smoother
    smoothk::K
end

Base.String(ks::NWKernelsmooth) = try
    string("Smoothed_NW_", String(ks.smoothk), "($(round(ks.smoothk.σ; digits=4)))")
catch
    string("Smoothed_NW_", String(ks.smoothk))
end
Base.show(io::IO, ks::NWKernelsmooth) = print(io, "NWKernelsmooth(", ks.smoothk, ")")

@stable function do_smoothing(x::AbstractVector{T}, y::AbstractVector{T}, smoother::NWKernelsmooth) where {T<:Real}
    N = length(x)
    W = evalKmatrix(smoother.smoothk, reshape(x, :, 1), reshape(x, :, 1))
    colmean = mean(W; dims=2)
    Wn = W ./ colmean
    return vec((Wn * y) ./ N)
end

@stable function do_smoothing!(out::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}, smoother::NWKernelsmooth) where {T<:Real}
    N = length(x)
    length(out) == N || throw(DimensionMismatch("out must have the same length as x"))
    W = evalKmatrix(smoother.smoothk, reshape(x, :, 1), reshape(x, :, 1))
    colmean = mean(W; dims=2)
    Wn = W ./ colmean
    out .= vec((Wn * y) ./ N)
    return out
end
