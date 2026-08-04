# Windowed local smoothers: LAS, LASb, LLSS, LLSSb + leave-one-out CV.
#
# All signatures are generic over `AbstractVector{T}` with `T <: Real` (views
# included) and the hot loops avoid allocations: `do_smoothing` allocates only
# the result vector, `do_smoothing!` allocates nothing.

"""
    Smoother

Abstract supertype for all smoothing algorithms.
"""
abstract type Smoother end

"""
    validate_inputs(x, y, window::Integer) -> Int

Validate inputs and return the half-window `k` (i.e. `(window - 1) ÷ 2`),
forcing the window to be odd.
"""
@stable function validate_inputs(x::AbstractArray, y::AbstractArray, window::Integer)
    length(x) == length(y) || throw(DimensionMismatch("x and y must have the same length"))
    window > 0 || throw(ArgumentError("Window size must be positive"))
    window < length(x) || throw(ArgumentError("Window size must be smaller than the data length"))
    w = iseven(window) ? window + 1 : window
    return Int((w - 1) ÷ 2)
end

"""
    LAS(window::Integer)

Local Average Smoothing: `result[i]` is the mean of `y` over the window of
`2k + 1` points around `i` (truncated at the boundaries).
"""
struct LAS <: Smoother
    window::Int
    LAS(window::Number) = new(Int(round(window)))
end

"""
    LASb(window::Integer)

Local Average Smoothing with boundary adjustment: at the edges the window is
shifted (rather than truncated) so that every point is averaged over the full
`2k + 1` window where possible.
"""
struct LASb <: Smoother
    window::Int
    LASb(window::Number) = new(Int(round(window)))
end

"""
    LLSS(window::Integer)

Local Linear Smoothing: fits a linear regression over each window and evaluates
it at the center point.
"""
struct LLSS <: Smoother
    window::Int
    LLSS(window::Number) = new(Int(round(window)))
end

"""
    LLSSb(window::Integer)

Local Linear Smoothing with boundary adjustment (window shifted at the edges).
"""
struct LLSSb <: Smoother
    window::Int
    LLSSb(window::Number) = new(Int(round(window)))
end

"""
    linear_regression(x, y, xi) -> T

Least-squares slope/intercept fit of `y` on `x`, evaluated at `xi`. Allocation
free (works on views).
"""
@inline @stable function linear_regression(x::AbstractVector{T}, y::AbstractVector{T}, xi::T) where {T<:Real}
    n = length(x)
    xm = zero(T)
    ym = zero(T)
    @inbounds for i in 1:n
        xm += x[i]
        ym += y[i]
    end
    xm /= n
    ym /= n
    c = zero(T)
    v = zero(T)
    @inbounds for i in 1:n
        dx = x[i] - xm
        c += dx * (y[i] - ym)
        v += dx * dx
    end
    β = iszero(v) ? zero(T) : c / v
    return ym - β * xm + β * xi
end

# Linear regression over the union of two index ranges (used by LLSSb).
@inline @stable function linear_regression_ranges(x::AbstractVector{T}, y::AbstractVector{T},
                                                   r1::UnitRange{Int}, r2::UnitRange{Int}, xi::T) where {T<:Real}
    n = length(r1) + length(r2)
    xm = zero(T)
    ym = zero(T)
    @inbounds for i in r1
        xm += x[i]; ym += y[i]
    end
    @inbounds for i in r2
        xm += x[i]; ym += y[i]
    end
    xm /= n
    ym /= n
    c = zero(T)
    v = zero(T)
    @inbounds for i in r1
        dx = x[i] - xm
        c += dx * (y[i] - ym)
        v += dx * dx
    end
    @inbounds for i in r2
        dx = x[i] - xm
        c += dx * (y[i] - ym)
        v += dx * dx
    end
    β = iszero(v) ? zero(T) : c / v
    return ym - β * xm + β * xi
end

@stable function do_smoothing(x::AbstractVector{T}, y::AbstractVector{T}, smoother::LAS) where {T<:Real}
    k = validate_inputs(x, y, smoother.window)
    n = length(y)
    result = Vector{T}(undef, n)
    @inbounds for i in 1:n
        lo = max(1, i - k)
        hi = min(n, i + k)
        s = zero(T)
        @simd for j in lo:hi
            s += y[j]
        end
        result[i] = s / (hi - lo + 1)
    end
    return result
end

@stable function do_smoothing!(out::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}, smoother::LAS) where {T<:Real}
    k = validate_inputs(x, y, smoother.window)
    n = length(y)
    length(out) == n || throw(DimensionMismatch("out must have the same length as y"))
    @inbounds for i in 1:n
        lo = max(1, i - k)
        hi = min(n, i + k)
        s = zero(T)
        @simd for j in lo:hi
            s += y[j]
        end
        out[i] = s / (hi - lo + 1)
    end
    return out
end

@stable function do_smoothing(x::AbstractVector{T}, y::AbstractVector{T}, smoother::LASb) where {T<:Real}
    k = validate_inputs(x, y, smoother.window)
    n = length(y)
    result = Vector{T}(undef, n)
    @inbounds for i in 1:n
        left_start = max(1, i - k) - min(0, n - i - k + 1)
        right_end = min(i + k, n) + min(0, i - k)
        s = zero(T)
        count = 0
        @simd for j in left_start:right_end
            s += y[j]
            count += 1
        end
        result[i] = s / count
    end
    return result
end

@stable function do_smoothing!(out::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}, smoother::LASb) where {T<:Real}
    k = validate_inputs(x, y, smoother.window)
    n = length(y)
    length(out) == n || throw(DimensionMismatch("out must have the same length as y"))
    @inbounds for i in 1:n
        left_start = max(1, i - k) - min(0, n - i - k + 1)
        right_end = min(i + k, n) + min(0, i - k)
        s = zero(T)
        count = 0
        @simd for j in left_start:right_end
            s += y[j]
            count += 1
        end
        out[i] = s / count
    end
    return out
end

@stable function do_smoothing(x::AbstractVector{T}, y::AbstractVector{T}, smoother::LLSS) where {T<:Real}
    k = validate_inputs(x, y, smoother.window)
    n = length(y)
    result = Vector{T}(undef, n)
    @inbounds for i in 1:n
        lo = max(1, i - k)
        hi = min(n, i + k)
        result[i] = linear_regression(@view(x[lo:hi]), @view(y[lo:hi]), x[i])
    end
    return result
end

@stable function do_smoothing!(out::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}, smoother::LLSS) where {T<:Real}
    k = validate_inputs(x, y, smoother.window)
    n = length(y)
    length(out) == n || throw(DimensionMismatch("out must have the same length as y"))
    @inbounds for i in 1:n
        lo = max(1, i - k)
        hi = min(n, i + k)
        out[i] = linear_regression(@view(x[lo:hi]), @view(y[lo:hi]), x[i])
    end
    return out
end

@stable function do_smoothing(x::AbstractVector{T}, y::AbstractVector{T}, smoother::LLSSb) where {T<:Real}
    k = validate_inputs(x, y, smoother.window)
    n = length(y)
    result = Vector{T}(undef, n)
    @inbounds for i in 1:n
        left = (max(1, i - k) - min(0, n - i - k + 1)):(i - 1)
        right = i:(min(i + k, n) + min(0, i - k))
        result[i] = linear_regression_ranges(x, y, left, right, x[i])
    end
    return result
end

@stable function do_smoothing!(out::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}, smoother::LLSSb) where {T<:Real}
    k = validate_inputs(x, y, smoother.window)
    n = length(y)
    length(out) == n || throw(DimensionMismatch("out must have the same length as y"))
    @inbounds for i in 1:n
        left = (max(1, i - k) - min(0, n - i - k + 1)):(i - 1)
        right = i:(min(i + k, n) + min(0, i - k))
        out[i] = linear_regression_ranges(x, y, left, right, x[i])
    end
    return out
end

"""
    loocv(x, y, smoother) -> (cv, ysmoothed)

Leave-one-out cross-validation for windowed smoothers
(`LAS`, `LASb`, `LLSS`, `LLSSb`). Returns the absolute LOO residuals and the
full smoothed curve.
"""
@stable function loocv(x::AbstractVector{T}, y::AbstractVector{T}, smoother::V) where {T<:Real, V<:Union{LAS,LASb,LLSS,LLSSb}}
    k = validate_inputs(x, y, smoother.window)
    n = length(x)
    ysmoothed = do_smoothing(x, y, smoother)
    cv = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        lo = max(1, i - k)
        hi = min(n, i + k)
        cnt = hi - lo + 1 - 1  # exclude i
        cnt > 0 || continue
        xm = zero(T)
        for j in lo:hi
            j == i && continue
            xm += x[j]
        end
        xm /= cnt
        v = zero(T)
        for j in lo:hi
            j == i && continue
            d = x[j] - xm
            v += d * d
        end
        v /= cnt
        denom = (1.0 - 1.0 / (2 * k)) - (v ≈ 0.0 ? 0.0 : (x[i] - xm) / v)
        cv[i] = abs(denom) < eps() ? 0.0 : (y[i] - ysmoothed[i]) / denom
    end
    return abs.(cv), ysmoothed
end

Base.show(io::IO, s::LAS) = print(io, "LAS($(s.window))")
Base.show(io::IO, s::LASb) = print(io, "LASb($(s.window))")
Base.show(io::IO, s::LLSS) = print(io, "LLSS($(s.window))")
Base.show(io::IO, s::LLSSb) = print(io, "LLSSb($(s.window))")
