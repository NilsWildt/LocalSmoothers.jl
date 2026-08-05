# Friedman's variable-span super smoother (supsmu).
#
# A running-lines smoother whose local bandwidth is chosen per point by
# cross-validation among three fixed spans, then interpolated. Port of the
# algorithm in acepack's supsmu.f90 / smooth.f90
# (https://github.com/vubiostat/acepack); the sliding-window statistics are kept
# with incremental add-right/remove-left running sums (O(n) per span) rather
# than recomputed per point.

"""
    Supsmu(; span = 0.0, bass = 0.0)

Friedman's super smoother. With `span = 0` the local bandwidth is selected per
point by cross-validation among three spans (0.05, 0.2, 0.5 of the data); a
fixed `span ∈ (0, 1]` uses that bandwidth everywhere. `bass ∈ [0, 10]` biases
the automatic selection toward larger spans (more smoothing).

`x` must be sorted ascending (as it is inside the ACE backfitting loop).
"""
struct Supsmu <: Smoother
    span::Float64
    bass::Float64
    Supsmu(; span::Real=0.0, bass::Real=0.0) = new(Float64(span), Float64(bass))
end

# The three candidate spans: tweeter (fine), midrange, woofer (coarse).
const _SUPSMU_SPANS = (0.05, 0.2, 0.5)

# Running-lines smoother at a fixed span. Returns (smoothed, |leave-one-out
# residual|). The window [j-ibw, j+ibw] slides monotonically, so each point is
# added once (on the right) and removed once (on the left): O(n) total.
function _running_line(x::AbstractVector{T}, y::AbstractVector{T}, span::Real) where {T<:Real}
    n = length(x)
    ibw = clamp(round(Int, 0.5 * span * n), 1, max(1, n - 1))
    smo = Vector{T}(undef, n)
    cvres = Vector{T}(undef, n)
    x̄ = sum(x) / n                       # centre x to condition the running sums
    Sx = Sy = Sxx = Sxy = zero(T)
    cnt = 0
    lo = 1
    hi = 0
    for j in 1:n
        nlo = max(1, j - ibw)
        nhi = min(n, j + ibw)
        while hi < nhi                    # add points entering on the right
            hi += 1
            xh = x[hi] - x̄
            Sx += xh; Sy += y[hi]; Sxx += xh * xh; Sxy += xh * y[hi]; cnt += 1
        end
        while lo < nlo                    # drop points leaving on the left
            xl = x[lo] - x̄
            Sx -= xl; Sy -= y[lo]; Sxx -= xl * xl; Sxy -= xl * y[lo]; cnt -= 1
            lo += 1
        end
        xm = Sx / cnt
        ym = Sy / cnt
        var = Sxx - Sx * xm               # Σ(x-x̄)² within the window, minus the mean part
        cvar = Sxy - Sx * ym
        a = var > 0 ? cvar / var : zero(T)
        dxj = (x[j] - x̄) - xm
        fit = a * dxj + ym
        smo[j] = fit
        h = var > 0 ? (one(T) / cnt + dxj * dxj / var) : one(T) / cnt
        denom = one(T) - h
        cvres[j] = abs(denom) > eps(T) ? abs(y[j] - fit) / denom : zero(T)
    end
    return smo, cvres
end

function do_smoothing(x::AbstractVector{T}, y::AbstractVector{T}, sm::Supsmu) where {T<:Real}
    n = length(x)
    length(y) == n || throw(DimensionMismatch("x and y must have the same length"))

    if sm.span > 0                        # fixed bandwidth
        return first(_running_line(x, y, sm.span))
    end

    # Smooth at each candidate span and denoise its CV-residual curve.
    fits = ntuple(i -> _running_line(x, y, _SUPSMU_SPANS[i]), 3)
    resid = ntuple(i -> first(_running_line(x, fits[i][2], _SUPSMU_SPANS[2])), 3)

    mid = _SUPSMU_SPANS[2]
    lo, wo = _SUPSMU_SPANS[1], _SUPSMU_SPANS[3]
    chosen = Vector{T}(undef, n)
    for j in 1:n
        best = 1
        rmin = resid[1][j]
        for i in 2:3
            if resid[i][j] < rmin
                rmin = resid[i][j]
                best = i
            end
        end
        s = _SUPSMU_SPANS[best]
        # Bass enhancement: nudge the span toward the woofer.
        if 0 < sm.bass ≤ 10 && resid[3][j] > 0 && rmin < resid[3][j]
            s += (wo - s) * max(1e-7, rmin / resid[3][j])^(10.0 - sm.bass)
        end
        chosen[j] = s
    end

    # Smooth the chosen-span curve so the bandwidth varies gently, then
    # interpolate the three fixed-span smooths at that span.
    chosen = first(_running_line(x, chosen, mid))
    result = Vector{T}(undef, n)
    for j in 1:n
        s = clamp(chosen[j], lo, wo)
        if s ≥ mid
            f = (s - mid) / (wo - mid)
            result[j] = (1 - f) * fits[2][1][j] + f * fits[3][1][j]
        else
            f = (mid - s) / (mid - lo)
            result[j] = (1 - f) * fits[2][1][j] + f * fits[1][1][j]
        end
    end
    return result
end

Base.show(io::IO, s::Supsmu) =
    print(io, s.span > 0 ? "Supsmu(span=$(s.span))" : "Supsmu(bass=$(s.bass))")
