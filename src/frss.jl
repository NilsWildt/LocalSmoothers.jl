# FRSS: multiscale smoothing (Friedman-style local regression over a hierarchy
# of scales selected by leave-one-out cross-validation).
#
# Bugs fixed relative to the original ACE.jl implementation:
#   - `smooth_residuals` had an arity mismatch at its call site (5 args vs 4)
#   - fields were abstract (`AbstractArray`) causing DispatchDoctor type
#     instability; now `Vector{Float64}` / `Float64`
#   - `interpolate_smooth` hard-coded a third scale; now generalizes to any
#     number of candidate scales

"""
    FRSS(initial_Js, medium_J, final_J)

Multiscale smoother. `initial_Js` is a collection of candidate scales used for
the initial smoothing passes (as a fraction of the sample size); `medium_J` and
`final_J` are the scales for the medium and final smoothing passes.
"""
mutable struct FRSS <: Smoother
    initial_Js::Vector{Float64}
    medium_J::Float64
    final_J::Float64

    function FRSS(inJs, medJ, finalJ)
        Js = sort!(unique(Float64.(inJs isa AbstractArray ? vec(inJs) : [inJs])))
        length(Js) >= 1 || throw(ArgumentError("initial_Js must contain at least one scale"))
        return new(Js, Float64(medJ), Float64(finalJ))
    end
end

Base.String(::FRSS) = "Smoothed_FRSS"
Base.show(io::IO, s::FRSS) = print(io, "FRSS(initial_Js=", s.initial_Js, ", medium_J=", s.medium_J, ", final_J=", s.final_J, ")")

@stable function perform_initial_smoothing(x::AbstractVector{T}, y::AbstractVector{T}, smoother::FRSS, Nx::Int) where {T<:Real}
    Js = smoother.initial_Js
    J = length(Js)
    initial_Js_array = repeat(reshape(Js, 1, J), Nx, 1)
    initial_smooth = Matrix{Float64}(undef, Nx, J)
    initial_cv_residuals = Matrix{Float64}(undef, Nx, J)
    @inbounds for i in 1:J
        lin_smoother = LLSSb(Nx * Js[i])
        cv, ys = loocv(x, y, lin_smoother)
        initial_cv_residuals[:, i] = cv
        initial_smooth[:, i] = ys
    end
    return initial_Js_array, initial_smooth, initial_cv_residuals
end

@stable function smooth_residuals(x::AbstractVector{T}, residuals::AbstractMatrix{Float64}, smoother::FRSS) where {T<:Real}
    J = size(residuals, 2)
    out = similar(residuals)
    lin_smoother = LLSSb(smoother.medium_J * size(residuals, 1))
    for j in 1:J
        out[:, j] = do_smoothing(x, view(residuals, :, j), lin_smoother)
    end
    return out
end

@stable function interpolate_smooth(initial_smooth::AbstractMatrix{Float64},
                                    smoothed_best_Js::AbstractVector{Float64},
                                    smoother::FRSS, Nx::Int)
    Js = smoother.initial_Js
    J = length(Js)
    out = Vector{Float64}(undef, Nx)
    @inbounds for i in 1:Nx
        Jtarget = smoothed_best_Js[i]
        if J == 1
            out[i] = initial_smooth[i, 1]
            continue
        end
        # nearest candidate scale
        jn = 1
        dmin = abs(Jtarget - Js[1])
        for j in 2:J
            d = abs(Jtarget - Js[j])
            if d < dmin
                dmin = d
                jn = j
            end
        end
        # nearest *other* candidate scale (for linear interpolation)
        j2 = if jn == 1
            2
        elseif jn == J
            J - 1
        else
            abs(Jtarget - Js[jn - 1]) <= abs(Jtarget - Js[jn + 1]) ? jn - 1 : jn + 1
        end
        J1 = Js[jn]
        J2 = Js[j2]
        s1 = initial_smooth[i, jn]
        s2 = initial_smooth[i, j2]
        out[i] = J2 == J1 ? s1 : s1 + (s2 - s1) * (Jtarget - J1) / (J2 - J1)
    end
    return out
end

@stable function do_smoothing(x::AbstractVector{T}, y::AbstractVector{T}, smoother::FRSS) where {T<:Real}
    length(x) == length(y) || throw(DimensionMismatch("x and y must have the same length"))
    Nx = length(x)

    # Step 1: initial smoothing at each candidate scale + LOO residuals
    initial_Js_array, initial_smooth, initial_cv_residuals =
        perform_initial_smoothing(x, y, smoother, Nx)

    # Step 2: smooth the residual curves at the medium scale
    initial_residuals_smoothed = smooth_residuals(x, initial_cv_residuals, smoother)

    # Step 3: pick the best scale per point
    best_Js = Vector{Float64}(undef, Nx)
    @inbounds for i in 1:Nx
        jmin = argmin(view(initial_residuals_smoothed, i, :))
        best_Js[i] = initial_Js_array[i, jmin]
    end

    # Step 4: smooth the chosen scales and clamp to the candidate range
    lin_smoother = LLSSb(smoother.medium_J * Nx)
    smoothed_best_Js = do_smoothing(x, best_Js, lin_smoother)
    clamp!(smoothed_best_Js, smoother.initial_Js[1], smoother.initial_Js[end])

    # Step 5: interpolate the initial smooths at the smoothed scales
    interpolated_smooth = interpolate_smooth(initial_smooth, smoothed_best_Js, smoother, Nx)

    # Step 6: final smoothing pass
    lin_smoother_final = LLSSb(smoother.final_J * Nx)
    return do_smoothing(x, interpolated_smooth, lin_smoother_final)
end
