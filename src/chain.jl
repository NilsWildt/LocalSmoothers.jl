# Chaining of smoothers + convenience dispatch.

"""
    do_smoothing(x, y, smoothers::AbstractVector{<:Smoother})

Apply a sequence of smoothers in order: each smoother operates on the output of
the previous one.
"""
function do_smoothing(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, smoothers::AbstractVector{<:Smoother})
    out = y
    for sm in smoothers
        out = do_smoothing(x, out, sm)
    end
    return out
end

"""
    do_smoothing(x, y, smoothers::Vararg{Smoother})

Apply a sequence of smoothers (variadic form).
"""
do_smoothing(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, sm1::Smoother, sm2::Smoother, sms::Smoother...) =
    do_smoothing(x, y, Smoother[sm1, sm2, sms...])

"""
    do_smoothing!(out, x, y, smoothers::AbstractVector{<:Smoother})

In-place variant: writes the result of the chained smoothing into `out`.
"""
function do_smoothing!(out::AbstractVector{<:Real}, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, smoothers::AbstractVector{<:Smoother})
    tmp = y
    for sm in smoothers
        do_smoothing!(out, x, tmp, sm)
        tmp = copy(out)
    end
    return out
end
