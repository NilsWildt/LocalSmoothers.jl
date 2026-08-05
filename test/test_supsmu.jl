using LocalSmoothers
using Test

@testitem "supsmu running-line matches a brute-force windowed fit" begin
    using Statistics
    n = 80
    x = collect(range(0.0, 1.0, length=n))
    y = sin.(6 .* x) .+ 0.1 .* sin.(50 .* x)     # deterministic
    span = 0.3
    ibw = clamp(round(Int, 0.5 * span * n), 1, n - 1)

    ref = similar(y)
    for j in 1:n
        lo = max(1, j - ibw)
        hi = min(n, j + ibw)
        xs = @view x[lo:hi]
        ys = @view y[lo:hi]
        xm = mean(xs)
        ym = mean(ys)
        v = sum((xi - xm)^2 for xi in xs)
        c = sum((xs[k] - xm) * (ys[k] - ym) for k in eachindex(xs, ys))
        a = v > 0 ? c / v : 0.0
        ref[j] = a * (x[j] - xm) + ym
    end

    smo, cvres = LocalSmoothers._running_line(x, y, span)
    @test smo ≈ ref rtol = 1e-9          # incremental running sums equal the direct fit
    @test all(isfinite, cvres) && all(≥(0), cvres)
end

@testitem "Supsmu smooths, recovers signal, handles edge cases" begin
    using Statistics
    n = 200
    x = collect(range(0.0, 1.0, length=n))
    truth = sin.(2π .* x)
    y = truth .+ 0.25 .* sin.(60π .* x)          # high-frequency deterministic ripple

    ŷ = do_smoothing(x, y, Supsmu())             # variable (CV) span
    @test length(ŷ) == n && all(isfinite, ŷ)
    @test sum(abs2, diff(ŷ)) < sum(abs2, diff(y))   # less total variation than the input
    @test cor(ŷ, truth) > cor(y, truth)             # closer to the underlying signal

    ŷf = do_smoothing(x, y, Supsmu(span=0.3))    # fixed span
    @test length(ŷf) == n && all(isfinite, ŷf)

    @test do_smoothing(x, fill(3.0, n), Supsmu()) ≈ fill(3.0, n)  # constant → constant
    @test all(isfinite, do_smoothing(x, y, Supsmu(bass=5)))       # bass enhancement runs
    @test_throws DimensionMismatch do_smoothing(x, y[1:end-1], Supsmu())
end
