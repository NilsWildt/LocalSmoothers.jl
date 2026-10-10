using LocalSmoothers
using Test

@testitem "validate_inputs" begin
    using Statistics
    x = collect(range(0.0, 1.0, length=101))
    y = sin.(x)
    @test LocalSmoothers.validate_inputs(x, y, 21) == 10
    @test LocalSmoothers.validate_inputs(x, y, 20) == 10   # even -> odd
    @test_throws DimensionMismatch LocalSmoothers.validate_inputs(x, y[1:50], 21)
    @test_throws ArgumentError LocalSmoothers.validate_inputs(x, y, 0)
    @test_throws ArgumentError LocalSmoothers.validate_inputs(x, y, 200)  # window >= n
end

@testitem "LAS basics" begin
    n = 200
    x = collect(range(0.0, 2.0, length=n))
    y = sin.(3 .* x) .+ 0.01 .* randn(n)

    out = do_smoothing(x, y, LAS(21))
    @test length(out) == n
    @test all(isfinite, out)

    # constant data is reproduced exactly
    c = fill(3.7, n)
    @test do_smoothing(x, c, LAS(21)) ≈ c
end

@testitem "LASb reproduces linear data in the interior" begin
    n = 200
    x = collect(range(-1.0, 1.0, length=n))
    y = 2.0 .* x .+ 1.0
    out = do_smoothing(x, y, LASb(21))
    @test all(isfinite, out)
    interior = 30:(n - 30)
    @test out[interior] ≈ y[interior] atol = 1e-12
    @test all(abs.(out[1:10] .- y[1:10]) .< 1.0)
end

@testitem "LLSS reproduces linear data exactly" begin
    n = 200
    x = collect(range(-1.0, 1.0, length=n))
    y = 2.0 .* x .+ 1.0
    for sm in (LLSS(15), LLSSb(15))
        out = do_smoothing(x, y, sm)
        @test out ≈ y atol = 1e-8
    end
end

@testitem "smoothers smooth noisy data" begin
    using Statistics
    n = 500
    x = collect(range(0.0, 4.0, length=n))
    f = x -> sin.(x)
    y = f.(x) .+ 0.2 .* randn(n)
    for sm in (LAS(41), LASb(41), LLSS(41), LLSSb(41))
        out = do_smoothing(x, y, sm)
        rmse = sqrt(mean((out .- f.(x)) .^ 2))
        @test rmse < 0.1
    end
end

@testitem "views and Float32 inputs work" begin
    n = 100
    x = collect(range(0.0, 1.0, length=n))
    y = exp.(x)
    out = do_smoothing(x, y, LAS(11))
    @test eltype(out) == Float64

    idx = 1:2:n
    outv = do_smoothing(@view(x[idx]), @view(y[idx]), LAS(7))
    outd = do_smoothing(x[idx], y[idx], LAS(7))
    @test outv ≈ outd

    x32 = Float32.(x)
    y32 = Float32.(y)
    out32 = do_smoothing(x32, y32, LAS(11))
    @test eltype(out32) == Float32
end

@testitem "in-place variants allocate nothing" begin
    n = 200
    x = collect(range(0.0, 1.0, length=n))
    y = sin.(6 .* x)
    out = similar(y)
    for sm in (LAS(21), LASb(21), LLSS(21), LLSSb(21))
        alloc = @allocated do_smoothing!(out, x, y, sm)
        @test alloc == 0
        @test out ≈ do_smoothing(x, y, sm)
    end
end

@testitem "do_smoothing allocates only the result" begin
    n = 200
    x = collect(range(0.0, 1.0, length=n))
    y = sin.(6 .* x)
    for sm in (LAS(21), LASb(21), LLSS(21), LLSSb(21))
        alloc = @allocated do_smoothing(x, y, sm)
        @test alloc <= 8 * n + 64   # one result vector of Float64
    end
end

@testitem "type stability (@inferred)" begin
    n = 100
    x = collect(range(0.0, 1.0, length=n))
    y = sin.(x)
    @inferred do_smoothing(x, y, LAS(11))
    @inferred do_smoothing(x, y, LASb(11))
    @inferred do_smoothing(x, y, LLSS(11))
    @inferred do_smoothing(x, y, LLSSb(11))
end

@testitem "input errors are thrown" begin
    x = collect(range(0.0, 1.0, length=50))
    y = sin.(x)
    @test_throws DimensionMismatch do_smoothing(x, y[1:40], LAS(5))
    @test_throws ArgumentError do_smoothing(x, y, LAS(100))  # window >= n
end

@testitem "boundary-adjusted smoothers keep the full window at the edges" begin
    n = 10
    x = collect(1.0:n)
    # LASb(5): 5-point windows, shifted inward at the edges -> 1:5 for i ≤ 3, 6:10 for i ≥ 8.
    @test do_smoothing(x, x, LASb(5)) ≈ [3, 3, 3, 4, 5, 6, 7, 8, 8, 8]
    y = x .^ 2
    linfit(r, xi) = ([ones(length(r)) x[r]] \ y[r])' * [1, xi]
    out = do_smoothing(x, y, LLSSb(5))
    @test out[1] ≈ linfit(1:5, x[1])
    @test out[2] ≈ linfit(1:5, x[2])
    @test out[10] ≈ linfit(6:10, x[10])
    @test do_smoothing!(similar(y), x, y, LASb(5)) ≈ do_smoothing(x, y, LASb(5))
    @test do_smoothing!(similar(y), x, y, LLSSb(5)) ≈ out
end

@testitem "loocv matches refitting without each point" begin
    n = 40
    x = collect(((1:n) ./ n) .^ 1.5)  # non-uniform spacing
    y = sin.(4 .* x) .+ 0.1 .* sin.(97 .* (1:n))
    k = 3  # window 7
    truncated(i) = max(1, i - k):min(n, i + k)
    shifted(i) = (lo = clamp(i - k, 1, n - 2k); lo:(lo + 2k))
    mean_fit(r, i) = sum(y[j] for j in r if j != i) / (length(r) - 1)
    function line_fit(r, i)
        o = [j for j in r if j != i]
        return ([ones(length(o)) x[o]] \ y[o])' * [1, x[i]]
    end
    for (sm, window, fit) in ((LAS(7), truncated, mean_fit), (LASb(7), shifted, mean_fit),
                              (LLSS(7), truncated, line_fit), (LLSSb(7), shifted, line_fit))
        cv, ys = loocv(x, y, sm)
        @test ys ≈ do_smoothing(x, y, sm)
        @test cv ≈ [abs(y[i] - fit(window(i), i)) for i in 1:n]
    end
    cv32, _ = loocv(Float32.(x), Float32.(y), LLSSb(7))
    @test eltype(cv32) == Float32
end

@testitem "loocv" begin
    n = 100
    x = collect(range(0.0, 1.0, length=n))
    y = sin.(x)
    for sm in (LAS(11), LASb(11), LLSS(11), LLSSb(11))
        cv, ys = loocv(x, y, sm)
        @test length(cv) == n
        @test length(ys) == n
        @test all(≥(0), cv)
        @test all(isfinite, cv)
    end
end
