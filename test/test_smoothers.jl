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
