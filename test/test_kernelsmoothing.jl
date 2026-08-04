using LocalSmoothers
using Test

@testitem "kernel constructors and String" begin
    g = Gaussian(0.5)
    @test g isa LocalSmoothers.SKernel
    @test String(NWKernelsmooth(g)) == "Smoothed_NW_k_gaussian(0.5)"
    @test occursin("Smoothed_Mercer", String(Kernelsmooth(g, 1e-6)))
    @test Kernelsmooth(g).reg == 1e-6
end

@testitem "NWKernelsmooth approximates the data for small bandwidth" begin
    using Statistics
    n = 200
    x = collect(range(0.0, 1.0, length=n))
    f = x -> sin.(2 .* pi .* x)
    y = f.(x)
    sm = NWKernelsmooth(Gaussian(0.002))  # bandwidth ~ grid spacing -> nearly interpolating
    out = do_smoothing(x, y, sm)
    @test length(out) == n
    @test all(isfinite, out)
    @test maximum(abs.(out .- y)) < 1e-2
end

@testitem "NWKernelsmooth matches a hand-computed value" begin
    x = [0.0, 0.5, 1.0]
    y = [0.0, 1.0, 0.0]
    σ = 0.3
    sm = NWKernelsmooth(Gaussian(σ))
    W = evalKmatrix(sm.smoothk, reshape(x, :, 1), reshape(x, :, 1))
    expected1 = sum(W[1, :] .* y) / sum(W[1, :])
    out = do_smoothing(x, y, sm)
    @test out[1] ≈ expected1 rtol = 1e-10
end

@testitem "NWKernelsmooth works with different kernels and views" begin
    n = 120
    x = collect(range(-1.0, 1.0, length=n))
    y = exp.(-x .^ 2) .+ 0.01 .* randn(n)
    for k in (Gaussian(0.2), Imq(0.5), Wendland(0.5, 2, 2), Epanechnikov(0.5))
        out = do_smoothing(x, y, NWKernelsmooth(k))
        @test all(isfinite, out)
        @test length(out) == n
    end
    idx = 1:2:n
    outv = do_smoothing(@view(x[idx]), @view(y[idx]), NWKernelsmooth(Gaussian(0.2)))
    outd = do_smoothing(x[idx], y[idx], NWKernelsmooth(Gaussian(0.2)))
    @test outv ≈ outd
end

@testitem "Kernelsmooth interpolates data points" begin
    n = 60
    x = collect(range(0.0, 1.0, length=n))
    y = sin.(3 .* x)
    sm = Kernelsmooth(Gaussian(0.1), 1e-10)
    out = do_smoothing(x, y, sm)
    @test length(out) == n
    @test all(isfinite, out)
    @test out ≈ y atol = 1e-3
end

@testitem "do_smoothing! for kernel smoothers matches allocating version" begin
    n = 100
    x = collect(range(0.0, 1.0, length=n))
    y = sin.(x)
    for sm in (Kernelsmooth(Gaussian(0.2), 1e-6), NWKernelsmooth(Gaussian(0.2)))
        out = similar(y)
        do_smoothing!(out, x, y, sm)
        @test out ≈ do_smoothing(x, y, sm)
    end
end

@testitem "kernel smoother type stability" begin
    n = 80
    x = collect(range(0.0, 1.0, length=n))
    y = sin.(x)
    @inferred do_smoothing(x, y, NWKernelsmooth(Gaussian(0.2)))
end
