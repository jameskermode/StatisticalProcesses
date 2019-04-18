module StatisticalProcesses

"""
Implementation of Gaussian and Student's T statistical processes for regression

Based on https://arxiv.org/abs/1801.06147 and http://proceedings.mlr.press/v33/shah14.pdf
"""

using LinearAlgebra
using Distributions
using ForwardDiff
using Optim
using SpecialFunctions

export GP, STP, predict, prior, posterior, loglikelihood, optimize_hypers

abstract type AbstractProcess end

struct GP <: AbstractProcess
    n::Integer
    x::AbstractArray
    y::AbstractArray
    K::AbstractMatrix
    C::AbstractMatrix
    invCy::AbstractArray
    k::Function
    hypers
    kernel_hypers
end

struct STP <: AbstractProcess
    n::Integer
    x::AbstractArray
    y::AbstractArray
    K::AbstractMatrix
    C::AbstractMatrix
    invCy::AbstractArray
    k::Function
    hypers
    kernel_hypers
end

Base.Broadcast.broadcastable(gp::GP) = Ref(gp)
Base.Broadcast.broadcastable(stp::STP) = Ref(stp)

function extract_kernel_hypers(k, hypers)
    ml = methods(k)
    kernel_hyper_names = Base.kwarg_decl(ml.ms[1], typeof(ml.mt.kwsorter))
    hypers = Dict(hypers)
    kernel_hypers = Dict(s => hypers[s] for s in kernel_hyper_names)
    return hypers, kernel_hypers
end

function GP(x, y, k; hypers...)
    n = length(x)
    hypers, kernel_hypers = extract_kernel_hypers(k, hypers)
    K = [ k(x[i], x[j]; kernel_hypers...) for i=1:n, j=1:n]
    σn = hypers[:σn]
    C = K + σn^2*I
    invCy = C \ y
    return GP(n, x, y, K, C, invCy, k, hypers, kernel_hypers)
end

function STP(x, y, k; hypers...)
    n = length(x)
    hypers, kernel_hypers = extract_kernel_hypers(k, hypers)
    K = [ k(x[i], x[j]; kernel_hypers...) for i=1:n, j=1:n]
    ν = hypers[:ν]
    @assert ν > 2
    σn = hypers[:σn]
    C = K + σn^2*I
    invCy = C \ y
    return STP(n, x, y, K, C, invCy, k, hypers, kernel_hypers)
end

# convert an STP to the "equivalent" GP - same kernel, hypers and training data
GP(stp::STP) = GP(stp.n, stp.x, stp.y, stp.K, stp.C,
                  stp.invCy, stp.k, stp.hypers, stp.kernel_hypers)


function predict(gp::AbstractProcess, xp::Number)
    Kxxp = [gp.k(gp.x[i], xp; gp.kernel_hypers...) for i=1:gp.n]
    κ = gp.k(xp, xp; gp.kernel_hypers...)
    μ = dot(Kxxp, gp.invCy)
    v = κ - dot(Kxxp, gp.C \ Kxxp)
    return (μ, v)
end

function predict(stp::STP, xp::Number)
    μ, v = invoke(predict, Tuple{AbstractProcess, Number}, stp, xp)
    ν = stp.hypers[:ν]
    scale = (ν + dot(stp.y, stp.invCy) - 2) / (ν + stp.n - 2)
    return (μ, scale * v)
end

function predict(gp::AbstractProcess, xp::AbstractArray)
    μv = predict.(gp, xp)
    μ, v = first.(μv), last.(μv)
    return μ, v
end

(gp::GP)(x) = predict(gp, x)

(stp::STP)(x) = predict(stp, x)

function prior(gp::GP, xp::AbstractArray)
    n = length(xp)
    K = [gp.k(xp[i], xp[j]; gp.kernel_hypers...) for i=1:n, j=1:n]
    σn = gp.hypers[:σn]
    return MultivariateNormal(K + σn^2*I)
end

function prior(stp::STP, xp::AbstractArray)
    n = length(xp)
    K_ss = [gp.k(xp[i], xp[j]; gp.kernel_hypers...) for i=1:n, j=1:n]
    ν = stp.hypers[:ν]
    Σ = (ν-2)/ν * (K_ss + stp.hypers[:σn]^2*I)
    return MvTDist(stp.ν, Σ)
end

function mean_and_covariance(gp::AbstractProcess, xp::AbstractArray)
    n = length(xp)
    K_ss = [gp.k(xp[i],   xp[j]; gp.kernel_hypers...) for i=1:n, j=1:n]
    K_s  = [gp.k(gp.x[i], xp[j]; gp.kernel_hypers...) for i=1:gp.n, j=1:n]

    c = cholesky(gp.C)
    Lk = c.L \ K_s
    μ = Lk' * (c.L \ gp.y)

    σn = gp.hypers[:σn]
    Σ = K_ss + σn^2*I - Lk' * Lk
    return μ, Σ
end

posterior(gp::GP, xp::AbstractArray) = MultivariateNormal(mean_and_covariance(gp, xp)...)

function posterior(stp::STP, xp::AbstractArray)
    μ, Σ = mean_and_covariance(stp, xp)
    ν = stp.hypers[:ν]
    scale = (ν + dot(stp.y, stp.invCy) - 2) / (ν + stp.n - 2)
    Σ = (ν-2)/ν * scale * (K_ss + stp.hypers[:σn]^2*I - Lk' * Lk)
    return MvTDist(ν, μ, Σ)
end

import Distributions: loglikelihood

loglikelihood(gp::GP) = -1/2*gp.y'*gp.invCy - 1/2*logdet(gp.C) - gp.n/2*log(2π)

function _old_loglikelihood(stp::STP)
    ν = stp.hypers[:ν]
    return (lgamma((ν + stp.n)/2) - lgamma(ν/2) - stp.n/2 * log(ν * π)
                 - 1/2*logdet((ν - 2)/ν * stp.C) - (ν + stp.n)/2*log(1 + (ν - 2)/ν * stp.y'*stp.invCy) / ν)
end

function loglikelihood(stp::STP)
    ν = stp.hypers[:ν]
    n = stp.n
    β = stp.y' * stp.invCy
    return ( -n/2*log((ν - 2)*π) - 1/2*logdet(stp.C) +
              log(gamma((ν+n)/2) / gamma(ν/2)) - (ν+n)/2*log(1 + β/(ν-2)) )
end

function optimize_hypers(gp::AbstractProcess; fix=[], initial=nothing, lower=nothing, upper=nothing)
    T = typeof(gp)
    variable_hypers = [p for p in keys(gp.hypers) if !(p in fix) ]
    @show variable_hypers

    if initial == nothing
        initial = [gp.hypers[h] for h in variable_hypers]
    end
    if lower == nothing
        lower = [1e-3 for h in variable_hypers]
    end
    if upper == nothing
        upper = [Inf for h in variable_hypers]
    end
    @show initial
    @show lower
    @show upper

    hypers = merge(gp.hypers,
                   Dict(k=>v for (k, v) in zip(variable_hypers, initial)))
    println("Initial: ", hypers)

    function L(p)
        hypers = merge(gp.hypers,
                       Dict(k=>v for (k, v) in zip(variable_hypers, p)))
        return -loglikelihood(T(gp.x, gp.y, gp.k; hypers...))
    end

    optfunc = OnceDifferentiable(L, initial; autodiff=:forward)
    res = optimize(optfunc, lower, upper, initial, Fminbox())
    @show res
    hypers = merge(gp.hypers,
                   Dict(k=>v for (k, v) in zip(variable_hypers, res.minimizer)))
    println("Optimised: ", hypers)

    opt_gp = T(gp.x, gp.y, gp.k; hypers...)
    @show loglikelihood(opt_gp)

    h = ForwardDiff.hessian(L, res.minimizer)
    L_laplace = -L(res.minimizer)*sqrt(((2π)^length(res.minimizer)) / det(h))
    println("Marginal likelihood (Laplace approximation): ", L_laplace)

    return hypers, (L, res, L_laplace), opt_gp
end


end # module
