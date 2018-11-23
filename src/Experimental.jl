"""
`module Experimental`

TODO: write documentation
"""
module Experimental

using JuLIP: AbstractAtoms, dofs, set_dofs!, atomdofs
using JuLIP.Potentials
using ForwardDiff

export newton!, constrained_bond_newton!

function newton(x, g, h; maxsteps=10, tol=1e-5)
    for i in 1:maxsteps
        gi = g(x)
        norm(gi, Inf) < tol && break
        x = x - h(x) \ gi
        @show i, norm(gi, Inf)
    end
    return x
end

function newton!(a::AbstractAtoms; maxsteps=20, tol=1e-10)
    x = newton(dofs(a),
               x -> gradient(a, x),
               x -> hessian(a, x))
    set_dofs!(a, x)
    return x
end

function constrained_bond_newton!(a::AbstractAtoms, i::Integer, j::Integer, bondlength::Float64;
                                  maxsteps=20, tol=1e-10)
    I1 = atomdofs(a, i)
    I2 = atomdofs(a, j)
    I1I2 = [I1; I2]

    # bondlength of target bond
    blen(x) = norm(x[I2] - x[I1])

    # constraint function
    c(x) = blen(x) - bondlength

    # gradient of constraint
    function dc(x)
        r = zeros(length(x))
        r[I1] = (x[I1]-x[I2])/blen(x)
        r[I2] = (x[I2]-x[I1])/blen(x)
        return r
    end

    # define some auxiliary index arrays that start at 1 to
    # allow the ForwardDiff hessian to be done only on relevent dofs, x[I1I2]
    _I1 = 1:length(I1)
    _I2 = length(I1)+1:length(I1)+length(I2)
    _blen(x) = norm(x[_I2] - x[_I1])
    _c(x) = _blen(x) - bondlength

    function ddc(x)
        s = spzeros(length(x), length(x))
        s[I1I2, I1I2] = ForwardDiff.hessian(_c, x[I1I2])
        return s
    end

    # Define Lagrangian L(x, λ) = E - λC and its gradient and hessian
    L(x, λ) = energy(a, x) - λ*c(x)
    L(z) = L(z[1:end-1], z[end])

    dL(x, λ) = [gradient(a, x) - λ * dc(x); -c(x)]
    dL(z) = dL(z[1:end-1], z[end])

    ddL(x, λ) = [(hessian(a, x) - λ * ddc(x)) -dc(x); -dc(x)' 0.]
    ddL(z) = ddL(z[1:end-1], z[end])

    # Use Newton scheme to find saddles L where ∇L = 0
    z = [dofs(a); -10.0*c(dofs(a))]
    z = newton(z, dL, ddL, maxsteps=maxsteps, tol=tol)
    set_dofs!(a, z[1:end-1])
    return z
end


# ========================================================================
#    A Prototype for a Very Fast Pair Potential Calculator
# ========================================================================

using JuLIP: Atoms, AbstractCalculator, neighbourlist,
             get_data, set_data!, has_data, JVecF, mat
import JuLIP: energy, forces

"""
`FastLJ`: This is a slightly unusual implementation of LJ, aiming to
make it as fast as possible.

Usage:
```
lj = JuLIP.Experimental.FastLJ(;r0 = r0, rcut = rcut, rbuf)
energy(lj, at)
forces(lj, at)
# etc...
```
* r0 : the usual LJ scaling factor
* rcut : cut-off radius (this calculator uses a C1-shift, not a spline)
* rbuf : the extra buffer used for assembling the neighbourlist
"""
struct FastLJ <: AbstractCalculator
    rcut::Float64        # cut-off radius
    rcut2::Float64       # rcut^2
    r0::Float64          # spatial scale
    fcut::Float64        # ϕlj(rcut)
    dfcut::Float64       # ∂ϕlj(rcut) / ∂(rcut^2)
    rbuf::Float64   # buffer for the neighbourlist
end

FastLJ(; r0=1.0, rcut=2.5*r0, rbuf=0.33*rcut) =
   FastLJ( rcut,
           rcut^2,
           r0,
           (r0/rcut)^12 - 2 * (r0/rcut)^6,
           6 / r0^2 * (-(r0/rcut)^14 + (r0/rcut)^8),
           rbuf )

function update!(V::FastLJ, at::Atoms{Float64, Int})
    # first check whether we already have a neighbourlist that is still usable
    if has_data(at, :nlist_fastlj_X)
        Xold = get_data(at, :nlist_fastlj_X)::Vector{JVecF}
        d2 = 0.0
        for n = 1:length(Xold)
            d2 = max(d2, sum(abs2, at.X[n] - Xold[n]))
        end
        if d2 < V.rbuf^2
            i = get_data(at, :nlist_fastlj_i)::Vector{Int}
            j = get_data(at, :nlist_fastlj_j)::Vector{Int}
            return i, j
        end
    end
    # if not, then assemble a new neighbourlist
    nlist = neighbourlist(at, V.rcut + V.rbuf)
    set_data!(at, :nlist_fastlj_X, copy(at.X))
    set_data!(at, :nlist_fastlj_i, nlist.i)
    set_data!(at, :nlist_fastlj_j, nlist.j)
    return nlist.i, nlist.j
end

function energy(V::FastLJ, at::Atoms{Float64, Int})
    i, j = update!(V, at)
    return energy_inner(V, at.X, i, j)
end

function energy_inner(V::FastLJ, X, i, j)
    E = 0.0
    for n = 1:length(i)
        @inbounds r2 = sum(abs2, X[i[n]]-X[j[n]])
        # evaluate LJ
        r2inv = (V.r0*V.r0)/r2
        r4inv = r2inv * r2inv
        r6inv = r2inv * r4inv
        r12inv = r6inv * r6inv
        lj1 = r12inv - 2 * r6inv
        # evaluate a cutoff
        lj2 = lj1 - V.fcut - V.dfcut * (r2 - V.rcut2)
        lj3 = lj2 * (r2 < V.rcut2)
        E += 0.5 * lj3
    end
    return E
end

forces(V::FastLJ, at::Atoms{Float64, Int}) =
    forces!(zeros(JVecF, length(at)), V, at)

function forces!(F, V::FastLJ, at::Atoms{Float64, Int})
    i, j = update!(V, at)
    return forces_inner!(F, V, at.X, i, j)
end

function forces_inner!(F, V::FastLJ, X, i, j)
    @assert length(F) == length(X)
    r02inv = 1.0 / V.r0^2
    for n = 1:length(i)
        R = X[i[n]]-X[j[n]]
        r2 = sum(abs2, R)
        r = sqrt(r2)
        # evaluate LJ
        r2inv = (V.r0*V.r0)/r2
        r4inv = r2inv * r2inv
        r6inv = r2inv * r4inv
        r8inv = r4inv * r4inv
        r14inv = r8inv * r6inv
        dlj1 = (6 * r02inv) * ( - r14inv + r8inv)
        # evaluate a cutoff
        dlj2 = (dlj1 - V.dfcut) * (r2 < V.rcut2)
        F[i[n]] -= dlj2 * R
        F[j[n]] += dlj2 * R
    end
    return F
end




end
