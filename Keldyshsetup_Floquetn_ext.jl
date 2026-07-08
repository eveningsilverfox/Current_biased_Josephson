module Keldyshsetup_Floquetn_ext

using MKL
using LinearAlgebra
using Statistics
using LaTeXStrings
using NLsolve
using OhMyThreads: TaskLocalValue
using Symbolics
using SparseArrays

BLAS.set_num_threads(1) # Avoid contention with threaded loop. Run with julia -t num_threads.
# The workload is many independent medium-size (N ~ 500) double-complex matmuls/solves
# (the per-frequency Dyson solve Gr = (I - gr*Sig)\gr, the Keldysh product Gr*Sig<*Ga, and
# the per-column derivative chains in the Jacobian).
#
# --- Total thread budget and oversubscription ---
# The total number of worker threads is fixed at startup by `julia -t num_threads`, i.e.
# num_threads = Threads.nthreads(). The two pools above run NESTED: when a Threads.@threads
# loop body calls a BLAS routine, the EFFECTIVE concurrency is the PRODUCT
#       Threads.nthreads()  x  BLAS.get_num_threads().
# If both are large this OVERSUBSCRIBES the machine -- e.g. `julia -t16` with MKL's default
# (~16 BLAS threads) launches ~16 x 16 = 256 threads fighting over ~16 cores, which thrashes
# caches and is much slower than 16. So the cores must be PARTITIONED between
#   (a) the Julia for-loop  -- OUTER parallelism: different loop iterations on different
#       threads (one whole matrix per thread), and
#   (b) BLAS               -- INNER parallelism: one mul!/*/\ split across threads.
# We want  nthreads(Julia) x BLAS_threads  ~ (physical cores), not their naive product.
#
# --- Design choice for THIS code ---
# 1. Matrices are only small to medium-sized, the regime where BLAS multithreading scales
#    POORLY (per-call fork/join overhead, limited arithmetic intensity): running whole
#    matrices concurrently beats splitting one matrix across threads. So we give BLAS a
#    SINGLE thread (BLAS.set_num_threads(1) above) and hand the ENTIRE thread budget to the
#    for loop:  nthreads(Julia) x 1 = nthreads, one matrix per thread, no oversubscription.
# 2. Of the two candidate loops to parallelize -- the Floquet-mode loop (2Nf+1 modes, or the
#    4Nf Jacobian derivative columns) and the frequency-grid loop (Nw0 = abs(Omega)/dw0 points) --
#    the number of Floquet modes (tens) is MUCH SMALLER than the number of frequency-grid
#    points (hundreds to thousands). So we put the Threads.@threads parallelism on the
#    FREQUENCY loop (chunked, see IbiasJacobian_Tfull / current_Floquet_Tfull): it exposes
#    far more independent tasks and keeps every thread busy. This is especially good on HPC
#    nodes with many threads -- more than the number of Floquet modes -- where the frequency
#    grid still has enough work to saturate all of them, whereas threading the (small)
#    Floquet/column loop would leave most threads idle.


## ============ Extended 4x4 Nambu(x)spin building blocks (classical spins / YSR) ==============
#
# These helpers extend the 2x2 Nambu (cup, cdn') framework of the original module to
# the full 4x4 Nambu(x)spin basis  c = (cup, cdn, cup', cdn') = kron(tau, sigma) needed
# to host classical-spin impurities and the resulting Yu-Shiba-Rusinov (YSR) states
# (arXiv:2602.15213, eqs. 7-8). tau = particle-hole, sigma = spin. The clean BCS lead
# is block-diagonal in the (cup,cdn') and (cdn,cup') sub-blocks (pairing +/-Delta);
# only the transverse exchange (Jx,Jy) couples them.

"""
    impurity4(J, K) -> Matrix{ComplexF64}

On-site classical-impurity matrix in the 4x4 Nambu(x)spin basis (cup, cdn, cup', cdn'),
implementing the on-site terms of eq. 7 of arXiv:2602.15213:

    Vimp = sum_k J_k (tau_u (x) sigma_k - tau_d (x) sigma_k*) + K (tau_z (x) sigma_0)
         = [ J.sigma      0      ]  +  K diag(1, 1, -1, -1).
           [   0      -(J.sigma)*]

`J = (Jx, Jy, Jz)` is the classical-spin exchange vector (units of Delta); `K` the
potential-scattering strength. The matrix is Hermitian. For `J || z` it is diagonal
(collinear -> the two Nambu(x)spin blocks stay decoupled); `Jx, Jy` couple the
(cup,cdn') and (cdn,cup') blocks and are what a YSR diode generally needs.
"""
function impurity4(J, K)
    sigx = ComplexF64[0 1; 1 0]; sigy = ComplexF64[0 -im; im 0]; sigz = ComplexF64[1 0; 0 -1];
    sig0 = ComplexF64[1 0; 0 1]; tauz = ComplexF64[1 0; 0 -1];
    Jsig = J[1]*sigx + J[2]*sigy + J[3]*sigz;            # J.sigma in spin space
    z2 = zeros(ComplexF64, 2, 2);
    Vexch = [Jsig z2; z2 -conj(Jsig)];                   # tau_u(x)(J.sigma) - tau_d(x)(J.sigma)*
    return Vexch + K * kron(tauz, sig0);                 # + potential scattering K tau_z(x)sigma_0
end

"""
    gsurf4(z, zeta, delta, Vimp) -> Matrix{ComplexF64}

Impurity-dressed lead surface Green's function (4x4) at complex energy `z` (= w+iGamma
for retarded, w-iGamma for advanced). Builds the clean BCS surface GF `g0` in the
(cup, cdn, cup', cdn') basis (block-diagonal in (cup,cdn')/(cdn,cup') with pairing
+/-delta), then applies the local Dyson dressing for an on-site impurity at the
contact site:  g = (1 - g0*Vimp)^{-1} g0.  `Vimp = 0` returns the clean `g0` and
reproduces the original 2x2 result in each block. Subgap poles of `g` are YSR states.
"""
function gsurf4(z, zeta, delta, Vimp)
    D = zeta * sqrt(delta^2 - z^2);
    g0 = (1/D) .* ComplexF64[-z 0 0 delta; 0 -z -delta 0; 0 -delta -z 0; delta 0 0 -z];
    return (I(4) - g0*Vimp) \ g0;
end

"""
    ysr_energies_analytical(J, K, zeta, delta) -> (Eminus, Eplus)

Closed-form subgap Yu-Shiba-Rusinov (YSR) bound-state energies of a single
classical-spin impurity `Vimp = impurity4(J, K)` on a BCS lead -- the in-gap
poles (|E| < delta) of the impurity-dressed surface GF [`gsurf4`](@ref). The
clean surface GF `g0` is spin-isotropic (singlet pairing), so the result depends
on the exchange only through its magnitude |J|. With dimensionless couplings
alpha = |J|/zeta (magnetic), beta = K/zeta (potential) and u = 1 - alpha^2 + beta^2,

    E_YSR = +/- delta * |u| / sqrt(u^2 + (2 alpha)^2).

The pair is particle-hole symmetric; the level crosses zero (singlet<->doublet
parity transition) at  alpha^2 = 1 + beta^2.  `J = K = 0` returns `+/-delta`
(state pinned at the gap edge -> no subgap bound state). Cross-checked against
[`ysr_energies_numerical`](@ref) to machine precision. Returns `(-|E_YSR|, +|E_YSR|)`.
"""
function ysr_energies_analytical(J, K, zeta, delta)
    alpha = norm(J) / zeta;                                  # dimensionless magnetic coupling |J|/zeta
    beta  = K / zeta;                                        # dimensionless potential coupling K/zeta
    u = 1 - alpha^2 + beta^2;
    E = delta * abs(u) / sqrt(u^2 + (2*alpha)^2);
    return (-E, E);
end

"""
    ysr_energies_numerical(J, K, zeta, delta; Ngrid=4001, edge=1e-4) -> (Eminus, Eplus)

Subgap YSR energies obtained by locating the in-gap poles of the impurity-dressed
surface GF [`gsurf4`](@ref) NUMERICALLY: `gsurf4 = (I - g0 Vimp)^{-1} g0` diverges
exactly where `det(I - g0(E) Vimp) = 0`, which is real for real subgap `E`. The
clean `g0` is taken from `gsurf4(E, zeta, delta, 0)`; the determinant is scanned on
`Ngrid` points across `(-delta, delta)` (kept `edge` away from the gap edges) and
each sign change is refined by bisection. Returns the outermost root pair
`(-|E_YSR|, +|E_YSR|)`, or `(-delta, delta)` if no subgap pole exists (no bound
state). Independent cross-check of [`ysr_energies_analytical`](@ref).
"""
function ysr_energies_numerical(J, K, zeta, delta; Ngrid = 4001, edge = 1e-4)
    Vimp = impurity4(J, K);
    g0clean(E) = gsurf4(E + 0im, zeta, delta, zeros(ComplexF64, 4, 4));   # clean surface GF g0
    fdet(E) = real(det(I(4) - g0clean(E)*Vimp));    # Gr denominator: vanishes at the YSR poles
    
    Egrid = range(-delta*(1-edge), delta*(1-edge), Ngrid);
    roots = Float64[];

    fprev = fdet(Egrid[1]); # Gr denominator at first E grid point
    for k in 2:Ngrid
        fcur = fdet(Egrid[k]);
        if fprev == 0.0 # Pole already present at first E grid point
            push!(roots, Egrid[k-1]);
        elseif fprev*fcur < 0 # Pole somewhere between E[k-1] and E[k] => bisect until root found
            a, b, fa = Egrid[k-1], Egrid[k], fprev;
            for _ in 1:100
                m = 0.5*(a+b); fm = fdet(m);
                (fa*fm <= 0) ? (b = m) : (a = m; fa = fm);
            end
            push!(roots, 0.5*(a+b));
        end
        fprev = fcur;
    end
    isempty(roots) && return (-delta, delta);                            # no subgap pole -> gap edge
    sort!(roots);
    return (roots[1], roots[end]);
end



## ============ Critical current ==============


"""
    currentPhi_eq_T2(war1, zeta, delta, T, Gamma, phi, JL, KL, JR, KR) -> Float64

Equilibrium (zero-bias) Josephson current at fixed phase difference `phi`, to
lowest order O(𝒯²), in the 4x4 Nambu(x)spin basis with per-lead classical-spin
(YSR) impurities `JL,KL` / `JR,KR`. Non-Floquet: the static hopping self-energies
`Sigrl, Siglr = 𝒯·diag(e^∓iφ/2, -e^±iφ/2)⊗σ0` are time-independent and the current
is a single frequency integral over `war1` of the Keldysh trace
`tr[(τz⊗σ0)(Σ gʳ Σ g< + Σ g< Σ gᵃ)]` (LR minus RL). Each lead carries its own
impurity-dressed surface GF `g_α = (I − g0 V_imp^α)⁻¹ g0`, assigned by the
alternation rule (propagator just after Σ_LR → R lead, after Σ_RL → L lead).
Reduces to 2x the 2x2 clean result when `JL=JR=0, KL=KR=0`. Building block of the
current–phase relation `I(φ)`; `maxᵩ I(φ)` is the critical current.

# Arguments
- `war1`: real-frequency grid (units of Δ).
- `zeta`, `delta`, `T`, `Gamma`: lead hopping ζ, gap Δ, transparency 𝒯, Dynes Γ.
- `phi`: phase difference φ.
- `JL,KL,JR,KR`: per-lead exchange `J=(Jx,Jy,Jz)` and potential `K` (`J=K=0`: clean).
"""
function currentPhi_eq_T2(war1, zeta, delta, T, Gamma, phi, JL, KL, JR, KR)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]);
    tz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1];
    sigx = ComplexF64[0 1; 1 0]; sigy = ComplexF64[0 -im; im 0]; sigz = ComplexF64[1 0; 0 -1]; sig0 = ComplexF64[1 0; 0 1]; tauz = ComplexF64[1 0; 0 -1];
    JsL = JL[1]*sigx + JL[2]*sigy + JL[3]*sigz; VimpL = [JsL zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsL)] + KL*kron(tauz,sig0);
    JsR = JR[1]*sigx + JR[2]*sigy + JR[3]*sigz; VimpR = [JsR zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsR)] + KR*kron(tauz,sig0);

    Sigrl = T .* kron(ComplexF64[exp(-im*phi/2) 0; 0 -exp(+im*phi/2)], sig0);
    Siglr = T .* kron(ComplexF64[exp(+im*phi/2) 0; 0 -exp(-im*phi/2)], sig0);

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        wwab = war1[hi]; z = wwab + im*Gamma; ff = (wwab < 0);
        g0 = (1/(zeta*sqrt(delta^2-z^2))) .* ComplexF64[-z 0 0 delta; 0 -z -delta 0; 0 -delta -z 0; delta 0 0 -z];
        grL = (I(4) - g0*VimpL) \ g0; gaL = grL';
        grR = (I(4) - g0*VimpR) \ g0; gaR = grR';
        # embedded (occupied-state) lesser GF, per lead
        gl0L = ff .* ( -(grL - gaL) ); SiglL = ff .* ( im*2*Gamma*I(4) + zeta^2 .* (tz*gl0L*tz) ); glL = grL*SiglL*gaL;
        gl0R = ff .* ( -(grR - gaR) ); SiglR = ff .* ( im*2*Gamma*I(4) + zeta^2 .* (tz*gl0R*tz) ); glR = grR*SiglR*gaR;
        # lead just after Sigma_LR -> R, after Sigma_RL -> L
        Idcwlr = real( tr( tz * ( Siglr * grR * Sigrl * glL ) ) ) +
                 real( tr( tz * ( Siglr * glR * Sigrl * gaL ) ) );
        Idcwrl = real( tr( tz * ( Sigrl * grL * Siglr * glR ) ) ) +
                 real( tr( tz * ( Sigrl * glL * Siglr * gaR ) ) );
        Idcw[hi] = Idcwlr - Idcwrl;
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end

"""
    currentPhi_eq_T4(war1, zeta, delta, T, Gamma, phi, JL, KL, JR, KR) -> Float64

Equilibrium Josephson current at fixed phase `phi`, expanded through O(𝒯⁴), in the
4x4 Nambu(x)spin basis with per-lead YSR impurities: the O(𝒯²) term of
[`currentPhi_eq_T2`](@ref) plus the four-vertex multiple-tunnelling diagrams
(chains `Σ gʳ Σ gʳ Σ gʳ Σ g<` etc.). Per-lead impurity-dressed surface GFs with the
alternation rule (LR chains see lead order R,L,R,L; RL chains L,R,L,R), simple FDT
lesser GF `g< = -(gʳ-gᵃ)θ(-ω)`. Reduces to 2x the 2x2 clean result when `J=K=0`.
"""
function currentPhi_eq_T4(war1, zeta, delta, T, Gamma, phi, JL, KL, JR, KR)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]);
    tz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1];
    sigx = ComplexF64[0 1; 1 0]; sigy = ComplexF64[0 -im; im 0]; sigz = ComplexF64[1 0; 0 -1]; sig0 = ComplexF64[1 0; 0 1]; tauz = ComplexF64[1 0; 0 -1];
    JsL = JL[1]*sigx + JL[2]*sigy + JL[3]*sigz; VimpL = [JsL zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsL)] + KL*kron(tauz,sig0);
    JsR = JR[1]*sigx + JR[2]*sigy + JR[3]*sigz; VimpR = [JsR zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsR)] + KR*kron(tauz,sig0);

    Sigrl = T .* kron(ComplexF64[exp(-im*phi/2) 0; 0 -exp(+im*phi/2)], sig0);
    Siglr = T .* kron(ComplexF64[exp(+im*phi/2) 0; 0 -exp(-im*phi/2)], sig0);

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        wwab = war1[hi]; z = wwab + im*Gamma; ff = (wwab < 0);
        g0 = (1/(zeta*sqrt(delta^2-z^2))) .* ComplexF64[-z 0 0 delta; 0 -z -delta 0; 0 -delta -z 0; delta 0 0 -z];
        grL = (I(4) - g0*VimpL) \ g0; gaL = grL'; glL = ff .* ( -(grL - gaL) );
        grR = (I(4) - g0*VimpR) \ g0; gaR = grR'; glR = ff .* ( -(grR - gaR) );

        # O(T^2): propagator just after Sigma_LR -> R, after Sigma_RL -> L
        Idcw2lr = real( tr( tz * ( Siglr * grR * Sigrl * glL ) ) ) +
                  real( tr( tz * ( Siglr * glR * Sigrl * gaL ) ) );
        Idcw2rl = real( tr( tz * ( Sigrl * grL * Siglr * glR ) ) ) +
                  real( tr( tz * ( Sigrl * glL * Siglr * gaR ) ) );
        # O(T^4): LR propagator leads [R,L,R,L], RL leads [L,R,L,R]
        Idcw4lr = real( tr( tz * ( Siglr * grR * Sigrl * grL * Siglr * grR * Sigrl * glL ) ) ) +
                  real( tr( tz * ( Siglr * grR * Sigrl * grL * Siglr * glR * Sigrl * gaL ) ) ) +
                  real( tr( tz * ( Siglr * grR * Sigrl * glL * Siglr * gaR * Sigrl * gaL ) ) ) +
                  real( tr( tz * ( Siglr * glR * Sigrl * gaL * Siglr * gaR * Sigrl * gaL ) ) );
        Idcw4rl = real( tr( tz * ( Sigrl * grL * Siglr * grR * Sigrl * grL * Siglr * glR ) ) ) +
                  real( tr( tz * ( Sigrl * grL * Siglr * grR * Sigrl * glL * Siglr * gaR ) ) ) +
                  real( tr( tz * ( Sigrl * grL * Siglr * glR * Sigrl * gaL * Siglr * gaR ) ) ) +
                  real( tr( tz * ( Sigrl * glL * Siglr * gaR * Sigrl * gaL * Siglr * gaR ) ) );
        Idcw[hi] = Idcw2lr + Idcw4lr - Idcw2rl - Idcw4rl;
    end

    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end

"""
    currentPhi_eq_Tfull(war1, zeta, delta, T, Gamma, phi, JL, KL, JR, KR) -> Float64

Equilibrium Josephson current at fixed phase `phi`, to ALL orders in the
transparency (full Dyson resummation, non-perturbative), in the 4x4 Nambu(x)spin
basis with per-lead YSR impurities. Assembles the 8x8 (two-lead ⊗ Nambu(x)spin)
junction self-energy `Sigrj` and the per-lead bare propagator
`grj = diag(g_L, g_R)` with `g_α = (I − g0 V_imp^α)⁻¹ g0`, solves the Dyson
equation `Grj = (I − grj·Sigrj)⁻¹ grj` and the Keldysh `Glj = Grj·Siglj·Grj†` at
each frequency, and integrates the current trace `tr[(τz⊗σ0)(Σ_LR G<_RL − Σ_RL
G<_LR)]`. The exact counterpart of [`currentPhi_eq_T2`](@ref)/`_T4`; reduces to 2x
the 2x2 clean result when `J=K=0`. Used by `Josephson_cphir.jl`.
"""
function currentPhi_eq_Tfull(war1, zeta, delta, T, Gamma, phi, JL, KL, JR, KR)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]);
    tz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1];
    sigx = ComplexF64[0 1; 1 0]; sigy = ComplexF64[0 -im; im 0]; sigz = ComplexF64[1 0; 0 -1]; sig0 = ComplexF64[1 0; 0 1]; tauz = ComplexF64[1 0; 0 -1];
    JsL = JL[1]*sigx + JL[2]*sigy + JL[3]*sigz; VimpL = [JsL zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsL)] + KL*kron(tauz,sig0);
    JsR = JR[1]*sigx + JR[2]*sigy + JR[3]*sigz; VimpR = [JsR zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsR)] + KR*kron(tauz,sig0);

    Sigrl = T .* kron(ComplexF64[exp(-im*phi/2) 0; 0 -exp(+im*phi/2)], sig0);
    Siglr = T .* kron(ComplexF64[exp(+im*phi/2) 0; 0 -exp(-im*phi/2)], sig0);
    Sigrj = [zeros(ComplexF64,4,4) Siglr; Sigrl zeros(ComplexF64,4,4)];

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        wwab = war1[hi]; z = wwab + im*Gamma; ff = (wwab < 0);
        g0 = (1/(zeta*sqrt(delta^2-z^2))) .* ComplexF64[-z 0 0 delta; 0 -z -delta 0; 0 -delta -z 0; delta 0 0 -z];
        grL = (I(4) - g0*VimpL) \ g0; gaL = grL'; gl0L = -(grL - gaL);
        grR = (I(4) - g0*VimpR) \ g0; gaR = grR'; gl0R = -(grR - gaR);

        Siglj = ff .* ( im*2*Gamma*I(8) + [ zeta^2 .* (tz*gl0L*tz) zeros(ComplexF64,4,4); zeros(ComplexF64,4,4) zeta^2 .* (tz*gl0R*tz) ] );
        grjwar = [grL zeros(ComplexF64,4,4); zeros(ComplexF64,4,4) grR];
        Grjwar = (I(8) - grjwar*Sigrj) \ grjwar;
        Gljwar = Grjwar * Siglj * Grjwar';

        Idcwlr = real( tr( tz * ( Siglr * Gljwar[5:8,1:4] ) ) );
        Idcwrl = real( tr( tz * ( Sigrl * Gljwar[1:4,5:8] ) ) );

        Idcw[hi] = Idcwlr - Idcwrl;
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end


## ============ V bias perturbative with explicit energy exchanges (MPT: non-renormalised jumps) ============




"""
    current_Vbias_Floquet_Tfull(war1, ev, zeta, delta, T, Gamma, JL, KL, JR, KR) -> Float64

Normal-state (Delta=0) Ohmic reference current via the full 4x4 Floquet machinery
([`current_Floquet_Tfull`](@ref)) with a single DC phase harmonic. Per-lead impurity
JL,KL,JR,KR threaded through.
"""
function current_Vbias_Floquet_Tfull(war1, ev, zeta, delta, T, Gamma, JL, KL, JR, KR)
    deltaw1 = abs(war1[2]-war1[1]);
    Nf = 20; Omega = ev; delta1 = 0;
    Nw0 = 2*ceil(Int, abs(Omega)/(2*deltaw1));                         # even cell count: PH-symmetric midpoint sampling
    war0 = -0.5*abs(Omega) .+ ((0:Nw0-1) .+ 0.5) .* (abs(Omega)/Nw0);  # midpoint rule: no sample on the T=0 occupation step
    VipI = zeros(ComplexF64, 4*Nf+1);
    VipI[2*Nf] = 1;
    If = Keldyshsetup_Floquetn_ext.current_Floquet_Tfull(war0, Omega, Nf, zeta, delta1, T, Gamma, VipI, JL, KL, JR, KR, 0);
    Idc = real(sum(diag(If)));
    return Idc
end


## ---- O(T^2) MPT family (4x4 Nambu(x)spin, per-lead YSR impurities) ----
# Each function is self-contained (no shared kernel), following the style of the 2x2
# module but with the impurity-dressed 4x4 surface GF g = (1 - g0 Vimp)^{-1} g0 built
# inline per lead. Chain lead assignment: the propagator just after a Sigma_LR vertex
# sits in the R lead, the one after Sigma_RL in the L lead (LR chain slots -> R,L,...;
# RL chain swaps Sigma_LR<->Sigma_RL and L<->R). With JL!=JR the LR and RL currents are
# unequal (diode), so both are summed explicitly (no factor-2 shortcut). The T2 family
# uses the embedded lesser GF g< = gr (2iGamma + zeta^2 tau_z(x)sig0 . g<0 . ...) ga.

"""
    current_Vbias_MPT_T2(war1, Omega, zeta, delta, T, Gamma, War, JL, KL, JR, KR) -> Float64

O(T^2) DC current of a voltage-biased junction (MPT, explicit energy exchange) in the
4x4 Nambu(x)spin basis (cup,cdn,cup',cdn') with per-lead classical-spin impurities
JL,KL (left) / JR,KR (right). Two tunnelling vertices carry energy kicks {+1,-1}.Omega.
`War`: Fourier coefficients of e^{i phi/2}. Reduces to 2x the original 2x2 result when
JL=JR, KL=KR.
"""
function current_Vbias_MPT_T2(war1, Omega, zeta, delta, T, Gamma, War, JL, KL, JR, KR)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]);
    tz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1];
    sigx = ComplexF64[0 1; 1 0]; sigy = ComplexF64[0 -im; im 0]; sigz = ComplexF64[1 0; 0 -1]; sig0 = ComplexF64[1 0; 0 1]; tauz = ComplexF64[1 0; 0 -1];
    JsL = JL[1]*sigx + JL[2]*sigy + JL[3]*sigz; VimpL = [JsL zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsL)] + KL*kron(tauz,sig0);
    JsR = JR[1]*sigx + JR[2]*sigy + JR[3]*sigz; VimpR = [JsR zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsR)] + KR*kron(tauz,sig0);

    Sigrl = zeros(ComplexF64, 2,4,4); Sigrl[1,:,:] = T .* kron([War[1] 0; 0 0],sig0); Sigrl[2,:,:] = T .* kron([0 0; 0 -conj(War[1])],sig0);
    Siglr = zeros(ComplexF64, 2,4,4); Siglr[1,:,:] = T .* kron([0 0; 0 -War[1]],sig0); Siglr[2,:,:] = T .* kron([conj(War[1]) 0; 0 0],sig0);

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        grL = zeros(ComplexF64, 3,4,4); glL = zeros(ComplexF64, 3,4,4); gaL = zeros(ComplexF64, 3,4,4);
        grR = zeros(ComplexF64, 3,4,4); glR = zeros(ComplexF64, 3,4,4); gaR = zeros(ComplexF64, 3,4,4);
        for ab = -1:1
            wwab = war1[hi] + ab*Omega; z = wwab + im*Gamma; ff = (wwab < 0);
            g0 = (1/(zeta*sqrt(delta^2-z^2))) .* ComplexF64[-z 0 0 delta; 0 -z -delta 0; 0 -delta -z 0; delta 0 0 -z];
            gr = (I(4) - g0*VimpL) \ g0; ga = gr'; gl0 = ff .* ( -(gr - ga) ); Sigl = ff .* ( im*2*Gamma*I(4) + zeta^2 .* (tz*gl0*tz) );
            grL[2-ab,:,:] = gr; gaL[2-ab,:,:] = ga; glL[2-ab,:,:] = gr*Sigl*ga;
            gr = (I(4) - g0*VimpR) \ g0; ga = gr'; gl0 = ff .* ( -(gr - ga) ); Sigl = ff .* ( im*2*Gamma*I(4) + zeta^2 .* (tz*gl0*tz) );
            grR[2-ab,:,:] = gr; gaR[2-ab,:,:] = ga; glR[2-ab,:,:] = gr*Sigl*ga;
        end

        Sar = -1 .* ones(Int32,2); indarr = 1:2;
        Idcwlr = 0.0; Idcwrl = 0.0;
        for ab1 = indarr
            Sar[ab1] = 1;
            Idcwlr = Idcwlr + real( tr( tz * ( Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * grR[2-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * glL[2-0,:,:] ) ) ) +
                              real( tr( tz * ( Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * glR[2-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gaL[2-0,:,:] ) ) );
            Idcwrl = Idcwrl + real( tr( tz * ( Sigrl[trunc(Int, 0.5*(3-Sar[2])),:,:] * grL[2-Sar[1],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[1])),:,:] * glR[2-0,:,:] ) ) ) +
                              real( tr( tz * ( Sigrl[trunc(Int, 0.5*(3-Sar[2])),:,:] * glL[2-Sar[1],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[1])),:,:] * gaR[2-0,:,:] ) ) );
        end
        Idcw[hi] = Idcwlr - Idcwrl;
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end

"""
    current_Vbias_MPT_T2_qp(war1, Omega, zeta, delta, T, Gamma, War, JL, KL, JR, KR) -> Float64

Quasiparticle channel of [`current_Vbias_MPT_T2`](@ref): keeps only the NORMAL
(e-e, h-h) blocks of each dressed lead GF (anomalous e-h blocks zeroed). 4x4 per-lead.
"""
function current_Vbias_MPT_T2_qp(war1, Omega, zeta, delta, T, Gamma, War, JL, KL, JR, KR)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]);
    tz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1];
    sigx = ComplexF64[0 1; 1 0]; sigy = ComplexF64[0 -im; im 0]; sigz = ComplexF64[1 0; 0 -1]; sig0 = ComplexF64[1 0; 0 1]; tauz = ComplexF64[1 0; 0 -1];
    JsL = JL[1]*sigx + JL[2]*sigy + JL[3]*sigz; VimpL = [JsL zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsL)] + KL*kron(tauz,sig0);
    JsR = JR[1]*sigx + JR[2]*sigy + JR[3]*sigz; VimpR = [JsR zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsR)] + KR*kron(tauz,sig0);

    Sigrl = zeros(ComplexF64, 2,4,4); Sigrl[1,:,:] = T .* kron([War[1] 0; 0 0],sig0); Sigrl[2,:,:] = T .* kron([0 0; 0 -conj(War[1])],sig0);
    Siglr = zeros(ComplexF64, 2,4,4); Siglr[1,:,:] = T .* kron([0 0; 0 -War[1]],sig0); Siglr[2,:,:] = T .* kron([conj(War[1]) 0; 0 0],sig0);

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        grL = zeros(ComplexF64, 3,4,4); glL = zeros(ComplexF64, 3,4,4); gaL = zeros(ComplexF64, 3,4,4);
        grR = zeros(ComplexF64, 3,4,4); glR = zeros(ComplexF64, 3,4,4); gaR = zeros(ComplexF64, 3,4,4);
        for ab = -1:1
            wwab = war1[hi] + ab*Omega; z = wwab + im*Gamma; ff = (wwab < 0);
            g0 = (1/(zeta*sqrt(delta^2-z^2))) .* ComplexF64[-z 0 0 delta; 0 -z -delta 0; 0 -delta -z 0; delta 0 0 -z];
            gr = (I(4) - g0*VimpL) \ g0; gr[1:2,3:4] .= 0; gr[3:4,1:2] .= 0; ga = gr'; gl0 = ff .* ( -(gr - ga) ); Sigl = ff .* ( im*2*Gamma*I(4) + zeta^2 .* (tz*gl0*tz) );
            grL[2-ab,:,:] = gr; gaL[2-ab,:,:] = ga; glL[2-ab,:,:] = gr*Sigl*ga;
            gr = (I(4) - g0*VimpR) \ g0; gr[1:2,3:4] .= 0; gr[3:4,1:2] .= 0; ga = gr'; gl0 = ff .* ( -(gr - ga) ); Sigl = ff .* ( im*2*Gamma*I(4) + zeta^2 .* (tz*gl0*tz) );
            grR[2-ab,:,:] = gr; gaR[2-ab,:,:] = ga; glR[2-ab,:,:] = gr*Sigl*ga;
        end

        Sar = -1 .* ones(Int32,2); indarr = 1:2;
        Idcwlr = 0.0; Idcwrl = 0.0;
        for ab1 = indarr
            Sar[ab1] = 1;
            Idcwlr = Idcwlr + real( tr( tz * ( Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * grR[2-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * glL[2-0,:,:] ) ) ) +
                              real( tr( tz * ( Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * glR[2-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gaL[2-0,:,:] ) ) );
            Idcwrl = Idcwrl + real( tr( tz * ( Sigrl[trunc(Int, 0.5*(3-Sar[2])),:,:] * grL[2-Sar[1],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[1])),:,:] * glR[2-0,:,:] ) ) ) +
                              real( tr( tz * ( Sigrl[trunc(Int, 0.5*(3-Sar[2])),:,:] * glL[2-Sar[1],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[1])),:,:] * gaR[2-0,:,:] ) ) );
        end
        Idcw[hi] = Idcwlr - Idcwrl;
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end

"""
    current_Vbias_MPT_T2_pair(war1, Omega, zeta, delta, T, Gamma, War, JL, KL, JR, KR) -> Float64

Pair channel of [`current_Vbias_MPT_T2`](@ref): keeps only the ANOMALOUS (e-h, h-e)
blocks of each dressed lead GF and, as in the 2x2 original, uses BOTH the +eV and -eV
phase harmonics War[2+1]/War[2-1] in the vertices to isolate the DC pair current.
4x4 per-lead.
"""
function current_Vbias_MPT_T2_pair(war1, Omega, zeta, delta, T, Gamma, War, JL, KL, JR, KR)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]);
    tz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1];
    sigx = ComplexF64[0 1; 1 0]; sigy = ComplexF64[0 -im; im 0]; sigz = ComplexF64[1 0; 0 -1]; sig0 = ComplexF64[1 0; 0 1]; tauz = ComplexF64[1 0; 0 -1];
    JsL = JL[1]*sigx + JL[2]*sigy + JL[3]*sigz; VimpL = [JsL zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsL)] + KL*kron(tauz,sig0);
    JsR = JR[1]*sigx + JR[2]*sigy + JR[3]*sigz; VimpR = [JsR zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsR)] + KR*kron(tauz,sig0);

    Sigrl = zeros(ComplexF64, 2,4,4); Sigrl[1,:,:] = T .* kron([War[2+1] 0; 0 -conj(War[2+1])],sig0); Sigrl[2,:,:] = T .* kron([War[2-1] 0; 0 -conj(War[2+1])],sig0);
    Siglr = zeros(ComplexF64, 2,4,4); Siglr[1,:,:] = T .* kron([conj(War[2+1]) 0; 0 -War[2-1]],sig0); Siglr[2,:,:] = T .* kron([conj(War[2+1]) 0; 0 -War[2-1]],sig0);

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        grL = zeros(ComplexF64, 3,4,4); glL = zeros(ComplexF64, 3,4,4); gaL = zeros(ComplexF64, 3,4,4);
        grR = zeros(ComplexF64, 3,4,4); glR = zeros(ComplexF64, 3,4,4); gaR = zeros(ComplexF64, 3,4,4);
        for ab = -1:1
            wwab = war1[hi] + ab*Omega; z = wwab + im*Gamma; ff = (wwab < 0);
            g0 = (1/(zeta*sqrt(delta^2-z^2))) .* ComplexF64[-z 0 0 delta; 0 -z -delta 0; 0 -delta -z 0; delta 0 0 -z];
            gr = (I(4) - g0*VimpL) \ g0; gr[1:2,1:2] .= 0; gr[3:4,3:4] .= 0; ga = gr'; gl0 = ff .* ( -(gr - ga) ); Sigl = ff .* ( im*2*Gamma*I(4) + zeta^2 .* (tz*gl0*tz) );
            grL[2-ab,:,:] = gr; gaL[2-ab,:,:] = ga; glL[2-ab,:,:] = gr*Sigl*ga;
            gr = (I(4) - g0*VimpR) \ g0; gr[1:2,1:2] .= 0; gr[3:4,3:4] .= 0; ga = gr'; gl0 = ff .* ( -(gr - ga) ); Sigl = ff .* ( im*2*Gamma*I(4) + zeta^2 .* (tz*gl0*tz) );
            grR[2-ab,:,:] = gr; gaR[2-ab,:,:] = ga; glR[2-ab,:,:] = gr*Sigl*ga;
        end

        Sar = -1 .* ones(Int32,2); indarr = 1:2;
        Idcwlr = 0.0; Idcwrl = 0.0;
        for ab1 = indarr
            Sar[ab1] = 1;
            Idcwlr = Idcwlr + real( tr( tz * ( Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * grR[2-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * glL[2-0,:,:] ) ) ) +
                              real( tr( tz * ( Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * glR[2-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gaL[2-0,:,:] ) ) );
            Idcwrl = Idcwrl + real( tr( tz * ( Sigrl[trunc(Int, 0.5*(3-Sar[2])),:,:] * grL[2-Sar[1],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[1])),:,:] * glR[2-0,:,:] ) ) ) +
                              real( tr( tz * ( Sigrl[trunc(Int, 0.5*(3-Sar[2])),:,:] * glL[2-Sar[1],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[1])),:,:] * gaR[2-0,:,:] ) ) );
        end
        Idcw[hi] = Idcwlr - Idcwrl;
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end




"""
    current_Vbias_MPT_T4(war1, Omega, zeta, delta, T, Gamma, War, JL, KL, JR, KR) -> Float64

O(T^4) DC current (MPT, explicit energy exchange), 4x4 Nambu(x)spin, per-lead YSR
impurities. Four tunnelling vertices carry the kicks {+1,+1,-1,-1}.Omega summed over
orderings; the LR and RL tunnelling chains are summed explicitly (lead just after
Sigma_LR -> R, after Sigma_RL -> L), so JL!=JR yields a diode asymmetry. Simple FDT
lesser GF g< = -(gr-ga) theta(-w). Reduces to 2x the 2x2 result when JL=JR.
"""
function current_Vbias_MPT_T4(war1, Omega, zeta, delta, T, Gamma, War, JL, KL, JR, KR)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]);
    tz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1];
    sigx = ComplexF64[0 1; 1 0]; sigy = ComplexF64[0 -im; im 0]; sigz = ComplexF64[1 0; 0 -1]; sig0 = ComplexF64[1 0; 0 1]; tauz = ComplexF64[1 0; 0 -1];
    JsL = JL[1]*sigx + JL[2]*sigy + JL[3]*sigz; VimpL = [JsL zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsL)] + KL*kron(tauz,sig0);
    JsR = JR[1]*sigx + JR[2]*sigy + JR[3]*sigz; VimpR = [JsR zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsR)] + KR*kron(tauz,sig0);

    Sigrl = zeros(ComplexF64, 2,4,4); Sigrl[1,:,:] = T .* kron([War[1] 0; 0 0],sig0); Sigrl[2,:,:] = T .* kron([0 0; 0 -conj(War[1])],sig0);
    Siglr = zeros(ComplexF64, 2,4,4); Siglr[1,:,:] = T .* kron([0 0; 0 -War[1]],sig0); Siglr[2,:,:] = T .* kron([conj(War[1]) 0; 0 0],sig0);

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        grL = zeros(ComplexF64,5,4,4); glL = zeros(ComplexF64,5,4,4); gaL = zeros(ComplexF64,5,4,4);
        grR = zeros(ComplexF64,5,4,4); glR = zeros(ComplexF64,5,4,4); gaR = zeros(ComplexF64,5,4,4);
        for ab = -2:2
            wwab = war1[hi] + ab*Omega; z = wwab + im*Gamma; ff = (wwab < 0);
            g0 = (1/(zeta*sqrt(delta^2-z^2))) .* ComplexF64[-z 0 0 delta; 0 -z -delta 0; 0 -delta -z 0; delta 0 0 -z];
            gr = (I(4) - g0*VimpL) \ g0; ga = gr'; grL[3-ab,:,:] = gr; gaL[3-ab,:,:] = ga; glL[3-ab,:,:] = ff .* ( -(gr-ga) );
            gr = (I(4) - g0*VimpR) \ g0; ga = gr'; grR[3-ab,:,:] = gr; gaR[3-ab,:,:] = ga; glR[3-ab,:,:] = ff .* ( -(gr-ga) );
        end

        indarr = 1:4;
        for ab1 = indarr
            for ab2 = indarr[1:end .!= ab1]
                Sar = -1 .* ones(Int32,4); Sar[ab1] = 1; Sar[ab2] = 1;
                iA = 3-Sar[1]-Sar[2]-Sar[3]; iB = 3-Sar[1]-Sar[2]; iC = 3-Sar[1]; iD = 3;
                s1 = trunc(Int, 0.5*(3-Sar[1])); s2 = trunc(Int, 0.5*(3-Sar[2])); s3 = trunc(Int, 0.5*(3-Sar[3])); s4 = trunc(Int, 0.5*(3-Sar[4]));
                # LR chain: GF slots (left->right) in leads R,L,R,L
                Idcw[hi] = Idcw[hi] + real( tr( tz * ( Siglr[s4,:,:]*grR[iA,:,:]*Sigrl[s3,:,:]*grL[iB,:,:]*Siglr[s2,:,:]*grR[iC,:,:]*Sigrl[s1,:,:]*glL[iD,:,:] ) ) ) +
                                      real( tr( tz * ( Siglr[s4,:,:]*grR[iA,:,:]*Sigrl[s3,:,:]*grL[iB,:,:]*Siglr[s2,:,:]*glR[iC,:,:]*Sigrl[s1,:,:]*gaL[iD,:,:] ) ) ) +
                                      real( tr( tz * ( Siglr[s4,:,:]*grR[iA,:,:]*Sigrl[s3,:,:]*glL[iB,:,:]*Siglr[s2,:,:]*gaR[iC,:,:]*Sigrl[s1,:,:]*gaL[iD,:,:] ) ) ) +
                                      real( tr( tz * ( Siglr[s4,:,:]*glR[iA,:,:]*Sigrl[s3,:,:]*gaL[iB,:,:]*Siglr[s2,:,:]*gaR[iC,:,:]*Sigrl[s1,:,:]*gaL[iD,:,:] ) ) );
                # RL chain (subtract): vertices Siglr<->Sigrl swapped, leads L,R,L,R
                Idcw[hi] = Idcw[hi] - real( tr( tz * ( Sigrl[s4,:,:]*grL[iA,:,:]*Siglr[s3,:,:]*grR[iB,:,:]*Sigrl[s2,:,:]*grL[iC,:,:]*Siglr[s1,:,:]*glR[iD,:,:] ) ) ) -
                                      real( tr( tz * ( Sigrl[s4,:,:]*grL[iA,:,:]*Siglr[s3,:,:]*grR[iB,:,:]*Sigrl[s2,:,:]*glL[iC,:,:]*Siglr[s1,:,:]*gaR[iD,:,:] ) ) ) -
                                      real( tr( tz * ( Sigrl[s4,:,:]*grL[iA,:,:]*Siglr[s3,:,:]*glR[iB,:,:]*Sigrl[s2,:,:]*gaL[iC,:,:]*Siglr[s1,:,:]*gaR[iD,:,:] ) ) ) -
                                      real( tr( tz * ( Sigrl[s4,:,:]*glL[iA,:,:]*Siglr[s3,:,:]*gaR[iB,:,:]*Sigrl[s2,:,:]*gaL[iC,:,:]*Siglr[s1,:,:]*gaR[iD,:,:] ) ) );
            end
        end
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end

"""
    current_Vbias_MPT_T4_qp(war1, Omega, zeta, delta, T, Gamma, War, JL, KL, JR, KR) -> Float64

Quasiparticle channel of [`current_Vbias_MPT_T4`](@ref): NORMAL (e-e, h-h) blocks of
each dressed lead GF only. 4x4 per-lead.
"""
function current_Vbias_MPT_T4_qp(war1, Omega, zeta, delta, T, Gamma, War, JL, KL, JR, KR)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]);
    tz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1];
    sigx = ComplexF64[0 1; 1 0]; sigy = ComplexF64[0 -im; im 0]; sigz = ComplexF64[1 0; 0 -1]; sig0 = ComplexF64[1 0; 0 1]; tauz = ComplexF64[1 0; 0 -1];
    JsL = JL[1]*sigx + JL[2]*sigy + JL[3]*sigz; VimpL = [JsL zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsL)] + KL*kron(tauz,sig0);
    JsR = JR[1]*sigx + JR[2]*sigy + JR[3]*sigz; VimpR = [JsR zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsR)] + KR*kron(tauz,sig0);

    Sigrl = zeros(ComplexF64, 2,4,4); Sigrl[1,:,:] = T .* kron([War[1] 0; 0 0],sig0); Sigrl[2,:,:] = T .* kron([0 0; 0 -conj(War[1])],sig0);
    Siglr = zeros(ComplexF64, 2,4,4); Siglr[1,:,:] = T .* kron([0 0; 0 -War[1]],sig0); Siglr[2,:,:] = T .* kron([conj(War[1]) 0; 0 0],sig0);

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        grL = zeros(ComplexF64,5,4,4); glL = zeros(ComplexF64,5,4,4); gaL = zeros(ComplexF64,5,4,4);
        grR = zeros(ComplexF64,5,4,4); glR = zeros(ComplexF64,5,4,4); gaR = zeros(ComplexF64,5,4,4);
        for ab = -2:2
            wwab = war1[hi] + ab*Omega; z = wwab + im*Gamma; ff = (wwab < 0);
            g0 = (1/(zeta*sqrt(delta^2-z^2))) .* ComplexF64[-z 0 0 delta; 0 -z -delta 0; 0 -delta -z 0; delta 0 0 -z];
            gr = (I(4) - g0*VimpL) \ g0; gr[1:2,3:4] .= 0; gr[3:4,1:2] .= 0; ga = gr'; grL[3-ab,:,:] = gr; gaL[3-ab,:,:] = ga; glL[3-ab,:,:] = ff .* ( -(gr-ga) );
            gr = (I(4) - g0*VimpR) \ g0; gr[1:2,3:4] .= 0; gr[3:4,1:2] .= 0; ga = gr'; grR[3-ab,:,:] = gr; gaR[3-ab,:,:] = ga; glR[3-ab,:,:] = ff .* ( -(gr-ga) );
        end

        indarr = 1:4;
        for ab1 = indarr
            for ab2 = indarr[1:end .!= ab1]
                Sar = -1 .* ones(Int32,4); Sar[ab1] = 1; Sar[ab2] = 1;
                iA = 3-Sar[1]-Sar[2]-Sar[3]; iB = 3-Sar[1]-Sar[2]; iC = 3-Sar[1]; iD = 3;
                s1 = trunc(Int, 0.5*(3-Sar[1])); s2 = trunc(Int, 0.5*(3-Sar[2])); s3 = trunc(Int, 0.5*(3-Sar[3])); s4 = trunc(Int, 0.5*(3-Sar[4]));
                Idcw[hi] = Idcw[hi] + real( tr( tz * ( Siglr[s4,:,:]*grR[iA,:,:]*Sigrl[s3,:,:]*grL[iB,:,:]*Siglr[s2,:,:]*grR[iC,:,:]*Sigrl[s1,:,:]*glL[iD,:,:] ) ) ) +
                                      real( tr( tz * ( Siglr[s4,:,:]*grR[iA,:,:]*Sigrl[s3,:,:]*grL[iB,:,:]*Siglr[s2,:,:]*glR[iC,:,:]*Sigrl[s1,:,:]*gaL[iD,:,:] ) ) ) +
                                      real( tr( tz * ( Siglr[s4,:,:]*grR[iA,:,:]*Sigrl[s3,:,:]*glL[iB,:,:]*Siglr[s2,:,:]*gaR[iC,:,:]*Sigrl[s1,:,:]*gaL[iD,:,:] ) ) ) +
                                      real( tr( tz * ( Siglr[s4,:,:]*glR[iA,:,:]*Sigrl[s3,:,:]*gaL[iB,:,:]*Siglr[s2,:,:]*gaR[iC,:,:]*Sigrl[s1,:,:]*gaL[iD,:,:] ) ) );
                Idcw[hi] = Idcw[hi] - real( tr( tz * ( Sigrl[s4,:,:]*grL[iA,:,:]*Siglr[s3,:,:]*grR[iB,:,:]*Sigrl[s2,:,:]*grL[iC,:,:]*Siglr[s1,:,:]*glR[iD,:,:] ) ) ) -
                                      real( tr( tz * ( Sigrl[s4,:,:]*grL[iA,:,:]*Siglr[s3,:,:]*grR[iB,:,:]*Sigrl[s2,:,:]*glL[iC,:,:]*Siglr[s1,:,:]*gaR[iD,:,:] ) ) ) -
                                      real( tr( tz * ( Sigrl[s4,:,:]*grL[iA,:,:]*Siglr[s3,:,:]*glR[iB,:,:]*Sigrl[s2,:,:]*gaL[iC,:,:]*Siglr[s1,:,:]*gaR[iD,:,:] ) ) ) -
                                      real( tr( tz * ( Sigrl[s4,:,:]*glL[iA,:,:]*Siglr[s3,:,:]*gaR[iB,:,:]*Sigrl[s2,:,:]*gaL[iC,:,:]*Siglr[s1,:,:]*gaR[iD,:,:] ) ) );
            end
        end
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end


"""
    current_Vbias_MPT_T4_aobo(war1, Omega, zeta, delta, T, Gamma, ao, bo, War, JL, KL, JR, KR) -> Float64

O(T^4), (ao,bo)-energy-exchange-channel current (MPT), 4x4 Nambu(x)spin, per-lead YSR
impurities. The four vertices carry the kicks {+ao,+bo,-ao,-bo}.Omega summed over all
orderings (Sar 3-loop); a `+` is a quasiparticle jump up in energy, a `-` a jump down
(net zero -> DC). LR and RL chains summed explicitly (lead after Sigma_LR -> R, after
Sigma_RL -> L). `War`: e^{i phi/2} coefficients ordered +nf:-2:-nf, nf=max(ao,bo).
Reduces to 2x the 2x2 result when JL=JR.
"""
function current_Vbias_MPT_T4_aobo(war1, Omega, zeta, delta, T, Gamma, ao, bo, War, JL, KL, JR, KR)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]);
    tz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1];
    sigx = ComplexF64[0 1; 1 0]; sigy = ComplexF64[0 -im; im 0]; sigz = ComplexF64[1 0; 0 -1]; sig0 = ComplexF64[1 0; 0 1]; tauz = ComplexF64[1 0; 0 -1];
    JsL = JL[1]*sigx + JL[2]*sigy + JL[3]*sigz; VimpL = [JsL zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsL)] + KL*kron(tauz,sig0);
    JsR = JR[1]*sigx + JR[2]*sigy + JR[3]*sigz; VimpR = [JsR zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsR)] + KR*kron(tauz,sig0);

    nf = maximum([ao, bo]); mm = ao + bo;
    Sigrl = zeros(ComplexF64, 2*nf+1,4,4); Siglr = zeros(ComplexF64, 2*nf+1,4,4);
    for h = -nf:2:nf
        Sigrl[nf+1-h,:,:] = T .* kron([War[nf+1-h] 0; 0 -conj(War[nf+1+h])],sig0);
        Siglr[nf+1-h,:,:] = T .* kron([conj(War[nf+1+h]) 0; 0 -War[nf+1-h]],sig0);
    end

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        grL = zeros(ComplexF64,2*mm+1,4,4); glL = zeros(ComplexF64,2*mm+1,4,4); gaL = zeros(ComplexF64,2*mm+1,4,4);
        grR = zeros(ComplexF64,2*mm+1,4,4); glR = zeros(ComplexF64,2*mm+1,4,4); gaR = zeros(ComplexF64,2*mm+1,4,4);
        for ab = -mm:mm
            wwab = war1[hi] + ab*Omega; z = wwab + im*Gamma; ff = (wwab < 0); k = mm+1-ab;
            g0 = (1/(zeta*sqrt(delta^2-z^2))) .* ComplexF64[-z 0 0 delta; 0 -z -delta 0; 0 -delta -z 0; delta 0 0 -z];
            gr = (I(4) - g0*VimpL) \ g0;  ga = gr'; grL[k,:,:] = gr; gaL[k,:,:] = ga; glL[k,:,:] = ff .* ( -(gr-ga) );
            gr = (I(4) - g0*VimpR) \ g0;  ga = gr'; grR[k,:,:] = gr; gaR[k,:,:] = ga; glR[k,:,:] = ff .* ( -(gr-ga) );
        end

        indarr = 1:4;
        for ab1 = indarr
            indarr1 = indarr[1:end .!= ab1];
            for ab2 = indarr1
                indarr2 = indarr1[ indarr1 .* (indarr1 .!= ab2) .!= 0 ];
                for ab3 = indarr2
                    Sar = -bo .* ones(Int32,4); Sar[ab1] = ao; Sar[ab2] = bo; Sar[ab3] = -ao;
                    iA = mm+1-Sar[1]-Sar[2]-Sar[3]; iB = mm+1-Sar[1]-Sar[2]; iC = mm+1-Sar[1]; iD = mm+1;
                    v1 = nf+1-Sar[1]; v2 = nf+1-Sar[2]; v3 = nf+1-Sar[3]; v4 = nf+1-Sar[4];
                    # LR chain: leads R,L,R,L
                    Idcw[hi] = Idcw[hi] + real( tr( tz * ( Siglr[v4,:,:]*grR[iA,:,:]*Sigrl[v3,:,:]*grL[iB,:,:]*Siglr[v2,:,:]*grR[iC,:,:]*Sigrl[v1,:,:]*glL[iD,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[v4,:,:]*grR[iA,:,:]*Sigrl[v3,:,:]*grL[iB,:,:]*Siglr[v2,:,:]*glR[iC,:,:]*Sigrl[v1,:,:]*gaL[iD,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[v4,:,:]*grR[iA,:,:]*Sigrl[v3,:,:]*glL[iB,:,:]*Siglr[v2,:,:]*gaR[iC,:,:]*Sigrl[v1,:,:]*gaL[iD,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[v4,:,:]*glR[iA,:,:]*Sigrl[v3,:,:]*gaL[iB,:,:]*Siglr[v2,:,:]*gaR[iC,:,:]*Sigrl[v1,:,:]*gaL[iD,:,:] ) ) );
                    # RL chain (subtract): vertices swapped, leads L,R,L,R
                    Idcw[hi] = Idcw[hi] - real( tr( tz * ( Sigrl[v4,:,:]*grL[iA,:,:]*Siglr[v3,:,:]*grR[iB,:,:]*Sigrl[v2,:,:]*grL[iC,:,:]*Siglr[v1,:,:]*glR[iD,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[v4,:,:]*grL[iA,:,:]*Siglr[v3,:,:]*grR[iB,:,:]*Sigrl[v2,:,:]*glL[iC,:,:]*Siglr[v1,:,:]*gaR[iD,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[v4,:,:]*grL[iA,:,:]*Siglr[v3,:,:]*glR[iB,:,:]*Sigrl[v2,:,:]*gaL[iC,:,:]*Siglr[v1,:,:]*gaR[iD,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[v4,:,:]*glL[iA,:,:]*Siglr[v3,:,:]*gaR[iB,:,:]*Sigrl[v2,:,:]*gaL[iC,:,:]*Siglr[v1,:,:]*gaR[iD,:,:] ) ) );
                end
            end
        end
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end

"""
    current_Vbias_MPT_T4_qp_aobo(war1, Omega, zeta, delta, T, Gamma, ao, bo, War, JL, KL, JR, KR) -> Float64

Quasiparticle (NORMAL e-e/h-h GF only) part of [`current_Vbias_MPT_T4_aobo`](@ref). 4x4 per-lead.
"""
function current_Vbias_MPT_T4_qp_aobo(war1, Omega, zeta, delta, T, Gamma, ao, bo, War, JL, KL, JR, KR)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]);
    tz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1];
    sigx = ComplexF64[0 1; 1 0]; sigy = ComplexF64[0 -im; im 0]; sigz = ComplexF64[1 0; 0 -1]; sig0 = ComplexF64[1 0; 0 1]; tauz = ComplexF64[1 0; 0 -1];
    JsL = JL[1]*sigx + JL[2]*sigy + JL[3]*sigz; VimpL = [JsL zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsL)] + KL*kron(tauz,sig0);
    JsR = JR[1]*sigx + JR[2]*sigy + JR[3]*sigz; VimpR = [JsR zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsR)] + KR*kron(tauz,sig0);

    nf = maximum([ao, bo]); mm = ao + bo;
    Sigrl = zeros(ComplexF64, 2*nf+1,4,4); Siglr = zeros(ComplexF64, 2*nf+1,4,4);
    for h = -nf:2:nf
        Sigrl[nf+1-h,:,:] = T .* kron([War[nf+1-h] 0; 0 -conj(War[nf+1+h])],sig0);
        Siglr[nf+1-h,:,:] = T .* kron([conj(War[nf+1+h]) 0; 0 -War[nf+1-h]],sig0);
    end

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        grL = zeros(ComplexF64,2*mm+1,4,4); glL = zeros(ComplexF64,2*mm+1,4,4); gaL = zeros(ComplexF64,2*mm+1,4,4);
        grR = zeros(ComplexF64,2*mm+1,4,4); glR = zeros(ComplexF64,2*mm+1,4,4); gaR = zeros(ComplexF64,2*mm+1,4,4);
        for ab = -mm:mm
            wwab = war1[hi] + ab*Omega; z = wwab + im*Gamma; ff = (wwab < 0); k = mm+1-ab;
            g0 = (1/(zeta*sqrt(delta^2-z^2))) .* ComplexF64[-z 0 0 delta; 0 -z -delta 0; 0 -delta -z 0; delta 0 0 -z];
            gr = (I(4) - g0*VimpL) \ g0; gr[1:2,3:4] .= 0; gr[3:4,1:2] .= 0; ga = gr'; grL[k,:,:] = gr; gaL[k,:,:] = ga; glL[k,:,:] = ff .* ( -(gr-ga) );
            gr = (I(4) - g0*VimpR) \ g0; gr[1:2,3:4] .= 0; gr[3:4,1:2] .= 0; ga = gr'; grR[k,:,:] = gr; gaR[k,:,:] = ga; glR[k,:,:] = ff .* ( -(gr-ga) );
        end

        indarr = 1:4;
        for ab1 = indarr
            indarr1 = indarr[1:end .!= ab1];
            for ab2 = indarr1
                indarr2 = indarr1[ indarr1 .* (indarr1 .!= ab2) .!= 0 ];
                for ab3 = indarr2
                    Sar = -bo .* ones(Int32,4); Sar[ab1] = ao; Sar[ab2] = bo; Sar[ab3] = -ao;
                    iA = mm+1-Sar[1]-Sar[2]-Sar[3]; iB = mm+1-Sar[1]-Sar[2]; iC = mm+1-Sar[1]; iD = mm+1;
                    v1 = nf+1-Sar[1]; v2 = nf+1-Sar[2]; v3 = nf+1-Sar[3]; v4 = nf+1-Sar[4];
                    # LR chain: leads R,L,R,L
                    Idcw[hi] = Idcw[hi] + real( tr( tz * ( Siglr[v4,:,:]*grR[iA,:,:]*Sigrl[v3,:,:]*grL[iB,:,:]*Siglr[v2,:,:]*grR[iC,:,:]*Sigrl[v1,:,:]*glL[iD,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[v4,:,:]*grR[iA,:,:]*Sigrl[v3,:,:]*grL[iB,:,:]*Siglr[v2,:,:]*glR[iC,:,:]*Sigrl[v1,:,:]*gaL[iD,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[v4,:,:]*grR[iA,:,:]*Sigrl[v3,:,:]*glL[iB,:,:]*Siglr[v2,:,:]*gaR[iC,:,:]*Sigrl[v1,:,:]*gaL[iD,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[v4,:,:]*glR[iA,:,:]*Sigrl[v3,:,:]*gaL[iB,:,:]*Siglr[v2,:,:]*gaR[iC,:,:]*Sigrl[v1,:,:]*gaL[iD,:,:] ) ) );
                    # RL chain (subtract): vertices swapped, leads L,R,L,R
                    Idcw[hi] = Idcw[hi] - real( tr( tz * ( Sigrl[v4,:,:]*grL[iA,:,:]*Siglr[v3,:,:]*grR[iB,:,:]*Sigrl[v2,:,:]*grL[iC,:,:]*Siglr[v1,:,:]*glR[iD,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[v4,:,:]*grL[iA,:,:]*Siglr[v3,:,:]*grR[iB,:,:]*Sigrl[v2,:,:]*glL[iC,:,:]*Siglr[v1,:,:]*gaR[iD,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[v4,:,:]*grL[iA,:,:]*Siglr[v3,:,:]*glR[iB,:,:]*Sigrl[v2,:,:]*gaL[iC,:,:]*Siglr[v1,:,:]*gaR[iD,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[v4,:,:]*glL[iA,:,:]*Siglr[v3,:,:]*gaR[iB,:,:]*Sigrl[v2,:,:]*gaL[iC,:,:]*Siglr[v1,:,:]*gaR[iD,:,:] ) ) );
                end
            end
        end
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end

"""
    current_Vbias_MPT_T4_pair_aobo(war1, Omega, zeta, delta, T, Gamma, ao, bo, War, JL, KL, JR, KR) -> Float64

Pair (ANOMALOUS e-h/h-e GF only) part of [`current_Vbias_MPT_T4_aobo`](@ref). 4x4 per-lead.
"""
function current_Vbias_MPT_T4_pair_aobo(war1, Omega, zeta, delta, T, Gamma, ao, bo, War, JL, KL, JR, KR)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]);
    tz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1];
    sigx = ComplexF64[0 1; 1 0]; sigy = ComplexF64[0 -im; im 0]; sigz = ComplexF64[1 0; 0 -1]; sig0 = ComplexF64[1 0; 0 1]; tauz = ComplexF64[1 0; 0 -1];
    JsL = JL[1]*sigx + JL[2]*sigy + JL[3]*sigz; VimpL = [JsL zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsL)] + KL*kron(tauz,sig0);
    JsR = JR[1]*sigx + JR[2]*sigy + JR[3]*sigz; VimpR = [JsR zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsR)] + KR*kron(tauz,sig0);

    nf = maximum([ao, bo]); mm = ao + bo;
    Sigrl = zeros(ComplexF64, 2*nf+1,4,4); Siglr = zeros(ComplexF64, 2*nf+1,4,4);
    for h = -nf:2:nf
        Sigrl[nf+1-h,:,:] = T .* kron([War[nf+1-h] 0; 0 -conj(War[nf+1+h])],sig0);
        Siglr[nf+1-h,:,:] = T .* kron([conj(War[nf+1+h]) 0; 0 -War[nf+1-h]],sig0);
    end

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        grL = zeros(ComplexF64,2*mm+1,4,4); glL = zeros(ComplexF64,2*mm+1,4,4); gaL = zeros(ComplexF64,2*mm+1,4,4);
        grR = zeros(ComplexF64,2*mm+1,4,4); glR = zeros(ComplexF64,2*mm+1,4,4); gaR = zeros(ComplexF64,2*mm+1,4,4);
        for ab = -mm:mm
            wwab = war1[hi] + ab*Omega; z = wwab + im*Gamma; ff = (wwab < 0); k = mm+1-ab;
            g0 = (1/(zeta*sqrt(delta^2-z^2))) .* ComplexF64[-z 0 0 delta; 0 -z -delta 0; 0 -delta -z 0; delta 0 0 -z];
            gr = (I(4) - g0*VimpL) \ g0; gr[1:2,1:2] .= 0; gr[3:4,3:4] .= 0; ga = gr'; grL[k,:,:] = gr; gaL[k,:,:] = ga; glL[k,:,:] = ff .* ( -(gr-ga) );
            gr = (I(4) - g0*VimpR) \ g0; gr[1:2,1:2] .= 0; gr[3:4,3:4] .= 0; ga = gr'; grR[k,:,:] = gr; gaR[k,:,:] = ga; glR[k,:,:] = ff .* ( -(gr-ga) );
        end

        indarr = 1:4;
        for ab1 = indarr
            indarr1 = indarr[1:end .!= ab1];
            for ab2 = indarr1
                indarr2 = indarr1[ indarr1 .* (indarr1 .!= ab2) .!= 0 ];
                for ab3 = indarr2
                    Sar = -bo .* ones(Int32,4); Sar[ab1] = ao; Sar[ab2] = bo; Sar[ab3] = -ao;
                    iA = mm+1-Sar[1]-Sar[2]-Sar[3]; iB = mm+1-Sar[1]-Sar[2]; iC = mm+1-Sar[1]; iD = mm+1;
                    v1 = nf+1-Sar[1]; v2 = nf+1-Sar[2]; v3 = nf+1-Sar[3]; v4 = nf+1-Sar[4];
                    # LR chain: leads R,L,R,L
                    Idcw[hi] = Idcw[hi] + real( tr( tz * ( Siglr[v4,:,:]*grR[iA,:,:]*Sigrl[v3,:,:]*grL[iB,:,:]*Siglr[v2,:,:]*grR[iC,:,:]*Sigrl[v1,:,:]*glL[iD,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[v4,:,:]*grR[iA,:,:]*Sigrl[v3,:,:]*grL[iB,:,:]*Siglr[v2,:,:]*glR[iC,:,:]*Sigrl[v1,:,:]*gaL[iD,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[v4,:,:]*grR[iA,:,:]*Sigrl[v3,:,:]*glL[iB,:,:]*Siglr[v2,:,:]*gaR[iC,:,:]*Sigrl[v1,:,:]*gaL[iD,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[v4,:,:]*glR[iA,:,:]*Sigrl[v3,:,:]*gaL[iB,:,:]*Siglr[v2,:,:]*gaR[iC,:,:]*Sigrl[v1,:,:]*gaL[iD,:,:] ) ) );
                    # RL chain (subtract): vertices swapped, leads L,R,L,R
                    Idcw[hi] = Idcw[hi] - real( tr( tz * ( Sigrl[v4,:,:]*grL[iA,:,:]*Siglr[v3,:,:]*grR[iB,:,:]*Sigrl[v2,:,:]*grL[iC,:,:]*Siglr[v1,:,:]*glR[iD,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[v4,:,:]*grL[iA,:,:]*Siglr[v3,:,:]*grR[iB,:,:]*Sigrl[v2,:,:]*glL[iC,:,:]*Siglr[v1,:,:]*gaR[iD,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[v4,:,:]*grL[iA,:,:]*Siglr[v3,:,:]*glR[iB,:,:]*Sigrl[v2,:,:]*gaL[iC,:,:]*Siglr[v1,:,:]*gaR[iD,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[v4,:,:]*glL[iA,:,:]*Siglr[v3,:,:]*gaR[iB,:,:]*Sigrl[v2,:,:]*gaL[iC,:,:]*Siglr[v1,:,:]*gaR[iD,:,:] ) ) );
                end
            end
        end
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end


"""
    current_Vbias_MPT_T6(war1, Omega, zeta, delta, T, Gamma, War, JL, KL, JR, KR) -> Float64

O(T^6) DC current (MPT, explicit energy exchange), 4x4 Nambu(x)spin, per-lead YSR
impurities. Six tunnelling vertices, kicks {+1,+1,+1,-1,-1,-1}.Omega summed over
orderings; LR and RL chains summed explicitly (lead after Sigma_LR -> R, after
Sigma_RL -> L). Reduces to 2x the 2x2 result when JL=JR.
"""
function current_Vbias_MPT_T6(war1, Omega, zeta, delta, T, Gamma, War, JL, KL, JR, KR)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]);
    tz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1];
    sigx = ComplexF64[0 1; 1 0]; sigy = ComplexF64[0 -im; im 0]; sigz = ComplexF64[1 0; 0 -1]; sig0 = ComplexF64[1 0; 0 1]; tauz = ComplexF64[1 0; 0 -1];
    JsL = JL[1]*sigx + JL[2]*sigy + JL[3]*sigz; VimpL = [JsL zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsL)] + KL*kron(tauz,sig0);
    JsR = JR[1]*sigx + JR[2]*sigy + JR[3]*sigz; VimpR = [JsR zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsR)] + KR*kron(tauz,sig0);

    Sigrl = zeros(ComplexF64, 2,4,4); Sigrl[1,:,:] = T .* kron([War[1] 0; 0 0],sig0); Sigrl[2,:,:] = T .* kron([0 0; 0 -conj(War[1])],sig0);
    Siglr = zeros(ComplexF64, 2,4,4); Siglr[1,:,:] = T .* kron([0 0; 0 -War[1]],sig0); Siglr[2,:,:] = T .* kron([conj(War[1]) 0; 0 0],sig0);

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        grL = zeros(ComplexF64,7,4,4); glL = zeros(ComplexF64,7,4,4); gaL = zeros(ComplexF64,7,4,4);
        grR = zeros(ComplexF64,7,4,4); glR = zeros(ComplexF64,7,4,4); gaR = zeros(ComplexF64,7,4,4);
        for ab = -3:3
            wwab = war1[hi] + ab*Omega; z = wwab + im*Gamma; ff = (wwab < 0);
            g0 = (1/(zeta*sqrt(delta^2-z^2))) .* ComplexF64[-z 0 0 delta; 0 -z -delta 0; 0 -delta -z 0; delta 0 0 -z];
            gr = (I(4) - g0*VimpL) \ g0;  ga = gr'; grL[4-ab,:,:] = gr; gaL[4-ab,:,:] = ga; glL[4-ab,:,:] = ff .* ( -(gr-ga) );
            gr = (I(4) - g0*VimpR) \ g0;  ga = gr'; grR[4-ab,:,:] = gr; gaR[4-ab,:,:] = ga; glR[4-ab,:,:] = ff .* ( -(gr-ga) );
        end

        indarr = 1:6;
        for ab1 = indarr
            indarr1 = indarr[1:end .!= ab1];
            for ab2 = indarr1
                indarr2 = indarr1[ indarr1 .* (indarr1 .!= ab2) .!= 0 ];
                for ab3 = indarr2
                    Sar = -1 .* ones(Int32,6); Sar[ab1] = 1; Sar[ab2] = 1; Sar[ab3] = 1;
                    iA = 4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5]; iB = 4-Sar[1]-Sar[2]-Sar[3]-Sar[4]; iC = 4-Sar[1]-Sar[2]-Sar[3]; iD = 4-Sar[1]-Sar[2]; iE = 4-Sar[1]; iF = 4;
                    s1 = trunc(Int, 0.5*(3-Sar[1])); s2 = trunc(Int, 0.5*(3-Sar[2])); s3 = trunc(Int, 0.5*(3-Sar[3])); s4 = trunc(Int, 0.5*(3-Sar[4])); s5 = trunc(Int, 0.5*(3-Sar[5])); s6 = trunc(Int, 0.5*(3-Sar[6]));
                    # LR chain: GF slots (left->right) leads R,L,R,L,R,L
                    Idcw[hi] = Idcw[hi] + real( tr( tz * ( Siglr[s6,:,:]*grR[iA,:,:]*Sigrl[s5,:,:]*grL[iB,:,:]*Siglr[s4,:,:]*grR[iC,:,:]*Sigrl[s3,:,:]*grL[iD,:,:]*Siglr[s2,:,:]*grR[iE,:,:]*Sigrl[s1,:,:]*glL[iF,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[s6,:,:]*grR[iA,:,:]*Sigrl[s5,:,:]*grL[iB,:,:]*Siglr[s4,:,:]*grR[iC,:,:]*Sigrl[s3,:,:]*grL[iD,:,:]*Siglr[s2,:,:]*glR[iE,:,:]*Sigrl[s1,:,:]*gaL[iF,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[s6,:,:]*grR[iA,:,:]*Sigrl[s5,:,:]*grL[iB,:,:]*Siglr[s4,:,:]*grR[iC,:,:]*Sigrl[s3,:,:]*glL[iD,:,:]*Siglr[s2,:,:]*gaR[iE,:,:]*Sigrl[s1,:,:]*gaL[iF,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[s6,:,:]*grR[iA,:,:]*Sigrl[s5,:,:]*grL[iB,:,:]*Siglr[s4,:,:]*glR[iC,:,:]*Sigrl[s3,:,:]*gaL[iD,:,:]*Siglr[s2,:,:]*gaR[iE,:,:]*Sigrl[s1,:,:]*gaL[iF,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[s6,:,:]*grR[iA,:,:]*Sigrl[s5,:,:]*glL[iB,:,:]*Siglr[s4,:,:]*gaR[iC,:,:]*Sigrl[s3,:,:]*gaL[iD,:,:]*Siglr[s2,:,:]*gaR[iE,:,:]*Sigrl[s1,:,:]*gaL[iF,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[s6,:,:]*glR[iA,:,:]*Sigrl[s5,:,:]*gaL[iB,:,:]*Siglr[s4,:,:]*gaR[iC,:,:]*Sigrl[s3,:,:]*gaL[iD,:,:]*Siglr[s2,:,:]*gaR[iE,:,:]*Sigrl[s1,:,:]*gaL[iF,:,:] ) ) );
                    # RL chain (subtract): vertices swapped, leads L,R,L,R,L,R
                    Idcw[hi] = Idcw[hi] - real( tr( tz * ( Sigrl[s6,:,:]*grL[iA,:,:]*Siglr[s5,:,:]*grR[iB,:,:]*Sigrl[s4,:,:]*grL[iC,:,:]*Siglr[s3,:,:]*grR[iD,:,:]*Sigrl[s2,:,:]*grL[iE,:,:]*Siglr[s1,:,:]*glR[iF,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[s6,:,:]*grL[iA,:,:]*Siglr[s5,:,:]*grR[iB,:,:]*Sigrl[s4,:,:]*grL[iC,:,:]*Siglr[s3,:,:]*grR[iD,:,:]*Sigrl[s2,:,:]*glL[iE,:,:]*Siglr[s1,:,:]*gaR[iF,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[s6,:,:]*grL[iA,:,:]*Siglr[s5,:,:]*grR[iB,:,:]*Sigrl[s4,:,:]*grL[iC,:,:]*Siglr[s3,:,:]*glR[iD,:,:]*Sigrl[s2,:,:]*gaL[iE,:,:]*Siglr[s1,:,:]*gaR[iF,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[s6,:,:]*grL[iA,:,:]*Siglr[s5,:,:]*grR[iB,:,:]*Sigrl[s4,:,:]*glL[iC,:,:]*Siglr[s3,:,:]*gaR[iD,:,:]*Sigrl[s2,:,:]*gaL[iE,:,:]*Siglr[s1,:,:]*gaR[iF,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[s6,:,:]*grL[iA,:,:]*Siglr[s5,:,:]*glR[iB,:,:]*Sigrl[s4,:,:]*gaL[iC,:,:]*Siglr[s3,:,:]*gaR[iD,:,:]*Sigrl[s2,:,:]*gaL[iE,:,:]*Siglr[s1,:,:]*gaR[iF,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[s6,:,:]*glL[iA,:,:]*Siglr[s5,:,:]*gaR[iB,:,:]*Sigrl[s4,:,:]*gaL[iC,:,:]*Siglr[s3,:,:]*gaR[iD,:,:]*Sigrl[s2,:,:]*gaL[iE,:,:]*Siglr[s1,:,:]*gaR[iF,:,:] ) ) );
                end
            end
        end
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end

"""
    current_Vbias_MPT_T6_pair(war1, Omega, zeta, delta, T, Gamma, War, JL, KL, JR, KR) -> Float64

Pair (ANOMALOUS e-h/h-e GF only) channel of [`current_Vbias_MPT_T6`](@ref). 4x4 per-lead.
"""
function current_Vbias_MPT_T6_pair(war1, Omega, zeta, delta, T, Gamma, War, JL, KL, JR, KR)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]);
    tz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1];
    sigx = ComplexF64[0 1; 1 0]; sigy = ComplexF64[0 -im; im 0]; sigz = ComplexF64[1 0; 0 -1]; sig0 = ComplexF64[1 0; 0 1]; tauz = ComplexF64[1 0; 0 -1];
    JsL = JL[1]*sigx + JL[2]*sigy + JL[3]*sigz; VimpL = [JsL zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsL)] + KL*kron(tauz,sig0);
    JsR = JR[1]*sigx + JR[2]*sigy + JR[3]*sigz; VimpR = [JsR zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsR)] + KR*kron(tauz,sig0);

    Sigrl = zeros(ComplexF64, 2,4,4); Sigrl[1,:,:] = T .* kron([War[1] 0; 0 0],sig0); Sigrl[2,:,:] = T .* kron([0 0; 0 -conj(War[1])],sig0);
    Siglr = zeros(ComplexF64, 2,4,4); Siglr[1,:,:] = T .* kron([0 0; 0 -War[1]],sig0); Siglr[2,:,:] = T .* kron([conj(War[1]) 0; 0 0],sig0);

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        grL = zeros(ComplexF64,7,4,4); glL = zeros(ComplexF64,7,4,4); gaL = zeros(ComplexF64,7,4,4);
        grR = zeros(ComplexF64,7,4,4); glR = zeros(ComplexF64,7,4,4); gaR = zeros(ComplexF64,7,4,4);
        for ab = -3:3
            wwab = war1[hi] + ab*Omega; z = wwab + im*Gamma; ff = (wwab < 0);
            g0 = (1/(zeta*sqrt(delta^2-z^2))) .* ComplexF64[-z 0 0 delta; 0 -z -delta 0; 0 -delta -z 0; delta 0 0 -z];
            gr = (I(4) - g0*VimpL) \ g0; gr[1:2,1:2] .= 0; gr[3:4,3:4] .= 0; ga = gr'; grL[4-ab,:,:] = gr; gaL[4-ab,:,:] = ga; glL[4-ab,:,:] = ff .* ( -(gr-ga) );
            gr = (I(4) - g0*VimpR) \ g0; gr[1:2,1:2] .= 0; gr[3:4,3:4] .= 0; ga = gr'; grR[4-ab,:,:] = gr; gaR[4-ab,:,:] = ga; glR[4-ab,:,:] = ff .* ( -(gr-ga) );
        end

        indarr = 1:6;
        for ab1 = indarr
            indarr1 = indarr[1:end .!= ab1];
            for ab2 = indarr1
                indarr2 = indarr1[ indarr1 .* (indarr1 .!= ab2) .!= 0 ];
                for ab3 = indarr2
                    Sar = -1 .* ones(Int32,6); Sar[ab1] = 1; Sar[ab2] = 1; Sar[ab3] = 1;
                    iA = 4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5]; iB = 4-Sar[1]-Sar[2]-Sar[3]-Sar[4]; iC = 4-Sar[1]-Sar[2]-Sar[3]; iD = 4-Sar[1]-Sar[2]; iE = 4-Sar[1]; iF = 4;
                    s1 = trunc(Int, 0.5*(3-Sar[1])); s2 = trunc(Int, 0.5*(3-Sar[2])); s3 = trunc(Int, 0.5*(3-Sar[3])); s4 = trunc(Int, 0.5*(3-Sar[4])); s5 = trunc(Int, 0.5*(3-Sar[5])); s6 = trunc(Int, 0.5*(3-Sar[6]));
                    # LR chain: GF slots (left->right) leads R,L,R,L,R,L
                    Idcw[hi] = Idcw[hi] + real( tr( tz * ( Siglr[s6,:,:]*grR[iA,:,:]*Sigrl[s5,:,:]*grL[iB,:,:]*Siglr[s4,:,:]*grR[iC,:,:]*Sigrl[s3,:,:]*grL[iD,:,:]*Siglr[s2,:,:]*grR[iE,:,:]*Sigrl[s1,:,:]*glL[iF,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[s6,:,:]*grR[iA,:,:]*Sigrl[s5,:,:]*grL[iB,:,:]*Siglr[s4,:,:]*grR[iC,:,:]*Sigrl[s3,:,:]*grL[iD,:,:]*Siglr[s2,:,:]*glR[iE,:,:]*Sigrl[s1,:,:]*gaL[iF,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[s6,:,:]*grR[iA,:,:]*Sigrl[s5,:,:]*grL[iB,:,:]*Siglr[s4,:,:]*grR[iC,:,:]*Sigrl[s3,:,:]*glL[iD,:,:]*Siglr[s2,:,:]*gaR[iE,:,:]*Sigrl[s1,:,:]*gaL[iF,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[s6,:,:]*grR[iA,:,:]*Sigrl[s5,:,:]*grL[iB,:,:]*Siglr[s4,:,:]*glR[iC,:,:]*Sigrl[s3,:,:]*gaL[iD,:,:]*Siglr[s2,:,:]*gaR[iE,:,:]*Sigrl[s1,:,:]*gaL[iF,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[s6,:,:]*grR[iA,:,:]*Sigrl[s5,:,:]*glL[iB,:,:]*Siglr[s4,:,:]*gaR[iC,:,:]*Sigrl[s3,:,:]*gaL[iD,:,:]*Siglr[s2,:,:]*gaR[iE,:,:]*Sigrl[s1,:,:]*gaL[iF,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[s6,:,:]*glR[iA,:,:]*Sigrl[s5,:,:]*gaL[iB,:,:]*Siglr[s4,:,:]*gaR[iC,:,:]*Sigrl[s3,:,:]*gaL[iD,:,:]*Siglr[s2,:,:]*gaR[iE,:,:]*Sigrl[s1,:,:]*gaL[iF,:,:] ) ) );
                    # RL chain (subtract): vertices swapped, leads L,R,L,R,L,R
                    Idcw[hi] = Idcw[hi] - real( tr( tz * ( Sigrl[s6,:,:]*grL[iA,:,:]*Siglr[s5,:,:]*grR[iB,:,:]*Sigrl[s4,:,:]*grL[iC,:,:]*Siglr[s3,:,:]*grR[iD,:,:]*Sigrl[s2,:,:]*grL[iE,:,:]*Siglr[s1,:,:]*glR[iF,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[s6,:,:]*grL[iA,:,:]*Siglr[s5,:,:]*grR[iB,:,:]*Sigrl[s4,:,:]*grL[iC,:,:]*Siglr[s3,:,:]*grR[iD,:,:]*Sigrl[s2,:,:]*glL[iE,:,:]*Siglr[s1,:,:]*gaR[iF,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[s6,:,:]*grL[iA,:,:]*Siglr[s5,:,:]*grR[iB,:,:]*Sigrl[s4,:,:]*grL[iC,:,:]*Siglr[s3,:,:]*glR[iD,:,:]*Sigrl[s2,:,:]*gaL[iE,:,:]*Siglr[s1,:,:]*gaR[iF,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[s6,:,:]*grL[iA,:,:]*Siglr[s5,:,:]*grR[iB,:,:]*Sigrl[s4,:,:]*glL[iC,:,:]*Siglr[s3,:,:]*gaR[iD,:,:]*Sigrl[s2,:,:]*gaL[iE,:,:]*Siglr[s1,:,:]*gaR[iF,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[s6,:,:]*grL[iA,:,:]*Siglr[s5,:,:]*glR[iB,:,:]*Sigrl[s4,:,:]*gaL[iC,:,:]*Siglr[s3,:,:]*gaR[iD,:,:]*Sigrl[s2,:,:]*gaL[iE,:,:]*Siglr[s1,:,:]*gaR[iF,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[s6,:,:]*glL[iA,:,:]*Siglr[s5,:,:]*gaR[iB,:,:]*Sigrl[s4,:,:]*gaL[iC,:,:]*Siglr[s3,:,:]*gaR[iD,:,:]*Sigrl[s2,:,:]*gaL[iE,:,:]*Siglr[s1,:,:]*gaR[iF,:,:] ) ) );
                end
            end
        end
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end

"""
    current_Vbias_MPT_T6_qp(war1, Omega, zeta, delta, T, Gamma, War, JL, KL, JR, KR) -> Float64

Quasiparticle (NORMAL e-e/h-h GF only) channel of [`current_Vbias_MPT_T6`](@ref). 4x4 per-lead.
"""
function current_Vbias_MPT_T6_qp(war1, Omega, zeta, delta, T, Gamma, War, JL, KL, JR, KR)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]);
    tz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1];
    sigx = ComplexF64[0 1; 1 0]; sigy = ComplexF64[0 -im; im 0]; sigz = ComplexF64[1 0; 0 -1]; sig0 = ComplexF64[1 0; 0 1]; tauz = ComplexF64[1 0; 0 -1];
    JsL = JL[1]*sigx + JL[2]*sigy + JL[3]*sigz; VimpL = [JsL zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsL)] + KL*kron(tauz,sig0);
    JsR = JR[1]*sigx + JR[2]*sigy + JR[3]*sigz; VimpR = [JsR zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) -conj(JsR)] + KR*kron(tauz,sig0);

    Sigrl = zeros(ComplexF64, 2,4,4); Sigrl[1,:,:] = T .* kron([War[1] 0; 0 0],sig0); Sigrl[2,:,:] = T .* kron([0 0; 0 -conj(War[1])],sig0);
    Siglr = zeros(ComplexF64, 2,4,4); Siglr[1,:,:] = T .* kron([0 0; 0 -War[1]],sig0); Siglr[2,:,:] = T .* kron([conj(War[1]) 0; 0 0],sig0);

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        grL = zeros(ComplexF64,7,4,4); glL = zeros(ComplexF64,7,4,4); gaL = zeros(ComplexF64,7,4,4);
        grR = zeros(ComplexF64,7,4,4); glR = zeros(ComplexF64,7,4,4); gaR = zeros(ComplexF64,7,4,4);
        for ab = -3:3
            wwab = war1[hi] + ab*Omega; z = wwab + im*Gamma; ff = (wwab < 0);
            g0 = (1/(zeta*sqrt(delta^2-z^2))) .* ComplexF64[-z 0 0 delta; 0 -z -delta 0; 0 -delta -z 0; delta 0 0 -z];
            gr = (I(4) - g0*VimpL) \ g0; gr[1:2,3:4] .= 0; gr[3:4,1:2] .= 0; ga = gr'; grL[4-ab,:,:] = gr; gaL[4-ab,:,:] = ga; glL[4-ab,:,:] = ff .* ( -(gr-ga) );
            gr = (I(4) - g0*VimpR) \ g0; gr[1:2,3:4] .= 0; gr[3:4,1:2] .= 0; ga = gr'; grR[4-ab,:,:] = gr; gaR[4-ab,:,:] = ga; glR[4-ab,:,:] = ff .* ( -(gr-ga) );
        end

        indarr = 1:6;
        for ab1 = indarr
            indarr1 = indarr[1:end .!= ab1];
            for ab2 = indarr1
                indarr2 = indarr1[ indarr1 .* (indarr1 .!= ab2) .!= 0 ];
                for ab3 = indarr2
                    Sar = -1 .* ones(Int32,6); Sar[ab1] = 1; Sar[ab2] = 1; Sar[ab3] = 1;
                    iA = 4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5]; iB = 4-Sar[1]-Sar[2]-Sar[3]-Sar[4]; iC = 4-Sar[1]-Sar[2]-Sar[3]; iD = 4-Sar[1]-Sar[2]; iE = 4-Sar[1]; iF = 4;
                    s1 = trunc(Int, 0.5*(3-Sar[1])); s2 = trunc(Int, 0.5*(3-Sar[2])); s3 = trunc(Int, 0.5*(3-Sar[3])); s4 = trunc(Int, 0.5*(3-Sar[4])); s5 = trunc(Int, 0.5*(3-Sar[5])); s6 = trunc(Int, 0.5*(3-Sar[6]));
                    # LR chain: GF slots (left->right) leads R,L,R,L,R,L
                    Idcw[hi] = Idcw[hi] + real( tr( tz * ( Siglr[s6,:,:]*grR[iA,:,:]*Sigrl[s5,:,:]*grL[iB,:,:]*Siglr[s4,:,:]*grR[iC,:,:]*Sigrl[s3,:,:]*grL[iD,:,:]*Siglr[s2,:,:]*grR[iE,:,:]*Sigrl[s1,:,:]*glL[iF,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[s6,:,:]*grR[iA,:,:]*Sigrl[s5,:,:]*grL[iB,:,:]*Siglr[s4,:,:]*grR[iC,:,:]*Sigrl[s3,:,:]*grL[iD,:,:]*Siglr[s2,:,:]*glR[iE,:,:]*Sigrl[s1,:,:]*gaL[iF,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[s6,:,:]*grR[iA,:,:]*Sigrl[s5,:,:]*grL[iB,:,:]*Siglr[s4,:,:]*grR[iC,:,:]*Sigrl[s3,:,:]*glL[iD,:,:]*Siglr[s2,:,:]*gaR[iE,:,:]*Sigrl[s1,:,:]*gaL[iF,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[s6,:,:]*grR[iA,:,:]*Sigrl[s5,:,:]*grL[iB,:,:]*Siglr[s4,:,:]*glR[iC,:,:]*Sigrl[s3,:,:]*gaL[iD,:,:]*Siglr[s2,:,:]*gaR[iE,:,:]*Sigrl[s1,:,:]*gaL[iF,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[s6,:,:]*grR[iA,:,:]*Sigrl[s5,:,:]*glL[iB,:,:]*Siglr[s4,:,:]*gaR[iC,:,:]*Sigrl[s3,:,:]*gaL[iD,:,:]*Siglr[s2,:,:]*gaR[iE,:,:]*Sigrl[s1,:,:]*gaL[iF,:,:] ) ) ) +
                                          real( tr( tz * ( Siglr[s6,:,:]*glR[iA,:,:]*Sigrl[s5,:,:]*gaL[iB,:,:]*Siglr[s4,:,:]*gaR[iC,:,:]*Sigrl[s3,:,:]*gaL[iD,:,:]*Siglr[s2,:,:]*gaR[iE,:,:]*Sigrl[s1,:,:]*gaL[iF,:,:] ) ) );
                    # RL chain (subtract): vertices swapped, leads L,R,L,R,L,R
                    Idcw[hi] = Idcw[hi] - real( tr( tz * ( Sigrl[s6,:,:]*grL[iA,:,:]*Siglr[s5,:,:]*grR[iB,:,:]*Sigrl[s4,:,:]*grL[iC,:,:]*Siglr[s3,:,:]*grR[iD,:,:]*Sigrl[s2,:,:]*grL[iE,:,:]*Siglr[s1,:,:]*glR[iF,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[s6,:,:]*grL[iA,:,:]*Siglr[s5,:,:]*grR[iB,:,:]*Sigrl[s4,:,:]*grL[iC,:,:]*Siglr[s3,:,:]*grR[iD,:,:]*Sigrl[s2,:,:]*glL[iE,:,:]*Siglr[s1,:,:]*gaR[iF,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[s6,:,:]*grL[iA,:,:]*Siglr[s5,:,:]*grR[iB,:,:]*Sigrl[s4,:,:]*grL[iC,:,:]*Siglr[s3,:,:]*glR[iD,:,:]*Sigrl[s2,:,:]*gaL[iE,:,:]*Siglr[s1,:,:]*gaR[iF,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[s6,:,:]*grL[iA,:,:]*Siglr[s5,:,:]*grR[iB,:,:]*Sigrl[s4,:,:]*glL[iC,:,:]*Siglr[s3,:,:]*gaR[iD,:,:]*Sigrl[s2,:,:]*gaL[iE,:,:]*Siglr[s1,:,:]*gaR[iF,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[s6,:,:]*grL[iA,:,:]*Siglr[s5,:,:]*glR[iB,:,:]*Sigrl[s4,:,:]*gaL[iC,:,:]*Siglr[s3,:,:]*gaR[iD,:,:]*Sigrl[s2,:,:]*gaL[iE,:,:]*Siglr[s1,:,:]*gaR[iF,:,:] ) ) ) -
                                          real( tr( tz * ( Sigrl[s6,:,:]*glL[iA,:,:]*Siglr[s5,:,:]*gaR[iB,:,:]*Sigrl[s4,:,:]*gaL[iC,:,:]*Siglr[s3,:,:]*gaR[iD,:,:]*Sigrl[s2,:,:]*gaL[iE,:,:]*Siglr[s1,:,:]*gaR[iF,:,:] ) ) );
                end
            end
        end
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end


## ============ I bias =============


"""
    surfacegr(zeta, delta, Gamma, ww, N) -> Matrix{ComplexF64}

Surface (edge) RETARDED Green's function of a semi-infinite 1D BCS lead at energy
`ww`, built by the transfer-matrix / eigendecomposition method of
Phys. Rev. B 83, 085412 (2011) (Eqs. 19–26): the decaying eigenvectors of the
transfer matrix select the surface GF. `N` = sites per unit cell (N=1 for a simple
chain), `zeta` hopping ζ, `delta` gap Δ, `Gamma` broadening Γ. Returns the 2N×2N
Nambu surface GF; for N=1 it coincides with the analytic BCS form used by
[`grwmnf`](@ref).
"""
function surfacegr(zeta, delta, Gamma, ww, N)
    "
    ##------PHYSICAL REVIEW B 83, 085412 (2011) [Eq. 19-26]------
    # Eq. 25: lim_{N->infty} ( (L1*z+0)/(L2*z+0) )^N->0, Lj in ascending order. Unitary matrix: det(TL)=1=> product of evals = 1. This means half>1 and half<1
    ";
    Nmin = trunc(Int, ceil(N/2))
    
    mu = 0;
    H0 = [mu delta; delta -mu] #Single site tip Hamiltonian
    tau3 = [1 0; 0 -1];
    Mh = kron((1.0+0.0*im)*I(N),zeta*tau3); iMh = inv(Mh);
    
    M0 = (ww+im*Gamma)*I(2*N) - kron((1.0+0.0*im)I(N), H0);
    TL = [zeros(ComplexF64, 2*N,2*N) iMh; -Mh M0*iMh];
    EIG = eigen(TL);
    PL = EIG.vectors;
    Lam = EIG.values;
    PL = PL[:, sortperm(Lam, by=abs)]
    sgr = PL[1:2*N,2*N+1:4*N] * inv(PL[2*N+1:4*N,2*N+1:4*N]);
    
    return sgr
end

"""
    grwmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR) -> Matrix{ComplexF64}

Bare RETARDED lead Green's function in the Floquet x (Nambu(x)spin) x lead basis,
4x4 per (lead, Floquet) block in the basis (cup, cdn, cup', cdn'). Block-diagonal in
the Floquet index m in [-Nf,Nf]; each 4x4 block is the impurity-dressed BCS surface
GF at the shifted energy w+m*Omega (see [`gsurf4`](@ref), [`impurity4`](@ref)). The
two leads are built SEPARATELY so they may carry different classical spins: left lead
uses exchange `JL=(Jx,Jy,Jz)` and potential `KL`, right lead `JR,KR`. With
JL=JR=0, KL=KR=0 each block reduces to (two spin copies of) the original 2x2 result.
Returns an 8(2Nf+1) x 8(2Nf+1) matrix (lead outer, Floquet middle, Nambu(x)spin inner).
"""
function grwmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)
    VL = Keldyshsetup_Floquetn_ext.impurity4(JL, KL);
    VR = Keldyshsetup_Floquetn_ext.impurity4(JR, KR);

    grwmn_dL = zeros(ComplexF64, 4*(2*Nf+1), 4*(2*Nf+1));
    grwmn_dR = zeros(ComplexF64, 4*(2*Nf+1), 4*(2*Nf+1));
    for ij = -Nf:Nf
        p = -ij+Nf+1; blk = 4*p-3:4*p;
        z = ww + ij*Omega + im*Gamma;
        grwmn_dL[blk, blk] = Keldyshsetup_Floquetn_ext.gsurf4(z, zeta, delta, VL);
        grwmn_dR[blk, blk] = Keldyshsetup_Floquetn_ext.gsurf4(z, zeta, delta, VR);
    end
    grwmn = [grwmn_dL zeros(size(grwmn_dL));
             zeros(size(grwmn_dL)) grwmn_dR];

    return grwmn
end

"""
    grwmnf_sp(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR) -> SparseMatrixCSC{ComplexF64,Int}

Sparse counterpart of [`grwmnf`](@ref). The bare (impurity-dressed) surface GF is
block-diagonal -- one 4x4 Nambu(x)spin block per Floquet mode, with the two leads
decoupled -- so it is built directly as a sparse block-diagonal matrix
`blkdiag(grwmn_dL, grwmn_dR)` instead of allocating the dense 8(2Nf+1)-square array
(and its two 4(2Nf+1)-square lead halves). Numerically identical to `grwmnf`; the
sparsity is what lets the Dyson solve in [`Grwmnf`](@ref) skip the dense factorization
once the hopping self-energy is also sparse.
"""
function grwmnf_sp(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)
    VL = Keldyshsetup_Floquetn_ext.impurity4(JL, KL);
    VR = Keldyshsetup_Floquetn_ext.impurity4(JR, KR);

    # Floquet mode ij sits at block position p = -ij+Nf+1, so p = 1..2Nf+1 <-> ij = Nf..-Nf.
    # Build the L-lead blocks then the R-lead blocks in that (reversed) order and stack them
    # block-diagonally == [grwmn_dL 0; 0 grwmn_dR].
    blocksL = [sparse(Keldyshsetup_Floquetn_ext.gsurf4(ww + ij*Omega + im*Gamma, zeta, delta, VL)) for ij = Nf:-1:-Nf];
    blocksR = [sparse(Keldyshsetup_Floquetn_ext.gsurf4(ww + ij*Omega + im*Gamma, zeta, delta, VR)) for ij = Nf:-1:-Nf];
    # Without the ..., blockdiag(blocksL) would pass a single argument (the Vector), which blockdiag doesn't accept.
    # blocksL = [A, B, C];  blocksR = [D, E];
    # blockdiag(blocksL..., blocksR...) = blockdiag(A, B, C, D, E)
    grwmn = blockdiag(blocksL..., blocksR...);

    return grwmn
end

"""
    gawmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR) -> Matrix{ComplexF64}

Bare ADVANCED lead Green's function, the -iGamma counterpart of [`grwmnf`](@ref)
(4x4 Nambu(x)spin, per-lead impurity dressing). Equivalent to `grwmnf(...)'`; in the
hot loops the code uses the dagger of the retarded GF instead.
"""
function gawmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)
    VL = Keldyshsetup_Floquetn_ext.impurity4(JL, KL);
    VR = Keldyshsetup_Floquetn_ext.impurity4(JR, KR);

    gawmn_dL = zeros(ComplexF64, 4*(2*Nf+1), 4*(2*Nf+1));
    gawmn_dR = zeros(ComplexF64, 4*(2*Nf+1), 4*(2*Nf+1));
    for ij = -Nf:Nf
        p = -ij+Nf+1; blk = 4*p-3:4*p;
        z = ww + ij*Omega - im*Gamma;
        gawmn_dL[blk, blk] = Keldyshsetup_Floquetn_ext.gsurf4(z, zeta, delta, VL);
        gawmn_dR[blk, blk] = Keldyshsetup_Floquetn_ext.gsurf4(z, zeta, delta, VR);
    end
    gawmn = [gawmn_dL zeros(size(gawmn_dL));
             zeros(size(gawmn_dL)) gawmn_dR];

    return gawmn
end

"""
    glwmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR) -> Matrix{ComplexF64}

Bare LESSER lead Green's function in the 4x4 Nambu(x)spin Floquet basis:
`g< = -(g^r - g^a)` restricted to occupied energies (w+m*Omega < 0, T=0),
block-diagonal in the Floquet index and in the lead. Built from the dressed retarded
GF [`grwmnf`](@ref) (g^a = g^r'), so the equilibrium filling of any YSR bound state
is included automatically.
"""
function glwmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)
    grwmn = Keldyshsetup_Floquetn_ext.grwmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR);
    gawmn = grwmn';

    glwmn = -( grwmn - gawmn );

    off = 4*(2*Nf+1);
    for ij = -Nf:Nf
        p = -ij+Nf+1; blkL = 4*p-3:4*p; blkR = off+4*p-3:off+4*p;
        ff = (ww+ij*Omega) < 0;
        glwmn[blkL, blkL] = ff .* glwmn[blkL, blkL];
        glwmn[blkR, blkR] = ff .* glwmn[blkR, blkR];
    end

    return glwmn
end

"""
    Grwmnf(ww, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR) -> Matrix{ComplexF64}

Full (dressed) RETARDED GF from the Floquet Dyson equation `Gr = (I - gr Sigr)^{-1} gr`,
with `gr` = [`grwmnf`](@ref) (4x4 Nambu(x)spin, per-lead impurity JL,KL,JR,KR) and
`Sigr` = [`Vwmnf`](@ref)`(Vip)`. Solved as a linear system of size 8(2Nf+1).
"""
function Grwmnf(ww, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR)
    # grwmn = Keldyshsetup_Floquetn_ext.grwmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR);
    grwmn = Keldyshsetup_Floquetn_ext.grwmnf_sp(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR);   # sparse block-diagonal bare GF
    # Vv = Keldyshsetup_Floquetn_ext.Vwmnf(Vip, Nf, T);
    Vv = Keldyshsetup_Floquetn_ext.Vwmnf_sp(Vip, Nf, T);   # sparse (banded, off-lead-diagonal) hopping self-energy
    Sigr = Vv;
    # A = I - grwmn*Sigr is now sparse; solve with a DENSE rhs (sparse\sparse is unsupported),
    # i.e. sparse LU factorization + dense solve, giving the (dense) Grwmn expected downstream.
    Grwmn = (I((2*Nf+1)*2*4) - grwmn*Sigr) \ Matrix(grwmn);
    return Grwmn
end

"""
    Gawmnf(ww, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR) -> Matrix{ComplexF64}

Full (dressed) ADVANCED GF `Ga = (I - ga Siga)^{-1} ga`, advanced counterpart of
[`Grwmnf`](@ref) (4x4). In practice the code uses `Grwmn'`, so this is currently unused.
"""
function Gawmnf(ww, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR)
    gawmn = Keldyshsetup_Floquetn_ext.gawmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR);
    Vv = Keldyshsetup_Floquetn_ext.Vwmnf(Vip, Nf, T);
    Siga = Vv;
    Gawmn = (I((2*Nf+1)*2*4) - gawmn*Siga) \ gawmn;
    return Gawmn
end

"""
    Siglf(ww, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR) -> Matrix{ComplexF64}

Lesser self-energy Sig< in the 4x4 Nambu(x)spin Floquet-lead basis: semi-infinite-lead
embedding via the surface term `zeta^2 (tauz(x)sig0) g< (tauz(x)sig0)` (method 1), with
the impurity-dressed `g<` from [`glwmnf`](@ref) so the lead+impurity is treated as one
equilibrium reservoir. The internal flag `Sigl_s` selecting the embedding scheme is
hardcoded to `1` (formerly a function argument); `Sigl_s == 2` (alternative embedding)
is not ported to the 4x4 basis.
"""
function Siglf(ww, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR)
    N8 = 8*(2*Nf+1); off = 4*(2*Nf+1);
    Siglbath = zeros(ComplexF64, N8, N8);
    Siglsurface = zeros(ComplexF64, N8, N8);

    ## Sig< from generic broadening: separate baths attached to each lattice site
    Siglbath_d = zeros(ComplexF64, off, off);
    for ij = -Nf:Nf
        p = -ij+Nf+1; blk = 4*p-3:4*p;
        ff = (ww+ij*Omega) < 0;
        Siglbath_d[blk, blk] = 1 .* ff .* ( +im*2*Gamma ) * I(4);
    end
    Siglbath = [Siglbath_d zeros(size(Siglbath_d));
                zeros(size(Siglbath_d)) Siglbath_d];

    ## Sig< from embedding the semi-infinite leads into the surface site (method 1)
    glwmn = Keldyshsetup_Floquetn_ext.glwmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR);

    tauzs = kron([1 0; 0 -1], [1 0; 0 1]);   # tau_z (x) sigma_0 = diag(1, 1, -1, -1)
    for ij = -Nf:Nf
        p = -ij+Nf+1; blkL = 4*p-3:4*p; blkR = off+4*p-3:off+4*p;
        Siglsurface[blkL, blkL] = zeta^2 .* ( tauzs * glwmn[blkL, blkL] * tauzs );
        Siglsurface[blkR, blkR] = zeta^2 .* ( tauzs * glwmn[blkR, blkR] * tauzs );
    end
    
    Sigl = Siglbath + Siglsurface;

    return Sigl
end

"""
    Glesser_Floquet_Tfull(ww, Omega, Nf, zeta, delta, T, Gamma, Vip,
                          JL, KL, JR, KR, Grwmn=nothing, Gawmn=nothing, Sigl=nothing) -> Matrix{ComplexF64}

Full (all-orders in T) LESSER GF via the Keldysh equation `G< = Gr Sig< Ga`, with `Gr`
from [`Grwmnf`](@ref) and `Sig<` from [`Siglf`](@ref) (4x4 Nambu(x)spin, per-lead
impurity JL,KL,JR,KR). Optional `Grwmn, Gawmn, Sigl` allow passing precomputed pieces.
"""
function Glesser_Floquet_Tfull(ww, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, Grwmn = nothing, Gawmn = nothing, Sigl = nothing)
    if isnothing(Grwmn)
        Grwmn = Keldyshsetup_Floquetn_ext.Grwmnf(ww, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR);
        Gawmn = Grwmn';
    end
    if isnothing(Sigl)
        Sigl = Keldyshsetup_Floquetn_ext.Siglf(ww, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR);
    end
    Glwmn = Grwmn*Sigl*Gawmn;
    return Glwmn
end

"""
    Glesser_Floquet_T2(ww, Omega, Nf, zeta, delta, T, Gamma, Vip,
                       grwmn=nothing, gawmn=nothing, glwmn=nothing) -> Matrix{ComplexF64}

LESSER Green's function truncated at O(𝒯²) in the transparency (Werthamer-type
tunnelling expansion): `G< ≈ gʳ V g< + g< V gᵃ`, with `V`=[`Vwmnf`](@ref)`(Vip)` and
bare propagators from [`grwmnf`](@ref)/[`glwmnf`](@ref). Optional bare GFs may be
passed in. Selected by the `ws=2` scheme.
"""
function Glesser_Floquet_T2(ww, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, grwmn = nothing, gawmn = nothing, glwmn = nothing)
    if isnothing(grwmn)
        grwmn = Keldyshsetup_Floquetn_ext.grwmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)
        gawmn = conj(transpose( grwmn ));
    end
    if isnothing(glwmn)
        glwmn = Keldyshsetup_Floquetn_ext.glwmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)
    end
    Vv = Keldyshsetup_Floquetn_ext.Vwmnf(Vip, Nf, T);

    Glwmn = grwmn*Vv*glwmn + glwmn*Vv*gawmn;

    return Glwmn
end

"""
    Glesser_Floquet_T4(ww, Omega, Nf, zeta, delta, T, Gamma, Vip,
                       grwmn=nothing, gawmn=nothing, glwmn=nothing) -> Matrix{ComplexF64}

LESSER Green's function of the transparency expansion kept through O(𝒯⁴) (the O(𝒯²)
term of [`Glesser_Floquet_T2`](@ref) plus the next chain order). Scheme `ws=4`.
"""
function Glesser_Floquet_T4(ww, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, grwmn = nothing, gawmn = nothing, glwmn = nothing)
    if isnothing(grwmn)
        grwmn = Keldyshsetup_Floquetn_ext.grwmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)
        gawmn = conj(transpose( grwmn ));
    end
    if isnothing(glwmn)
        glwmn = Keldyshsetup_Floquetn_ext.glwmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)
    end
    Vv = Keldyshsetup_Floquetn_ext.Vwmnf(Vip, Nf, T);
    gr_s = grwmn*Vv; s_ga = Vv*gawmn;
    gr_s_gr_s = gr_s*gr_s; s_ga_s_ga = s_ga*s_ga;

    Glwmn = (gr_s*glwmn + glwmn*s_ga) + # O(T²) O(T⁴) terms don't enter current. So it doesn't appear anywhere.
            (gr_s_gr_s*gr_s*glwmn + gr_s_gr_s*glwmn*s_ga + gr_s*glwmn*s_ga_s_ga + glwmn*s_ga*s_ga_s_ga);

    return Glwmn
end

"""
    Glesser_Floquet_T6(ww, Omega, Nf, zeta, delta, T, Gamma, Vip,
                       grwmn=nothing, gawmn=nothing, glwmn=nothing) -> Matrix{ComplexF64}

LESSER Green's function of the transparency expansion kept through O(𝒯⁶). Scheme `ws=6`.
"""
function Glesser_Floquet_T6(ww, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, grwmn = nothing, gawmn = nothing, glwmn = nothing)
    if isnothing(grwmn)
        grwmn = Keldyshsetup_Floquetn_ext.grwmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)
        gawmn = conj(transpose( grwmn ));
    end
    if isnothing(glwmn)
        glwmn = Keldyshsetup_Floquetn_ext.glwmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)
    end
    Vv = Keldyshsetup_Floquetn_ext.Vwmnf(Vip, Nf, T);
    gr_s = grwmn*Vv; s_ga = Vv*gawmn;
    gr_s_gr_s = gr_s*gr_s; s_ga_s_ga = s_ga*s_ga;

    Glwmn = (gr_s*glwmn + glwmn*s_ga) + # O(T²) O(T⁴) terms don't enter current. So it doesn't appear anywhere.
            (gr_s_gr_s*gr_s*glwmn + gr_s_gr_s*glwmn*s_ga + gr_s*glwmn*s_ga_s_ga + glwmn*s_ga*s_ga_s_ga) +
            (gr_s_gr_s*gr_s_gr_s*gr_s*glwmn + gr_s_gr_s*gr_s_gr_s*glwmn*s_ga + gr_s_gr_s*gr_s*glwmn*s_ga_s_ga + gr_s_gr_s*glwmn*s_ga*s_ga_s_ga + gr_s*glwmn*s_ga_s_ga*s_ga_s_ga + glwmn*s_ga*s_ga_s_ga*s_ga_s_ga);

    return Glwmn
end

"""
    Glesser_Floquet_T8(ww, Omega, Nf, zeta, delta, T, Gamma, Vip,
                       grwmn=nothing, gawmn=nothing, glwmn=nothing) -> Matrix{ComplexF64}

LESSER Green's function of the transparency expansion kept through O(𝒯⁸). Scheme `ws=8`.
"""
function Glesser_Floquet_T8(ww, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, grwmn = nothing, gawmn = nothing, glwmn = nothing)
    if isnothing(grwmn)
        grwmn = Keldyshsetup_Floquetn_ext.grwmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)
        gawmn = conj(transpose( grwmn ));
    end
    if isnothing(glwmn)
        glwmn = Keldyshsetup_Floquetn_ext.glwmnf(ww, Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)
    end
    Vv = Keldyshsetup_Floquetn_ext.Vwmnf(Vip, Nf, T);
    gr_s = grwmn*Vv; s_ga = Vv*gawmn;

    Glwmn = (gr_s*glwmn + glwmn*s_ga) + # O(T²) O(T⁴) terms don't enter current. So it doesn't appear anywhere.
            (gr_s*gr_s*gr_s*glwmn + gr_s*gr_s*glwmn*s_ga + gr_s*glwmn*s_ga*s_ga + glwmn*s_ga*s_ga*s_ga) +
            (gr_s*gr_s*gr_s*gr_s*gr_s*glwmn + gr_s*gr_s*gr_s*gr_s*glwmn*s_ga + gr_s*gr_s*gr_s*glwmn*s_ga*s_ga + gr_s*gr_s*glwmn*s_ga*s_ga*s_ga + gr_s*glwmn*s_ga*s_ga*s_ga*s_ga + glwmn*s_ga*s_ga*s_ga*s_ga*s_ga) +
            (gr_s*gr_s*gr_s*gr_s*gr_s*gr_s*gr_s*glwmn + gr_s*gr_s*gr_s*gr_s*gr_s*gr_s*glwmn*s_ga + gr_s*gr_s*gr_s*gr_s*gr_s*glwmn*s_ga*s_ga + gr_s*gr_s*gr_s*gr_s*glwmn*s_ga*s_ga*s_ga + gr_s*gr_s*gr_s*glwmn*s_ga*s_ga*s_ga*s_ga + gr_s*gr_s*glwmn*s_ga*s_ga*s_ga*s_ga*s_ga + gr_s*glwmn*s_ga*s_ga*s_ga*s_ga*s_ga*s_ga + glwmn*s_ga*s_ga*s_ga*s_ga*s_ga*s_ga*s_ga);

    return Glwmn
end


"""
    current_Floquet_Tfull(war0, Omega, Nf, zeta, delta, T, Gamma, Vip,
                          JL, KL, JR, KR, iterprint) -> Matrix{ComplexF64}

Floquet-mode-resolved current for phase solution `Vip`, exact in T (4x4 Nambu(x)spin).
At each energy in `war0` it forms `G<` via [`Glesser_Floquet_Tfull`](@ref) and the
current matrix `-Vi G<` (Vi = [`Viwmnf`](@ref)), takes the Nambu(x)spin trace (4 diag
elements) per Floquet pair (m,n) as lead-L minus lead-R, and integrates over the
one-period grid. Returns `Iif[m,n]`: diagonal sum = DC current, off-diagonals = AC.
"""
function current_Floquet_Tfull(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, iterprint)
    Nw0 = length(war0); deltaw0 = abs(war0[2]-war0[1]);

    Vvi = Keldyshsetup_Floquetn_ext.Viwmnf(Vip, Nf, T); # current vertex: tau_z(x)sig0 folded in
    off = 4*(2*Nf+1);                                   # lead-R offset

    Iiwf = zeros(ComplexF64, 2*Nf+1,2*Nf+1,Nw0);
    Threads.@threads for ij = 1:Nw0
        Glwmn = Keldyshsetup_Floquetn_ext.Glesser_Floquet_Tfull(war0[ij], Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR)
        Itemp0 = -Vvi*Glwmn;
        Threads.@threads for jk = 1:2*Nf+1
            for kl = 1:2*Nf+1
                # Nambu(x)spin trace (4 diagonal components) of Floquet pair (jk,kl): lead L minus lead R
                LL = Itemp0[4*jk-3,4*kl-3] + Itemp0[4*jk-2,4*kl-2] + Itemp0[4*jk-1,4*kl-1] + Itemp0[4*jk,4*kl];
                RR = Itemp0[off+4*jk-3,off+4*kl-3] + Itemp0[off+4*jk-2,off+4*kl-2] + Itemp0[off+4*jk-1,off+4*kl-1] + Itemp0[off+4*jk,off+4*kl];
                Iiwf[jk,kl,ij] = LL - RR;
            end
        end
    end

    Iif = zeros(ComplexF64, 2*Nf+1,2*Nf+1);
    Threads.@threads for jk = 1:2*Nf+1
        for kl = 1:2*Nf+1
            Iif[jk,kl] = (deltaw0 / (2*pi)) * sum(Iiwf[jk,kl,:]);
        end
    end

    return Iif
end

"""
    current_Floquet_T2(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, iterprint) -> Matrix{ComplexF64}

Floquet-mode-resolved current for phase solution `Vip`, truncated at O(𝒯²): like
[`current_Floquet_Tfull`](@ref) but with `G<` from [`Glesser_Floquet_T2`](@ref).
"""
function current_Floquet_T2(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, iterprint)
    Nw0 = length(war0); deltaw0 = abs(war0[2]-war0[1]);

    Vvi = Keldyshsetup_Floquetn_ext.Viwmnf(Vip, Nf, T); #Has extra tau_z Nambu Pauli matrix needed in current calculation
    
    Iiwf = zeros(ComplexF64, 2*Nf+1,2*Nf+1,Nw0);
    Threads.@threads for ij = 1:Nw0
        grwmn = Keldyshsetup_Floquetn_ext.grwmnf(war0[ij], Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR);
        glwmn = Keldyshsetup_Floquetn_ext.glwmnf(war0[ij], Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR);
        
        Glwmn = Keldyshsetup_Floquetn_ext.Glesser_Floquet_T2(war0[ij], Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, grwmn, conj(transpose(grwmn)), glwmn);
        Itemp0 = -Vvi*Glwmn; #Has extra tau_z Nambu Pauli matrix needed in current calculation
        ##--m2--
        Threads.@threads for jk = 1:2*Nf+1
            for kl = 1:2*Nf+1          #VLR*G<RL                                               #VRL*G<LR
                Iiwf[jk,kl,ij] = (Itemp0[4*jk-3,4*kl-3]+Itemp0[4*jk-2,4*kl-2]+Itemp0[4*jk-1,4*kl-1]+Itemp0[4*jk,4*kl]) - (Itemp0[4*(2*Nf+1)+4*jk-3,4*(2*Nf+1)+4*kl-3]+Itemp0[4*(2*Nf+1)+4*jk-2,4*(2*Nf+1)+4*kl-2]+Itemp0[4*(2*Nf+1)+4*jk-1,4*(2*Nf+1)+4*kl-1]+Itemp0[4*(2*Nf+1)+4*jk,4*(2*Nf+1)+4*kl]); #Trace in Nambu space for each Floquet mode (jk)
            end
        end
    end

    Iif = zeros(ComplexF64, 2*Nf+1,2*Nf+1);
    Threads.@threads for jk = 1:2*Nf+1
        for kl = 1:2*Nf+1
            Iif[jk,kl] = (deltaw0 / (2*pi)) * sum(Iiwf[jk,kl,:]);
        end
    end
    
    return Iif
end

"""
    current_Floquet_T4(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, iterprint) -> Matrix{ComplexF64}

Floquet-mode-resolved current truncated at O(𝒯⁴) (uses [`Glesser_Floquet_T4`](@ref)).
"""
function current_Floquet_T4(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, iterprint)
    Nw0 = length(war0); deltaw0 = abs(war0[2]-war0[1]);

    Vvi = Keldyshsetup_Floquetn_ext.Viwmnf(Vip, Nf, T); #Has extra tau_z Nambu Pauli matrix needed in current calculation
    
    Iiwf = zeros(ComplexF64, 2*Nf+1,2*Nf+1,Nw0);
    Threads.@threads for ij = 1:Nw0
        grwmn = Keldyshsetup_Floquetn_ext.grwmnf(war0[ij], Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR);
        glwmn = Keldyshsetup_Floquetn_ext.glwmnf(war0[ij], Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR);
        
        Glwmn = Keldyshsetup_Floquetn_ext.Glesser_Floquet_T4(war0[ij], Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, grwmn, conj(transpose(grwmn)), glwmn);
        Itemp0 = -Vvi*Glwmn; #Has extra tau_z Nambu Pauli matrix needed in current calculation
        ##--m2--
        Threads.@threads for jk = 1:2*Nf+1
            for kl = 1:2*Nf+1          #VLR*G<RL                                               #VRL*G<LR
                Iiwf[jk,kl,ij] = (Itemp0[4*jk-3,4*kl-3]+Itemp0[4*jk-2,4*kl-2]+Itemp0[4*jk-1,4*kl-1]+Itemp0[4*jk,4*kl]) - (Itemp0[4*(2*Nf+1)+4*jk-3,4*(2*Nf+1)+4*kl-3]+Itemp0[4*(2*Nf+1)+4*jk-2,4*(2*Nf+1)+4*kl-2]+Itemp0[4*(2*Nf+1)+4*jk-1,4*(2*Nf+1)+4*kl-1]+Itemp0[4*(2*Nf+1)+4*jk,4*(2*Nf+1)+4*kl]); #Trace in Nambu space for each Floquet mode (jk)
            end
        end
    end

    Iif = zeros(ComplexF64, 2*Nf+1,2*Nf+1);
    Threads.@threads for jk = 1:2*Nf+1
        for kl = 1:2*Nf+1
            Iif[jk,kl] = (deltaw0 / (2*pi)) * sum(Iiwf[jk,kl,:]);
        end
    end
    
    return Iif
end

"""
    current_Floquet_T6(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, iterprint) -> Matrix{ComplexF64}

Floquet-mode-resolved current truncated at O(𝒯⁶) (uses [`Glesser_Floquet_T6`](@ref)).
"""
function current_Floquet_T6(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, iterprint)
    Nw0 = length(war0); deltaw0 = abs(war0[2]-war0[1]);

    Vvi = Keldyshsetup_Floquetn_ext.Viwmnf(Vip, Nf, T); #Has extra tau_z Nambu Pauli matrix needed in current calculation
    
    Iiwf = zeros(ComplexF64, 2*Nf+1,2*Nf+1,Nw0);
    Threads.@threads for ij = 1:Nw0
        grwmn = Keldyshsetup_Floquetn_ext.grwmnf(war0[ij], Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR);
        glwmn = Keldyshsetup_Floquetn_ext.glwmnf(war0[ij], Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR);
        
        Glwmn = Keldyshsetup_Floquetn_ext.Glesser_Floquet_T6(war0[ij], Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, grwmn, conj(transpose(grwmn)), glwmn);
        Itemp0 = -Vvi*Glwmn; #Has extra tau_z Nambu Pauli matrix needed in current calculation
        ##--m2--
        Threads.@threads for jk = 1:2*Nf+1
            for kl = 1:2*Nf+1          #VLR*G<RL                                               #VRL*G<LR
                Iiwf[jk,kl,ij] = (Itemp0[4*jk-3,4*kl-3]+Itemp0[4*jk-2,4*kl-2]+Itemp0[4*jk-1,4*kl-1]+Itemp0[4*jk,4*kl]) - (Itemp0[4*(2*Nf+1)+4*jk-3,4*(2*Nf+1)+4*kl-3]+Itemp0[4*(2*Nf+1)+4*jk-2,4*(2*Nf+1)+4*kl-2]+Itemp0[4*(2*Nf+1)+4*jk-1,4*(2*Nf+1)+4*kl-1]+Itemp0[4*(2*Nf+1)+4*jk,4*(2*Nf+1)+4*kl]); #Trace in Nambu space for each Floquet mode (jk)
            end
        end
    end

    Iif = zeros(ComplexF64, 2*Nf+1,2*Nf+1);
    Threads.@threads for jk = 1:2*Nf+1
        for kl = 1:2*Nf+1
            Iif[jk,kl] = (deltaw0 / (2*pi)) * sum(Iiwf[jk,kl,:]);
        end
    end
    
    return Iif
end

"""
    current_Floquet_T8(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, iterprint) -> Matrix{ComplexF64}

Floquet-mode-resolved current truncated at O(𝒯⁸) (uses [`Glesser_Floquet_T8`](@ref)).
"""
function current_Floquet_T8(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, iterprint)
    Nw0 = length(war0); deltaw0 = abs(war0[2]-war0[1]);

    Vvi = Keldyshsetup_Floquetn_ext.Viwmnf(Vip, Nf, T); #Has extra tau_z Nambu Pauli matrix needed in current calculation
    
    Iiwf = zeros(ComplexF64, 2*Nf+1,2*Nf+1,Nw0);
    Threads.@threads for ij = 1:Nw0
        grwmn = Keldyshsetup_Floquetn_ext.grwmnf(war0[ij], Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR);
        glwmn = Keldyshsetup_Floquetn_ext.glwmnf(war0[ij], Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR);
        
        Glwmn = Keldyshsetup_Floquetn_ext.Glesser_Floquet_T8(war0[ij], Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, grwmn, conj(transpose(grwmn)), glwmn);
        Itemp0 = -Vvi*Glwmn; #Has extra tau_z Nambu Pauli matrix needed in current calculation
        ##--m2--
        Threads.@threads for jk = 1:2*Nf+1
            for kl = 1:2*Nf+1          #VLR*G<RL                                               #VRL*G<LR
                Iiwf[jk,kl,ij] = (Itemp0[4*jk-3,4*kl-3]+Itemp0[4*jk-2,4*kl-2]+Itemp0[4*jk-1,4*kl-1]+Itemp0[4*jk,4*kl]) - (Itemp0[4*(2*Nf+1)+4*jk-3,4*(2*Nf+1)+4*kl-3]+Itemp0[4*(2*Nf+1)+4*jk-2,4*(2*Nf+1)+4*kl-2]+Itemp0[4*(2*Nf+1)+4*jk-1,4*(2*Nf+1)+4*kl-1]+Itemp0[4*(2*Nf+1)+4*jk,4*(2*Nf+1)+4*kl]); #Trace in Nambu space for each Floquet mode (jk)
            end
        end
    end

    Iif = zeros(ComplexF64, 2*Nf+1,2*Nf+1);
    Threads.@threads for jk = 1:2*Nf+1
        for kl = 1:2*Nf+1
            Iif[jk,kl] = (deltaw0 / (2*pi)) * sum(Iiwf[jk,kl,:]);
        end
    end
    
    return Iif
end

"""
    Vwmnf(Vip, Nf, T) -> Matrix{ComplexF64}

Hopping (tunneling) self-energy in the 4x4 Nambu(x)spin Floquet-lead basis,
implementing eq. 8 of arXiv:2602.15213: each lead-offdiagonal 4x4 block is
`W tau_u(x)sig0 - W* tau_d(x)sig0 = kron(diag(W,-W*), sig0)`, spin-conserving, with
`W` the phase Fourier coefficient `Vip` scaled by transparency `T`. `Vip` holds the
Fourier coefficients of exp(-i phi/2) (length 4Nf+1; only odd eV harmonics nonzero).
"""
function Vwmnf(Vip, Nf, T)
    sig0 = ComplexF64[1 0; 0 1];
    Vwmn_d  = zeros(ComplexF64, 4*(2*Nf+1), 4*(2*Nf+1));
    Vwmn_d1 = zeros(ComplexF64, 4*(2*Nf+1), 4*(2*Nf+1));
    for ij = -Nf:Nf
        for jk = -Nf:Nf
            rij = 4*(-ij+Nf+1)-3:4*(-ij+Nf+1); cjk = 4*(-jk+Nf+1)-3:4*(-jk+Nf+1);
            Vwmn_d[rij, cjk]  = T .* kron(ComplexF64[ Vip[-(ij-jk)+2*Nf+1] 0; 0 -conj(Vip[(ij-jk)+2*Nf+1]) ], sig0);
            Vwmn_d1[rij, cjk] = T .* kron(ComplexF64[ conj(Vip[(ij-jk)+2*Nf+1]) 0; 0 -Vip[-(ij-jk)+2*Nf+1] ], sig0);
        end
    end
    Vwmn = [zeros(size(Vwmn_d)) Vwmn_d;
            Vwmn_d1 zeros(size(Vwmn_d))];
    return Vwmn
end

"""
    Vwmnf_sp(Vip, Nf, T) -> SparseMatrixCSC{ComplexF64,Int}

Sparse counterpart of [`Vwmnf`](@ref). The hopping self-energy is off-diagonal in lead
(`[0 Vwmn_d; Vwmn_d1 0]`), banded in Floquet (block `(ij,jk)` carries only the harmonics
`Vip[±(ij-jk)]`, nonzero for odd `ij-jk` within the support of `Vip`), and each 4x4
Nambu(x)spin block is diagonal (`kron(diag,sig0)`). So it is assembled directly from
its (few) nonzero diagonal entries instead of the dense 8(2Nf+1)-square array, keeping
allocations much smaller. Numerically identical to `Vwmnf`; keeping it sparse lets the Dyson LHS
`I - grwmn*Sigr` in [`Grwmnf`](@ref) stay sparse for a banded (rather than dense) solve.
"""
function Vwmnf_sp(Vip, Nf, T)
    Is = Int[]; Js = Int[]; Vs = ComplexF64[];
    for ij = -Nf:Nf
        for jk = -Nf:Nf
            a = Vip[-(ij-jk)+2*Nf+1]; b = Vip[(ij-jk)+2*Nf+1];
            dvals  = (T*a, T*a, -T*conj(b), -T*conj(b));    # Vwmn_d  block = T*diag(a, a, -conj(b), -conj(b))
            d1vals = (T*conj(b), T*conj(b), -T*a, -T*a);    # Vwmn_d1 block = T*diag(conj(b), conj(b), -a, -a)
            rL = 4*(-ij+Nf+1)-3; cR = 4*(2*Nf+1) + 4*(-jk+Nf+1)-3; # Vwmn_d  -> top-right   (L rows, R cols)
            rR = 4*(2*Nf+1) + 4*(-ij+Nf+1)-3; cL = 4*(-jk+Nf+1)-3; # Vwmn_d1 -> bottom-left (R rows, L cols)
            for t = 0:3
                if dvals[t+1] != 0
                    push!(Is, rL+t); push!(Js, cR+t); push!(Vs, dvals[t+1]);
                end
                if d1vals[t+1] != 0
                    push!(Is, rR+t); push!(Js, cL+t); push!(Vs, d1vals[t+1]);
                end
            end
        end
    end
    return sparse(Is, Js, Vs, 2*4*(2*Nf+1), 2*4*(2*Nf+1));
end

"""
    Viwmnf(Vip, Nf, T) -> Matrix{ComplexF64}

Current-vertex variant of [`Vwmnf`](@ref): same hopping but with the hole-Nambu sign
unflipped, i.e. `tau_z(x)sig0` folded in (`Viwmnf = (tau_z(x)sig0) Vwmnf`), as needed
by the current operator `I = tr[(tau_z(x)sig0)(.)]`. Used to form `-Viwmnf G<`.
"""
function Viwmnf(Vip, Nf, T)
    sig0 = ComplexF64[1 0; 0 1];
    Vwmn_d  = zeros(ComplexF64, 4*(2*Nf+1), 4*(2*Nf+1));
    Vwmn_d1 = zeros(ComplexF64, 4*(2*Nf+1), 4*(2*Nf+1));
    for ij = -Nf:Nf
        for jk = -Nf:Nf
            rij = 4*(-ij+Nf+1)-3:4*(-ij+Nf+1); cjk = 4*(-jk+Nf+1)-3:4*(-jk+Nf+1);
            Vwmn_d[rij, cjk]  = T .* kron(ComplexF64[ Vip[-(ij-jk)+2*Nf+1] 0; 0 conj(Vip[(ij-jk)+2*Nf+1]) ], sig0);
            Vwmn_d1[rij, cjk] = T .* kron(ComplexF64[ conj(Vip[(ij-jk)+2*Nf+1]) 0; 0 Vip[-(ij-jk)+2*Nf+1] ], sig0);
        end
    end
    Vwmn = [zeros(size(Vwmn_d)) Vwmn_d;
            Vwmn_d1 zeros(size(Vwmn_d))];
    return Vwmn
end

"""
    IbiasResidual_Tfull(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma,
                        JL, KL, JR, KR, iterprint) -> Vector

Residual F(x) of the current-bias self-consistency, exact in T (4x4 Nambu(x)spin,
per-lead impurity JL,KL,JR,KR). The unknown real vector `Vipi` (length 4Nf) packs the
real and imag parts of the 2Nf odd phase harmonics W_m -- UNCHANGED by the spin
extension. Returns the 4Nf real equations: (1) unitarity of exp(-i phi/2); (2) gauge
phi(0)=0; (3) vanishing of every AC current harmonic I_{2h}=0, with the current from
[`current_Floquet_Tfull`](@ref).
"""
function IbiasResidual_Tfull(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, iterprint)
    Vip = zeros(ComplexF64, 4*Nf+1);
    for jk = 1:2*Nf
        Vip[2*jk] = Vipi[jk] + im*Vipi[(2*Nf)+jk]; #Vipi only has odd harmonics of eV
    end

    eqns = zeros(ComplexF64, 4*Nf);

    #--exp(-iphi/2)*exp(+iphi/2)=1--
    Threads.@threads for hi = 1:Nf-1
        for ij = 1:(4*Nf+1-2*hi) #Assume higher harmonics are 0
            eqns[hi] = eqns[hi] + ( real(Vip[ij+2*hi])*real(Vip[ij]) + imag(Vip[ij+2*hi])*imag(Vip[ij]) );
            eqns[Nf-1+hi] = eqns[Nf-1+hi] + ( -real(Vip[ij+2*hi])*imag(Vip[ij]) + imag(Vip[ij+2*hi])*real(Vip[ij]) );
        end
    end
    for ij = 1:4*Nf+1
        eqns[2*Nf-1] = eqns[2*Nf-1] + ( real(Vip[ij])*real(Vip[ij]) + imag(Vip[ij])*imag(Vip[ij]) );
    end
    eqns[2*Nf-1] = eqns[2*Nf-1] - 1;

    #--phi(t=0)=0--
    eqns[2*Nf] = sum( imag.(Vip) );

    #--Current (4x4 Nambu(x)spin, per-lead impurity)--
    Iif = Keldyshsetup_Floquetn_ext.current_Floquet_Tfull(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, iterprint);
    Ifa = zeros(ComplexF64, 4*Nf+1);
    for kl = -Nf:Nf
        for lm = -Nf:Nf
            Ifa[-(kl-lm)+(2*Nf+1)] = Ifa[-(kl-lm)+(2*Nf+1)] + Iif[-kl+Nf+1,-lm+Nf+1];
        end
    end
    Threads.@threads for hi = 1:Nf
        eqns[(2*Nf)+hi] = real(Ifa[-2*hi+(2*Nf+1)]);
        eqns[(2*Nf+Nf)+hi] = imag(Ifa[-2*hi+(2*Nf+1)]);
    end

    println("iterprint = ",iterprint)
    println("eq norm = ",norm(eqns))

    return eqns
end

"""
    IbiasResidual_T2(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, iterprint) -> Vector{Float64}

Current-bias residual `F(x)` with the current rows evaluated at O(𝒯²) (via
[`current_Floquet_T2`](@ref)); the unitarity and gauge rows are identical to
[`IbiasResidual_Tfull`](@ref). Used by [`phisolve`](@ref) for `ws=2`.
"""
function IbiasResidual_T2(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, iterprint)
    Vip = zeros(ComplexF64, 4*Nf+1);
    for jk = 1:2*Nf
        Vip[2*jk] = Vipi[jk] + im*Vipi[(2*Nf)+jk]; #Vipi only has odd harmonics of eV
    end
       
    eqns = zeros(ComplexF64, 4*Nf);

    #--exp(-iphi/2)*exp(+iphi/2)=1--
    Threads.@threads for hi = 1:Nf-1
        for ij = 1:(4*Nf+1-2*hi) #Assume higher harmonics are 0
            eqns[hi] = eqns[hi] + ( real(Vip[ij+2*hi])*real(Vip[ij]) + imag(Vip[ij+2*hi])*imag(Vip[ij]) );
            eqns[Nf-1+hi] = eqns[Nf-1+hi] + ( -real(Vip[ij+2*hi])*imag(Vip[ij]) + imag(Vip[ij+2*hi])*real(Vip[ij]) );
        end
    end
    for ij = 1:4*Nf+1
        eqns[2*Nf-1] = eqns[2*Nf-1] + ( real(Vip[ij])*real(Vip[ij]) + imag(Vip[ij])*imag(Vip[ij]) );
    end
    eqns[2*Nf-1] = eqns[2*Nf-1] - 1;

    #--phi(t=0)=0--
    eqns[2*Nf] = sum( imag.(Vip) );
    
    #--Current--
    Iif = Keldyshsetup_Floquetn_ext.current_Floquet_T2(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, iterprint);
    Ifa = zeros(ComplexF64, 4*Nf+1);
    for kl = -Nf:Nf
        for lm = -Nf:Nf
            Ifa[-(kl-lm)+(2*Nf+1)] = Ifa[-(kl-lm)+(2*Nf+1)] + Iif[-kl+Nf+1,-lm+Nf+1];
        end
    end
    Threads.@threads for hi = 1:Nf #odd harmonics are 0 anyway (gets better as no. of Floquet modes increases). So manually set the non-zero even harmonics to 0.
        eqns[(2*Nf)+hi] = real(Ifa[-2*hi+(2*Nf+1)]);
        eqns[(2*Nf+Nf)+hi] = imag(Ifa[-2*hi+(2*Nf+1)]);
    end

    println("iterprint = ",iterprint)
    println("eq norm = ",norm(eqns))

    return eqns
end

"""
    IbiasResidual_T4(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, iterprint) -> Vector{Float64}

Current-bias residual with current rows at O(𝒯⁴) ([`current_Floquet_T4`](@ref));
constraint rows as in [`IbiasResidual_Tfull`](@ref). Used by [`phisolve`](@ref) for `ws=4`.
"""
function IbiasResidual_T4(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, iterprint)
    Vip = zeros(ComplexF64, 4*Nf+1);
    for jk = 1:2*Nf
        Vip[2*jk] = Vipi[jk] + im*Vipi[(2*Nf)+jk]; #Vipi only has odd harmonics of eV
    end
       
    eqns = zeros(ComplexF64, 4*Nf);

    #--exp(-iphi/2)*exp(+iphi/2)=1--
    Threads.@threads for hi = 1:Nf-1
        for ij = 1:(4*Nf+1-2*hi) #Assume higher harmonics are 0
            eqns[hi] = eqns[hi] + ( real(Vip[ij+2*hi])*real(Vip[ij]) + imag(Vip[ij+2*hi])*imag(Vip[ij]) );
            eqns[Nf-1+hi] = eqns[Nf-1+hi] + ( -real(Vip[ij+2*hi])*imag(Vip[ij]) + imag(Vip[ij+2*hi])*real(Vip[ij]) );
        end
    end
    for ij = 1:4*Nf+1
        eqns[2*Nf-1] = eqns[2*Nf-1] + ( real(Vip[ij])*real(Vip[ij]) + imag(Vip[ij])*imag(Vip[ij]) );
    end
    eqns[2*Nf-1] = eqns[2*Nf-1] - 1;

    #--phi(t=0)=0--
    eqns[2*Nf] = sum( imag.(Vip) );
    
    #--Current--
    Iif = Keldyshsetup_Floquetn_ext.current_Floquet_T4(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, iterprint);
    Ifa = zeros(ComplexF64, 4*Nf+1);
    for kl = -Nf:Nf
        for lm = -Nf:Nf
            Ifa[-(kl-lm)+(2*Nf+1)] = Ifa[-(kl-lm)+(2*Nf+1)] + Iif[-kl+Nf+1,-lm+Nf+1];
        end
    end
    Threads.@threads for hi = 1:Nf #odd harmonics are 0 anyway (gets better as no. of Floquet modes increases). So manually set the non-zero even harmonics to 0.
        eqns[(2*Nf)+hi] = real(Ifa[-2*hi+(2*Nf+1)]);
        eqns[(2*Nf+Nf)+hi] = imag(Ifa[-2*hi+(2*Nf+1)]);
    end

    println("iterprint = ",iterprint)
    println("eq norm = ",norm(eqns))

    return eqns
end


#-----Jacobian calculation

"""
    IbiasJacobian_Tfull(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma,
                        JL, KL, JR, KR, iterprint) -> Matrix

Analytic Jacobian J_{ij}=dF_i/dx_j of [`IbiasResidual_Tfull`](@ref) (4Nf x 4Nf),
4x4 Nambu(x)spin with per-lead impurity JL,KL,JR,KR. Unitarity/gauge rows are closed
form (unchanged by the spin extension). Current rows use
`dG</dx_j = Gr M_j G< + G< M_j Ga` (valid since Sig< is x-independent and V linear in
x; M_j=dV/dx_j sparse, constant), reorganized so -Vi.Gr and -Vi.G< are formed once per
energy. The Nambu(x)spin trace per Floquet pair sums 4 diagonal components.
"""
function IbiasJacobian_Tfull(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, iterprint)
    # --- Unpack the real unknown vector into complex phase Fourier coefficients ---
    Vip = zeros(ComplexF64, 4*Nf+1);
    for jk = 1:2*Nf
        Vip[2*jk] = Vipi[jk] + im*Vipi[(2*Nf)+jk]; #Vipi only has odd harmonics of eV
    end

    jaceqns = zeros(ComplexF64, 4*Nf, 4*Nf);

    # ===== Rows 1..2Nf: derivatives of the algebraic phase constraints (closed form) =====
    # (UNCHANGED by the 4x4 extension -- the unknowns are the scalar phase harmonics.)
    Threads.@threads for hi = 1:Nf-1
        ct = 0;
        for ij = 1:(4*Nf+1) #variable index, Real components of V
            if ij%2 == 0
                ct = ct + 1;
                if ij+2*hi<=4*Nf+1
                    jaceqns[hi,ct] = jaceqns[hi,ct] + real(Vip[ij+2*hi]);
                    jaceqns[Nf-1+hi,ct] = jaceqns[Nf-1+hi,ct] + imag(Vip[ij+2*hi]);
                end
                if ij-2*hi>=1
                    jaceqns[hi,ct] = jaceqns[hi,ct] + real(Vip[ij-2*hi]);
                    jaceqns[Nf-1+hi,ct] = jaceqns[Nf-1+hi,ct] - imag(Vip[ij-2*hi]);
                end
            end
        end
        for ij = (4*Nf+1)+1:2*(4*Nf+1) #variable index, Imag components of V
            if (ij-(4*Nf+1))%2 == 0
                ct = ct + 1;
                if ij-(4*Nf+1)+2*hi<=4*Nf+1
                    jaceqns[hi,ct] = jaceqns[hi,ct] + imag(Vip[ij-(4*Nf+1)+2*hi]);
                    jaceqns[Nf-1+hi,ct] = jaceqns[Nf-1+hi,ct] - real(Vip[ij-(4*Nf+1)+2*hi]);
                end
                if ij-(4*Nf+1)-2*hi>=1
                    jaceqns[hi,ct] = jaceqns[hi,ct] + imag(Vip[ij-(4*Nf+1)-2*hi]);
                    jaceqns[Nf-1+hi,ct] = jaceqns[Nf-1+hi,ct] + real(Vip[ij-(4*Nf+1)-2*hi]);
                end
            end
        end
    end
    #-- Normalization (row 2Nf-1) and gauge (row 2Nf) --
    jaceqns[2*Nf-1,1:(2*Nf)] .+= 2*real.(Vip[2:2:(4*Nf)]);
    jaceqns[2*Nf-1,(2*Nf)+1:2*(2*Nf)] .+= 2*imag.(Vip[2:2:(4*Nf)]);
    for ij = (2*Nf)+1:2*(2*Nf)
        jaceqns[2*Nf,ij] = 1;
    end

    # ===== Rows 2Nf+1..4Nf: derivatives of the AC current harmonics I_{2h} =====
    Vvi = Keldyshsetup_Floquetn_ext.Viwmnf(Vip, Nf, T); # current vertex (tau_z(x)sig0-folded hopping)
    Nw0 = length(war0); deltaw0 = abs(war0[2]-war0[1]);
    off = 4*(2*Nf+1); # lead-R offset in the 8(2Nf+1)-dim matrices

    jacIif = zeros(ComplexF64, 2*Nf+1,2*Nf+1,2*(2*Nf)); #accumulated over frequency on the fly
    jacIfa = zeros(ComplexF64, 2*(2*Nf),4*Nf+1);

    #-- M_ij = dV/dx_ij: exact (linear) forward difference; each is a single sparse Floquet off-diagonal. --
    deltaV = 1; dV = deltaV .* I(4*Nf);
    Vipn = zeros(ComplexF64, 4*Nf+1);
    Miar = zeros(ComplexF64, 2*4*(2*Nf+1),2*4*(2*Nf+1),4*Nf); Mar = zeros(ComplexF64, 2*4*(2*Nf+1),2*4*(2*Nf+1),4*Nf);
    for ij = 1:2*(2*Nf)
        Vipin = Vipi + dV[:,ij];
        for jk = 1:2*Nf
            Vipn[2*jk] = Vipin[jk] + im*Vipin[(2*Nf)+jk];
        end
        @views Miar[:,:,ij] = ( Keldyshsetup_Floquetn_ext.Viwmnf(Vipn, Nf, T) - Keldyshsetup_Floquetn_ext.Viwmnf(Vip, Nf, T) ) ./ deltaV;
        @views Mar[:,:,ij] = ( Keldyshsetup_Floquetn_ext.Vwmnf(Vipn, Nf, T) - Keldyshsetup_Floquetn_ext.Vwmnf(Vip, Nf, T) ) ./ deltaV;
    end
    Mar = Mar .* (abs.(Mar) .> 0.5*T*deltaV); Miar = -Miar .* (abs.(Miar) .> 0.5*T*deltaV); # drop numerical zeros; fold -1 of d(-Vi.G<) into Miar

    Marsp = [sparse(@view Mar[:,:,ij]) for ij = 1:4*Nf]; Miarsp = [sparse(@view Miar[:,:,ij]) for ij = 1:4*Nf];
    #= ---- OLD: serial over Nw0 (frequency), threaded over the 4Nf columns. Kept for reference. ----
    Glwmntemp = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Glwmn = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); VviGrwmn = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); VviGlwmn = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1));
    tls = TaskLocalValue{Vector{Matrix{ComplexF64}}}( () ->
                     begin
                         t1, t2, t3 = (zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)) for _ in 1:3)
                         [t1, t2, t3]
                     end)
    for gh = 1:Nw0
        if gh%10 == 0
            println(" (jac) outer iter = ",iterprint)
            println(" (jac) w iter/Nw0 = $(gh)/$(Nw0)")
        end
        # Propagators at this frequency, shared by all 4Nf column-derivatives:
        Grwmn = Keldyshsetup_Floquetn_ext.Grwmnf(war0[gh], Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR);
        Gawmn = conj(transpose(Grwmn));
        Sigl = Keldyshsetup_Floquetn_ext.Siglf(war0[gh], Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR);
        mul!(Glwmntemp, Grwmn, Sigl); mul!(Glwmn, Glwmntemp, Gawmn);               # G< = Gr Sig< Ga
        mul!(VviGrwmn, -Vvi, Grwmn); mul!(VviGlwmn, -Vvi, Glwmn);                  # -Vi.Gr , -Vi.G<

        Threads.@threads for ij = 1:(4*Nf)
            t1, t2, t3 = tls[]
            mul!(t1, Marsp[ij], Gawmn);  mul!(t3, VviGlwmn, t1)        # -Vi.G< (dV.Ga)
            mul!(t2, Marsp[ij], Glwmn);  mul!(t3, VviGrwmn, t2, 1, 1)  # + -Vi.Gr (dV.G<)
            mul!(t3, Miarsp[ij], Glwmn, 1, 1)                         # + (-dVi).G<

            # Nambu(x)spin trace (4 diag components) per Floquet pair: lead L (LL) minus lead R (RR)
            for jk = 1:(2*Nf+1)
                for kl = 1:(2*Nf+1)
                    @views jacIif[jk,kl,ij] += (t3[4*jk-3,4*kl-3]+t3[4*jk-2,4*kl-2]+t3[4*jk-1,4*kl-1]+t3[4*jk,4*kl]) - (t3[off+4*jk-3,off+4*kl-3]+t3[off+4*jk-2,off+4*kl-2]+t3[off+4*jk-1,off+4*kl-1]+t3[off+4*jk,off+4*kl]);
                end
            end
        end
    end
    =#

    # ---- NEW: thread over frequency CHUNKS (parallel width = up to Nw0). ----
    # Each chunk owns a private accumulator `partials[c]` and its own scratch matrices (plain
    # locals, allocated once per chunk and reused via mul! across the chunk's frequencies and the
    # 4Nf columns) -- no TaskLocalValue needed, and per-chunk allocation is negligible vs compute.
    # Pair with BLAS.set_num_threads(1) so this Julia-level parallelism isn't oversubscribed.
    N8 = 8*(2*Nf+1);
    nchunks = min(Nw0, max(Threads.nthreads(), 1));
    cb = round.(Int, range(0, Nw0; length = nchunks+1)); # contiguous chunk boundaries of 1:Nw0
    partials = [zeros(ComplexF64, 2*Nf+1, 2*Nf+1, 2*(2*Nf)) for _ = 1:nchunks];
    # Same result as the dense path below, but skips the cross-lead (LR, RL) quadrants of t3
    # that the trace never reads: compute ONLY the lead-diagonal LL and RR quadrants via
    # row/col-restricted GEMMs, (X*Y)[Lrows,Lcols] = X[Lrows,:]*Y[:,Lcols]. Halves the dense
    # 8(2Nf+1)^3 GEMM work (4 quadrant GEMMs of size off-square vs 2 full N8-square). The row
    # slices [1:off]/[off+1:N8] and column slices are unit-stride views, so BLAS needs no copy.
    Threads.@threads for c = 1:nchunks
        jacloc = partials[c];                            # private accumulator (alias, mutated in place)
        Glwmntemp = zeros(ComplexF64, N8, N8); Glwmn = zeros(ComplexF64, N8, N8);
        VviGrwmn  = zeros(ComplexF64, N8, N8); VviGlwmn = zeros(ComplexF64, N8, N8);
        t1 = zeros(ComplexF64, N8, N8); t2 = zeros(ComplexF64, N8, N8); tC = zeros(ComplexF64, N8, N8);
        tLL = zeros(ComplexF64, off, off); tRR = zeros(ComplexF64, off, off);
        VviGlwmnL = @view VviGlwmn[1:off, :];    VviGrwmnL = @view VviGrwmn[1:off, :];       # lead-L rows
        VviGlwmnR = @view VviGlwmn[off+1:N8, :]; VviGrwmnR = @view VviGrwmn[off+1:N8, :];     # lead-R rows
        t1L = @view t1[:, 1:off];    t2L = @view t2[:, 1:off];    tC_LL = @view tC[1:off, 1:off];       # lead-L cols
        t1R = @view t1[:, off+1:N8]; t2R = @view t2[:, off+1:N8]; tC_RR = @view tC[off+1:N8, off+1:N8];  # lead-R cols
        for gh = (cb[c]+1):cb[c+1]
            Grwmn = Keldyshsetup_Floquetn_ext.Grwmnf(war0[gh], Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR);
            Gawmn = conj(transpose(Grwmn));
            Sigl = Keldyshsetup_Floquetn_ext.Siglf(war0[gh], Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR);
            mul!(Glwmntemp, Grwmn, Sigl); mul!(Glwmn, Glwmntemp, Gawmn);               # G< = Gr Sig< Ga
            mul!(VviGrwmn, -Vvi, Grwmn); mul!(VviGlwmn, -Vvi, Glwmn);                  # -Vi.Gr , -Vi.G<
            for ij = 1:(4*Nf)
                # tLL = VviGlwmn[Lrows,:]·t1[:,Lcols] + VviGrwmn[Lrows,:]·t2[:,Lcols] + (Miarsp·Glwmn)[Lrows,Lcols]
                # tRR = VviGlwmn[Rrows,:]·t1[:,Rcols] + VviGrwmn[Rrows,:]·t2[:,Rcols] + (Miarsp·Glwmn)[Rrows,Rcols]
                mul!(t1, Marsp[ij], Gawmn); mul!(t2, Marsp[ij], Glwmn); mul!(tC, Miarsp[ij], Glwmn);
                mul!(tLL, VviGlwmnL, t1L); mul!(tLL, VviGrwmnL, t2L, 1, 1); tLL .+= tC_LL   # LL quadrant = t3[1:off,1:off]
                mul!(tRR, VviGlwmnR, t1R); mul!(tRR, VviGrwmnR, t2R, 1, 1); tRR .+= tC_RR   # RR quadrant = t3[off+1:N8,off+1:N8]
                # Nambu(x)spin trace (4 diag components) per Floquet pair: lead L (LL) minus lead R (RR)
                for jk = 1:(2*Nf+1)
                    for kl = 1:(2*Nf+1)
                        @views jacloc[jk,kl,ij] += (tLL[4*jk-3,4*kl-3]+tLL[4*jk-2,4*kl-2]+tLL[4*jk-1,4*kl-1]+tLL[4*jk,4*kl]) - (tRR[4*jk-3,4*kl-3]+tRR[4*jk-2,4*kl-2]+tRR[4*jk-1,4*kl-1]+tRR[4*jk,4*kl]);
                    end
                end
            end
        end
    end
    # Threads.@threads for c = 1:nchunks
    #     jacloc = partials[c];                            # private accumulator (alias, mutated in place)
    #     Glwmntemp = zeros(ComplexF64, N8, N8); Glwmn = zeros(ComplexF64, N8, N8);
    #     VviGrwmn  = zeros(ComplexF64, N8, N8); VviGlwmn = zeros(ComplexF64, N8, N8);
    #     t1 = zeros(ComplexF64, N8, N8); t2 = zeros(ComplexF64, N8, N8); t3 = zeros(ComplexF64, N8, N8);
    #     for gh = (cb[c]+1):cb[c+1]
    #         Grwmn = Keldyshsetup_Floquetn_ext.Grwmnf(war0[gh], Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR);
    #         Gawmn = conj(transpose(Grwmn));
    #         Sigl = Keldyshsetup_Floquetn_ext.Siglf(war0[gh], Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR);
    #         mul!(Glwmntemp, Grwmn, Sigl); mul!(Glwmn, Glwmntemp, Gawmn);               # G< = Gr Sig< Ga
    #         mul!(VviGrwmn, -Vvi, Grwmn); mul!(VviGlwmn, -Vvi, Glwmn);                  # -Vi.Gr , -Vi.G<
    #         for ij = 1:(4*Nf)
    #             mul!(t1, Marsp[ij], Gawmn);  mul!(t3, VviGlwmn, t1)        # -Vi.G< (dV.Ga)
    #             mul!(t2, Marsp[ij], Glwmn);  mul!(t3, VviGrwmn, t2, 1, 1)  # + -Vi.Gr (dV.G<)
    #             mul!(t3, Miarsp[ij], Glwmn, 1, 1)                         # + (-dVi).G<

    #             # Nambu(x)spin trace (4 diag components) per Floquet pair: lead L (LL) minus lead R (RR)
    #             for jk = 1:(2*Nf+1)
    #                 for kl = 1:(2*Nf+1)
    #                     @views jacloc[jk,kl,ij] += (t3[4*jk-3,4*kl-3]+t3[4*jk-2,4*kl-2]+t3[4*jk-1,4*kl-1]+t3[4*jk,4*kl]) - (t3[off+4*jk-3,off+4*kl-3]+t3[off+4*jk-2,off+4*kl-2]+t3[off+4*jk-1,off+4*kl-1]+t3[off+4*jk,off+4*kl]);
    #                 end
    #             end
    #         end
    #     end
    # end
    jacIif = sum(partials);                              # reduce the per-chunk partials

    jacIif .*= deltaw0 / (2*pi); # complete the (1/2pi) frequency integral

    # Collapse (m,n) Floquet-mode derivatives into current-harmonic derivatives (s=n-m)
    for kl = -Nf:Nf
        for lm = -Nf:Nf
            jacIfa[:,-(kl-lm)+(2*Nf+1)] = jacIfa[:,-(kl-lm)+(2*Nf+1)] + jacIif[-kl+Nf+1,-lm+Nf+1,:];
        end
    end

    # Current rows: bias condition I_{2h}=0 -> rows 2Nf+hi / 3Nf+hi are dRe/dIm I_{2h}
    Threads.@threads for hi = 1:Nf
        jaceqns[(2*Nf)+hi,:] = transpose(real(jacIfa[:,-2*hi+(2*Nf+1)]));
        jaceqns[(2*Nf+Nf)+hi,:] = transpose(imag(jacIfa[:,-2*hi+(2*Nf+1)]));
    end

    println(" (jac) iterprint = ",iterprint)

    return jaceqns
end

"""
    IbiasJacobian_T2(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, iterprint) -> Matrix{Float64}

Analytic Jacobian of [`IbiasResidual_T2`](@ref): same construction as
[`IbiasJacobian_Tfull`](@ref) but with the current-row derivatives built from the
bare propagators / O(𝒯²)-truncated `G<`. Used by [`phisolve`](@ref) for `ws=2`.
"""
function IbiasJacobian_T2(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, iterprint)
    # --- Unpack the real unknown vector into complex phase Fourier coefficients ---
    # Vipi = [ Re(W_1)…Re(W_2Nf) ; Im(W_1)…Im(W_2Nf) ], the 2Nf odd harmonics of exp(-iφ/2).
    # Stored in the even slots of Vip (odd harmonics of eV); even harmonics stay zero.
    Vip = zeros(ComplexF64, 4*Nf+1);
    for jk = 1:2*Nf
        Vip[2*jk] = Vipi[jk] + im*Vipi[(2*Nf)+jk]; #Vipi only has odd harmonics of eV
    end

    # Jacobian to be filled: jaceqns[i,j] = ∂F_i/∂x_j, where F are the 4Nf residual
    # equations (constraints + vanishing AC current) and x are the 4Nf real unknowns.
    jaceqns = zeros(ComplexF64, 4*Nf, 4*Nf);
    
    # ===== Rows 1..2Nf: derivatives of the algebraic phase constraints (closed form) =====
    # Column index ct runs over the unknowns: ct=1..2Nf are ∂/∂Re(W_r), ct=2Nf+1..4Nf are ∂/∂Im(W_r).

    #-- Unitarity exp(-iφ/2)·exp(+iφ/2)=1  ⟹  C_{2h} = Σ_r W_{r+2h} W*_r = 0 (h≠0).
    #   Row hi = ∂Re(C_{2h})/∂x ; row Nf-1+hi = ∂Im(C_{2h})/∂x , for h = hi = 1..Nf-1.
    Threads.@threads for hi = 1:Nf-1
        ct = 0;
        for ij = 1:(4*Nf+1) #variable (for derivative in Jacobian) index. Real components of V
            if ij%2 == 0 # only even slots carry the (odd-harmonic) coefficients
                ct = ct + 1;
                if ij+2*hi<=4*Nf+1 # convolution partner shifted up by 2h
                    jaceqns[hi,ct] = jaceqns[hi,ct] + real(Vip[ij+2*hi]);
                    jaceqns[Nf-1+hi,ct] = jaceqns[Nf-1+hi,ct] + imag(Vip[ij+2*hi]);
                end
                if ij-2*hi>=1 # convolution partner shifted down by 2h
                    jaceqns[hi,ct] = jaceqns[hi,ct] + real(Vip[ij-2*hi]);
                    jaceqns[Nf-1+hi,ct] = jaceqns[Nf-1+hi,ct] - imag(Vip[ij-2*hi]);
                end
            end
        end
        for ij = (4*Nf+1)+1:2*(4*Nf+1) #variable (for derivative in Jacobian) index. Imag components of V
            if (ij-(4*Nf+1))%2 == 0
                ct = ct + 1;
                if ij-(4*Nf+1)+2*hi<=4*Nf+1
                    jaceqns[hi,ct] = jaceqns[hi,ct] + imag(Vip[ij-(4*Nf+1)+2*hi]);
                    jaceqns[Nf-1+hi,ct] = jaceqns[Nf-1+hi,ct] - real(Vip[ij-(4*Nf+1)+2*hi]);
                end
                if ij-(4*Nf+1)-2*hi>=1
                    jaceqns[hi,ct] = jaceqns[hi,ct] + imag(Vip[ij-(4*Nf+1)-2*hi]);
                    jaceqns[Nf-1+hi,ct] = jaceqns[Nf-1+hi,ct] + real(Vip[ij-(4*Nf+1)-2*hi]);
                end
            end
        end
    end
    #-- Normalization (row 2Nf-1): ∂(Σ_r|W_r|²-1)/∂Re(W_r)=2Re(W_r), /∂Im(W_r)=2Im(W_r). --
    jaceqns[2*Nf-1,1:(2*Nf)] .+= 2*real(Vip[2:2:(4*Nf)]);
    jaceqns[2*Nf-1,(2*Nf)+1:2*(2*Nf)] .+= 2*imag(Vip[2:2:(4*Nf)]);

    #-- Gauge (row 2Nf): φ(t=0)=0 ⟹ Σ_m Im(W_m)=0, so ∂/∂Im(W)=1, ∂/∂Re(W)=0 --
    for ij = (2*Nf)+1:2*(2*Nf) #variable (for derivative in Jacobian) index
        jaceqns[2*Nf,ij] = 1;
    end

    # ===== Rows 2Nf+1..4Nf: derivatives of the AC current harmonics I_{2h} =====
    Vvi = Keldyshsetup_Floquetn_ext.Viwmnf(Vip, Nf, T); # current vertex (τ₃-folded hopping)
    Nw0 = length(war0); deltaw0 = abs(war0[2]-war0[1]); # frequency grid over one Floquet zone
                
    jacIif = zeros(ComplexF64, 2*Nf+1,2*Nf+1,2*(2*Nf)); #accumulated over frequency on the fly
    jacIfa = zeros(ComplexF64, 2*(2*Nf),4*Nf+1);
    
    #-- Derivatives of the hopping self-energies w.r.t. each unknown. Because V (Vwmnf)
    #   and the current vertex Vᵢ (Viwmnf) are LINEAR in W, M_ij=∂V/∂x_ij are CONSTANT
    #   matrices, obtained here by an exact forward difference: perturb unknown ij by
    #   deltaV=1 and difference the hopping matrix. Each derivative is a single Floquet
    #   off-diagonal (hence very sparse — exploited below).
    deltaV = 1; dV = deltaV .* I(4*Nf); #2Nf real + 2Nf odd
    Vipn = zeros(ComplexF64, 4*Nf+1); #4*Nf+1 needed for Vwmnf, Viwmnf. Actually use only 4*Nf variables
    Miar = zeros(ComplexF64, 2*4*(2*Nf+1),2*4*(2*Nf+1),4*Nf); Mar = zeros(ComplexF64, 2*4*(2*Nf+1),2*4*(2*Nf+1),4*Nf); # Mar[:,:,ij]=∂V/∂x_ij, Miar[:,:,ij]=∂Vᵢ/∂x_ij
    for ij = 1:2*(2*Nf) #variable (for derivative in Jacobian) index
        Vipin = Vipi + dV[:,ij]; # perturb the ij-th real unknown
        for jk = 1:2*Nf
            Vipn[2*jk] = Vipin[jk] + im*Vipin[(2*Nf)+jk]; #Vipi only has even harmonics of eV, (-Nf*eV:Nf*eV, Nf is even)
        end
        @views Miar[:,:,ij] = ( Keldyshsetup_Floquetn_ext.Viwmnf(Vipn, Nf, T) - Keldyshsetup_Floquetn_ext.Viwmnf(Vip, Nf, T) ) ./ deltaV; # exact (linear) derivative
        @views Mar[:,:,ij] = ( Keldyshsetup_Floquetn_ext.Vwmnf(Vipn, Nf, T) - Keldyshsetup_Floquetn_ext.Vwmnf(Vip, Nf, T) ) ./ deltaV;
    end
    Mar = Mar .* (abs.(Mar) .> 0.5*T*deltaV); Miar = -Miar .* (abs.(Miar) .> 0.5*T*deltaV); # drop numerical-zero entries; fold the leading minus of ∂(-VᵢG<)/∂x into Miar
    
    Marsp = [sparse(@view Mar[:,:,ij]) for ij = 1:4*Nf]; Miarsp = [sparse(@view Miar[:,:,ij]) for ij = 1:4*Nf]; #dV/dx is a single (sparse) Floquet off-diagonal
    #= ---- OLD: serial over Nw0 (frequency), threaded over the 4Nf columns. Kept for reference. ----
    Glwmn_w2 = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Vvigrwmn = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Vviglwmn = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1));
    tls = TaskLocalValue{Vector{Matrix{ComplexF64}}}( () ->
                     begin
                         t1, t2, t3 = (zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)) for _ in 1:3)
                         [t1, t2, t3]
                     end)
    for gh = 1:Nw0
        grwmn = Keldyshsetup_Floquetn_ext.grwmnf(war0[gh], Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)
        gawmn = conj(transpose( grwmn ));
        glwmn = Keldyshsetup_Floquetn_ext.glwmnf(war0[gh], Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)
        Glwmn_w2 .= Keldyshsetup_Floquetn_ext.Glesser_Floquet_T2(war0[gh], Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, grwmn, gawmn, glwmn);
        mul!(Vvigrwmn, -Vvi, grwmn); mul!(Vviglwmn, -Vvi, glwmn);
        Threads.@threads for ij = 1:(4*Nf)
            t1, t2, t3 = tls[]
            mul!(t1, Marsp[ij], gawmn);  mul!(t3, Vviglwmn, t1)
            mul!(t2, Marsp[ij], glwmn);  mul!(t3, Vvigrwmn, t2, 1, 1)
            mul!(t3, Miarsp[ij], Glwmn_w2, 1, 1)
            for jk = 1:(2*Nf+1)
                for kl = 1:(2*Nf+1)
                    @views jacIif[jk,kl,ij] += (t3[4*jk-3,4*kl-3]+t3[4*jk-2,4*kl-2]+t3[4*jk-1,4*kl-1]+t3[4*jk,4*kl]) - (t3[4*(2*Nf+1)+4*jk-3,4*(2*Nf+1)+4*kl-3]+t3[4*(2*Nf+1)+4*jk-2,4*(2*Nf+1)+4*kl-2]+t3[4*(2*Nf+1)+4*jk-1,4*(2*Nf+1)+4*kl-1]+t3[4*(2*Nf+1)+4*jk,4*(2*Nf+1)+4*kl]);
                end
            end
        end
    end
    =#

    # ---- NEW: thread over frequency CHUNKS (parallel width = up to Nw0). Private scratch + accumulator. ----
    N8 = 8*(2*Nf+1);
    nchunks = min(Nw0, max(Threads.nthreads(), 1));
    cb = round.(Int, range(0, Nw0; length = nchunks+1));
    partials = [zeros(ComplexF64, 2*Nf+1, 2*Nf+1, 2*(2*Nf)) for _ = 1:nchunks];
    Threads.@threads for c = 1:nchunks
        jacloc = partials[c];
        Glwmn_w2 = zeros(ComplexF64, N8, N8); Vvigrwmn = zeros(ComplexF64, N8, N8); Vviglwmn = zeros(ComplexF64, N8, N8);
        t1 = zeros(ComplexF64, N8, N8); t2 = zeros(ComplexF64, N8, N8); t3 = zeros(ComplexF64, N8, N8);
        for gh = (cb[c]+1):cb[c+1]
            # Bare lead propagators (the O(𝒯²) scheme expands G< in powers of T):
            grwmn = Keldyshsetup_Floquetn_ext.grwmnf(war0[gh], Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)            # bare retarded gr
            gawmn = conj(transpose( grwmn ));                                                        # bare advanced ga = (gr)†
            glwmn = Keldyshsetup_Floquetn_ext.glwmnf(war0[gh], Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)            # bare lesser g<
            Glwmn_w2 .= Keldyshsetup_Floquetn_ext.Glesser_Floquet_T2(war0[gh], Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, grwmn, gawmn, glwmn); # O(𝒯²) lesser GF
            mul!(Vvigrwmn, -Vvi, grwmn); mul!(Vviglwmn, -Vvi, glwmn);                                 # -Vᵢ·gr , -Vᵢ·g<
            for ij = 1:(4*Nf) #variable (for derivative in Jacobian) index
                mul!(t1, Marsp[ij], gawmn);  mul!(t3, Vviglwmn, t1) # -Vᵢ·g< · (∂V/∂x · ga)
                mul!(t2, Marsp[ij], glwmn);  mul!(t3, Vvigrwmn, t2, 1, 1) # + -Vᵢ·gr · (∂V/∂x · g<)
                mul!(t3, Miarsp[ij], Glwmn_w2, 1, 1) # + (-∂Vᵢ/∂x) · G<₍₂₎
                for jk = 1:(2*Nf+1)
                    for kl = 1:(2*Nf+1)
                        @views jacloc[jk,kl,ij] += (t3[4*jk-3,4*kl-3]+t3[4*jk-2,4*kl-2]+t3[4*jk-1,4*kl-1]+t3[4*jk,4*kl]) - (t3[4*(2*Nf+1)+4*jk-3,4*(2*Nf+1)+4*kl-3]+t3[4*(2*Nf+1)+4*jk-2,4*(2*Nf+1)+4*kl-2]+t3[4*(2*Nf+1)+4*jk-1,4*(2*Nf+1)+4*kl-1]+t3[4*(2*Nf+1)+4*jk,4*(2*Nf+1)+4*kl]);
                    end
                end
            end
        end
    end
    jacIif = sum(partials);

    jacIif .*= deltaw0 / (2*pi); # complete the (1/2π)∫dω frequency integral

    # Collapse the (m,n) Floquet-mode current derivatives into current-harmonic
    # derivatives: harmonic s = n-m, so sum the antidiagonals of jacIif. jacIfa[j, s]
    # = ∂I_s/∂x_j.
    for kl = -Nf:Nf
        for lm = -Nf:Nf
            jacIfa[:,-(kl-lm)+(2*Nf+1)] = jacIfa[:,-(kl-lm)+(2*Nf+1)] + jacIif[-kl+Nf+1,-lm+Nf+1,:];
        end
    end

    # Current rows of the Jacobian: the bias condition is I_{2h}=0, so rows 2Nf+hi /
    # 3Nf+hi are ∂Re(I_{2h})/∂x and ∂Im(I_{2h})/∂x for h=hi=1..Nf (even harmonics 2h).
    Threads.@threads for hi = 1:Nf
        jaceqns[(2*Nf)+hi,:] = transpose(real(jacIfa[:,-2*hi+(2*Nf+1)]));
        jaceqns[(2*Nf+Nf)+hi,:] = transpose(imag(jacIfa[:,-2*hi+(2*Nf+1)]));
    end

    println(" (jac) iterprint = ",iterprint)

    return jaceqns
end

"""
    IbiasJacobian_T4(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, iterprint) -> Matrix{Float64}

Analytic Jacobian of [`IbiasResidual_T4`](@ref) (current-row derivatives at O(𝒯⁴)).
Used by [`phisolve`](@ref) for `ws=4`.
"""
function IbiasJacobian_T4(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, iterprint)
    # --- Unpack the real unknown vector into complex phase Fourier coefficients ---
    # Vipi = [ Re(W_1)…Re(W_2Nf) ; Im(W_1)…Im(W_2Nf) ], the 2Nf odd harmonics of exp(-iφ/2).
    # Stored in the even slots of Vip (odd harmonics of eV); even harmonics stay zero.
    Vip = zeros(ComplexF64, 4*Nf+1);
    for jk = 1:2*Nf
        Vip[2*jk] = Vipi[jk] + im*Vipi[(2*Nf)+jk]; #Vipi only has odd harmonics of eV
    end

    # Jacobian to be filled: jaceqns[i,j] = ∂F_i/∂x_j, where F are the 4Nf residual
    # equations (constraints + vanishing AC current) and x are the 4Nf real unknowns.
    jaceqns = zeros(ComplexF64, 4*Nf, 4*Nf);
    
    # ===== Rows 1..2Nf: derivatives of the algebraic phase constraints (closed form) =====
    # Column index ct runs over the unknowns: ct=1..2Nf are ∂/∂Re(W_r), ct=2Nf+1..4Nf are ∂/∂Im(W_r).

    #-- Unitarity exp(-iφ/2)·exp(+iφ/2)=1  ⟹  C_{2h} = Σ_r W_{r+2h} W*_r = 0 (h≠0).
    #   Row hi = ∂Re(C_{2h})/∂x ; row Nf-1+hi = ∂Im(C_{2h})/∂x , for h = hi = 1..Nf-1.
    Threads.@threads for hi = 1:Nf-1
        ct = 0;
        for ij = 1:(4*Nf+1) #variable (for derivative in Jacobian) index. Real components of V
            if ij%2 == 0 # only even slots carry the (odd-harmonic) coefficients
                ct = ct + 1;
                if ij+2*hi<=4*Nf+1 # convolution partner shifted up by 2h
                    jaceqns[hi,ct] = jaceqns[hi,ct] + real(Vip[ij+2*hi]);
                    jaceqns[Nf-1+hi,ct] = jaceqns[Nf-1+hi,ct] + imag(Vip[ij+2*hi]);
                end
                if ij-2*hi>=1 # convolution partner shifted down by 2h
                    jaceqns[hi,ct] = jaceqns[hi,ct] + real(Vip[ij-2*hi]);
                    jaceqns[Nf-1+hi,ct] = jaceqns[Nf-1+hi,ct] - imag(Vip[ij-2*hi]);
                end
            end
        end
        for ij = (4*Nf+1)+1:2*(4*Nf+1) #variable (for derivative in Jacobian) index. Imag components of V
            if (ij-(4*Nf+1))%2 == 0
                ct = ct + 1;
                if ij-(4*Nf+1)+2*hi<=4*Nf+1
                    jaceqns[hi,ct] = jaceqns[hi,ct] + imag(Vip[ij-(4*Nf+1)+2*hi]);
                    jaceqns[Nf-1+hi,ct] = jaceqns[Nf-1+hi,ct] - real(Vip[ij-(4*Nf+1)+2*hi]);
                end
                if ij-(4*Nf+1)-2*hi>=1
                    jaceqns[hi,ct] = jaceqns[hi,ct] + imag(Vip[ij-(4*Nf+1)-2*hi]);
                    jaceqns[Nf-1+hi,ct] = jaceqns[Nf-1+hi,ct] + real(Vip[ij-(4*Nf+1)-2*hi]);
                end
            end
        end
    end
    #-- Normalization (row 2Nf-1): ∂(Σ_r|W_r|²-1)/∂Re(W_r)=2Re(W_r), /∂Im(W_r)=2Im(W_r). --
    jaceqns[2*Nf-1,1:(2*Nf)] .+= 2*real(Vip[2:2:(4*Nf)]);
    jaceqns[2*Nf-1,(2*Nf)+1:2*(2*Nf)] .+= 2*imag(Vip[2:2:(4*Nf)]);

    #-- Gauge (row 2Nf): φ(t=0)=0 ⟹ Σ_m Im(W_m)=0, so ∂/∂Im(W)=1, ∂/∂Re(W)=0 --
    for ij = (2*Nf)+1:2*(2*Nf) #variable (for derivative in Jacobian) index
        jaceqns[2*Nf,ij] = 1;
    end

    # ===== Rows 2Nf+1..4Nf: derivatives of the AC current harmonics I_{2h} =====
    Vvi = Keldyshsetup_Floquetn_ext.Viwmnf(Vip, Nf, T); # current vertex (τ₃-folded hopping)
    Vv = Keldyshsetup_Floquetn_ext.Vwmnf(Vip, Nf, T);   # hopping self-energy (enters the higher-order G< expansion)

    Nw0 = length(war0); deltaw0 = abs(war0[2]-war0[1]); # frequency grid over one Floquet zone
                
    jacIif = zeros(ComplexF64, 2*Nf+1,2*Nf+1,2*(2*Nf)); #accumulated over frequency on the fly
    jacIfa = zeros(ComplexF64, 2*(2*Nf),4*Nf+1);
    
    #-- Derivatives of the hopping self-energies w.r.t. each unknown. Because V (Vwmnf)
    #   and the current vertex Vᵢ (Viwmnf) are LINEAR in W, M_ij=∂V/∂x_ij are CONSTANT
    #   matrices, obtained here by an exact forward difference: perturb unknown ij by
    #   deltaV=1 and difference the hopping matrix. Each derivative is a single Floquet
    #   off-diagonal (hence very sparse — exploited below).
    deltaV = 1; dV = deltaV .* I(4*Nf); #2Nf real + 2Nf odd
    Vipn = zeros(ComplexF64, 4*Nf+1); #4*Nf+1 needed for Vwmnf, Viwmnf. Actually use only 4*Nf variables
    Miar = zeros(ComplexF64, 2*4*(2*Nf+1),2*4*(2*Nf+1),4*Nf); Mar = zeros(ComplexF64, 2*4*(2*Nf+1),2*4*(2*Nf+1),4*Nf); # Mar[:,:,ij]=∂V/∂x_ij, Miar[:,:,ij]=∂Vᵢ/∂x_ij
    for ij = 1:2*(2*Nf) #variable (for derivative in Jacobian) index
        Vipin = Vipi + dV[:,ij]; # perturb the ij-th real unknown
        for jk = 1:2*Nf
            Vipn[2*jk] = Vipin[jk] + im*Vipin[(2*Nf)+jk]; #Vipi only has even harmonics of eV, (-Nf*eV:Nf*eV, Nf is even)
        end
        @views Miar[:,:,ij] = ( Keldyshsetup_Floquetn_ext.Viwmnf(Vipn, Nf, T) - Keldyshsetup_Floquetn_ext.Viwmnf(Vip, Nf, T) ) ./ deltaV; # exact (linear) derivative
        @views Mar[:,:,ij] = ( Keldyshsetup_Floquetn_ext.Vwmnf(Vipn, Nf, T) - Keldyshsetup_Floquetn_ext.Vwmnf(Vip, Nf, T) ) ./ deltaV;
    end
    Mar = Mar .* (abs.(Mar) .> 0.5*T*deltaV); Miar = -Miar .* (abs.(Miar) .> 0.5*T*deltaV); # drop numerical-zero entries; fold the leading minus of ∂(-VᵢG<)/∂x into Miar

    # Preallocated intermediates for the O(𝒯⁴) derivative. Naming convention: 's' denotes the
    # hopping self-energy Vv (=Σ), and r/l/a denote gr/g</ga, so e.g. gr_s_gl = gr·Σ·g< and
    # Vvigr_s_gr = -Vᵢ·gr·Σ·gr. These chains are the building blocks of the O(𝒯⁴) lesser-GF
    # expansion and its derivative; they are formed once per frequency below.
    #= ---- OLD: serial over Nw0 (frequency), threaded over the 4Nf columns. Kept for reference. ----
    Glwmn_w4 = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1));
    Vvigr = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Vvigl = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Vvigr_s_gr = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Vvigr_s_gl = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Vvigl_s_ga = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1));
    Vvigr_s_gr_s_gr = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Vvigr_s_gr_s_gl = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Vvigr_s_gl_s_ga = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Vvigl_s_ga_s_ga = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1));
    gr_s_gr_s_gl = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); gr_s_gr_s_gr = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); gr_s_gr = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); gr_s_gl = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1));
    gr_s_gl_s_ga = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); gl_s_ga = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1));
    ga_s_ga_s_ga = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); gl_s_ga_s_ga = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); ga_s_ga = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1));
    

    for gh = 1:Nw0
        if gh%10 == 0
            println(" (jac) ev iter = ",iterprint)
            println(" (jac) w iter/Nw0 = $(gh)/$(Nw0)")
        end
        
        # Bare propagators at this frequency:
        grwmn = Keldyshsetup_Floquetn_ext.grwmnf(war0[gh], Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)            # bare retarded gr
        gawmn = conj(transpose( grwmn ));                                                        # bare advanced ga
        glwmn = Keldyshsetup_Floquetn_ext.glwmnf(war0[gh], Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)            # bare lesser g<
        Glwmn_w4 .= Keldyshsetup_Floquetn_ext.Glesser_Floquet_T4(war0[gh], Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, grwmn, gawmn, glwmn); # O(𝒯⁴) lesser GF
        # Chained products gr·Σ·g<, gr·Σ·gr·Σ·g<, … (suffix = factor sequence) entering G<₍₄₎:
        gr_s_gr = grwmn*Vv*grwmn; gr_s_gl = grwmn*Vv*glwmn; gr_s_gr_s_gl = gr_s_gr*Vv*glwmn; gr_s_gr_s_gr = gr_s_gr*Vv*grwmn;
        gr_s_gl_s_ga = gr_s_gl*Vv*gawmn; gl_s_ga = glwmn*Vv*gawmn;
        gl_s_ga_s_ga = gl_s_ga*Vv*gawmn; ga_s_ga = gawmn*Vv*gawmn;
        ga_s_ga_s_ga = ga_s_ga*Vv*gawmn;
        # Pre-multiply each chain by the vertex -Vᵢ (done once per ω, reused for every column ij):
        mul!(Vvigr, -Vvi, grwmn); mul!(Vvigl, -Vvi, glwmn); mul!(Vvigr_s_gr, -Vvi, gr_s_gr); mul!(Vvigr_s_gl, -Vvi, gr_s_gl); mul!(Vvigl_s_ga, -Vvi, gl_s_ga);
        mul!(Vvigr_s_gr_s_gr, -Vvi, gr_s_gr_s_gr); mul!(Vvigr_s_gr_s_gl, -Vvi, gr_s_gr_s_gl); mul!(Vvigr_s_gl_s_ga, -Vvi, gr_s_gl_s_ga); mul!(Vvigl_s_ga_s_ga, -Vvi, gl_s_ga_s_ga);

        # ∂G<₍₄₎/∂x_ij by the product rule: in each chain of G<₍₄₎, replace exactly one Σ (=Vv) by
        # M_ij = ∂V/∂x_ij. Each line below collects the single-Σ replacements at one expansion order;
        # the final term is (-∂Vᵢ/∂x)·G<₍₄₎ (the vertex derivative, with its minus folded into Miar).
        Threads.@threads for ij = 1:(4*Nf) #variable (for derivative in Jacobian) index
            t3 = Vvigr*Mar[:,:,ij]*glwmn + Vvigl*Mar[:,:,ij]*gawmn + # O(𝒯²) part: replace the single Σ
                 Vvigr*Mar[:,:,ij]*gr_s_gr_s_gl + Vvigr_s_gr*Mar[:,:,ij]*gr_s_gl + Vvigr_s_gr_s_gr*Mar[:,:,ij]*glwmn + # O(𝒯⁴): Σ in the gr-gr-g< chain
                 Vvigr*Mar[:,:,ij]*gr_s_gl_s_ga + Vvigr_s_gr*Mar[:,:,ij]*gl_s_ga + Vvigr_s_gr_s_gl*Mar[:,:,ij]*gawmn + # Σ in the gr-g<-ga chain
                 Vvigr*Mar[:,:,ij]*gl_s_ga_s_ga + Vvigr_s_gl*Mar[:,:,ij]*ga_s_ga + Vvigr_s_gl_s_ga*Mar[:,:,ij]*gawmn + # Σ in the g<-ga-ga chain
                 Vvigl*Mar[:,:,ij]*ga_s_ga_s_ga + Vvigl_s_ga*Mar[:,:,ij]*ga_s_ga + Vvigl_s_ga_s_ga*Mar[:,:,ij]*gawmn + # Σ in the ga-ga-ga chain
                 Miar[:,:,ij]*Glwmn_w4; # vertex-derivative term: (-∂Vᵢ/∂x)·G<₍₄₎  (O(𝒯²) part included in Glwmn_w4)

            # Reduce the N×N derivative matrix t3 = ∂(current operator)/∂x_ij to the
            # Floquet-mode-resolved current derivative: for each Floquet block (jk,kl) take
            # the Nambu trace of the left-lead (LL) block minus the right-lead (RR) block,
            # and accumulate the frequency integral on the fly (sum over gh).
            for jk = 1:(2*Nf+1)
                for kl = 1:(2*Nf+1)          # LL Nambu trace (rows/cols 1..2(2Nf+1))     minus RR block (offset 2(2Nf+1))
                    @views jacIif[jk,kl,ij] += (t3[4*jk-3,4*kl-3]+t3[4*jk-2,4*kl-2]+t3[4*jk-1,4*kl-1]+t3[4*jk,4*kl]) - (t3[4*(2*Nf+1)+4*jk-3,4*(2*Nf+1)+4*kl-3]+t3[4*(2*Nf+1)+4*jk-2,4*(2*Nf+1)+4*kl-2]+t3[4*(2*Nf+1)+4*jk-1,4*(2*Nf+1)+4*kl-1]+t3[4*(2*Nf+1)+4*jk,4*(2*Nf+1)+4*kl]); #Trace in Nambu space for each Floquet mode (jk)
                end
            end
        end
    end
    =#

    # ---- NEW: thread over frequency CHUNKS (parallel width = up to Nw0). Private chains/scratch + accumulator. ----
    nchunks = min(Nw0, max(Threads.nthreads(), 1));
    cb = round.(Int, range(0, Nw0; length = nchunks+1));
    partials = [zeros(ComplexF64, 2*Nf+1, 2*Nf+1, 2*(2*Nf)) for _ = 1:nchunks];
    Threads.@threads for c = 1:nchunks
        jacloc = partials[c];
        Glwmn_w4 = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1));
        Vvigr = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Vvigl = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Vvigr_s_gr = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Vvigr_s_gl = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Vvigl_s_ga = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1));
        Vvigr_s_gr_s_gr = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Vvigr_s_gr_s_gl = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Vvigr_s_gl_s_ga = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); Vvigl_s_ga_s_ga = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1));
        gr_s_gr_s_gl = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); gr_s_gr_s_gr = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); gr_s_gr = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); gr_s_gl = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1));
        gr_s_gl_s_ga = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); gl_s_ga = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1));
        ga_s_ga_s_ga = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); gl_s_ga_s_ga = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1)); ga_s_ga = zeros(ComplexF64, 8*(2*Nf+1),8*(2*Nf+1));
    
        for gh = (cb[c]+1):cb[c+1]
        
            # Bare propagators at this frequency:
            grwmn = Keldyshsetup_Floquetn_ext.grwmnf(war0[gh], Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)            # bare retarded gr
            gawmn = conj(transpose( grwmn ));                                                        # bare advanced ga
            glwmn = Keldyshsetup_Floquetn_ext.glwmnf(war0[gh], Omega, Nf, zeta, delta, Gamma, JL, KL, JR, KR)            # bare lesser g<
            Glwmn_w4 .= Keldyshsetup_Floquetn_ext.Glesser_Floquet_T4(war0[gh], Omega, Nf, zeta, delta, T, Gamma, Vip, JL, KL, JR, KR, grwmn, gawmn, glwmn); # O(𝒯⁴) lesser GF
            # Chained products gr·Σ·g<, gr·Σ·gr·Σ·g<, … (suffix = factor sequence) entering G<₍₄₎:
            gr_s_gr = grwmn*Vv*grwmn; gr_s_gl = grwmn*Vv*glwmn; gr_s_gr_s_gl = gr_s_gr*Vv*glwmn; gr_s_gr_s_gr = gr_s_gr*Vv*grwmn;
            gr_s_gl_s_ga = gr_s_gl*Vv*gawmn; gl_s_ga = glwmn*Vv*gawmn;
            gl_s_ga_s_ga = gl_s_ga*Vv*gawmn; ga_s_ga = gawmn*Vv*gawmn;
            ga_s_ga_s_ga = ga_s_ga*Vv*gawmn;
            # Pre-multiply each chain by the vertex -Vᵢ (done once per ω, reused for every column ij):
            mul!(Vvigr, -Vvi, grwmn); mul!(Vvigl, -Vvi, glwmn); mul!(Vvigr_s_gr, -Vvi, gr_s_gr); mul!(Vvigr_s_gl, -Vvi, gr_s_gl); mul!(Vvigl_s_ga, -Vvi, gl_s_ga);
            mul!(Vvigr_s_gr_s_gr, -Vvi, gr_s_gr_s_gr); mul!(Vvigr_s_gr_s_gl, -Vvi, gr_s_gr_s_gl); mul!(Vvigr_s_gl_s_ga, -Vvi, gr_s_gl_s_ga); mul!(Vvigl_s_ga_s_ga, -Vvi, gl_s_ga_s_ga);

            # ∂G<₍₄₎/∂x_ij by the product rule: in each chain of G<₍₄₎, replace exactly one Σ (=Vv) by
            # M_ij = ∂V/∂x_ij. Each line below collects the single-Σ replacements at one expansion order;
            # the final term is (-∂Vᵢ/∂x)·G<₍₄₎ (the vertex derivative, with its minus folded into Miar).
            for ij = 1:(4*Nf) #variable (for derivative in Jacobian) index
                t3 = Vvigr*Mar[:,:,ij]*glwmn + Vvigl*Mar[:,:,ij]*gawmn + # O(𝒯²) part: replace the single Σ
                     Vvigr*Mar[:,:,ij]*gr_s_gr_s_gl + Vvigr_s_gr*Mar[:,:,ij]*gr_s_gl + Vvigr_s_gr_s_gr*Mar[:,:,ij]*glwmn + # O(𝒯⁴): Σ in the gr-gr-g< chain
                     Vvigr*Mar[:,:,ij]*gr_s_gl_s_ga + Vvigr_s_gr*Mar[:,:,ij]*gl_s_ga + Vvigr_s_gr_s_gl*Mar[:,:,ij]*gawmn + # Σ in the gr-g<-ga chain
                     Vvigr*Mar[:,:,ij]*gl_s_ga_s_ga + Vvigr_s_gl*Mar[:,:,ij]*ga_s_ga + Vvigr_s_gl_s_ga*Mar[:,:,ij]*gawmn + # Σ in the g<-ga-ga chain
                     Vvigl*Mar[:,:,ij]*ga_s_ga_s_ga + Vvigl_s_ga*Mar[:,:,ij]*ga_s_ga + Vvigl_s_ga_s_ga*Mar[:,:,ij]*gawmn + # Σ in the ga-ga-ga chain
                     Miar[:,:,ij]*Glwmn_w4; # vertex-derivative term: (-∂Vᵢ/∂x)·G<₍₄₎  (O(𝒯²) part included in Glwmn_w4)

                # Reduce the N×N derivative matrix t3 = ∂(current operator)/∂x_ij to the
                # Floquet-mode-resolved current derivative: for each Floquet block (jk,kl) take
                # the Nambu trace of the left-lead (LL) block minus the right-lead (RR) block,
                # and accumulate the frequency integral on the fly (sum over gh).
                for jk = 1:(2*Nf+1)
                    for kl = 1:(2*Nf+1)          # LL Nambu trace (rows/cols 1..2(2Nf+1))     minus RR block (offset 2(2Nf+1))
                        @views jacloc[jk,kl,ij] += (t3[4*jk-3,4*kl-3]+t3[4*jk-2,4*kl-2]+t3[4*jk-1,4*kl-1]+t3[4*jk,4*kl]) - (t3[4*(2*Nf+1)+4*jk-3,4*(2*Nf+1)+4*kl-3]+t3[4*(2*Nf+1)+4*jk-2,4*(2*Nf+1)+4*kl-2]+t3[4*(2*Nf+1)+4*jk-1,4*(2*Nf+1)+4*kl-1]+t3[4*(2*Nf+1)+4*jk,4*(2*Nf+1)+4*kl]); #Trace in Nambu space for each Floquet mode (jk)
                    end
                end
            end
        end
    end
    jacIif = sum(partials);

    jacIif .*= deltaw0 / (2*pi); # complete the (1/2π)∫dω frequency integral

    # Collapse the (m,n) Floquet-mode current derivatives into current-harmonic
    # derivatives: harmonic s = n-m, so sum the antidiagonals of jacIif. jacIfa[j, s]
    # = ∂I_s/∂x_j.
    for kl = -Nf:Nf
        for lm = -Nf:Nf
            jacIfa[:,-(kl-lm)+(2*Nf+1)] = jacIfa[:,-(kl-lm)+(2*Nf+1)] + jacIif[-kl+Nf+1,-lm+Nf+1,:];
        end
    end

    # Current rows of the Jacobian: the bias condition is I_{2h}=0, so rows 2Nf+hi /
    # 3Nf+hi are ∂Re(I_{2h})/∂x and ∂Im(I_{2h})/∂x for h=hi=1..Nf (even harmonics 2h).
    Threads.@threads for hi = 1:Nf
        jaceqns[(2*Nf)+hi,:] = transpose(real(jacIfa[:,-2*hi+(2*Nf+1)]));
        jaceqns[(2*Nf+Nf)+hi,:] = transpose(imag(jacIfa[:,-2*hi+(2*Nf+1)]));
    end

    println(" (jac) iterprint = ",iterprint)

    return jaceqns
end


"""
    Vt(Vipi, Nf, tar0, Omega) -> Vector{Float64}

Reconstructs the time-domain phase `φ(t)` on the grid `tar0` from the solved Fourier
coefficients `Vipi`: `φ(t) = 2i·log(Σ_m W_m e^{−imΩt})`, with 2π branch unwrapping so
the result is continuous. The voltage waveform is `V(t)=½ dφ/dt`. Used to plot the
self-consistent voltage of a current-biased junction.
"""
function Vt(Vipi, Nf, tar0, Omega)
    Vip = zeros(ComplexF64, 4*Nf+1);
    for jk = 1:2*Nf
        Vip[2*jk] = Vipi[jk] + im*Vipi[(2*Nf)+jk]; #Vipi only has odd harmonics of eV
    end

    Nt0 = length(tar0); eVt0 = zeros(ComplexF64, Nt0); Vt = zeros(Float64, Nt0);
    ctar = zeros(Float64, Nt0);
    for hi = 1:Nt0 
        for ij = 2*Nf:-1:-2*Nf
            eVt0[hi] = eVt0[hi] + Vip[-ij+(2*Nf+1)]*exp(-im*ij*Omega*tar0[hi]);
        end
        Vt[hi] = real( 2*im*log(eVt0[hi]) );
        if hi>1
            if Vt[hi]-Vt[hi-1]>0.9*2*pi
                ctar[hi:Nt0] = ctar[hi:Nt0] .- 1;
            end
            if Vt[hi]-Vt[hi-1]<-0.9*2*pi
                ctar[hi:Nt0] = ctar[hi:Nt0] .+ 1;
            end
        end
    end
    Vt = Vt + ctar .* 4*pi;

    return Vt
end

"""
    phisolve(ws, dw0, evar, Nf, zeta, delta, T, Gamma,
             JL, KL, JR, KR) -> (Iv, Vipsol, residualarr)

Main current-bias driver for the 4x4 Nambu(x)spin (YSR) module. For each bias eV in
`evar` (swept high->low with continuation), solves the self-consistency F(x)=0 with
`NLsolve.nlsolve` (:trust_region) using the analytic [`IbiasJacobian_Tfull`](@ref).
Per-lead classical-spin impurities are `JL,KL` (left) and `JR,KR` (right). Only the
exact-Dyson scheme `ws=0` is supported; perturbative schemes are not yet ported.

Seeding: the largest-|V| point uses the trivial seed and every lower point continues from
its neighbour.

Keyword arguments `ftols` (1e-12), `xtols` (1e-10), `itermax` (40) override the nlsolve
exit criteria, e.g. a larger `itermax` for diagnostic runs where the trust-region crawl
needs more than 40 iterations to traverse an ill-conditioned stretch; `max_scansteps` (0),
`tol_accept` (1e-9), `itermax_scan` (100) control rescue mode. Being keywords, they can
be supplied by name in any order, e.g. `phisolve(...; itermax = 60, max_scansteps = 10)`.

Row equilibration (`scale_current`, default `false`): when enabled, the current-nulling rows
`2Nf+1:4Nf` of both `F` and its Jacobian are multiplied by `RN` (from [`RN_full`](@ref)) so
they become O(Delta) like the unitarity/gauge rows (raw scale `Ic ~ Delta/RN`). This leaves
the root unchanged (row scaling is exact-Newton invariant), so it cannot create a missing
root. It DOES change the `:trust_region` PATH, though: the dogleg merit `||D f||^2` has
gradient `J'D^2 f` skewed by the `RN^2` weighting, so trial steps are rejected and the
Jacobian is recomputed several times per accepted iteration (~4x the Jacobian evaluations,
~1.6-3.7x wall-time in tests). That path change may help OR hurt reaching an existing root at
hard points (empirically TBD), so it is OFF by default (most points converge fine and faster
unscaled) and kept as an opt-in. With it on, `residualarr` and the `ftol`/`tol_accept` tests
are in the scaled metric.

Rescue mode (`max_scansteps > 0`, ws=0): whenever a bias point misses `tol_accept`, it is
re-solved by a Gamma-homotopy at fixed V -- geometrically ramp the Dynes broadening from
`Gam0` (large: subgap MAR/YSR resonances smeared, the branch is smooth and easy) down to
the physical `Gamma` in `max_scansteps` stages, the LAST stage being at the physical
`Gamma`, chaining seeds and starting from the continuation seed. Large Gamma damps out
the sharp resonances that likely make the solution hard to reach. After a failed rescue the
next point inherits the unconverged continuation seed and its own rescue catches it.
`Gam0` (default 2e-1) sets the homotopy start.

# Returns
- `Iv`: DC current vs bias;  `Vipsol`: solved phase coefficients per bias;  `residualarr`: final residual norm.
"""
function phisolve(ws, dw0, evar, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR;
                  ftols = 1e-16, xtols = 1e-13, itermax = 60, max_scansteps = 0, tol_accept = 1e-9, itermax_scan = 100, scale_current = false, Gam0 = 2e-1)
    Nev = length(evar);

    If = zeros(ComplexF64, Nev,2*Nf+1,2*Nf+1); Ifa = zeros(ComplexF64, Nev,4*Nf+1); Iv = zeros(Float64, Nev);
    Vipsol = zeros(Float64, Nev, 2*(2*Nf)); #(2*Nf) real + (2*Nf) imag, only odd multiples of eV kept
    Vipseed = zeros(Float64, 2*(2*Nf));
    residualarr = zeros(Float64, Nev);

    #--Current-row equilibration (OFF by default; opt-in via scale_current=true)-------------
    # Rows 2Nf+1:4Nf are the current-nulling equations I_{2h}=0 (natural scale Ic ~ Delta/RN);
    # rows 1:2Nf (unitarity/gauge) are O(1). Multiplying the current rows AND their Jacobian
    # rows by RN equilibrates them to O(Delta). This does NOT move the root (row scaling is
    # exact-Newton invariant), so it cannot create a missing root. It DOES change NLsolve's
    # :trust_region PATH, though: the dogleg merit ||D f||^2 has gradient J'D^2 f skewed by
    # D^2 (RN^2 on the current rows), so trial steps are rejected and the Jacobian is recomputed
    # several times per accepted iteration (~4x the Jacobian evals, ~1.6-3.7x wall-time). That
    # path change CAN alter whether an existing root is reached at hard points -- may help or
    # hurt, TBD empirically. Default OFF (most points converge fine and faster unscaled); kept
    # as an opt-in for the hard band. RN is bias- and x-independent (computed once).
    RN = scale_current ? Keldyshsetup_Floquetn_ext.RN_full(Nf, dw0, zeta, delta, T, Gamma, JL, KL, JR, KR) : 1.0;
    rowscale = ones(Float64, 4*Nf); rowscale[(2*Nf+1):(4*Nf)] .= RN;
    scale_current && println("-- current-row equilibration ON: RN = $RN (note: inflates Jacobian evals under :trust_region)")

    for hi = Nev:-1:1
        println("ev iter = ",hi)

        ev = evar[hi]; Omega = ev;
        Nw0 = 2*ceil(Int, abs(Omega)/(2*dw0));                             # even cell count: PH-symmetric midpoint sampling
        war0 = -0.5*abs(Omega) .+ ((0:Nw0-1) .+ 0.5) .* (abs(Omega)/Nw0);  # midpoint rule: no sample on the T=0 occupation step

        # active-ws residual/Jacobian for this bias point, and their row-equilibrated forms (Fs,Js)
        Fraw = ws == 4 ? (x -> Keldyshsetup_Floquetn_ext.IbiasResidual_T4(x, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, hi)) :
               ws == 2 ? (x -> Keldyshsetup_Floquetn_ext.IbiasResidual_T2(x, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, hi)) :
                         (x -> Keldyshsetup_Floquetn_ext.IbiasResidual_Tfull(x, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, hi));
        Jraw = ws == 4 ? (x -> Keldyshsetup_Floquetn_ext.IbiasJacobian_T4(x, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, hi)) :
               ws == 2 ? (x -> Keldyshsetup_Floquetn_ext.IbiasJacobian_T2(x, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, hi)) :
                         (x -> Keldyshsetup_Floquetn_ext.IbiasJacobian_Tfull(x, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, hi));
        Fs = scale_current ? (x -> rowscale .* Fraw(x)) : Fraw;   # scale only if opted-in (else zero overhead)
        Js = scale_current ? (x -> rowscale .* Jraw(x)) : Jraw;

        if hi == Nev
            Vipseed[Nf] = 1;               # default: trivial seed at the largest |V|
        else
            Vipseed .= Vipsol[hi+1,:];     # default: continuation from the adjacent bias point
        end

        fseed = Fraw(Vipseed); fvalue = scale_current ? norm(rowscale .* fseed) : norm(fseed);   # seed residual for the skip test

        if fvalue < ftols # seed from higher bias point already satisfies the tolerance, skip the nlsolve
            Vipsol[hi,:] = Vipseed;
            residualarr[hi] = fvalue;
        else # seed from higher bias point does not satisfy the tolerance, run the solver
            t0 = time()
            res = nlsolve(Fs, Js, Vipseed, show_trace=true, method = :trust_region, ftol = ftols; xtol = xtols, iterations = itermax);
            t1 = time()
            println("time = ",t1-t0)
            Vipsol[hi,:] = res.zero;
            residualarr[hi] = res.residual_norm;
        end

        #--Homotopy rescue (max_scansteps > 0, ws=0 only): the direct solve missed tol_accept, so
        #  re-solve at fixed V by a Gamma-homotopy -- geometrically ramp the Dynes broadening from
        #  Gam0 (large: subgap MAR/YSR resonances smeared, branch smooth) down to the physical Gamma
        #  in max_scansteps stages (LAST stage = physical Gamma), chaining seeds from the continuation seed.
        if max_scansteps > 0 && residualarr[hi] > tol_accept && ws == 0
            println("-- Homotopy rescue at ev iter = $hi (direct residual = $(residualarr[hi]))")
            println("   Gamma-homotopy $Gam0 -> $Gamma over $max_scansteps stages (last stage at the physical Gamma)")
            seedh = copy(Vipseed);       # start from the continuation seed
            resnormh = NaN;
            for st = 1:max_scansteps
                # geometric ramp: stage 1 = Gam0, stage max_scansteps = Gamma (physical)
                Gammas = max_scansteps == 1 ? Gamma : Gam0 * (Gamma/Gam0)^((st-1)/(max_scansteps-1));
                println("   scanstep $st/$max_scansteps : Gamma = $Gammas")
                resh = nlsolve(x -> rowscale .* Keldyshsetup_Floquetn_ext.IbiasResidual_Tfull(x, war0, Omega, Nf, zeta, delta, T, Gammas, JL, KL, JR, KR, hi), x -> rowscale .* Keldyshsetup_Floquetn_ext.IbiasJacobian_Tfull(x, war0, Omega, Nf, zeta, delta, T, Gammas, JL, KL, JR, KR, hi), seedh, show_trace=true, method = :trust_region, ftol = ftols; xtol = xtols, iterations = itermax_scan);
                seedh = resh.zero; resnormh = resh.residual_norm;
                println("   scanstep $st/$max_scansteps residual = $(resnormh)")
            end
            if resnormh < tol_accept
                Vipsol[hi,:] = seedh; residualarr[hi] = resnormh;
                println("-- Homotopy rescue SUCCEEDED at ev iter = $hi (residual = $resnormh)")
            else
                println("-- Homotopy rescue FAILED at ev iter = $hi (final residual = $resnormh); terminating this bias point")
                continue # keep the direct result in Vipsol/residualarr as the failure record; Iv stays 0
            end
        end

        #find current using solution, verify only DC component present
        VipI = zeros(ComplexF64, 4*Nf+1);
        for kl = 1:2*Nf
            VipI[2*kl] = Vipsol[hi,kl] + im*Vipsol[hi,(2*Nf)+kl];
        end

        if ws == 4
            If[hi,:,:] = Keldyshsetup_Floquetn_ext.current_Floquet_T4(war0, Omega, Nf, zeta, delta, T, Gamma, VipI, JL, KL, JR, KR, hi);
        elseif ws == 2
            If[hi,:,:] = Keldyshsetup_Floquetn_ext.current_Floquet_T2(war0, Omega, Nf, zeta, delta, T, Gamma, VipI, JL, KL, JR, KR, hi);
        elseif ws == 0
            If[hi,:,:] = Keldyshsetup_Floquetn_ext.current_Floquet_Tfull(war0, Omega, Nf, zeta, delta, T, Gamma, VipI, JL, KL, JR, KR, hi);
        end

        for kl = -Nf:Nf
            for lm = -Nf:Nf
                Ifa[hi,-(kl-lm)+(2*Nf+1)] = Ifa[hi,-(kl-lm)+(2*Nf+1)] + If[hi,-kl+Nf+1,-lm+Nf+1];
            end
        end

        Iv[hi] = real(sum(diag(If[hi,:,:])));
    end

    return Iv, Vipsol, residualarr
end


"""
    RN_full(Nf, dw0, zeta, delta, T, Gamma, JL, KL, JR, KR) -> Float64

Normal-state resistance `R_N`: Ohmic DC current with the gap set to zero (delta1=0) at
a small reference voltage via [`current_Floquet_Tfull`](@ref), then `R_N = V/I`. Pass
JL=JR=0, KL=KR=0 for the clean Ohmic reference, or the actual impurity to normalise by
the same junction's normal-state resistance.
"""
function RN_full(Nf, dw0, zeta, delta, T, Gamma, JL, KL, JR, KR)
    Omega = maximum([0.1, delta/3]); delta1 = 0;
    Nw0 = 2*ceil(Int, abs(Omega)/(2*dw0));                             # even cell count: PH-symmetric midpoint sampling
    war0 = -0.5*abs(Omega) .+ ((0:Nw0-1) .+ 0.5) .* (abs(Omega)/Nw0);  # midpoint rule: no sample on the T=0 occupation step

    VipI = zeros(ComplexF64, 4*Nf+1);
    VipI[2*Nf] = 1;

    If = Keldyshsetup_Floquetn_ext.current_Floquet_Tfull(war0, Omega, Nf, zeta, delta1, T, Gamma, VipI, JL, KL, JR, KR, 0);
    Idc = real(sum(diag(If)));
    GN = (Idc-0)/(Omega-0); RN = 1/GN;
    return RN
end




end
