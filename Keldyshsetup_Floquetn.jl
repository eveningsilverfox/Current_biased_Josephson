module Keldyshsetup_Floquetn

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


## ============ Perturbative current symbolic expression ==============

"""
    currentSym_T2()

Symbolic (Symbolics.jl) derivation of the structure of the O(𝒯²) DC Josephson
current. Builds the lowest-order current as a Nambu-space trace of the hopping
self-energies (`Siglr`, `Sigrl`) and symbolic Green's-function blocks `gij`, then
`simplify`s it, as an analytic aid for reading off which GF / phase-harmonic
combinations contribute at second order in the transparency. Not used by any
solver. (Note: returns the bare symbol `I` rather than the assembled `Ia` — a
pre-existing quirk.)
"""
function currentSym_T2()
    @variables Wa, Wmac, Wb, Wmbc;
    @variables g11a, g12a, g21a, g22a, g11b, g12b, g21b, g22b;

    Siglra = [Wa 0; 0 -Wmac]; # Sigrl1m = [Wm1 0; 0 -W1c];
    Siglrb = [Wb 0; 0 -Wmbc];
    Sigrla = [Wmac 0; 0 -Wa];
    Sigrlb = [Wmbc 0; 0 -Wb];

    #Pure DC bias has only one element as exp(-iphi(t)/2=-ieVdct) contains only eVdc. This kills pure pair current unless eVdc=0, in which case both elements are present.
    #The AC voltage for a DC current bias has both, as exp(-iphi(t)/2) contains both +ne<V> and -ne<V>! This generates a pure DC pair current regardless of the voltage!

    Ia = Symbolics.tr( [1 0; 0 -1] * Siglra * [g11a g12a; g21a g22a] * Sigrlb * [g11b g12b; g21b g22b] );
    simplify(Ia,expand=true)



    return I
end

"""
    currentSym_T4()

Symbolic (Symbolics.jl) derivation of the structure of the O(𝒯⁴) DC current: the
four-vertex Nambu-space trace of the hopping self-energies and symbolic GF blocks,
`simplify`-expanded. Analytic aid only (not called by any solver); the order-𝒯⁴
counterpart of [`currentSym_T2`](@ref). Returns the assembled expression `Ia`.
"""
function currentSym_T4()
    @variables Wa, Wmac, Wb, Wmbc, Wc, Wmcc, Wd, Wmdc;
    @variables g11a, g12a, g21a, g22a, g11b, g12b, g21b, g22b, g11c, g12c, g21c, g22c, g11d, g12d, g21d, g22d ;

    Siglra = [Wa 0; 0 -Wmac]; # Sigrl1m = [Wm1 0; 0 -W1c];
    Siglrb = [Wb 0; 0 -Wmbc];
    Siglrc = [Wc 0; 0 -Wmcc];
    Siglrd = [Wd 0; 0 -Wmdc];
    Sigrla = [Wmac 0; 0 -Wa];
    Sigrlb = [Wmbc 0; 0 -Wb];
    Sigrlc = [Wmcc 0; 0 -Wc];
    Sigrld = [Wmdc 0; 0 -Wd];
    #Pure DC bias has only one element as exp(-iphi(t)/2=-ieVdct) contains only eVdc. This kills pure pair current unless eVdc=0, in which case both elements are present.
    #The AC voltage for a DC current bias has both, as exp(-iphi(t)/2) contains both +ne<V> and -ne<V>! This generates a pure DC pair current regardless of the voltage!

    Ia = Symbolics.tr( [1 0; 0 -1] * Siglra * [g11a g12a; g21a g22a] * Sigrlb * [g11b g12b; g21b g22b] * Siglrc * [g11c g12c; g21c g22c] * Sigrld * [g11d g12d; g21d g22d] );
    simplify(Ia,expand=true)


    return Ia
end



## ============ Critical current ==============


"""
    currentPhi_eq_T2(war1, zeta, delta, T, Gamma, phi) -> Float64

Equilibrium (zero-bias) Josephson current at fixed phase difference `phi`, to
lowest order O(𝒯²) in the transparency. Non-Floquet: the phase is static, so the
hopping self-energies `Sigrl, Siglr = 𝒯·diag(e^∓iφ/2, -e^±iφ/2)` are time-
independent and the current is a single frequency integral over `war1` of the
Keldysh trace `tr[τ3(Σ gʳ Σ g< + Σ g< Σ gᵃ)]` (LR minus RL), using the BCS surface
GF and the bath+surface lesser self-energy. Building block of the current–phase
relation `I(φ)`; `maxᵩ I(φ)` is the critical current.

# Arguments
- `war1`: real-frequency grid (units of Δ).
- `zeta`, `delta`, `T`, `Gamma`: lead hopping ζ, gap Δ, transparency 𝒯, Dynes Γ.
- `phi`: phase difference φ.
"""
function currentPhi_eq_T2(war1, zeta, delta, T, Gamma, phi)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]); tau3 = [1 0; 0 -1];

    Sigrl = zeros(ComplexF64, 2,2); Sigrl =  T .* [exp(-im*phi/2) 0; 0 -exp(+im*phi/2)];
    Siglr = zeros(ComplexF64, 2,2); Siglr =  T .* [exp(+im*phi/2) 0; 0 -exp(-im*phi/2)];

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        # fac = exp(-(ww+ij*Omega)^2 / (zeta^2));
        fac = 1;
        grwar = fac * 1/( zeta*sqrt(delta^2-(war1[hi]+im*Gamma)^2) ) .* [-(war1[hi]+im*Gamma) delta; delta -(war1[hi]+im*Gamma)];
        gawar = conj(transpose(grwar));

        # glwar = -2*im*(wwab .< 0) .* imag.(grwar);
        glwar0 = -2*im*(war1[hi] .< 0) .* imag.(grwar); Sigl = ( im*2*Gamma*I(2) + zeta^2 .* ( tau3 * glwar0 * tau3 ) ) .* (war1[hi] .< 0);
        glwar = grwar * Sigl * conj(transpose(grwar));
        
        Idcwlr = real( tr( tau3 * ( Siglr * grwar * Sigrl * glwar ) ) ) + 
                 real( tr( tau3 * ( Siglr * glwar * Sigrl * gawar ) ) );
        Idcwrl = real( tr( tau3 * ( Sigrl * grwar * Siglr * glwar ) ) ) + 
                 real( tr( tau3 * ( Sigrl * glwar * Siglr * gawar ) ) );
        
        Idcw[hi] = Idcwlr - Idcwrl;
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end

"""
    currentPhi_eq_T4(war1, zeta, delta, T, Gamma, phi) -> Float64

Equilibrium Josephson current at fixed phase `phi`, expanded through O(𝒯⁴): the
O(𝒯²) term of [`currentPhi_eq_T2`](@ref) plus the four-vertex multiple-tunnelling
diagrams (chains `Σ gʳ Σ gʳ Σ gʳ Σ g<` etc.). Non-Floquet, equilibrium
(occupied-state) lesser GFs.
"""
function currentPhi_eq_T4(war1, zeta, delta, T, Gamma, phi)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]); tau3 = [1 0; 0 -1];

    Sigrl = zeros(ComplexF64, 2,2); Sigrl =  T .* [exp(-im*phi/2) 0; 0 -exp(+im*phi/2)];
    Siglr = zeros(ComplexF64, 2,2); Siglr =  T .* [exp(+im*phi/2) 0; 0 -exp(-im*phi/2)];

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        # fac = exp(-(ww+ij*Omega)^2 / (zeta^2));
        fac = 1;

        wwab = war1[hi];
        grwar = fac * 1/( zeta*sqrt(delta^2-(wwab+im*Gamma)^2) ) .* [-(wwab+im*Gamma) delta; delta -(wwab+im*Gamma)];
        glwar = -2*im*(wwab .< 0) .* imag.(grwar);
        gawar = conj(transpose(grwar));
        
        Idcw2lr = real( tr( tau3 * ( Siglr * grwar * Sigrl * glwar ) ) ) + 
                  real( tr( tau3 * ( Siglr * glwar * Sigrl * gawar ) ) );
        Idcw2rl = real( tr( tau3 * ( Sigrl * grwar * Siglr * glwar ) ) ) + 
                  real( tr( tau3 * ( Sigrl * glwar * Siglr * gawar ) ) );
        
        Idcw4lr = real( tr( tau3 * ( Siglr * grwar * Sigrl * grwar * Siglr * grwar * Sigrl * glwar ) ) ) +
                  real( tr( tau3 * ( Siglr * grwar * Sigrl * grwar * Siglr * glwar * Sigrl * gawar ) ) ) + 
                  real( tr( tau3 * ( Siglr * grwar * Sigrl * glwar * Siglr * gawar * Sigrl * gawar ) ) ) + 
                  real( tr( tau3 * ( Siglr * glwar * Sigrl * gawar * Siglr * gawar * Sigrl * gawar ) ) ); 
        Idcw4rl = real( tr( tau3 * ( Sigrl * grwar * Siglr * grwar * Sigrl * grwar * Siglr * glwar ) ) ) +
                  real( tr( tau3 * ( Sigrl * grwar * Siglr * grwar * Sigrl * glwar * Siglr * gawar ) ) ) + 
                  real( tr( tau3 * ( Sigrl * grwar * Siglr * glwar * Sigrl * gawar * Siglr * gawar ) ) ) + 
                  real( tr( tau3 * ( Sigrl * glwar * Siglr * gawar * Sigrl * gawar * Siglr * gawar ) ) );
        Idcw[hi] = Idcw[hi] + Idcw2lr + Idcw4lr - Idcw2rl - Idcw4rl;
    end
    
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end

"""
    currentPhi_eq_Tfull(war1, zeta, delta, T, Gamma, phi) -> Float64

Equilibrium Josephson current at fixed phase `phi`, to ALL orders in the
transparency (full Dyson resummation, non-perturbative). Assembles the 4×4
(two-lead ⊗ Nambu) junction self-energy `Sigrj`, solves the Dyson equation
`Grj = (I − grj·Sigrj)⁻¹ grj` and the Keldysh `Glj = Grj·Siglj·Grj†` at each
frequency, and integrates the current trace. The exact counterpart of
[`currentPhi_eq_T2`](@ref)…`_T8`; used by `Josephson_cphir.jl` to build the exact
current–phase relation.
"""
function currentPhi_eq_Tfull(war1, zeta, delta, T, Gamma, phi)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]); tau3 = [1 0; 0 -1];

    Sigrl = zeros(ComplexF64, 2,2); Sigrl =  T .* [exp(-im*phi/2) 0; 0 -exp(+im*phi/2)];
    Siglr = zeros(ComplexF64, 2,2); Siglr =  T .* [exp(+im*phi/2) 0; 0 -exp(-im*phi/2)];
    
    Sigrj = [zeros(ComplexF64,2,2) Siglr; Sigrl zeros(ComplexF64,2,2)];
    
    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        # fac = exp(-(ww+ij*Omega)^2 / (zeta^2));
        fac = 1;

        wwab = war1[hi];
        grwar = fac * 1/( zeta*sqrt(delta^2-(wwab+im*Gamma)^2) ) .* [-(wwab+im*Gamma) delta; delta -(wwab+im*Gamma)];
        glwar = -2*im*(wwab .< 0) .* imag.(grwar);
        gawar = conj(transpose(grwar));
        
        Siglj = ( im*2*Gamma*I(4) + [ zeta^2 .* ( tau3 * glwar * tau3 ) zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) zeta^2 .* ( tau3 * glwar * tau3 ) ] ) .* (wwab .< 0);
        grjwar = [grwar zeros(ComplexF64,2,2); zeros(ComplexF64,2,2) grwar];
        Grjwar = (I(4) - grjwar*Sigrj) \ grjwar;
        Gljwar = Grjwar * Siglj * conj(transpose(Grjwar));

        Idcwlr = real( tr( tau3 * ( Siglr * Gljwar[3:4,1:2] ) ) );
        Idcwrl = real( tr( tau3 * ( Sigrl * Gljwar[1:2,3:4] ) ) );
        
        Idcw[hi] = Idcw[hi] + Idcwlr - Idcwrl;
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end


## ============ V bias perturbative with explicit energy exchanges (MPT: non-renormalised jumps) ============

"""
    current_Vbias_Floquet_Tfull(war1, ev, zeta, delta, T, Gamma) -> Float64

DC current of a NORMAL (Δ=0) voltage-biased junction at bias `ev`, computed with
the full Floquet machinery: calls [`current_Floquet_Tfull`](@ref) with a single
phase harmonic (`VipI[2Nf]=1`, i.e. pure DC voltage) and sums the diagonal. A
convenience wrapper for the Ohmic/normal-state reference current (fixes `Nf=20`,
builds the one-period grid `war0`). Despite the `_Floquet` tag it lives in the MPT
section as the all-orders reference.
"""
function current_Vbias_Floquet_Tfull(war1, ev, zeta, delta, T, Gamma)
    deltaw1 = abs(war1[2]-war1[1]); tau3 = [1 0; 0 -1];

    Nf = 20; Omega = ev; delta1 = 0;
    
    Nw0 = trunc(Int, abs(Omega)/deltaw1); 
    # war0 = range(0, (Nw0-1)*Omega/Nw0, Nw0);
    war0 = -0.5*abs(Omega) .+ range(0, (Nw0-1)*abs(Omega)/Nw0, Nw0);

    VipI = zeros(ComplexF64, 4*Nf+1);
    VipI[2*Nf] = 1; 
    
    If = Keldyshsetup_Floquetn.current_Floquet_Tfull(war0, Omega, Nf, zeta, delta1, T, Gamma, VipI, 0);
    Ifa = zeros(ComplexF64, 4*Nf+1);
    for kl = -Nf:Nf
        for lm = -Nf:Nf
            Ifa[-(kl-lm)+(2*Nf+1)] = Ifa[-(kl-lm)+(2*Nf+1)] + If[-kl+Nf+1,-lm+Nf+1];
        end
    end
    Idc = real(sum(diag(If)));

    return Idc
end


"""
    current_Vbias_MPT_T2(war1, Omega, zeta, delta, T, Gamma, War) -> Float64

Second-order O(𝒯²) DC current of a voltage-biased junction at bias `Omega = e⟨V⟩`,
via the explicit multiparticle-tunnelling (MPT) expansion with manual energy-
exchange bookkeeping. Each hopping vertex `Sigrl/Siglr` carries a '+energy' and a
'−energy' component (slots `[1,:,:]`/`[2,:,:]`); the inner `Sar` loop enforces net-
zero energy exchange across the two vertices so only DC survives. Uses the full BCS
surface GF (normal + anomalous) and the bath+surface lesser self-energy.

`War` holds the Fourier coefficient(s) of `e^{iφ/2}` (a single harmonic for pure DC
voltage). Returns the DC current `I_LR − I_RL`.
"""
function current_Vbias_MPT_T2(war1, Omega, zeta, delta, T, Gamma, War)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]); tau3 = [1 0; 0 -1];

    Sigrl = zeros(ComplexF64, 2,2,2); Sigrl[1,:,:] = T .* [War[1] 0; 0 0]; Sigrl[2,:,:] = T .* [0 0; 0 -conj(War[1])]; #Valid if War only has a single element, Omega=eV_dc
    Siglr = zeros(ComplexF64, 2,2,2); Siglr[1,:,:] = T .* [0 0; 0 -War[1]]; Siglr[2,:,:] = T .* [conj(War[1]) 0; 0 0];  #Both Sigrl and Siglr ordered such that Sigrl(lr)[1,:,:] adds and Sigrl(lr)[2,:,:] removes energy

    Idcw = zeros(Float64,Nw1);
    grw = zeros(ComplexF64,Nw1,2,2);
    glw = zeros(ComplexF64,Nw1,2,2);
    Threads.@threads for hi = 1:Nw1
        # fac = exp(-(ww+ij*Omega)^2 / (zeta^2));
        fac = 1;

        grwar = zeros(ComplexF64, 3,2,2);
        glwar = zeros(ComplexF64, 3,2,2);
        gawar = zeros(ComplexF64, 3,2,2);

        for ab = -1:1
            wwab = war1[hi] + ab*Omega;
            grwar[2-ab,:,:] = fac * 1/( zeta*sqrt(delta^2-(wwab+im*Gamma)^2) ) .* [-(wwab+im*Gamma) delta; delta -(wwab+im*Gamma)];
            gawar[2-ab,:,:] = conj(transpose(grwar[2-ab,:,:]));

            # glwar[2-ab,:,:] = -2*im*(wwab .< 0) .* imag.(grwar[2-ab,:,:]);
            glwar0 = -2*im*(wwab .< 0) .* imag.(grwar[2-ab,:,:]); Sigl = ( im*2*Gamma*I(2) + zeta^2 .* ( tau3 * glwar0 * tau3 ) ) .* (wwab .< 0);
            glwar[2-ab,:,:] = grwar[2-ab,:,:] * Sigl * conj(transpose(grwar[2-ab,:,:]));
            
        end
        grw[hi,:,:] = grwar[2,:,:];
        glw[hi,:,:] = glwar[2,:,:];
        # Idcw[hi] = real( tr( tau3*( Sm*grpw*Sp*glw  + Sm*glpw*Sp*gaw ) ) ) + real( tr( tau3*( Sp*grmw*Sm*glw  + Sp*glmw*Sm*gaw ) ) );                  

        Sar = -1 .* ones(Int32,2);
        indarr = 1:2;
        Idcwlr = 0; Idcwrl = 0;
        for ab1 = indarr 
            Sar[ab1] = 1; #Net must be zero, so Sar[ab1] adds energy while Sar[!ab1] removes the same energy
            Idcwlr = Idcwlr + real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * grwar[2-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * glwar[2-0,:,:] ) ) ) + 
                              real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * glwar[2-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[2-0,:,:] ) ) );
            Idcwrl = Idcwrl + real( tr( tau3 * ( Sigrl[trunc(Int, 0.5*(3-Sar[2])),:,:] * grwar[2-Sar[1],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[1])),:,:] * glwar[2-0,:,:] ) ) ) +
                              real( tr( tau3 * ( Sigrl[trunc(Int, 0.5*(3-Sar[2])),:,:] * glwar[2-Sar[1],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[2-0,:,:] ) ) );
        end
        Idcw[hi] = Idcwlr - Idcwrl;
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end

"""
    current_Vbias_MPT_T2_qp(war1, Omega, zeta, delta, T, Gamma, War) -> Float64

Quasiparticle-channel part of [`current_Vbias_MPT_T2`](@ref): the same O(𝒯²) MPT
current but with the lead GF reduced to its NORMAL (diagonal, τ₀) component — the
anomalous Δ off-diagonal is dropped — isolating single-quasiparticle tunnelling.
"""
function current_Vbias_MPT_T2_qp(war1, Omega, zeta, delta, T, Gamma, War)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]); tau3 = [1 0; 0 -1];

    Sigrl = zeros(ComplexF64, 2,2,2); Sigrl[1,:,:] = T .* [War[1] 0; 0 0]; Sigrl[2,:,:] = T .* [0 0; 0 -conj(War[1])]; #Valid if War only has a single element, Omega=eV_dc
    Siglr = zeros(ComplexF64, 2,2,2); Siglr[1,:,:] = T .* [0 0; 0 -War[1]]; Siglr[2,:,:] = T .* [conj(War[1]) 0; 0 0];  #Both Sigrl and Siglr ordered such that Sigrl(lr)[1,:,:] adds and Sigrl(lr)[2,:,:] removes energy

    Idcw = zeros(Float64,Nw1);
    grw = zeros(ComplexF64,Nw1,2,2);
    glw = zeros(ComplexF64,Nw1,2,2);
    Threads.@threads for hi = 1:Nw1
        # fac = exp(-(ww+ij*Omega)^2 / (zeta^2));
        fac = 1;

        grwar = zeros(ComplexF64, 3,2,2);
        glwar = zeros(ComplexF64, 3,2,2);
        gawar = zeros(ComplexF64, 3,2,2);

        for ab = -1:1
            wwab = war1[hi] + ab*Omega;
            grwar[2-ab,:,:] = fac * 1/( zeta*sqrt(delta^2-(wwab+im*Gamma)^2) ) .* [-(wwab+im*Gamma) 0; 0 -(wwab+im*Gamma)];
            gawar[2-ab,:,:] = conj(transpose(grwar[2-ab,:,:]));

            # glwar[2-ab,:,:] = -2*im*(wwab .< 0) .* imag.(grwar[2-ab,:,:]);
            glwar0 = -2*im*(wwab .< 0) .* imag.(grwar[2-ab,:,:]); Sigl = ( im*2*Gamma*I(2) + zeta^2 .* ( tau3 * glwar0 * tau3 ) ) .* (wwab .< 0);
            glwar[2-ab,:,:] = grwar[2-ab,:,:] * Sigl * conj(transpose(grwar[2-ab,:,:]));
            
        end
        grw[hi,:,:] = grwar[2,:,:];
        glw[hi,:,:] = glwar[2,:,:];
        # Idcw[hi] = real( tr( tau3*( Sm*grpw*Sp*glw  + Sm*glpw*Sp*gaw ) ) ) + real( tr( tau3*( Sp*grmw*Sm*glw  + Sp*glmw*Sm*gaw ) ) );                  

        Sar = -1 .* ones(Int32,2);
        indarr = 1:2;
        Idcwlr = 0; Idcwrl = 0;
        for ab1 = indarr 
            Sar[ab1] = 1; #Net must be zero, so Sar[ab1] adds energy while Sar[!ab1] removes the same energy
            Idcwlr = Idcwlr + real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * grwar[2-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * glwar[2-0,:,:] ) ) ) + 
                              real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * glwar[2-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[2-0,:,:] ) ) );
            Idcwrl = Idcwrl + real( tr( tau3 * ( Sigrl[trunc(Int, 0.5*(3-Sar[2])),:,:] * grwar[2-Sar[1],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[1])),:,:] * glwar[2-0,:,:] ) ) ) +
                              real( tr( tau3 * ( Sigrl[trunc(Int, 0.5*(3-Sar[2])),:,:] * glwar[2-Sar[1],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[2-0,:,:] ) ) );
        end
        Idcw[hi] = Idcwlr - Idcwrl;
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end

"""
    current_Vbias_MPT_T2_pair(war1, Omega, zeta, delta, T, Gamma, War) -> Float64

Pair (Josephson) channel part of [`current_Vbias_MPT_T2`](@ref): the same O(𝒯²)
MPT current but keeping only the ANOMALOUS (off-diagonal, Δ·τ₁) part of the lead GF
and using both the +eV and −eV phase harmonics (`War[2±1]`) in the vertices,
isolating the DC pair current.
"""
function current_Vbias_MPT_T2_pair(war1, Omega, zeta, delta, T, Gamma, War)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]); tau3 = [1 0; 0 -1];

    Sigrl = zeros(ComplexF64, 2,2,2); Sigrl[1,:,:] = T .* [War[2+1] 0; 0 -conj(War[2+1])]; Sigrl[2,:,:] = T .* [War[2-1] 0; 0 -conj(War[2+1])]; #Valid if War only has a single element, Omega=eV_dc
    Siglr = zeros(ComplexF64, 2,2,2); Siglr[1,:,:] = T .* [conj(War[2+1]) 0; 0 -War[2-1]]; Siglr[2,:,:] = T .* [conj(War[2+1]) 0; 0 -War[2-1]];  #Both Sigrl and Siglr ordered such that Sigrl(lr)[1,:,:] adds and Sigrl(lr)[2,:,:] removes energy

    Idcw = zeros(Float64,Nw1);
    grw = zeros(ComplexF64,Nw1,2,2);
    glw = zeros(ComplexF64,Nw1,2,2);
    Threads.@threads for hi = 1:Nw1
        # fac = exp(-(ww+ij*Omega)^2 / (zeta^2));
        fac = 1;

        grwar = zeros(ComplexF64, 3,2,2);
        glwar = zeros(ComplexF64, 3,2,2);
        gawar = zeros(ComplexF64, 3,2,2);

        for ab = -1:1
            wwab = war1[hi] + ab*Omega;
            grwar[2-ab,:,:] = fac * 1/( zeta*sqrt(delta^2-(wwab+im*Gamma)^2) ) .* [0 delta; delta 0];
            gawar[2-ab,:,:] = conj(transpose(grwar[2-ab,:,:]));

            # glwar[2-ab,:,:] = -2*im*(wwab .< 0) .* imag.(grwar[2-ab,:,:]);
            glwar0 = -2*im*(wwab .< 0) .* imag.(grwar[2-ab,:,:]); Sigl = ( im*2*Gamma*I(2) + zeta^2 .* ( tau3 * glwar0 * tau3 ) ) .* (wwab .< 0);
            glwar[2-ab,:,:] = grwar[2-ab,:,:] * Sigl * conj(transpose(grwar[2-ab,:,:]));
            
        end
        grw[hi,:,:] = grwar[2,:,:];
        glw[hi,:,:] = glwar[2,:,:];
        # Idcw[hi] = real( tr( tau3*( Sm*grpw*Sp*glw  + Sm*glpw*Sp*gaw ) ) ) + real( tr( tau3*( Sp*grmw*Sm*glw  + Sp*glmw*Sm*gaw ) ) );                  

        Sar = -1 .* ones(Int32,2);
        indarr = 1:2;
        Idcwlr = 0; Idcwrl = 0;
        for ab1 = indarr 
            Sar[ab1] = 1; #Net must be zero, so Sar[ab1] adds energy while Sar[!ab1] removes the same energy
            Idcwlr = Idcwlr + real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * grwar[2-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * glwar[2-0,:,:] ) ) ) + 
                              real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * glwar[2-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[2-0,:,:] ) ) );
            Idcwrl = Idcwrl + real( tr( tau3 * ( Sigrl[trunc(Int, 0.5*(3-Sar[2])),:,:] * grwar[2-Sar[1],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[1])),:,:] * glwar[2-0,:,:] ) ) ) +
                              real( tr( tau3 * ( Sigrl[trunc(Int, 0.5*(3-Sar[2])),:,:] * glwar[2-Sar[1],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[2-0,:,:] ) ) );
        end
        Idcw[hi] = Idcwlr - Idcwrl;
    end
    Idc = (deltaw1/(2*pi)) * sum(Idcw);

    return Idc
end

"""
    current_Vbias_MPT_T4(war1, Omega, zeta, delta, T, Gamma, War) -> Float64

Fourth-order O(𝒯⁴) DC current of a voltage-biased junction (MPT expansion with
explicit energy-exchange bookkeeping), full BCS surface GF. Sums the four-vertex
tunnelling diagrams over all energy-conserving exchange permutations. Order 𝒯⁴ is
where two-quasiparticle processes first enter the subharmonic gap structure.
"""
function current_Vbias_MPT_T4(war1, Omega, zeta, delta, T, Gamma, War)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]); tau3 = [1 0; 0 -1];

    Sigrl = zeros(ComplexF64, 2,2,2); Sigrl[1,:,:] = T .* [War[1] 0; 0 0]; Sigrl[2,:,:] = T .* [0 0; 0 -conj(War[1])];  #Valid if War only has a single element, Omega=eV_dc
    Siglr = zeros(ComplexF64, 2,2,2); Siglr[1,:,:] = T .* [0 0; 0 -War[1]]; Siglr[2,:,:] = T .* [conj(War[1]) 0; 0 0];  #Both Sigrl and Siglr ordered such that Sigrl(lr)[1,:,:] adds and Sigrl(lr)[2,:,:] removes energy

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        # fac = exp(-(ww+ij*Omega)^2 / (zeta^2));
        fac = 1;

        grwar = zeros(ComplexF64, 5,2,2);
        glwar = zeros(ComplexF64, 5,2,2);
        gawar = zeros(ComplexF64, 5,2,2);

        for ab = -2:2
            wwab = war1[hi] + ab*Omega;
            grwar[3-ab,:,:] = fac * 1/( zeta*sqrt(delta^2-(wwab+im*Gamma)^2) ) .* [-(wwab+im*Gamma) delta; delta -(wwab+im*Gamma)];
            glwar[3-ab,:,:] = -2*im*(wwab .< 0) .* imag.(grwar[3-ab,:,:]);
            gawar[3-ab,:,:] = conj(transpose(grwar[3-ab,:,:]));
        end

        # Idcw[hi] = real( tr( tau3*( Sm*grpw*Sp*grw*Sm*grpw*Sp*glw + Sm*grpw*Sp*grw*Sm*glpw*Sp*gaw + Sm*grpw*Sp*glw*Sm*gapw*Sp*gaw + Sm*glpw*Sp*gaw*Sm*gapw*Sp*gaw ) ) ) + 
        #            real( tr( tau3*( Sm*grpw*Sm*gr2pw*Sp*grpw*Sp*glw + Sm*grpw*Sm*gr2pw*Sp*glpw*Sp*gaw + Sm*grpw*Sm*gl2pw*Sp*gapw*Sp*gaw + Sm*glpw*Sm*ga2pw*Sp*gapw*Sp*gaw ) ) ) + 
        #            real( tr( tau3*( Sp*grmw*Sm*grw*Sm*grpw*Sp*glw + Sp*grmw*Sm*grw*Sm*glpw*Sp*gaw + Sp*grmw*Sm*glw*Sm*gapw*Sp*gaw + Sp*glmw*Sm*gaw*Sm*gapw*Sp*gaw ) ) ) ;                   

        Sar = -1 .* ones(Int32,4);
        indarr = 1:4;
        for ab1 = indarr
            for ab2 = indarr[1:end .!= ab1] #println("ab1 = $(ab1) ab2 = $(ab2)")
                Sar = -1 .* ones(Int32,4); Sar[ab1] = 1; Sar[ab2] = 1;
                Idcw[hi] = Idcw[hi] + real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * grwar[3-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * grwar[3-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * grwar[3-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * glwar[3-0,:,:] ) ) ) + 
                                      real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * grwar[3-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * grwar[3-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * glwar[3-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[3-0,:,:] ) ) ) +
                                      real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * grwar[3-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * glwar[3-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * gawar[3-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[3-0,:,:] ) ) ) +
                                      real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * glwar[3-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * gawar[3-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * gawar[3-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[3-0,:,:] ) ) );
            end 
        end
    end
    Idc = 2*(deltaw1/(2*pi)) * sum(Idcw); #2: LR - RL. See 2dnd order current, Lr and RL and equal in magnitude and opposite in sign.
    
    return Idc
end

"""
    current_Vbias_MPT_T4_qp(war1, Omega, zeta, delta, T, Gamma, War) -> Float64

Quasiparticle-channel (normal-GF-only) part of [`current_Vbias_MPT_T4`](@ref): the
O(𝒯⁴) MPT current with only the diagonal (τ₀) lead GF retained.
"""
function current_Vbias_MPT_T4_qp(war1, Omega, zeta, delta, T, Gamma, War)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]); tau3 = [1 0; 0 -1];

    Sigrl = zeros(ComplexF64, 2,2,2); Sigrl[1,:,:] = T .* [War[1] 0; 0 0]; Sigrl[2,:,:] = T .* [0 0; 0 -conj(War[1])];  #Valid if War only has a single element, Omega=eV_dc
    Siglr = zeros(ComplexF64, 2,2,2); Siglr[1,:,:] = T .* [0 0; 0 -War[1]]; Siglr[2,:,:] = T .* [conj(War[1]) 0; 0 0];  #Both Sigrl and Siglr ordered such that Sigrl(lr)[1,:,:] adds and Sigrl(lr)[2,:,:] removes energy

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        # fac = exp(-(ww+ij*Omega)^2 / (zeta^2));
        fac = 1;

        grwar = zeros(ComplexF64, 5,2,2);
        glwar = zeros(ComplexF64, 5,2,2);
        gawar = zeros(ComplexF64, 5,2,2);

        for ab = -2:2
            wwab = war1[hi] + ab*Omega;
            grwar[3-ab,:,:] = fac * 1/( zeta*sqrt(delta^2-(wwab+im*Gamma)^2) ) .* [-(wwab+im*Gamma) 0; 0 -(wwab+im*Gamma)];
            glwar[3-ab,:,:] = -2*im*(wwab .< 0) .* imag.(grwar[3-ab,:,:]);
            gawar[3-ab,:,:] = conj(transpose(grwar[3-ab,:,:]));
        end

        # Idcw[hi] = real( tr( tau3*( Sm*grpw*Sp*grw*Sm*grpw*Sp*glw + Sm*grpw*Sp*grw*Sm*glpw*Sp*gaw + Sm*grpw*Sp*glw*Sm*gapw*Sp*gaw + Sm*glpw*Sp*gaw*Sm*gapw*Sp*gaw ) ) ) + 
        #            real( tr( tau3*( Sm*grpw*Sm*gr2pw*Sp*grpw*Sp*glw + Sm*grpw*Sm*gr2pw*Sp*glpw*Sp*gaw + Sm*grpw*Sm*gl2pw*Sp*gapw*Sp*gaw + Sm*glpw*Sm*ga2pw*Sp*gapw*Sp*gaw ) ) ) + 
        #            real( tr( tau3*( Sp*grmw*Sm*grw*Sm*grpw*Sp*glw + Sp*grmw*Sm*grw*Sm*glpw*Sp*gaw + Sp*grmw*Sm*glw*Sm*gapw*Sp*gaw + Sp*glmw*Sm*gaw*Sm*gapw*Sp*gaw ) ) ) ;                   

        Sar = -1 .* ones(Int32,4);
        indarr = 1:4;
        for ab1 = indarr
            for ab2 = indarr[1:end .!= ab1] #println("ab1 = $(ab1) ab2 = $(ab2)")
                Sar = -1 .* ones(Int32,4); Sar[ab1] = 1; Sar[ab2] = 1;
                Idcw[hi] = Idcw[hi] + real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * grwar[3-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * grwar[3-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * grwar[3-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * glwar[3-0,:,:] ) ) ) + 
                                      real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * grwar[3-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * grwar[3-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * glwar[3-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[3-0,:,:] ) ) ) +
                                      real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * grwar[3-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * glwar[3-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * gawar[3-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[3-0,:,:] ) ) ) +
                                      real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * glwar[3-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * gawar[3-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * gawar[3-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[3-0,:,:] ) ) );
            end 
        end
    end
    Idc = 2*(deltaw1/(2*pi)) * sum(Idcw); #2: LR - RL. See 2dnd order current, Lr and RL and equal in magnitude and opposite in sign.
    
    return Idc
end


"""
    current_Vbias_MPT_T4_aobo(war1, Omega, zeta, delta, T, Gamma, ao, bo, War) -> Float64

O(𝒯⁴) DC current of a voltage-biased junction restricted to one energy-exchange
channel labelled by the harmonic indices `ao`, `bo`.

Energy-exchange bookkeeping. Under a DC bias the hopping is time-periodic, so each of
the four tunnelling vertices of this 4th-order (two-Andreev-reflection) process
exchanges an integer number of energy quanta Ω = eV with the bias field. The four
vertices carry the kicks {+ao, +bo, −ao, −bo}·Ω. Each sign configuration is a distinct
Andreev process: at a `+` vertex the Bogoliubov quasiparticle JUMPS UP in energy by
(index)·eV, at a `−` vertex it JUMPS DOWN. The kicks sum to zero (+ao +bo −ao −bo = 0),
so the quasiparticle returns to its starting energy — the condition for a stationary
(DC) current. Between vertices it propagates at the running energy ω + (cumulative
kicks)·Ω, at which the lead GF `grwar[...]` is evaluated; the anomalous (off-diagonal)
GF components are the Andreev particle↔hole reflections.

This routine sums over ALL vertex orderings of the four kicks (the `Sar` permutation
loop) and over the Keldysh placement of the lesser GF, i.e. the full (ao,bo)-channel
current. Single fixed orderings are [`current_Vbias_MPT_T4_fghfhg_mbomaoboao`](@ref) /
[`current_Vbias_MPT_T4_fghfhg_maomboboao`](@ref); the normal-/anomalous-GF projections
are [`current_Vbias_MPT_T4_qp_aobo`](@ref) / [`current_Vbias_MPT_T4_pair_aobo`](@ref).
Uses the full BCS surface GF [`surfacegr`](@ref).

`War`: Fourier coefficients of `e^{iφ/2}`, ordered `+nf:-2:-nf` with `nf=max(ao,bo)`.
Returns `2·I_LR` (factor 2: equal-and-opposite LR/RL parts).
"""
function current_Vbias_MPT_T4_aobo(war1, Omega, zeta, delta, T, Gamma, ao, bo, War) #War: Fourier coefficients of exp(i*phi/2), in the order +nf:-2:-nf, where nf = maximum([ao, bo])
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]); tau3 = [1 0; 0 -1];

    nf = maximum([ao, bo]);
    Sigrl = zeros(ComplexF64, 2*nf+1,2,2); Siglr = zeros(ComplexF64, 2*nf+1,2,2); 
    for hi = -nf:2:nf 
        Sigrl[nf+1-hi,:,:] = T .* [War[nf+1-hi] 0; 0 -conj(War[nf+1+hi])]; 
        Siglr[nf+1-hi,:,:] = T .* [conj(War[nf+1+hi]) 0; 0 -War[nf+1-hi]];
    end

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        # fac = exp(-(ww+ij*Omega)^2 / (zeta^2));
        fac = 1;

        grwar = zeros(ComplexF64, 2*(ao+bo)+1,2,2);
        glwar = zeros(ComplexF64, 2*(ao+bo)+1,2,2);
        gawar = zeros(ComplexF64, 2*(ao+bo)+1,2,2);

        for ab = -(ao+bo):(ao+bo)
            wwab = war1[hi] + ab*Omega;
            # grwar[(ao+bo)+1-ab,:,:] = Keldyshsetup_Floquetn.surfacegr(zeta, delta, Gamma, wwab, 1);
            grwar[(ao+bo)+1-ab,:,:] = fac * 1/( zeta*sqrt(delta^2-(wwab+im*Gamma)^2) ) .* [-(wwab+im*Gamma) delta; delta -(wwab+im*Gamma)];
            glwar[(ao+bo)+1-ab,:,:] = -2*im*(wwab .< 0) .* imag.(grwar[(ao+bo)+1-ab,:,:]);
            gawar[(ao+bo)+1-ab,:,:] = conj(transpose(grwar[(ao+bo)+1-ab,:,:]));
        end

        indarr = 1:4;
        for ab1 = indarr
            indarr1 = indarr[1:end .!= ab1];
            for ab2 = indarr1
                indarr2 = indarr1[ indarr1 .* (indarr1 .!= ab2) .!= 0 ];
                for ab3 = indarr2 #println("ab1 = $(ab1) ab2 = $(ab2) ab3 = $(ab3)")
                    Sar = -bo .* ones(Int32,4); Sar[ab1] = ao; Sar[ab2] = bo; Sar[ab3] = -ao; #default: all -bo. Then choose +ao, +bo, -ao spots, going over all permutations
                    Idcw[hi] = Idcw[hi] + real( tr( tau3 * ( Siglr[nf+1-Sar[4],:,:] * grwar[(ao+bo)+1-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[nf+1-Sar[3],:,:] * grwar[(ao+bo)+1-Sar[1]-Sar[2],:,:] * Siglr[nf+1-Sar[2],:,:] * grwar[(ao+bo)+1-Sar[1],:,:] * Sigrl[nf+1-Sar[1],:,:] * glwar[(ao+bo)+1-0,:,:] ) ) ) + 
                                          real( tr( tau3 * ( Siglr[nf+1-Sar[4],:,:] * grwar[(ao+bo)+1-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[nf+1-Sar[3],:,:] * grwar[(ao+bo)+1-Sar[1]-Sar[2],:,:] * Siglr[nf+1-Sar[2],:,:] * glwar[(ao+bo)+1-Sar[1],:,:] * Sigrl[nf+1-Sar[1],:,:] * gawar[(ao+bo)+1-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[nf+1-Sar[4],:,:] * grwar[(ao+bo)+1-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[nf+1-Sar[3],:,:] * glwar[(ao+bo)+1-Sar[1]-Sar[2],:,:] * Siglr[nf+1-Sar[2],:,:] * gawar[(ao+bo)+1-Sar[1],:,:] * Sigrl[nf+1-Sar[1],:,:] * gawar[(ao+bo)+1-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[nf+1-Sar[4],:,:] * glwar[(ao+bo)+1-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[nf+1-Sar[3],:,:] * gawar[(ao+bo)+1-Sar[1]-Sar[2],:,:] * Siglr[nf+1-Sar[2],:,:] * gawar[(ao+bo)+1-Sar[1],:,:] * Sigrl[nf+1-Sar[1],:,:] * gawar[(ao+bo)+1-0,:,:] ) ) );
                end
            end 
        end
    end
    Idc = 2*(deltaw1/(2*pi)) * sum(Idcw); #2: LR - RL. See 2dnd order current, Lr and RL and equal in magnitude and opposite in sign.
    
    return Idc
end

"""
    current_Vbias_MPT_T4_qp_aobo(war1, Omega, zeta, delta, T, Gamma, ao, bo, War) -> Float64

(ao,bo)-channel O(𝒯⁴) current keeping only the NORMAL (quasiparticle) part of the
lead GF — the per-channel quasiparticle counterpart of
[`current_Vbias_MPT_T4_aobo`](@ref). The only MPT routine with a live call site
(`Josephson_Vbias_Floquetn_direct.jl`).
"""
function current_Vbias_MPT_T4_qp_aobo(war1, Omega, zeta, delta, T, Gamma, ao, bo, War) #War: Fourier coefficients of exp(i*phi/2), in the order +nf:-2:-nf, where nf = maximum([ao, bo])
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]); tau3 = [1 0; 0 -1];

    nf = maximum([ao, bo]);
    Sigrl = zeros(ComplexF64, 2*nf+1,2,2); Siglr = zeros(ComplexF64, 2*nf+1,2,2); 
    for hi = -nf:2:nf 
        Sigrl[nf+1-hi,:,:] = T .* [War[nf+1-hi] 0; 0 -conj(War[nf+1+hi])]; 
        Siglr[nf+1-hi,:,:] = T .* [conj(War[nf+1+hi]) 0; 0 -War[nf+1-hi]];
    end

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        # fac = exp(-(ww+ij*Omega)^2 / (zeta^2));
        fac = 1;

        grwar = zeros(ComplexF64, 2*(ao+bo)+1,2,2);
        glwar = zeros(ComplexF64, 2*(ao+bo)+1,2,2);
        gawar = zeros(ComplexF64, 2*(ao+bo)+1,2,2);

        for ab = -(ao+bo):(ao+bo)
            wwab = war1[hi] + ab*Omega;
            # grwar[(ao+bo)+1-ab,:,:] = Keldyshsetup_Floquetn.surfacegr(zeta, delta, Gamma, wwab, 1); grwar[(ao+bo)+1-ab,1,2] = 0; grwar[(ao+bo)+1-ab,2,1] = 0;
            grwar[(ao+bo)+1-ab,:,:] = fac * 1/( zeta*sqrt(delta^2-(wwab+im*Gamma)^2) ) .* [-(wwab+im*Gamma) 0; 0 -(wwab+im*Gamma)];
            glwar[(ao+bo)+1-ab,:,:] = -2*im*(wwab .< 0) .* imag.(grwar[(ao+bo)+1-ab,:,:]);
            gawar[(ao+bo)+1-ab,:,:] = conj(transpose(grwar[(ao+bo)+1-ab,:,:]));
        end

        indarr = 1:4;
        for ab1 = indarr
            indarr1 = indarr[1:end .!= ab1];
            for ab2 = indarr1
                indarr2 = indarr1[ indarr1 .* (indarr1 .!= ab2) .!= 0 ];
                for ab3 = indarr2 #println("ab1 = $(ab1) ab2 = $(ab2) ab3 = $(ab3)")
                    Sar = -bo .* ones(Int32,4); Sar[ab1] = ao; Sar[ab2] = bo; Sar[ab3] = -ao; #default: all -bo. Then choose +ao, +bo, -ao spots, going over all permutations
                    Idcw[hi] = Idcw[hi] + real( tr( tau3 * ( Siglr[nf+1-Sar[4],:,:] * grwar[(ao+bo)+1-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[nf+1-Sar[3],:,:] * grwar[(ao+bo)+1-Sar[1]-Sar[2],:,:] * Siglr[nf+1-Sar[2],:,:] * grwar[(ao+bo)+1-Sar[1],:,:] * Sigrl[nf+1-Sar[1],:,:] * glwar[(ao+bo)+1-0,:,:] ) ) ) + 
                                          real( tr( tau3 * ( Siglr[nf+1-Sar[4],:,:] * grwar[(ao+bo)+1-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[nf+1-Sar[3],:,:] * grwar[(ao+bo)+1-Sar[1]-Sar[2],:,:] * Siglr[nf+1-Sar[2],:,:] * glwar[(ao+bo)+1-Sar[1],:,:] * Sigrl[nf+1-Sar[1],:,:] * gawar[(ao+bo)+1-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[nf+1-Sar[4],:,:] * grwar[(ao+bo)+1-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[nf+1-Sar[3],:,:] * glwar[(ao+bo)+1-Sar[1]-Sar[2],:,:] * Siglr[nf+1-Sar[2],:,:] * gawar[(ao+bo)+1-Sar[1],:,:] * Sigrl[nf+1-Sar[1],:,:] * gawar[(ao+bo)+1-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[nf+1-Sar[4],:,:] * glwar[(ao+bo)+1-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[nf+1-Sar[3],:,:] * gawar[(ao+bo)+1-Sar[1]-Sar[2],:,:] * Siglr[nf+1-Sar[2],:,:] * gawar[(ao+bo)+1-Sar[1],:,:] * Sigrl[nf+1-Sar[1],:,:] * gawar[(ao+bo)+1-0,:,:] ) ) );
                end
            end 
        end
    end
    Idc = 2*(deltaw1/(2*pi)) * sum(Idcw); #2: LR - RL. See 2dnd order current, Lr and RL and equal in magnitude and opposite in sign.
    
    return Idc
end

""" 
    current_Vbias_MPT_T4_pair_aobo(war1, Omega, zeta, delta, T, Gamma, ao, bo, War) -> Float64

(ao,bo)-channel O(𝒯⁴) current keeping only the ANOMALOUS (pair) part of the lead GF
— the per-channel pair counterpart of [`current_Vbias_MPT_T4_aobo`](@ref).
"""
function current_Vbias_MPT_T4_pair_aobo(war1, Omega, zeta, delta, T, Gamma, ao, bo, War) #War: Fourier coefficients of exp(i*phi/2), in the order +nf:-2:-nf, where nf = maximum([ao, bo])
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]); tau3 = [1 0; 0 -1];

    nf = maximum([ao, bo]);
    Sigrl = zeros(ComplexF64, 2*nf+1,2,2); Siglr = zeros(ComplexF64, 2*nf+1,2,2); 
    for hi = -nf:2:nf 
        Sigrl[nf+1-hi,:,:] = T .* [War[nf+1-hi] 0; 0 -conj(War[nf+1+hi])]; 
        Siglr[nf+1-hi,:,:] = T .* [conj(War[nf+1+hi]) 0; 0 -War[nf+1-hi]];
    end

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        # fac = exp(-(ww+ij*Omega)^2 / (zeta^2));
        fac = 1;

        grwar = zeros(ComplexF64, 2*(ao+bo)+1,2,2);
        glwar = zeros(ComplexF64, 2*(ao+bo)+1,2,2);
        gawar = zeros(ComplexF64, 2*(ao+bo)+1,2,2);

        for ab = -(ao+bo):(ao+bo)
            wwab = war1[hi] + ab*Omega;
            # grwar[(ao+bo)+1-ab,:,:] = Keldyshsetup_Floquetn.surfacegr(zeta, delta, Gamma, wwab, 1); grwar[(ao+bo)+1-ab,1,1] = 0; grwar[(ao+bo)+1-ab,2,2] = 0;
            grwar[(ao+bo)+1-ab,:,:] = fac * 1/( zeta*sqrt(delta^2-(wwab+im*Gamma)^2) ) .* [0 delta; delta 0];
            glwar[(ao+bo)+1-ab,:,:] = -2*im*(wwab .< 0) .* imag.(grwar[(ao+bo)+1-ab,:,:]);
            gawar[(ao+bo)+1-ab,:,:] = conj(transpose(grwar[(ao+bo)+1-ab,:,:]));
        end

        indarr = 1:4;
        for ab1 = indarr
            indarr1 = indarr[1:end .!= ab1];
            for ab2 = indarr1
                indarr2 = indarr1[ indarr1 .* (indarr1 .!= ab2) .!= 0 ];
                for ab3 = indarr2 #println("ab1 = $(ab1) ab2 = $(ab2) ab3 = $(ab3)")
                    Sar = -bo .* ones(Int32,4); Sar[ab1] = ao; Sar[ab2] = bo; Sar[ab3] = -ao; #default: all -bo. Then choose +ao, +bo, -ao spots, going over all permutations
                    Idcw[hi] = Idcw[hi] + real( tr( tau3 * ( Siglr[nf+1-Sar[4],:,:] * grwar[(ao+bo)+1-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[nf+1-Sar[3],:,:] * grwar[(ao+bo)+1-Sar[1]-Sar[2],:,:] * Siglr[nf+1-Sar[2],:,:] * grwar[(ao+bo)+1-Sar[1],:,:] * Sigrl[nf+1-Sar[1],:,:] * glwar[(ao+bo)+1-0,:,:] ) ) ) + 
                                          real( tr( tau3 * ( Siglr[nf+1-Sar[4],:,:] * grwar[(ao+bo)+1-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[nf+1-Sar[3],:,:] * grwar[(ao+bo)+1-Sar[1]-Sar[2],:,:] * Siglr[nf+1-Sar[2],:,:] * glwar[(ao+bo)+1-Sar[1],:,:] * Sigrl[nf+1-Sar[1],:,:] * gawar[(ao+bo)+1-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[nf+1-Sar[4],:,:] * grwar[(ao+bo)+1-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[nf+1-Sar[3],:,:] * glwar[(ao+bo)+1-Sar[1]-Sar[2],:,:] * Siglr[nf+1-Sar[2],:,:] * gawar[(ao+bo)+1-Sar[1],:,:] * Sigrl[nf+1-Sar[1],:,:] * gawar[(ao+bo)+1-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[nf+1-Sar[4],:,:] * glwar[(ao+bo)+1-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[nf+1-Sar[3],:,:] * gawar[(ao+bo)+1-Sar[1]-Sar[2],:,:] * Siglr[nf+1-Sar[2],:,:] * gawar[(ao+bo)+1-Sar[1],:,:] * Sigrl[nf+1-Sar[1],:,:] * gawar[(ao+bo)+1-0,:,:] ) ) );
                end
            end 
        end
    end
    Idc = 2*(deltaw1/(2*pi)) * sum(Idcw); #2: LR - RL. See 2dnd order current, Lr and RL and equal in magnitude and opposite in sign.
    
    return Idc
end

"""
    current_Vbias_MPT_T6(war1, Omega, zeta, delta, T, Gamma, War) -> Float64

Sixth-order O(𝒯⁶) DC current of a voltage-biased junction (MPT expansion, full BCS
surface GF), summing the six-vertex tunnelling diagrams over all energy-conserving
exchange permutations. `War`: Fourier coefficients of `e^{iφ/2}`.
"""
function current_Vbias_MPT_T6(war1, Omega, zeta, delta, T, Gamma, War) #War: exp(i*phi/2)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]); tau3 = [1 0; 0 -1];

    Sigrl = zeros(ComplexF64, 2,2,2); Sigrl[1,:,:] = T .* [War[1] 0; 0 0]; Sigrl[2,:,:] = T .* [0 0; 0 -conj(War[1])]; 
    Siglr = zeros(ComplexF64, 2,2,2); Siglr[1,:,:] = T .* [0 0; 0 -War[1]]; Siglr[2,:,:] = T .* [conj(War[1]) 0; 0 0];

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        # fac = exp(-(ww+ij*Omega)^2 / (zeta^2));
        fac = 1;

        grwar = zeros(ComplexF64, 7,2,2);
        glwar = zeros(ComplexF64, 7,2,2);
        gawar = zeros(ComplexF64, 7,2,2);

        for ab = -3:3
            wwab = war1[hi] + ab*Omega;
            grwar[4-ab,:,:] = fac * 1/( zeta*sqrt(delta^2-(wwab+im*Gamma)^2) ) .* [-(wwab+im*Gamma) delta; delta -(wwab+im*Gamma)];
            glwar[4-ab,:,:] = -2*im*(wwab .< 0) .* imag.(grwar[4-ab,:,:]);
            gawar[4-ab,:,:] = conj(transpose(grwar[4-ab,:,:]));
        end         

        Sar = -1 .* ones(Int32,6);
        indarr = 1:6;
        for ab1 = indarr
            indarr1 = indarr[1:end .!= ab1];
            for ab2 = indarr1
                indarr2 = indarr1[ indarr1 .* (indarr1 .!= ab2) .!= 0 ];
                for ab3 = indarr2 # println("ab1 = $(ab1) ab2 = $(ab2) ab3 = $(ab3)")
                    Sar = -1 .* ones(Int32,6); Sar[ab1] = 1; Sar[ab2] = 1; Sar[ab3] = 1;

                    Idcw[hi] = Idcw[hi] + real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * grwar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * grwar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * glwar[4-0,:,:] ) ) ) + 
                                          real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * grwar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * glwar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[4-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * glwar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * gawar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[4-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * glwar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * gawar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * gawar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[4-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * glwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * gawar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * gawar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * gawar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[4-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * glwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * gawar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * gawar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * gawar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * gawar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[4-0,:,:] ) ) ) ;
                end
            end 
        end
    end
    Idc = 2*(deltaw1/(2*pi)) * sum(Idcw); #2: LR - RL. See 2dnd order current, Lr and RL and equal in magnitude and opposite in sign.
    
    return Idc
end

"""
    current_Vbias_MPT_T6_pair(war1, Omega, zeta, delta, T, Gamma, War) -> Float64

Pair (anomalous-GF-only) channel part of [`current_Vbias_MPT_T6`](@ref).
"""
function current_Vbias_MPT_T6_pair(war1, Omega, zeta, delta, T, Gamma, War) #War: exp(i*phi/2)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]); tau3 = [1 0; 0 -1];

    Sigrl = zeros(ComplexF64, 2,2,2); Sigrl[1,:,:] = T .* [War[1] 0; 0 0]; Sigrl[2,:,:] = T .* [0 0; 0 -conj(War[1])]; 
    Siglr = zeros(ComplexF64, 2,2,2); Siglr[1,:,:] = T .* [0 0; 0 -War[1]]; Siglr[2,:,:] = T .* [conj(War[1]) 0; 0 0];

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        # fac = exp(-(ww+ij*Omega)^2 / (zeta^2));
        fac = 1;

        grwar = zeros(ComplexF64, 7,2,2);
        glwar = zeros(ComplexF64, 7,2,2);
        gawar = zeros(ComplexF64, 7,2,2);

        for ab = -3:3
            wwab = war1[hi] + ab*Omega;
            grwar[4-ab,:,:] = fac * 1/( zeta*sqrt(delta^2-(wwab+im*Gamma)^2) ) .* [0 delta; delta 0];
            glwar[4-ab,:,:] = -2*im*(wwab .< 0) .* imag.(grwar[4-ab,:,:]);
            gawar[4-ab,:,:] = conj(transpose(grwar[4-ab,:,:]));
        end         

        Sar = -1 .* ones(Int32,6);
        indarr = 1:6;
        for ab1 = indarr
            indarr1 = indarr[1:end .!= ab1];
            for ab2 = indarr1
                indarr2 = indarr1[ indarr1 .* (indarr1 .!= ab2) .!= 0 ];
                for ab3 = indarr2 # println("ab1 = $(ab1) ab2 = $(ab2) ab3 = $(ab3)")
                    Sar = -1 .* ones(Int32,6); Sar[ab1] = 1; Sar[ab2] = 1; Sar[ab3] = 1;

                    Idcw[hi] = Idcw[hi] + real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * grwar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * grwar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * glwar[4-0,:,:] ) ) ) + 
                                          real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * grwar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * glwar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[4-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * glwar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * gawar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[4-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * glwar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * gawar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * gawar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[4-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * glwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * gawar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * gawar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * gawar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[4-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * glwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * gawar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * gawar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * gawar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * gawar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[4-0,:,:] ) ) ) ;
                end
            end 
        end
    end
    Idc = 2*(deltaw1/(2*pi)) * sum(Idcw); #2: LR - RL. See 2dnd order current, Lr and RL and equal in magnitude and opposite in sign.
    
    return Idc
end

"""
    current_Vbias_MPT_T6_qp(war1, Omega, zeta, delta, T, Gamma, War) -> Float64

Quasiparticle-channel (normal-GF-only) part of [`current_Vbias_MPT_T6`](@ref).
"""
function current_Vbias_MPT_T6_qp(war1, Omega, zeta, delta, T, Gamma, War) #War: exp(i*phi/2)
    Nw1 = length(war1); deltaw1 = abs(war1[2]-war1[1]); tau3 = [1 0; 0 -1];

    Sigrl = zeros(ComplexF64, 2,2,2); Sigrl[1,:,:] = T .* [War[1] 0; 0 0]; Sigrl[2,:,:] = T .* [0 0; 0 -conj(War[1])]; 
    Siglr = zeros(ComplexF64, 2,2,2); Siglr[1,:,:] = T .* [0 0; 0 -War[1]]; Siglr[2,:,:] = T .* [conj(War[1]) 0; 0 0];

    Idcw = zeros(Float64,Nw1);
    Threads.@threads for hi = 1:Nw1
        # fac = exp(-(ww+ij*Omega)^2 / (zeta^2));
        fac = 1;

        grwar = zeros(ComplexF64, 7,2,2);
        glwar = zeros(ComplexF64, 7,2,2);
        gawar = zeros(ComplexF64, 7,2,2);

        for ab = -3:3
            wwab = war1[hi] + ab*Omega;
            grwar[4-ab,:,:] = fac * 1/( zeta*sqrt(delta^2-(wwab+im*Gamma)^2) ) .* [-(wwab+im*Gamma) 0; 0 -(wwab+im*Gamma)];
            glwar[4-ab,:,:] = -2*im*(wwab .< 0) .* imag.(grwar[4-ab,:,:]);
            gawar[4-ab,:,:] = conj(transpose(grwar[4-ab,:,:]));
        end         

        Sar = -1 .* ones(Int32,6);
        indarr = 1:6;
        for ab1 = indarr
            indarr1 = indarr[1:end .!= ab1];
            for ab2 = indarr1
                indarr2 = indarr1[ indarr1 .* (indarr1 .!= ab2) .!= 0 ];
                for ab3 = indarr2 # println("ab1 = $(ab1) ab2 = $(ab2) ab3 = $(ab3)")
                    Sar = -1 .* ones(Int32,6); Sar[ab1] = 1; Sar[ab2] = 1; Sar[ab3] = 1;

                    Idcw[hi] = Idcw[hi] + real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * grwar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * grwar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * glwar[4-0,:,:] ) ) ) + 
                                          real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * grwar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * glwar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[4-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * glwar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * gawar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[4-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * glwar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * gawar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * gawar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[4-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * grwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * glwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * gawar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * gawar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * gawar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[4-0,:,:] ) ) ) +
                                          real( tr( tau3 * ( Siglr[trunc(Int, 0.5*(3-Sar[6])),:,:] * glwar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4]-Sar[5],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[5])),:,:] * gawar[4-Sar[1]-Sar[2]-Sar[3]-Sar[4],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[4])),:,:] * gawar[4-Sar[1]-Sar[2]-Sar[3],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[3])),:,:] * gawar[4-Sar[1]-Sar[2],:,:] * Siglr[trunc(Int, 0.5*(3-Sar[2])),:,:] * gawar[4-Sar[1],:,:] * Sigrl[trunc(Int, 0.5*(3-Sar[1])),:,:] * gawar[4-0,:,:] ) ) ) ;
                end
            end 
        end
    end
    Idc = 2*(deltaw1/(2*pi)) * sum(Idcw); #2: LR - RL. See 2dnd order current, Lr and RL and equal in magnitude and opposite in sign.
    
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
    grwmnf(ww, Omega, Nf, zeta, delta, Gamma) -> Matrix{ComplexF64}

Bare RETARDED lead Green's function in the Floquet–Nambu–lead basis. Block-diagonal
in the Floquet index m ∈ [−Nf,Nf]: each 2×2 block is the analytic BCS surface GF at
the shifted energy ω+mΩ,
`gʳ(ω+mΩ) = [−(ω+mΩ+iΓ)τ₀ + Δτ₁] / (ζ·√(Δ²−(ω+mΩ+iΓ)²))`,
duplicated over the two leads (L,R). `ww`=ω, `Omega`=Ω drive frequency. Returns a
4(2Nf+1)×4(2Nf+1) matrix.
"""
function grwmnf(ww, Omega, Nf, zeta, delta, Gamma)
    grwmn_d = zeros(ComplexF64, 2*(2*Nf+1),2*(2*Nf+1));

    for ij = -Nf:Nf   
        # fac = exp(-(ww+ij*Omega)^2 / (zeta^2));
        fac = 1;
        grwmn_d[2*(-ij+Nf+1)-1:2*(-ij+Nf+1),2*(-ij+Nf+1)-1:2*(-ij+Nf+1)] = fac * 1/( zeta*sqrt(delta^2-(ww+ij*Omega+im*Gamma)^2) ) .* [-(ww+ij*Omega+im*Gamma) delta;
                                                                                                                                        delta -(ww+ij*Omega+im*Gamma)];
        # grwmn_d[2*(-ij+Nf+1)-1:2*(-ij+Nf+1),2*(-ij+Nf+1)-1:2*(-ij+Nf+1)] = Keldyshsetup_Floquetn.surfacegr(zeta, delta, Gamma, ww+ij*Omega, 1);
    end
    grwmn = [grwmn_d zeros(size(grwmn_d));
             zeros(size(grwmn_d)) grwmn_d];

    return grwmn
end

"""
    gawmnf(ww, Omega, Nf, zeta, delta, Gamma) -> Matrix{ComplexF64}

Bare ADVANCED lead Green's function in the Floquet–Nambu–lead basis: the −iΓ
counterpart of [`grwmnf`](@ref) (equivalently `grwmnf(...)'`).
"""
function gawmnf(ww, Omega, Nf, zeta, delta, Gamma)
    gawmn_d = zeros(ComplexF64, 2*(2*Nf+1),2*(2*Nf+1));

    for ij = -Nf:Nf
        # fac = exp(-(ww+ij*Omega)^2 / (zeta^2));
        fac = 1;
        gawmn_d[2*(-ij+Nf+1)-1:2*(-ij+Nf+1),2*(-ij+Nf+1)-1:2*(-ij+Nf+1)] = fac * 1/( zeta*sqrt(delta^2-(ww+ij*Omega-im*Gamma)^2) ) .* [-(ww+ij*Omega-im*Gamma) delta;
                                                                                                                                        delta -(ww+ij*Omega-im*Gamma)];
    end
    gawmn = [gawmn_d zeros(size(gawmn_d));
             zeros(size(gawmn_d)) gawmn_d];

    return gawmn
end

"""
    glwmnf(ww, Omega, Nf, zeta, delta, Gamma) -> Matrix{ComplexF64}

Bare LESSER lead Green's function in the Floquet basis: `g< = −(gʳ−gᵃ)` restricted
to occupied energies (ω+mΩ < 0, zero temperature), block-diagonal in the Floquet
index.
"""
function glwmnf(ww, Omega, Nf, zeta, delta, Gamma)
    grwmn = Keldyshsetup_Floquetn.grwmnf(ww, Omega, Nf, zeta, delta, Gamma);
    # gawmn = Keldyshsetup_Floquetn.gawmnf(ww, Omega, Nf, delta, Gamma);
    gawmn = grwmn';

    glwmn = -( grwmn - gawmn );

    for ij = -Nf:Nf
        ff = (ww+ij*Omega) < 0;
        glwmn[2*(-ij+Nf+1)-1:2*(-ij+Nf+1), 2*(-ij+Nf+1)-1:2*(-ij+Nf+1)] = ff .* glwmn[2*(-ij+Nf+1)-1:2*(-ij+Nf+1), 2*(-ij+Nf+1)-1:2*(-ij+Nf+1)];
        glwmn[2*(2*Nf+1) + 2*(-ij+Nf+1)-1:2*(2*Nf+1) + 2*(-ij+Nf+1), 2*(2*Nf+1) + 2*(-ij+Nf+1)-1:2*(2*Nf+1) + 2*(-ij+Nf+1)] = ff .* glwmn[2*(2*Nf+1) + 2*(-ij+Nf+1)-1:2*(2*Nf+1) + 2*(-ij+Nf+1), 2*(2*Nf+1) + 2*(-ij+Nf+1)-1:2*(2*Nf+1) + 2*(-ij+Nf+1)];
    end
    
    return glwmn
end

"""
    Grwmnf(ww, Omega, Nf, zeta, delta, T, Gamma, Vip) -> Matrix{ComplexF64}

Full (dressed) RETARDED Green's function from the Floquet Dyson equation
`Gʳ = (I − gʳ·Σʳ)⁻¹ gʳ`, where `gʳ`=[`grwmnf`](@ref) and `Σʳ`=[`Vwmnf`](@ref)`(Vip)`
is the hopping self-energy carrying the phase harmonics `Vip` and transparency `T`.
Solved as a linear system.
"""
function Grwmnf(ww, Omega, Nf, zeta, delta, T, Gamma, Vip)
    grwmn = Keldyshsetup_Floquetn.grwmnf(ww, Omega, Nf, zeta, delta, Gamma);
    Vv = Keldyshsetup_Floquetn.Vwmnf(Vip, Nf, T);
    Sigr = Vv;

    # Grwmn = inv(I((2*Nf+1)*2*2)-grwmn*Sigr) * grwmn;
    Grwmn = (I((2*Nf+1)*2*2)-grwmn*Sigr) \ grwmn;

    return Grwmn
end

"""
    Gawmnf(ww, Omega, Nf, zeta, delta, T, Gamma, Vip) -> Matrix{ComplexF64}

Full (dressed) ADVANCED Green's function, `Gᵃ = (I − gᵃ·Σᵃ)⁻¹ gᵃ`, the advanced
counterpart of [`Grwmnf`](@ref). In practice the code uses `Grwmn'` instead, so this
routine is currently unused.
"""
function Gawmnf(ww, Omega, Nf, zeta, delta, T, Gamma, Vip)
    gawmn = Keldyshsetup_Floquetn.gawmnf(ww, Omega, Nf, zeta, delta, Gamma);
    Vv = Keldyshsetup_Floquetn.Vwmnf(Vip, Nf, T);
    Siga = Vv;

    # Gawmn = inv(I((2*Nf+1)*2*2)-gawmn*Siga) * gawmn;
    Gawmn = (I((2*Nf+1)*2*2)-gawmn*Siga) \ gawmn;

    return Gawmn
end

"""
    Siglf(ww, Omega, Nf, zeta, delta, T, Gamma) -> Matrix{ComplexF64}

Lesser self-energy Σ< in the Floquet–Nambu–lead basis. It embeds the
semi-infinite-lead occupation into the contact site via the surface term
`ζ²·τ₃ g< τ₃` (method 1). The internal flag `Sigl_s` selects the embedding scheme
and is hardcoded to `1` (it was formerly a function argument; `Sigl_s == 2` would
select an alternative retarded-self-energy-difference embedding). Supplies the
occupation information for the Keldysh equation in [`Glesser_Floquet_Tfull`](@ref).
"""
function Siglf(ww, Omega, Nf, zeta, delta, T, Gamma)
    Sigl_s = 1;  # surface-embedding flag (hardcoded; formerly a function argument)
    Siglbath = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); 
    Siglsurface = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); 
        
    ## Sig< due to generic broadening from separate baths attached to each lattice site
    Siglbath_d = zeros(ComplexF64, 2*(2*Nf+1),2*(2*Nf+1));
    for ij = -Nf:Nf
        ff = (ww+ij*Omega) < 0;
        Siglbath_d[2*(-ij+Nf+1)-1:2*(-ij+Nf+1), 2*(-ij+Nf+1)-1:2*(-ij+Nf+1)] = 1 .* ff .* ( +im*2*Gamma ) * I(2);
    end
    Siglbath = [Siglbath_d zeros(size(Siglbath_d));
                zeros(size(Siglbath_d)) Siglbath_d];

    if Sigl_s == 1
        ## Sig< due to embedding the semi infinite leads into the surface site. Method 1
        glwmn = Keldyshsetup_Floquetn.glwmnf(ww, Omega, Nf, zeta, delta, Gamma);
        
        tau3 = [1 0; 0 -1];
        for ij = -Nf:Nf
            Siglsurface[2*(-ij+Nf+1)-1:2*(-ij+Nf+1), 2*(-ij+Nf+1)-1:2*(-ij+Nf+1)] = zeta^2 .* ( tau3 * glwmn[2*(-ij+Nf+1)-1:2*(-ij+Nf+1), 2*(-ij+Nf+1)-1:2*(-ij+Nf+1)] * tau3 );
            Siglsurface[2*(2*Nf+1)+2*(-ij+Nf+1)-1:2*(2*Nf+1)+2*(-ij+Nf+1), 2*(2*Nf+1)+2*(-ij+Nf+1)-1:2*(2*Nf+1)+2*(-ij+Nf+1)] = zeta^2 .* ( tau3 * glwmn[2*(2*Nf+1)+2*(-ij+Nf+1)-1:2*(2*Nf+1)+2*(-ij+Nf+1), 2*(2*Nf+1)+2*(-ij+Nf+1)-1:2*(2*Nf+1)+2*(-ij+Nf+1)] * tau3 );
        end
    elseif Sigl_s == 2
        ## Sig< due to embedding the semi infinite leads into the surface site. Method 2
        grwmn = Keldyshsetup_Floquetn.grwmnf(ww, Omega, Nf, zeta, delta, Gamma);
        gr0wmn = Keldyshsetup_Floquetn.gr0wmnf(ww, Omega, Nf, zeta, delta, Gamma);
        
        Sigrsurface = -(inv(grwmn)-inv(gr0wmn));
        for ij = -Nf:Nf
            ff = (ww+ij*Omega) < 0;
            Siglsurface[2*(-ij+Nf+1)-1:2*(-ij+Nf+1), 2*(-ij+Nf+1)-1:2*(-ij+Nf+1)] = ff * (-2) * im .* imag.(Sigrsurface[2*(-ij+Nf+1)-1:2*(-ij+Nf+1), 2*(-ij+Nf+1)-1:2*(-ij+Nf+1)]);
            Siglsurface[2*(2*Nf+1)+2*(-ij+Nf+1)-1:2*(2*Nf+1)+2*(-ij+Nf+1), 2*(2*Nf+1)+2*(-ij+Nf+1)-1:2*(2*Nf+1)+2*(-ij+Nf+1)] = ff * (-2) * im .* imag.(Sigrsurface[2*(2*Nf+1)+2*(-ij+Nf+1)-1:2*(2*Nf+1)+2*(-ij+Nf+1), 2*(2*Nf+1)+2*(-ij+Nf+1)-1:2*(2*Nf+1)+2*(-ij+Nf+1)]);
        end
    end

    Sigl = Siglbath + Siglsurface;

    return Sigl
end

"""
    Glesser_Floquet_Tfull(ww, Omega, Nf, zeta, delta, T, Gamma, Vip,
                          Grwmn=nothing, Gawmn=nothing, Sigl=nothing) -> Matrix{ComplexF64}

Full (all-orders in 𝒯) LESSER Green's function via the Keldysh equation
`G< = Gʳ·Σ<·Gᵃ`, with `Gʳ` from [`Grwmnf`](@ref) and `Σ<` from [`Siglf`](@ref). The
optional `Grwmn`, `Gawmn`, `Sigl` allow passing precomputed pieces to avoid
recomputation. This is the exact-Dyson scheme (`ws=0`).
"""
function Glesser_Floquet_Tfull(ww, Omega, Nf, zeta, delta, T, Gamma, Vip, Grwmn = nothing, Gawmn = nothing, Sigl = nothing)
    if isnothing(Grwmn)
        Grwmn = Keldyshsetup_Floquetn.Grwmnf(ww, Omega, Nf, zeta, delta, T, Gamma, Vip);
        # Gawmn = Keldyshsetup_Floquetn.Gawmnf(ww, Omega, Nf, zeta, delta, T, Gamma, Vip);
        Gawmn = Grwmn';
    end

    if isnothing(Sigl)
        Sigl = Keldyshsetup_Floquetn.Siglf(ww, Omega, Nf, zeta, delta, T, Gamma);
    end

    Glwmn = Grwmn*Sigl*Gawmn;
    # Glwmn = Array(cu(Grwmn)*cu(Sigl)*cu(Gawmn))

    # Glwmn = (I(2*2*(2*Nf+1))+Grwmn*Vv)*glwmn*(I(2*2*(2*Nf+1))+Vv*Gawmn);
    # Glwmn = (I(2*2*(2*Nf+1))+Grwmn*Vv)*glwmn*(I(2*2*(2*Nf+1))+Vv*Gawmn) + Grwmn*Sigl*Gawmn;
    # Glwmn = Grwmn*( inv(grwmn)*glwmn*inv(conj(transpose(grwmn))) )*Gawmn;
    # Glwmn = Grwmn*( inv(grwmn)*glwmn*inv(conj(transpose(grwmn))) )*Gawmn + Grwmn*Sigl*Gawmn;
    # Glwmn = (Grwmn*inv(grwmn)) * (glwmn + glwmn*Vv*Gawmn + grwmn*Sigl*Gawmn);
    
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
function Glesser_Floquet_T2(ww, Omega, Nf, zeta, delta, T, Gamma, Vip, grwmn = nothing, gawmn = nothing, glwmn = nothing)
    if isnothing(grwmn)
        grwmn = Keldyshsetup_Floquetn.grwmnf(ww, Omega, Nf, zeta, delta, Gamma)
        gawmn = conj(transpose( grwmn ));
    end
    if isnothing(glwmn)
        glwmn = Keldyshsetup_Floquetn.glwmnf(ww, Omega, Nf, zeta, delta, Gamma)
    end
    Vv = Keldyshsetup_Floquetn.Vwmnf(Vip, Nf, T);

    Glwmn = grwmn*Vv*glwmn + glwmn*Vv*gawmn;

    return Glwmn
end

"""
    Glesser_Floquet_T4(ww, Omega, Nf, zeta, delta, T, Gamma, Vip,
                       grwmn=nothing, gawmn=nothing, glwmn=nothing) -> Matrix{ComplexF64}

LESSER Green's function of the transparency expansion kept through O(𝒯⁴) (the O(𝒯²)
term of [`Glesser_Floquet_T2`](@ref) plus the next chain order). Scheme `ws=4`.
"""
function Glesser_Floquet_T4(ww, Omega, Nf, zeta, delta, T, Gamma, Vip, grwmn = nothing, gawmn = nothing, glwmn = nothing)
    if isnothing(grwmn)
        grwmn = Keldyshsetup_Floquetn.grwmnf(ww, Omega, Nf, zeta, delta, Gamma)
        gawmn = conj(transpose( grwmn ));
    end
    if isnothing(glwmn)
        glwmn = Keldyshsetup_Floquetn.glwmnf(ww, Omega, Nf, zeta, delta, Gamma)
    end
    Vv = Keldyshsetup_Floquetn.Vwmnf(Vip, Nf, T);
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
function Glesser_Floquet_T6(ww, Omega, Nf, zeta, delta, T, Gamma, Vip, grwmn = nothing, gawmn = nothing, glwmn = nothing)
    if isnothing(grwmn)
        grwmn = Keldyshsetup_Floquetn.grwmnf(ww, Omega, Nf, zeta, delta, Gamma)
        gawmn = conj(transpose( grwmn ));
    end
    if isnothing(glwmn)
        glwmn = Keldyshsetup_Floquetn.glwmnf(ww, Omega, Nf, zeta, delta, Gamma)
    end
    Vv = Keldyshsetup_Floquetn.Vwmnf(Vip, Nf, T);
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
function Glesser_Floquet_T8(ww, Omega, Nf, zeta, delta, T, Gamma, Vip, grwmn = nothing, gawmn = nothing, glwmn = nothing)
    if isnothing(grwmn)
        grwmn = Keldyshsetup_Floquetn.grwmnf(ww, Omega, Nf, zeta, delta, Gamma)
        gawmn = conj(transpose( grwmn ));
    end
    if isnothing(glwmn)
        glwmn = Keldyshsetup_Floquetn.glwmnf(ww, Omega, Nf, zeta, delta, Gamma)
    end
    Vv = Keldyshsetup_Floquetn.Vwmnf(Vip, Nf, T);
    gr_s = grwmn*Vv; s_ga = Vv*gawmn;

    Glwmn = (gr_s*glwmn + glwmn*s_ga) + # O(T²) O(T⁴) terms don't enter current. So it doesn't appear anywhere.
            (gr_s*gr_s*gr_s*glwmn + gr_s*gr_s*glwmn*s_ga + gr_s*glwmn*s_ga*s_ga + glwmn*s_ga*s_ga*s_ga) +
            (gr_s*gr_s*gr_s*gr_s*gr_s*glwmn + gr_s*gr_s*gr_s*gr_s*glwmn*s_ga + gr_s*gr_s*gr_s*glwmn*s_ga*s_ga + gr_s*gr_s*glwmn*s_ga*s_ga*s_ga + gr_s*glwmn*s_ga*s_ga*s_ga*s_ga + glwmn*s_ga*s_ga*s_ga*s_ga*s_ga) +
            (gr_s*gr_s*gr_s*gr_s*gr_s*gr_s*gr_s*glwmn + gr_s*gr_s*gr_s*gr_s*gr_s*gr_s*glwmn*s_ga + gr_s*gr_s*gr_s*gr_s*gr_s*glwmn*s_ga*s_ga + gr_s*gr_s*gr_s*gr_s*glwmn*s_ga*s_ga*s_ga + gr_s*gr_s*gr_s*glwmn*s_ga*s_ga*s_ga*s_ga + gr_s*gr_s*glwmn*s_ga*s_ga*s_ga*s_ga*s_ga + gr_s*glwmn*s_ga*s_ga*s_ga*s_ga*s_ga*s_ga + glwmn*s_ga*s_ga*s_ga*s_ga*s_ga*s_ga*s_ga);

    return Glwmn
end


"""
    current_Floquet_Tfull(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, iterprint) -> Matrix{ComplexF64}

Floquet-mode-resolved current for a given phase solution `Vip`, exact in 𝒯. At each
energy in `war0` it forms `G<` via [`Glesser_Floquet_Tfull`](@ref) and the current
matrix `−Vᵢ·G<` (with `Vᵢ`=[`Viwmnf`](@ref)), takes the Nambu trace (LL−RR) per
Floquet pair (m,n), and integrates over the one-period grid `war0`. Returns the
matrix `Iif[m,n]`; its diagonal sum is the DC current and the off-diagonals are the
AC harmonics. The core current evaluator used by the drivers and by
[`IbiasResidual_Tfull`](@ref).
"""
function current_Floquet_Tfull(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, iterprint)
    Nw0 = length(war0); deltaw0 = abs(war0[2]-war0[1]);

    Vvi = Keldyshsetup_Floquetn.Viwmnf(Vip, Nf, T); #Has extra tau_z Nambu Pauli matrix needed in current calculation
    
    Iiwf = zeros(ComplexF64, 2*Nf+1,2*Nf+1,Nw0);
    Threads.@threads for ij = 1:Nw0
        # println(ij)
        Glwmn = Keldyshsetup_Floquetn.Glesser_Floquet_Tfull(war0[ij], Omega, Nf, zeta, delta, T, Gamma, Vip)
        Itemp0 = -Vvi*Glwmn; #Has extra tau_z Nambu Pauli matrix needed in current calculation
        ##--m2--
        Threads.@threads for jk = 1:2*Nf+1
            for kl = 1:2*Nf+1          #VLR*G<RL                                               #VRL*G<LR
                Iiwf[jk,kl,ij] = (Itemp0[2*jk-1,2*kl-1]+Itemp0[2*jk,2*kl]) - (Itemp0[2*(2*Nf+1)+2*jk-1,2*(2*Nf+1)+2*kl-1]+Itemp0[2*(2*Nf+1)+2*jk,2*(2*Nf+1)+2*kl]); #Trace in Nambu space for each Floquet mode (jk)
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
    current_Floquet_T2(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, iterprint) -> Matrix{ComplexF64}

Floquet-mode-resolved current for phase solution `Vip`, truncated at O(𝒯²): like
[`current_Floquet_Tfull`](@ref) but with `G<` from [`Glesser_Floquet_T2`](@ref).
"""
function current_Floquet_T2(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, iterprint)
    Nw0 = length(war0); deltaw0 = abs(war0[2]-war0[1]);

    Vvi = Keldyshsetup_Floquetn.Viwmnf(Vip, Nf, T); #Has extra tau_z Nambu Pauli matrix needed in current calculation
    
    Iiwf = zeros(ComplexF64, 2*Nf+1,2*Nf+1,Nw0);
    Threads.@threads for ij = 1:Nw0
        grwmn = Keldyshsetup_Floquetn.grwmnf(war0[ij], Omega, Nf, zeta, delta, Gamma);
        glwmn = Keldyshsetup_Floquetn.glwmnf(war0[ij], Omega, Nf, zeta, delta, Gamma);
        
        Glwmn = Keldyshsetup_Floquetn.Glesser_Floquet_T2(war0[ij], Omega, Nf, zeta, delta, T, Gamma, Vip, grwmn, conj(transpose(grwmn)), glwmn);
        Itemp0 = -Vvi*Glwmn; #Has extra tau_z Nambu Pauli matrix needed in current calculation
        ##--m2--
        Threads.@threads for jk = 1:2*Nf+1
            for kl = 1:2*Nf+1          #VLR*G<RL                                               #VRL*G<LR
                Iiwf[jk,kl,ij] = (Itemp0[2*jk-1,2*kl-1]+Itemp0[2*jk,2*kl]) - (Itemp0[2*(2*Nf+1)+2*jk-1,2*(2*Nf+1)+2*kl-1]+Itemp0[2*(2*Nf+1)+2*jk,2*(2*Nf+1)+2*kl]); #Trace in Nambu space for each Floquet mode (jk)
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
    current_Floquet_T4(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, iterprint) -> Matrix{ComplexF64}

Floquet-mode-resolved current truncated at O(𝒯⁴) (uses [`Glesser_Floquet_T4`](@ref)).
"""
function current_Floquet_T4(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, iterprint)
    Nw0 = length(war0); deltaw0 = abs(war0[2]-war0[1]);

    Vvi = Keldyshsetup_Floquetn.Viwmnf(Vip, Nf, T); #Has extra tau_z Nambu Pauli matrix needed in current calculation
    
    Iiwf = zeros(ComplexF64, 2*Nf+1,2*Nf+1,Nw0);
    Threads.@threads for ij = 1:Nw0
        grwmn = Keldyshsetup_Floquetn.grwmnf(war0[ij], Omega, Nf, zeta, delta, Gamma);
        glwmn = Keldyshsetup_Floquetn.glwmnf(war0[ij], Omega, Nf, zeta, delta, Gamma);
        
        Glwmn = Keldyshsetup_Floquetn.Glesser_Floquet_T4(war0[ij], Omega, Nf, zeta, delta, T, Gamma, Vip, grwmn, conj(transpose(grwmn)), glwmn);
        Itemp0 = -Vvi*Glwmn; #Has extra tau_z Nambu Pauli matrix needed in current calculation
        ##--m2--
        Threads.@threads for jk = 1:2*Nf+1
            for kl = 1:2*Nf+1          #VLR*G<RL                                               #VRL*G<LR
                Iiwf[jk,kl,ij] = (Itemp0[2*jk-1,2*kl-1]+Itemp0[2*jk,2*kl]) - (Itemp0[2*(2*Nf+1)+2*jk-1,2*(2*Nf+1)+2*kl-1]+Itemp0[2*(2*Nf+1)+2*jk,2*(2*Nf+1)+2*kl]); #Trace in Nambu space for each Floquet mode (jk)
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
    current_Floquet_T6(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, iterprint) -> Matrix{ComplexF64}

Floquet-mode-resolved current truncated at O(𝒯⁶) (uses [`Glesser_Floquet_T6`](@ref)).
"""
function current_Floquet_T6(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, iterprint)
    Nw0 = length(war0); deltaw0 = abs(war0[2]-war0[1]);

    Vvi = Keldyshsetup_Floquetn.Viwmnf(Vip, Nf, T); #Has extra tau_z Nambu Pauli matrix needed in current calculation
    
    Iiwf = zeros(ComplexF64, 2*Nf+1,2*Nf+1,Nw0);
    Threads.@threads for ij = 1:Nw0
        grwmn = Keldyshsetup_Floquetn.grwmnf(war0[ij], Omega, Nf, zeta, delta, Gamma);
        glwmn = Keldyshsetup_Floquetn.glwmnf(war0[ij], Omega, Nf, zeta, delta, Gamma);
        
        Glwmn = Keldyshsetup_Floquetn.Glesser_Floquet_T6(war0[ij], Omega, Nf, zeta, delta, T, Gamma, Vip, grwmn, conj(transpose(grwmn)), glwmn);
        Itemp0 = -Vvi*Glwmn; #Has extra tau_z Nambu Pauli matrix needed in current calculation
        ##--m2--
        Threads.@threads for jk = 1:2*Nf+1
            for kl = 1:2*Nf+1          #VLR*G<RL                                               #VRL*G<LR
                Iiwf[jk,kl,ij] = (Itemp0[2*jk-1,2*kl-1]+Itemp0[2*jk,2*kl]) - (Itemp0[2*(2*Nf+1)+2*jk-1,2*(2*Nf+1)+2*kl-1]+Itemp0[2*(2*Nf+1)+2*jk,2*(2*Nf+1)+2*kl]); #Trace in Nambu space for each Floquet mode (jk)
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
    current_Floquet_T8(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, iterprint) -> Matrix{ComplexF64}

Floquet-mode-resolved current truncated at O(𝒯⁸) (uses [`Glesser_Floquet_T8`](@ref)).
"""
function current_Floquet_T8(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, iterprint)
    Nw0 = length(war0); deltaw0 = abs(war0[2]-war0[1]);

    Vvi = Keldyshsetup_Floquetn.Viwmnf(Vip, Nf, T); #Has extra tau_z Nambu Pauli matrix needed in current calculation
    
    Iiwf = zeros(ComplexF64, 2*Nf+1,2*Nf+1,Nw0);
    Threads.@threads for ij = 1:Nw0
        grwmn = Keldyshsetup_Floquetn.grwmnf(war0[ij], Omega, Nf, zeta, delta, Gamma);
        glwmn = Keldyshsetup_Floquetn.glwmnf(war0[ij], Omega, Nf, zeta, delta, Gamma);
        
        Glwmn = Keldyshsetup_Floquetn.Glesser_Floquet_T8(war0[ij], Omega, Nf, zeta, delta, T, Gamma, Vip, grwmn, conj(transpose(grwmn)), glwmn);
        Itemp0 = -Vvi*Glwmn; #Has extra tau_z Nambu Pauli matrix needed in current calculation
        ##--m2--
        Threads.@threads for jk = 1:2*Nf+1
            for kl = 1:2*Nf+1          #VLR*G<RL                                               #VRL*G<LR
                Iiwf[jk,kl,ij] = (Itemp0[2*jk-1,2*kl-1]+Itemp0[2*jk,2*kl]) - (Itemp0[2*(2*Nf+1)+2*jk-1,2*(2*Nf+1)+2*kl-1]+Itemp0[2*(2*Nf+1)+2*jk,2*(2*Nf+1)+2*kl]); #Trace in Nambu space for each Floquet mode (jk)
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

Hopping self-energy Σʳ in the Floquet–Nambu–lead basis, off-diagonal in L/R, built
from the phase Fourier coefficients `Vip` (of `e^{−iφ/2}`) and transparency `T`:
`[V]ᴸᴿ_{mn} = 𝒯·diag(W_{n−m}, −W*_{m−n})`,
`[V]ᴿᴸ_{mn} = 𝒯·diag(W*_{m−n}, −W_{n−m})`.
Returns a 4(2Nf+1)×4(2Nf+1) matrix. Linear in `Vip`.
"""
function Vwmnf(Vip, Nf, T)
    Vwmn_d = zeros(ComplexF64, 2*(2*Nf+1),2*(2*Nf+1));
    Vwmn_d1 = zeros(ComplexF64, 2*(2*Nf+1),2*(2*Nf+1));

    for ij = -Nf:Nf
        for jk = -Nf:Nf
            Vwmn_d[2*(-ij+Nf+1)-1:2*(-ij+Nf+1),2*(-jk+Nf+1)-1:2*(-jk+Nf+1)] = T .* [ Vip[-(ij-jk)+2*Nf+1] 0;
                                                                                 0 -conj(Vip[(ij-jk)+2*Nf+1]) ];
            Vwmn_d1[2*(-ij+Nf+1)-1:2*(-ij+Nf+1),2*(-jk+Nf+1)-1:2*(-jk+Nf+1)] = T .* [ conj(Vip[(ij-jk)+2*Nf+1]) 0;
                                                                                  0 -Vip[-(ij-jk)+2*Nf+1] ];                                                                                 
        end
    end

    Vwmn = [zeros(size(Vwmn_d)) Vwmn_d;
            Vwmn_d1 zeros(size(Vwmn_d))];

    return Vwmn
end

"""
    Viwmnf(Vip, Nf, T) -> Matrix{ComplexF64}

Current-vertex variant of [`Vwmnf`](@ref): the same hopping matrix but with the
lower (hole) Nambu sign flipped (i.e. `τ₃`-folded), as required by the current
operator `I = tr[τ₃(…)]`. Used to form the current `−Viwmnf·G<`.
"""
function Viwmnf(Vip, Nf, T)
    Vwmn_d = zeros(ComplexF64, 2*(2*Nf+1),2*(2*Nf+1));
    Vwmn_d1 = zeros(ComplexF64, 2*(2*Nf+1),2*(2*Nf+1));

    for ij = -Nf:Nf
        for jk = -Nf:Nf
            Vwmn_d[2*(-ij+Nf+1)-1:2*(-ij+Nf+1),2*(-jk+Nf+1)-1:2*(-jk+Nf+1)] = T .* [ Vip[-(ij-jk)+2*Nf+1] 0;
                                                                                 0 conj(Vip[(ij-jk)+2*Nf+1]) ];
            Vwmn_d1[2*(-ij+Nf+1)-1:2*(-ij+Nf+1),2*(-jk+Nf+1)-1:2*(-jk+Nf+1)] = T .* [ conj(Vip[(ij-jk)+2*Nf+1]) 0;
                                                                                  0 Vip[-(ij-jk)+2*Nf+1] ];     
        end
    end
    
    Vwmn = [zeros(size(Vwmn_d)) Vwmn_d;
            Vwmn_d1 zeros(size(Vwmn_d))];

    return Vwmn
end

"""
    IbiasResidual_Tfull(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, iterprint) -> Vector{Float64}

Residual `F(x)` of the current-bias self-consistency problem, exact in 𝒯. The
unknown real vector `Vipi` (length 4Nf) packs the real and imaginary parts of the
2Nf odd phase harmonics `W_m`. Returns the 4Nf real equations:
1. unitarity/normalization of `e^{−iφ/2}` — convolution `C_{2h}=Σ_r W_{r+2h} W*_r = δ_{h,0}`;
2. gauge `φ(0)=0` — `Σ_m Im W_m = 0`;
3. vanishing of every AC current harmonic `I_{2h}=0`, with the current from
   [`current_Floquet_Tfull`](@ref).
A root of `F` is the self-consistent phase whose voltage carries only DC current.
"""
function IbiasResidual_Tfull(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, iterprint)
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
    Iif = Keldyshsetup_Floquetn.current_Floquet_Tfull(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, iterprint);
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
    IbiasResidual_T2(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, iterprint) -> Vector{Float64}

Current-bias residual `F(x)` with the current rows evaluated at O(𝒯²) (via
[`current_Floquet_T2`](@ref)); the unitarity and gauge rows are identical to
[`IbiasResidual_Tfull`](@ref). Used by [`phisolve`](@ref) for `ws=2`.
"""
function IbiasResidual_T2(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, iterprint)
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
    Iif = Keldyshsetup_Floquetn.current_Floquet_T2(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, iterprint);
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
    IbiasResidual_T4(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, iterprint) -> Vector{Float64}

Current-bias residual with current rows at O(𝒯⁴) ([`current_Floquet_T4`](@ref));
constraint rows as in [`IbiasResidual_Tfull`](@ref). Used by [`phisolve`](@ref) for `ws=4`.
"""
function IbiasResidual_T4(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, iterprint)
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
    Iif = Keldyshsetup_Floquetn.current_Floquet_T4(war0, Omega, Nf, zeta, delta, T, Gamma, Vip, iterprint);
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
    IbiasJacobian_Tfull(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, iterprint) -> Matrix{Float64}

Analytic Jacobian `J_{ij}=∂F_i/∂x_j` of [`IbiasResidual_Tfull`](@ref) (4Nf×4Nf).
The unitarity/gauge-constraint rows are differentiated in closed form (polynomial in
`W`). The current rows use
`∂G</∂x_j = Gʳ·M_j·G< + G<·M_j·Gᵃ`
(valid because `Σ<` is independent of `x` and `V` is linear in `x`, with
`M_j=∂V/∂x_j` sparse and constant), reorganized so `−Vᵢ·Gʳ` and `−Vᵢ·G<` are formed
once per energy. Passed to `NLsolve` for the trust-region Newton solve.
"""
function IbiasJacobian_Tfull(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, iterprint)
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
    jaceqns[2*Nf-1,1:(2*Nf)] .+= 2*real.(Vip[2:2:(4*Nf)]);
    jaceqns[2*Nf-1,(2*Nf)+1:2*(2*Nf)] .+= 2*imag.(Vip[2:2:(4*Nf)]);

    #-- Gauge (row 2Nf): φ(t=0)=0 ⟹ Σ_m Im(W_m)=0, so ∂/∂Im(W)=1, ∂/∂Re(W)=0 --
    for ij = (2*Nf)+1:2*(2*Nf) #variable (for derivative in Jacobian) index
        jaceqns[2*Nf,ij] = 1;
    end

    # ===== Rows 2Nf+1..4Nf: derivatives of the AC current harmonics I_{2h} =====
    Vvi = Keldyshsetup_Floquetn.Viwmnf(Vip, Nf, T); # current vertex (τ₃-folded hopping)
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
    Miar = zeros(ComplexF64, 2*2*(2*Nf+1),2*2*(2*Nf+1),4*Nf); Mar = zeros(ComplexF64, 2*2*(2*Nf+1),2*2*(2*Nf+1),4*Nf); # Mar[:,:,ij]=∂V/∂x_ij, Miar[:,:,ij]=∂Vᵢ/∂x_ij
    for ij = 1:2*(2*Nf) #variable (for derivative in Jacobian) index
        Vipin = Vipi + dV[:,ij]; # perturb the ij-th real unknown
        for jk = 1:2*Nf
            Vipn[2*jk] = Vipin[jk] + im*Vipin[(2*Nf)+jk]; #Vipi only has even harmonics of eV, (-Nf*eV:Nf*eV, Nf is even)
        end
        @views Miar[:,:,ij] = ( Keldyshsetup_Floquetn.Viwmnf(Vipn, Nf, T) - Keldyshsetup_Floquetn.Viwmnf(Vip, Nf, T) ) ./ deltaV; # exact (linear) derivative
        @views Mar[:,:,ij] = ( Keldyshsetup_Floquetn.Vwmnf(Vipn, Nf, T) - Keldyshsetup_Floquetn.Vwmnf(Vip, Nf, T) ) ./ deltaV;
    end
    Mar = Mar .* (abs.(Mar) .> 0.5*T*deltaV); Miar = -Miar .* (abs.(Miar) .> 0.5*T*deltaV); # drop numerical-zero entries; fold the leading minus of ∂(-VᵢG<)/∂x into Miar
    
    Marsp = [sparse(@view Mar[:,:,ij]) for ij = 1:4*Nf]; Miarsp = [sparse(@view Miar[:,:,ij]) for ij = 1:4*Nf]; #dV/dx is a single (sparse) Floquet off-diagonal
    #= ---- OLD: serial over Nw0 (frequency), threaded over the 4Nf columns. Kept for reference. ----
    Glwmntemp = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Glwmn = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); VviGrwmn = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); VviGlwmn = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1));
    tls = TaskLocalValue{Vector{Matrix{ComplexF64}}}( () ->
                     begin
                         t1, t2, t3 = (zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)) for _ in 1:3)
                         [t1, t2, t3]
                     end)
    for gh = 1:Nw0
        if gh%10 == 0
            println(" (jac) outer iter = ",iterprint)
            println(" (jac) w iter/Nw0 = $(gh)/$(Nw0)")
        end
        Grwmn = Keldyshsetup_Floquetn.Grwmnf(war0[gh], Omega, Nf, zeta, delta, T, Gamma, Vip);
        Gawmn = conj(transpose(Grwmn));
        Sigl = Siglf(war0[gh], Omega, Nf, zeta, delta, T, Gamma);
        mul!(Glwmntemp, Grwmn, Sigl); mul!(Glwmn, Glwmntemp, Gawmn);
        mul!(VviGrwmn, -Vvi, Grwmn); mul!(VviGlwmn, -Vvi, Glwmn);
        Threads.@threads for ij = 1:(4*Nf)
            t1, t2, t3 = tls[]
            mul!(t1, Marsp[ij], Gawmn);  mul!(t3, VviGlwmn, t1)
            mul!(t2, Marsp[ij], Glwmn);  mul!(t3, VviGrwmn, t2, 1, 1)
            mul!(t3, Miarsp[ij], Glwmn, 1, 1)
            for jk = 1:(2*Nf+1)
                for kl = 1:(2*Nf+1)
                    @views jacIif[jk,kl,ij] += (t3[2*jk-1,2*kl-1]+t3[2*jk,2*kl]) - (t3[2*(2*Nf+1)+2*jk-1,2*(2*Nf+1)+2*kl-1]+t3[2*(2*Nf+1)+2*jk,2*(2*Nf+1)+2*kl]);
                end
            end
        end
    end
    =#

    # ---- NEW: thread over frequency CHUNKS (parallel width = up to Nw0). ----
    # Each chunk owns a private accumulator partials[c] and its own scratch (plain locals, allocated
    # once per chunk and reused via mul!) -- no TaskLocalValue. Pair with BLAS.set_num_threads(1).
    # (For heterogeneous P/E cores, raise nchunks and use `Threads.@threads :dynamic`.)
    N4 = 4*(2*Nf+1);
    nchunks = min(Nw0, max(Threads.nthreads(), 1));
    cb = round.(Int, range(0, Nw0; length = nchunks+1)); # contiguous chunk boundaries of 1:Nw0
    partials = [zeros(ComplexF64, 2*Nf+1, 2*Nf+1, 2*(2*Nf)) for _ = 1:nchunks];
    Threads.@threads for c = 1:nchunks
        jacloc = partials[c];
        Glwmntemp = zeros(ComplexF64, N4, N4); Glwmn = zeros(ComplexF64, N4, N4);
        VviGrwmn  = zeros(ComplexF64, N4, N4); VviGlwmn = zeros(ComplexF64, N4, N4);
        t1 = zeros(ComplexF64, N4, N4); t2 = zeros(ComplexF64, N4, N4); t3 = zeros(ComplexF64, N4, N4);
        for gh = (cb[c]+1):cb[c+1]
            Grwmn = Keldyshsetup_Floquetn.Grwmnf(war0[gh], Omega, Nf, zeta, delta, T, Gamma, Vip); # full retarded GF (Dyson, exact in T)
            Gawmn = conj(transpose(Grwmn));                                            # advanced GF = (Gr)†
            Sigl = Siglf(war0[gh], Omega, Nf, zeta, delta, T, Gamma);  # lesser self-energy Σ<
            mul!(Glwmntemp, Grwmn, Sigl); mul!(Glwmn, Glwmntemp, Gawmn);               # Keldysh lesser GF: G< = Gr·Σ<·Ga
            mul!(VviGrwmn, -Vvi, Grwmn); mul!(VviGlwmn, -Vvi, Glwmn);                  # vertex prefactors -Vᵢ·Gr and -Vᵢ·G<
            for ij = 1:(4*Nf) #variable (for derivative in Jacobian) index
                mul!(t1, Marsp[ij], Gawmn);  mul!(t3, VviGlwmn, t1) # -Vᵢ·G< · (∂V/∂x · Ga)
                mul!(t2, Marsp[ij], Glwmn);  mul!(t3, VviGrwmn, t2, 1, 1) # + -Vᵢ·Gr · (∂V/∂x · G<)
                mul!(t3, Miarsp[ij], Glwmn, 1, 1) # + (-∂Vᵢ/∂x) · G<
                for jk = 1:(2*Nf+1)
                    for kl = 1:(2*Nf+1)          # LL Nambu trace minus RR block (offset 2(2Nf+1))
                        @views jacloc[jk,kl,ij] += (t3[2*jk-1,2*kl-1]+t3[2*jk,2*kl]) - (t3[2*(2*Nf+1)+2*jk-1,2*(2*Nf+1)+2*kl-1]+t3[2*(2*Nf+1)+2*jk,2*(2*Nf+1)+2*kl]);
                    end
                end
            end
        end
    end
    jacIif = sum(partials);                              # reduce the per-chunk partials

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
    IbiasJacobian_T2(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, iterprint) -> Matrix{Float64}

Analytic Jacobian of [`IbiasResidual_T2`](@ref): same construction as
[`IbiasJacobian_Tfull`](@ref) but with the current-row derivatives built from the
bare propagators / O(𝒯²)-truncated `G<`. Used by [`phisolve`](@ref) for `ws=2`.
"""
function IbiasJacobian_T2(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, iterprint)
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
    Vvi = Keldyshsetup_Floquetn.Viwmnf(Vip, Nf, T); # current vertex (τ₃-folded hopping)
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
    Miar = zeros(ComplexF64, 2*2*(2*Nf+1),2*2*(2*Nf+1),4*Nf); Mar = zeros(ComplexF64, 2*2*(2*Nf+1),2*2*(2*Nf+1),4*Nf); # Mar[:,:,ij]=∂V/∂x_ij, Miar[:,:,ij]=∂Vᵢ/∂x_ij
    for ij = 1:2*(2*Nf) #variable (for derivative in Jacobian) index
        Vipin = Vipi + dV[:,ij]; # perturb the ij-th real unknown
        for jk = 1:2*Nf
            Vipn[2*jk] = Vipin[jk] + im*Vipin[(2*Nf)+jk]; #Vipi only has even harmonics of eV, (-Nf*eV:Nf*eV, Nf is even)
        end
        @views Miar[:,:,ij] = ( Keldyshsetup_Floquetn.Viwmnf(Vipn, Nf, T) - Keldyshsetup_Floquetn.Viwmnf(Vip, Nf, T) ) ./ deltaV; # exact (linear) derivative
        @views Mar[:,:,ij] = ( Keldyshsetup_Floquetn.Vwmnf(Vipn, Nf, T) - Keldyshsetup_Floquetn.Vwmnf(Vip, Nf, T) ) ./ deltaV;
    end
    Mar = Mar .* (abs.(Mar) .> 0.5*T*deltaV); Miar = -Miar .* (abs.(Miar) .> 0.5*T*deltaV); # drop numerical-zero entries; fold the leading minus of ∂(-VᵢG<)/∂x into Miar
    
    Marsp = [sparse(@view Mar[:,:,ij]) for ij = 1:4*Nf]; Miarsp = [sparse(@view Miar[:,:,ij]) for ij = 1:4*Nf]; #dV/dx is a single (sparse) Floquet off-diagonal
    #= ---- OLD: serial over Nw0 (frequency), threaded over the 4Nf columns. Kept for reference. ----
    Glwmn_w2 = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Vvigrwmn = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Vviglwmn = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1));
    tls = TaskLocalValue{Vector{Matrix{ComplexF64}}}( () ->
                     begin
                         t1, t2, t3 = (zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)) for _ in 1:3)
                         [t1, t2, t3]
                     end)
    for gh = 1:Nw0
        grwmn = Keldyshsetup_Floquetn.grwmnf(war0[gh], Omega, Nf, zeta, delta, Gamma)
        gawmn = conj(transpose( grwmn ));
        glwmn = Keldyshsetup_Floquetn.glwmnf(war0[gh], Omega, Nf, zeta, delta, Gamma)
        Glwmn_w2 .= Keldyshsetup_Floquetn.Glesser_Floquet_T2(war0[gh], Omega, Nf, zeta, delta, T, Gamma, Vip, grwmn, gawmn, glwmn);
        mul!(Vvigrwmn, -Vvi, grwmn); mul!(Vviglwmn, -Vvi, glwmn);
        Threads.@threads for ij = 1:(4*Nf)
            t1, t2, t3 = tls[]
            mul!(t1, Marsp[ij], gawmn);  mul!(t3, Vviglwmn, t1)
            mul!(t2, Marsp[ij], glwmn);  mul!(t3, Vvigrwmn, t2, 1, 1)
            mul!(t3, Miarsp[ij], Glwmn_w2, 1, 1)
            for jk = 1:(2*Nf+1)
                for kl = 1:(2*Nf+1)
                    @views jacIif[jk,kl,ij] += (t3[2*jk-1,2*kl-1]+t3[2*jk,2*kl]) - (t3[2*(2*Nf+1)+2*jk-1,2*(2*Nf+1)+2*kl-1]+t3[2*(2*Nf+1)+2*jk,2*(2*Nf+1)+2*kl]);
                end
            end
        end
    end
    =#

    # ---- NEW: thread over frequency CHUNKS (parallel width = up to Nw0). Private scratch + accumulator. ----
    N4 = 4*(2*Nf+1);
    nchunks = min(Nw0, max(Threads.nthreads(), 1));
    cb = round.(Int, range(0, Nw0; length = nchunks+1));
    partials = [zeros(ComplexF64, 2*Nf+1, 2*Nf+1, 2*(2*Nf)) for _ = 1:nchunks];
    Threads.@threads for c = 1:nchunks
        jacloc = partials[c];
        Glwmn_w2 = zeros(ComplexF64, N4, N4); Vvigrwmn = zeros(ComplexF64, N4, N4); Vviglwmn = zeros(ComplexF64, N4, N4);
        t1 = zeros(ComplexF64, N4, N4); t2 = zeros(ComplexF64, N4, N4); t3 = zeros(ComplexF64, N4, N4);
        for gh = (cb[c]+1):cb[c+1]
            # Bare lead propagators (the O(𝒯²) scheme expands G< in powers of T):
            grwmn = Keldyshsetup_Floquetn.grwmnf(war0[gh], Omega, Nf, zeta, delta, Gamma)            # bare retarded gr
            gawmn = conj(transpose( grwmn ));                                                        # bare advanced ga = (gr)†
            glwmn = Keldyshsetup_Floquetn.glwmnf(war0[gh], Omega, Nf, zeta, delta, Gamma)            # bare lesser g<
            Glwmn_w2 .= Keldyshsetup_Floquetn.Glesser_Floquet_T2(war0[gh], Omega, Nf, zeta, delta, T, Gamma, Vip, grwmn, gawmn, glwmn); # O(𝒯²) lesser GF
            mul!(Vvigrwmn, -Vvi, grwmn); mul!(Vviglwmn, -Vvi, glwmn);                                 # -Vᵢ·gr , -Vᵢ·g<
            for ij = 1:(4*Nf) #variable (for derivative in Jacobian) index
                mul!(t1, Marsp[ij], gawmn);  mul!(t3, Vviglwmn, t1) # -Vᵢ·g< · (∂V/∂x · ga)
                mul!(t2, Marsp[ij], glwmn);  mul!(t3, Vvigrwmn, t2, 1, 1) # + -Vᵢ·gr · (∂V/∂x · g<)
                mul!(t3, Miarsp[ij], Glwmn_w2, 1, 1) # + (-∂Vᵢ/∂x) · G<₍₂₎
                for jk = 1:(2*Nf+1)
                    for kl = 1:(2*Nf+1)
                        @views jacloc[jk,kl,ij] += (t3[2*jk-1,2*kl-1]+t3[2*jk,2*kl]) - (t3[2*(2*Nf+1)+2*jk-1,2*(2*Nf+1)+2*kl-1]+t3[2*(2*Nf+1)+2*jk,2*(2*Nf+1)+2*kl]);
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
    IbiasJacobian_T4(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, iterprint) -> Matrix{Float64}

Analytic Jacobian of [`IbiasResidual_T4`](@ref) (current-row derivatives at O(𝒯⁴)).
Used by [`phisolve`](@ref) for `ws=4`.
"""
function IbiasJacobian_T4(Vipi, war0, Omega, Nf, zeta, delta, T, Gamma, iterprint)
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
    Vvi = Keldyshsetup_Floquetn.Viwmnf(Vip, Nf, T); # current vertex (τ₃-folded hopping)
    Vv = Keldyshsetup_Floquetn.Vwmnf(Vip, Nf, T);   # hopping self-energy (enters the higher-order G< expansion)

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
    Miar = zeros(ComplexF64, 2*2*(2*Nf+1),2*2*(2*Nf+1),4*Nf); Mar = zeros(ComplexF64, 2*2*(2*Nf+1),2*2*(2*Nf+1),4*Nf); # Mar[:,:,ij]=∂V/∂x_ij, Miar[:,:,ij]=∂Vᵢ/∂x_ij
    for ij = 1:2*(2*Nf) #variable (for derivative in Jacobian) index
        Vipin = Vipi + dV[:,ij]; # perturb the ij-th real unknown
        for jk = 1:2*Nf
            Vipn[2*jk] = Vipin[jk] + im*Vipin[(2*Nf)+jk]; #Vipi only has even harmonics of eV, (-Nf*eV:Nf*eV, Nf is even)
        end
        @views Miar[:,:,ij] = ( Keldyshsetup_Floquetn.Viwmnf(Vipn, Nf, T) - Keldyshsetup_Floquetn.Viwmnf(Vip, Nf, T) ) ./ deltaV; # exact (linear) derivative
        @views Mar[:,:,ij] = ( Keldyshsetup_Floquetn.Vwmnf(Vipn, Nf, T) - Keldyshsetup_Floquetn.Vwmnf(Vip, Nf, T) ) ./ deltaV;
    end
    Mar = Mar .* (abs.(Mar) .> 0.5*T*deltaV); Miar = -Miar .* (abs.(Miar) .> 0.5*T*deltaV); # drop numerical-zero entries; fold the leading minus of ∂(-VᵢG<)/∂x into Miar

    # Preallocated intermediates for the O(𝒯⁴) derivative. Naming convention: 's' denotes the
    # hopping self-energy Vv (=Σ), and r/l/a denote gr/g</ga, so e.g. gr_s_gl = gr·Σ·g< and
    # Vvigr_s_gr = -Vᵢ·gr·Σ·gr. These chains are the building blocks of the O(𝒯⁴) lesser-GF
    # expansion and its derivative; they are formed once per frequency below.
    #= ---- OLD: serial over Nw0 (frequency), threaded over the 4Nf columns. Kept for reference. ----
    Glwmn_w4 = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1));
    Vvigr = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Vvigl = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Vvigr_s_gr = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Vvigr_s_gl = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Vvigl_s_ga = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1));
    Vvigr_s_gr_s_gr = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Vvigr_s_gr_s_gl = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Vvigr_s_gl_s_ga = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Vvigl_s_ga_s_ga = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1));
    gr_s_gr_s_gl = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); gr_s_gr_s_gr = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); gr_s_gr = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); gr_s_gl = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1));
    gr_s_gl_s_ga = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); gl_s_ga = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1));
    ga_s_ga_s_ga = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); gl_s_ga_s_ga = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); ga_s_ga = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1));
    

    for gh = 1:Nw0
        if gh%10 == 0
            println(" (jac) ev iter = ",iterprint)
            println(" (jac) w iter/Nw0 = $(gh)/$(Nw0)")
        end
        
        # Bare propagators at this frequency:
        grwmn = Keldyshsetup_Floquetn.grwmnf(war0[gh], Omega, Nf, zeta, delta, Gamma)            # bare retarded gr
        gawmn = conj(transpose( grwmn ));                                                        # bare advanced ga
        glwmn = Keldyshsetup_Floquetn.glwmnf(war0[gh], Omega, Nf, zeta, delta, Gamma)            # bare lesser g<
        Glwmn_w4 .= Keldyshsetup_Floquetn.Glesser_Floquet_T4(war0[gh], Omega, Nf, zeta, delta, T, Gamma, Vip, grwmn, gawmn, glwmn); # O(𝒯⁴) lesser GF
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
                    @views jacIif[jk,kl,ij] += (t3[2*jk-1,2*kl-1]+t3[2*jk,2*kl]) - (t3[2*(2*Nf+1)+2*jk-1,2*(2*Nf+1)+2*kl-1]+t3[2*(2*Nf+1)+2*jk,2*(2*Nf+1)+2*kl]); #Trace in Nambu space for each Floquet mode (jk)
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
        Glwmn_w4 = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1));
        Vvigr = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Vvigl = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Vvigr_s_gr = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Vvigr_s_gl = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Vvigl_s_ga = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1));
        Vvigr_s_gr_s_gr = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Vvigr_s_gr_s_gl = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Vvigr_s_gl_s_ga = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); Vvigl_s_ga_s_ga = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1));
        gr_s_gr_s_gl = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); gr_s_gr_s_gr = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); gr_s_gr = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); gr_s_gl = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1));
        gr_s_gl_s_ga = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); gl_s_ga = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1));
        ga_s_ga_s_ga = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); gl_s_ga_s_ga = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1)); ga_s_ga = zeros(ComplexF64, 4*(2*Nf+1),4*(2*Nf+1));
    
        for gh = (cb[c]+1):cb[c+1]
        
            # Bare propagators at this frequency:
            grwmn = Keldyshsetup_Floquetn.grwmnf(war0[gh], Omega, Nf, zeta, delta, Gamma)            # bare retarded gr
            gawmn = conj(transpose( grwmn ));                                                        # bare advanced ga
            glwmn = Keldyshsetup_Floquetn.glwmnf(war0[gh], Omega, Nf, zeta, delta, Gamma)            # bare lesser g<
            Glwmn_w4 .= Keldyshsetup_Floquetn.Glesser_Floquet_T4(war0[gh], Omega, Nf, zeta, delta, T, Gamma, Vip, grwmn, gawmn, glwmn); # O(𝒯⁴) lesser GF
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
                        @views jacloc[jk,kl,ij] += (t3[2*jk-1,2*kl-1]+t3[2*jk,2*kl]) - (t3[2*(2*Nf+1)+2*jk-1,2*(2*Nf+1)+2*kl-1]+t3[2*(2*Nf+1)+2*jk,2*(2*Nf+1)+2*kl]); #Trace in Nambu space for each Floquet mode (jk)
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
             Vipsolseed=nothing, Nevseed=nothing) -> (Iv, Vipsol, residualarr)

Main current-bias driver. For each bias `eV` in `evar` (swept high→low with
continuation: the converged phase at one bias seeds the next), solves the self-
consistency `F(x)=0` with `NLsolve.nlsolve` (`:trust_region`), selecting the
transparency-order scheme by the flag `ws` (0 = exact Dyson, 2/4/6/8 = truncated):
it calls the matching [`IbiasResidual_Tfull`](@ref)/`_T2…` and, for `ws=0,2,4`, the
analytic [`IbiasJacobian_Tfull`](@ref)/`_T2`/`_T4`. `dw0` sets the energy-grid
spacing; optional `Vipsolseed`/`Nevseed` warm-start the highest-bias point.

# Returns
- `Iv`: DC current vs bias.
- `Vipsol`: solved phase coefficients per bias (rows).
- `residualarr`: final residual norm per bias.
"""
function phisolve(ws, dw0, evar, Nf, zeta, delta, T, Gamma, Vipsolseed = nothing, Nevseed = nothing)
    Nev = length(evar);

    If = zeros(ComplexF64, Nev,2*Nf+1,2*Nf+1); Ifa = zeros(ComplexF64, Nev,4*Nf+1); Iv = zeros(Float64, Nev);
    Vipsol = zeros(Float64, Nev, 2*(2*Nf)); #(2*Nf) real + (2*Nf) imag #I_mn, Gr_mn, m,n \in -Nf:+Nf, => V_m m \in -2Nf:2Nf. Only odd multiples of eV kept
    Vipseed = zeros(Float64, 2*(2*Nf)); #(2*Nf) real + (2*Nf) imag
    residualarr = zeros(Float64, Nev);

    ftols = 5e-13; xtols = 1e-13; itermax = 200;

    for hi = Nev:-1:1
        println("ev iter = ",hi)
        
        ev = evar[hi]; Omega = ev;
        Nw0 = trunc(Int, abs(Omega)/dw0); 
        # war0 = range(0, (Nw0-1)*Omega/Nw0, Nw0);
        war0 = -0.5*abs(Omega) .+ range(0, (Nw0-1)*abs(Omega)/Nw0, Nw0);
        
        if hi == Nev
            if Vipsolseed == nothing
                Vipseed[Nf] = 1;
            else
                Vipseed = Vipsolseed[Nevseed,:];
            end
        else
            Vipseed .= Vipsol[hi+1,:];
        end
    
        if hi>trunc(Int,0.6*Nev)
            ftols = 5e-17;
        end
    
        if ws == 4
            fvalue = norm(Keldyshsetup_Floquetn.IbiasResidual_T4(Vipseed, war0, Omega, Nf, zeta, delta, T, Gamma,  hi));
        elseif ws == 2
            fvalue = norm(Keldyshsetup_Floquetn.IbiasResidual_T2(Vipseed, war0, Omega, Nf, zeta, delta, T, Gamma,  hi));
        elseif ws == 0
            fvalue = norm(Keldyshsetup_Floquetn.IbiasResidual_Tfull(Vipseed, war0, Omega, Nf, zeta, delta, T, Gamma, hi));
        end
        
        if fvalue < ftols
            Vipsol[hi,:] = Vipseed;
            residualarr[hi] = fvalue;
        else     
            t0 = time()
            
            if ws == 4
                res = nlsolve(x -> Keldyshsetup_Floquetn.IbiasResidual_T4(x, war0, Omega, Nf, zeta, delta, T, Gamma, hi), x -> Keldyshsetup_Floquetn.IbiasJacobian_T4(x, war0, Omega, Nf, zeta, delta, T, Gamma, hi), Vipseed, show_trace=true, method = :trust_region, ftol = ftols; xtol = xtols, iterations = itermax);
            elseif ws == 2 
                res = nlsolve(x -> Keldyshsetup_Floquetn.IbiasResidual_T2(x, war0, Omega, Nf, zeta, delta, T, Gamma, hi), x -> Keldyshsetup_Floquetn.IbiasJacobian_T2(x, war0, Omega, Nf, zeta, delta, T, Gamma, hi), Vipseed, show_trace=true, method = :trust_region, ftol = ftols; xtol = xtols, iterations = itermax);
            elseif ws == 0 
                res = nlsolve(x -> Keldyshsetup_Floquetn.IbiasResidual_Tfull(x, war0, Omega, Nf, zeta, delta, T, Gamma, hi), x -> Keldyshsetup_Floquetn.IbiasJacobian_Tfull(x, war0, Omega, Nf, zeta, delta, T, Gamma, hi), Vipseed, show_trace=true, method = :trust_region, ftol = ftols; xtol = xtols, iterations = itermax);
            end
            
            t1 = time()
            println("time = ",t1-t0)
            Vipsol[hi,:] = res.zero;
            residualarr[hi] = res.residual_norm;
    
        end
        
        #find current using solution, verify only DC component present
        VipI = zeros(ComplexF64, 4*Nf+1);
        for kl = 1:2*Nf
            VipI[2*kl] = Vipsol[hi,kl] + im*Vipsol[hi,(2*Nf)+kl];
        end
    
        if ws == 4 
            If[hi,:,:] = Keldyshsetup_Floquetn.current_Floquet_T4(war0, Omega, Nf, zeta, delta, T, Gamma, VipI, hi);
        elseif ws == 2 
            If[hi,:,:] = Keldyshsetup_Floquetn.current_Floquet_T2(war0, Omega, Nf, zeta, delta, T, Gamma, VipI, hi);
        elseif ws == 0 
            If[hi,:,:] = Keldyshsetup_Floquetn.current_Floquet_Tfull(war0, Omega, Nf, zeta, delta, T, Gamma, VipI, hi);
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
    RN_full(Nf, dw0, zeta, delta, T, Gamma) -> Float64

Normal-state resistance `R_N`: computes the Ohmic DC current of the junction with
the gap set to zero (`delta1=0`) at a small reference voltage via
[`current_Floquet_Tfull`](@ref), then `R_N = V/I`. Used to normalize the I–V curves
as `I·eR_N/Δ`.
"""
function RN_full(Nf, dw0, zeta, delta, T, Gamma)
    tau3 = [1 0; 0 -1];
    
    #---RN---
    Omega = maximum([0.1, delta/3]); delta1 = 0;
    
    Nw0 = trunc(Int, abs(Omega)/dw0); 
    war0 = -0.5*abs(Omega) .+ range(0, (Nw0-1)*abs(Omega)/Nw0, Nw0);
    
    VipI = zeros(ComplexF64, 4*Nf+1);
    VipI[2*Nf] = 1; 
    
    If = Keldyshsetup_Floquetn.current_Floquet_Tfull(war0, Omega, Nf, zeta, delta1, T, Gamma, VipI, 0);
    Ifa = zeros(ComplexF64, 4*Nf+1);
    for kl = -Nf:Nf
        for lm = -Nf:Nf
            Ifa[-(kl-lm)+(2*Nf+1)] = Ifa[-(kl-lm)+(2*Nf+1)] + If[-kl+Nf+1,-lm+Nf+1];
        end
    end
    Idc = real(sum(diag(If)));
    GN = (Idc-0)/(Omega-0); RN = 1/GN;

    return RN
end




end