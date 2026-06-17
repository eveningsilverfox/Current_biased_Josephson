include("Keldyshsetup_Floquetn_ext.jl")
using .Keldyshsetup_Floquetn_ext

using MKL
using LinearAlgebra
using Statistics
using Plots
using LaTeXStrings
using NLsolve
using JLD

# ---------------------------------------------------------------------------
# DC current-biased single-channel Josephson junction with classical-spin (YSR)
# impurities -- 4x4 Nambu(x)spin extension of Josephson_Ibias_Floquetn.jl.
# The AC voltage V(t) (phase harmonics W_m) is solved SELF-CONSISTENTLY by
# demanding that only DC current flows (I_{2h}=0). Exact-Dyson scheme (ws=0) only.
# ---------------------------------------------------------------------------

#size
Nf = 22;

#energies
mu = 0; delta = 1; zeta = 5; T = 0.6; Gamma = 1e-2;
dw0 = minimum([0.015, Gamma/2.0]);

#classical-spin impurities (units of Delta): J=(Jx,Jy,Jz) exchange, K potential
#  J=K=0  -> non-magnetic (reproduces 2x the original 2x2 self-consistent I-V)
#  collinear YSR:        JL=JR=[0,0,Jz]
#  non-collinear/diode:  rotate JR vs JL, e.g. JR=Jz*[sin(th),0,cos(th)]
JL = [0.0, 0.0, 4.0]; KL = 0.0;
JR = [0.0, 0.0, 4.0]; KR = 0.0;

#voltage
Nev = 90; evar = delta*range(0.36, 3.2, Nev);

#time (for phase reconstruction)
tmax = 100; dt = 2*pi/(Nf*maximum(evar)); Nt0 = trunc(Int, tmax/dt); tar0 = range(0, tmax, Nt0);

#Lesser self energy
Sigl_s = 1;

#Scheme (only ws=0 supported in the 4x4 ext module)
ws = 0;

#naming
str1 = "Nf22_Ibias_ext_delta1_zeta5_T0p6_Gam1e-2_V0p36_3p0_90";
str2 = "n_" * str1;

## ----------Self-consistent solve----------
Vipsolseed = nothing; Nevseed = nothing;
Iv, Vipsol, residualarr = Keldyshsetup_Floquetn_ext.phisolve(ws, dw0, evar, Nf, zeta, delta, T, Gamma, Sigl_s, JL, KL, JR, KR, Vipsolseed, Nevseed)

dIdv = zeros(Float64, Nev);
for hi = 1:Nev-1
    dIdv[hi] = (Iv[hi+1]-Iv[hi]) ./ (evar[2]-evar[1]);
end
dIdv[Nev] = dIdv[Nev-1] + (dIdv[Nev-1]-dIdv[Nev-2])

# Lower/upper envelopes of the (hysteretic) IV curve
Ivl = zeros(Float64, Nev); Ivu = zeros(Float64, Nev);
Ivl[Nev] = Iv[Nev];
for hi = Nev-1:-1:1
    Ivl[hi] = Iv[hi] < Ivl[hi+1] ? Iv[hi] : Ivl[hi+1];
end
Ivu[1] = Iv[1];
for hi = 2:Nev
    Ivu[hi] = Iv[hi] > Ivu[hi-1] ? Iv[hi] : Ivu[hi-1];
end

# Reconstruct phase V(t) and complex harmonic array
Vt = zeros(Float64, Nev,Nt0);
for hi = Nev:-1:1
    Vt[hi,:] .= real.(Keldyshsetup_Floquetn_ext.Vt(Vipsol[hi,:], Nf, tar0, evar[hi]));
end
Vipsol_complex = zeros(ComplexF64, Nev,4*Nf+1);
for hi = Nev:-1:1
    for kl = 1:2*Nf
        Vipsol_complex[hi,2*kl] = Vipsol[hi,kl] + im*Vipsol[hi,(2*Nf)+kl];
    end
end

## ------------RN--------------
RN = Keldyshsetup_Floquetn_ext.RN_full(Nf, dw0, zeta, delta, T, Gamma, JL, KL, JR, KR);

## ------------Saving----------------
# save("Vipsol_" * str2 * ".jld", "Vipsol", Vipsol);
# save("IV_Ibias_" * str2 * ".jld", "Iv", Iv);

## ----------Plots----------
p2 = plot(evar/delta, Iv .* RN, lc=:blue, lw=1.5, framestyle=:box)
vline!([2/1],linestyle=:dash,lc=:gray, label="")
vline!([2/2],linestyle=:dash,lc=:gray, label="")
vline!([2/3],linestyle=:dash,lc=:gray, label="")
vline!([2/4],linestyle=:dash,lc=:gray, label="")
vline!([2/5],linestyle=:dash,lc=:gray, label="")
xlabel!(L"eV/\Delta"); ylabel!(L"IeR_N/\Delta")
plot!(legend=:none, titlefontsize=20, tickfontsize=17, guidefontsize=17)

p2v = plot(evar/delta, dIdv .* RN, lc=:blue, lw=1.5, framestyle=:box, legend=:topleft)
vline!([2/1],linestyle=:dash,lc=:red, label="")
vline!([2/2],linestyle=:dash,lc=:red, label="")
vline!([2/3],linestyle=:dash,lc=:red, label="")
vline!([2/4],linestyle=:dash,lc=:red, label="")
vline!([2/5],linestyle=:dash,lc=:red, label="")
xlabel!(L"eV/\Delta"); ylabel!(L"(dI/dV)eR_N/\Delta")
plot!(legend=:none, titlefontsize=20, tickfontsize=17, guidefontsize=17)
