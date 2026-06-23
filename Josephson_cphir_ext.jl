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
# Equilibrium current-phase relation I(phi) and critical current Ic vs
# transparency, in the 4x4 Nambu(x)spin basis with classical-spin (YSR)
# impurities (per-lead JL,KL,JR,KR). 4x4 extension of Josephson_cphir.jl.
# The currentPhi_eq family is T2/_T4/_Tfull (all 4x4 YSR-aware).
# With J=K=0 every quantity reduces to 2x the original 2x2 result.
# ---------------------------------------------------------------------------

#energies
mu = 0; delta = 1; zeta = 20; Gamma = 1e-2;
nT = 100; Tar = zeta .* range(0.02, 1.0, nT);
dw1 = Gamma/5; wmax = 1*zeta; Nw1 = trunc(Int, 2*wmax/dw1); war1 = range(-wmax, wmax, Nw1); #Converges rapidly as wmax increases over delta.

#classical-spin impurities (units of Delta): J = (Jx,Jy,Jz) exchange, K potential
#  J=K=0  -> non-magnetic (reproduces 2x the original 2x2 I(phi))
#  collinear YSR:        JL=JR=[0,0,Jz]
#  non-collinear/diode:  rotate JR vs JL, e.g. JR=Jz*[sin(th),0,cos(th)]
JL = [0.0, 0.0, 0.0]; KL = 0.0;
JR = [0.0, 0.0, 0.0]; KR = 0.0;

#Phase
Nphi = 50; phiar = 2*pi*range(0.0, 1.0, Nphi);

#naming
str1 = "ext_delta1_zeta20_Gam1e-2";

## ----------Current----------
cphi2Tar = zeros(Float64, nT,Nphi); cphi4Tar = zeros(Float64, nT,Nphi); cphifTar = zeros(Float64, nT,Nphi);
Ic2Tar = zeros(Float64, nT); Ic4Tar = zeros(Float64, nT); IcfTar = zeros(Float64, nT);
RNTar = zeros(Float64, nT);
for gh = 1:nT
    println("gh = $(gh)")
    for hi = 1:Nphi
        # cphi2Tar[gh,hi] = Keldyshsetup_Floquetn_ext.currentPhi_eq_T2(war1, zeta, delta, Tar[gh], Gamma, phiar[hi], JL, KL, JR, KR);
        # cphi4Tar[gh,hi] = Keldyshsetup_Floquetn_ext.currentPhi_eq_T4(war1, zeta, delta, Tar[gh], Gamma, phiar[hi], JL, KL, JR, KR);
        cphifTar[gh,hi] = Keldyshsetup_Floquetn_ext.currentPhi_eq_Tfull(war1, zeta, delta, Tar[gh], Gamma, phiar[hi], JL, KL, JR, KR);
    end
    Ic2Tar[gh] = maximum(cphi2Tar);
    Ic4Tar[gh] = maximum(cphi4Tar);
    IcfTar[gh] = maximum(cphifTar);

    RNTar[gh] = Keldyshsetup_Floquetn_ext.RN_full(22, dw1, zeta, delta, Tar[gh], Gamma, JL, KL, JR, KR);
end

## ----------Plots----------
p3 = scatter(Tar ./ zeta, 2 .* (Tar ./ zeta) .^ 2, label=L"an. \mathcal{T}^2", mc=:black, ms=6, ma=1, framestyle = :box)
plot!(Tar ./ zeta, IcfTar .* RNTar, label=L"ex. \mathcal{T}^\infty", lc=:gray, lw=1.5, framestyle = :box)
xlabel!(L"T"); ylabel!(L"I_c")
plot!(legend=:bottomleft, legendfontsize=14, titlefontsize=20, tickfontsize=17, guidefontsize = 17)
display(p3)
savefig(plot!(p3, dpi=450), "cphi_" * str1 * ".png")

p5 = plot(Tar ./ zeta, RNTar, label=L"R_N", lc=:blue, lw=1.5, framestyle = :box)
xlabel!(L"T/\zeta"); ylabel!(L"R_N")
plot!(legendfontsize=14, titlefontsize=20, tickfontsize=17, guidefontsize = 20, legend=:topright)
display(p5)
savefig(plot!(p5, dpi=450), "RN_" * str1 * ".png")
