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
JL = [0.0, 0.0, 4.0]; KL = 0.0;
JR = [0.0, 0.0, 4.0]; KR = 2.0;

#Phase
Nphi = 50; phiar = 2*pi*range(0.0, 1.0, Nphi);

#naming
fnum(x) = x isa Integer ? string(x) : replace(string(round(x, sigdigits=4)), "." => "p");   # numeric value -> filename token ('.' -> 'p')
fvec(v) = join(fnum.(v), "-");                                                               # vector value -> components joined by '-'
str1 = "ext_delta$(fnum(delta))_zeta$(fnum(zeta))_Gam$(fnum(Gamma))_JL$(fvec(JL))_KL$(fnum(KL))_JR$(fvec(JR))_KR$(fnum(KR))";

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
tidx = 1:10:nT;
p1 = plot(framestyle=:box);
for gh in tidx
    plot!(p1, phiar ./ pi, cphifTar[gh,:], lw=1.5, label=L"T/\zeta=%$(round(Tar[gh]/zeta, digits=2))");
end
xlabel!(L"\phi/\pi"); ylabel!(L"I(\phi)")
plot!(legend=:outertopright, titlefontsize=20, tickfontsize=17, guidefontsize=17, size=(650,400))
display(p1)
savefig(plot!(p1, dpi=450), "Iphi_" * str1 * ".png")

p3 = plot(Tar ./ zeta, IcfTar .* RNTar, lc=:gray, lw=1.5, framestyle = :box)
xlabel!(L"T"); ylabel!(L"I_c")
plot!(legend=:none, titlefontsize=20, tickfontsize=17, guidefontsize = 17)
display(p3)
savefig(plot!(p3, dpi=450), "cphi_" * str1 * ".png")

p5 = plot(Tar ./ zeta, RNTar, lc=:blue, lw=1.5, framestyle = :box)
xlabel!(L"T/\zeta"); ylabel!(L"R_N")
plot!(legend=:none, titlefontsize=20, tickfontsize=17, guidefontsize = 20, legend=:topright)
display(p5)
savefig(plot!(p5, dpi=450), "RN_" * str1 * ".png")
