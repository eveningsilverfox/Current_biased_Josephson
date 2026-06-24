include("Keldyshsetup_Floquetn.jl")
using .Keldyshsetup_Floquetn

using MKL
using LinearAlgebra
using Statistics
using Plots
using LaTeXStrings
using NLsolve
using JLD



#energies
mu = 0; delta = 1; zeta = 20; Gamma = 1e-2; 
nT = 100; Tar = zeta .* range(0.02, 1.0, nT);
dw1 = Gamma/5; wmax = 1*zeta; Nw1 = trunc(Int, 2*wmax/dw1); war1 = range(-wmax, wmax, Nw1); #Converges rapidly as wmax increases over delta.

#Phase
Nphi = 50; phiar = 2*pi*range(0.0, 1.0, Nphi);

#naming
fnum(x) = x isa Integer ? string(x) : replace(string(round(x, sigdigits=4)), "." => "p");   # numeric value -> filename token ('.' -> 'p')
str1 = "delta$(fnum(delta))_zeta$(fnum(zeta))_Gam$(fnum(Gamma))";

## ----------Current----------

cphi2Tar = zeros(Float64, nT,Nphi); cphi4Tar = zeros(Float64, nT,Nphi); cphi6Tar = zeros(Float64, nT,Nphi); cphi8Tar = zeros(Float64, nT,Nphi); cphifTar = zeros(Float64, nT,Nphi);
Ic2Tar = zeros(Float64, nT); Ic4Tar = zeros(Float64, nT); Ic6Tar = zeros(Float64, nT); Ic8Tar = zeros(Float64, nT); IcfTar = zeros(Float64, nT); IcRNexTar = zeros(Float64, nT); 
RNTar = zeros(Float64, nT); 
for gh = 1:nT
    println("gh = $(gh)")
    for hi = 1:Nphi
        # cphi2Tar[gh,hi] = Keldyshsetup_Floquetn.currentPhi_eq_T2(war1, zeta, delta, Tar[gh], Gamma, phiar[hi]);
        # cphi4Tar[gh,hi] = Keldyshsetup_Floquetn.currentPhi_eq_T4(war1, zeta, delta, Tar[gh], Gamma, phiar[hi]);
        # cphi6Tar[gh,hi] = Keldyshsetup_Floquetn.currentPhi_eq_T6(war1, zeta, delta, Tar[gh], Gamma, phiar[hi]);
        # cphi8Tar[gh,hi] = Keldyshsetup_Floquetn.currentPhi_eq_T8(war1, zeta, delta, Tar[gh], Gamma, phiar[hi]);
        cphifTar[gh,hi] = Keldyshsetup_Floquetn.currentPhi_eq_Tfull(war1, zeta, delta, Tar[gh], Gamma, phiar[hi]);
    end
    Ic2Tar[gh] = maximum(cphi2Tar);
    Ic4Tar[gh] = maximum(cphi4Tar);
    Ic6Tar[gh] = maximum(cphi6Tar);
    Ic8Tar[gh] = maximum(cphi8Tar);
    IcfTar[gh] = maximum(cphifTar);

    RNTar[gh] = Keldyshsetup_Floquetn.RN_full(22, dw1, zeta, delta, Tar[gh], Gamma);
end


## ----------Plots----------

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

