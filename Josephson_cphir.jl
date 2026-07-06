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
dw1 = Gamma/5; wmax = 1*zeta; #Converges rapidly as wmax increases over delta.
Nw1 = 2*ceil(Int, wmax/dw1); war1 = -wmax .+ ((0:Nw1-1) .+ 0.5) .* (2*wmax/Nw1); # even-count midpoint sampling: PH-symmetric, no sample on the T=0 step at w=0

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

