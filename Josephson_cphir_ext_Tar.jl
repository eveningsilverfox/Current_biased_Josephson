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
# Transparency-sweep version of Josephson_cphir_ext.jl. 
# ---------------------------------------------------------------------------

#energies
mu = 0; delta = 1; zeta = 5; Gamma = 1e-2;
dw1 = Gamma/5; wmax = 2*zeta; #Converges rapidly as wmax increases over delta.
Nw1 = 2*ceil(Int, wmax/dw1); war1 = -wmax .+ ((0:Nw1-1) .+ 0.5) .* (2*wmax/Nw1); # even-count midpoint sampling: PH-symmetric, no sample on the T=0 step at w=0

#transparency sweep: NT points, log-spaced from deep in the tunnel limit up to 1.0
#(log spacing so the small-T tunnel regime is well resolved; for linear use range(Tmin,1.0,NT))
NT = 2; Tmin = 1e-4; Tmax = 2.5; Tar = 10 .^ range(log10(Tmin), log10(Tmax), NT);

#classical-spin impurities (units of Delta): J = (Jx,Jy,Jz) exchange, K potential
#  J=K=0  -> non-magnetic (reproduces 2x the original 2x2 I(phi))
#  collinear YSR:        JL=JR=[0,0,Jz]
#  non-collinear/diode:  rotate JR vs JL, e.g. JR=Jz*[sin(th),0,cos(th)]
JL = [0.0, 0.0, 5.0]; KL = 1.0;
JR = [0.0, 0.0, 0.0]; KR = 0.0;

#Phase
Nphi = 50; phiar = 2*pi*range(0.0, 1.0, Nphi);

#naming
fnum(x) = x isa Integer ? string(x) : replace(string(round(x, sigdigits=4)), "." => "p");   # numeric value -> filename token ('.' -> 'p')
fvec(v) = join(fnum.(v), "-");                                                               # vector value -> components joined by '-'
str1 = "ext_delta$(fnum(delta))_zeta$(fnum(zeta))_Gam$(fnum(Gamma))_JL$(fvec(JL))_KL$(fnum(KL))_JR$(fvec(JR))_KR$(fnum(KR))";

## ----------Current vs transparency----------
cphiTar = zeros(Float64, NT, Nphi);   # CPR I(phi) at each transparency
IcpTar  = zeros(Float64, NT);         # forward  Ic+
IcmTar  = zeros(Float64, NT);         # backward Ic-
etaTar  = zeros(Float64, NT);         # diode efficiency (0 if reciprocal)
RNTar   = zeros(Float64, NT);         # normal-state resistance
for gh = 1:NT
    T = Tar[gh]
    for hi = 1:Nphi
        cphiTar[gh,hi] = Keldyshsetup_Floquetn_ext.currentPhi_eq_Tfull(war1, zeta, delta, T, Gamma, phiar[hi], JL, KL, JR, KR);
    end
    IcpTar[gh] =  maximum(cphiTar[gh,:]);
    IcmTar[gh] = -minimum(cphiTar[gh,:]);
    etaTar[gh] = (IcpTar[gh] - IcmTar[gh]) / (IcpTar[gh] + IcmTar[gh]);
    RNTar[gh]  = Keldyshsetup_Floquetn_ext.RN_full(22, dw1, zeta, delta, T, Gamma, JL, KL, JR, KR);
    println("T = $(round(T, sigdigits=4)) : Ic+ = $(round(IcpTar[gh], sigdigits=4)), Ic- = $(round(IcmTar[gh], sigdigits=4)), eta = $(round(etaTar[gh], sigdigits=3)), RN = $(round(RNTar[gh], sigdigits=4))");
end


## ----------Plots----------
# (1) critical current x R_N vs transparency; Ic+ and Ic- overlaid (equal => no static diode)
p1 = plot(Tar, IcpTar .* RNTar, lc = "#1f5fb4", lw = 2.4, marker = :circle, ms = 4, label = L"I_c^{+}eR_N/\Delta")
plot!(p1, Tar, IcmTar .* RNTar, lc = :red, lw = 2.4, alpha = 0.45, marker = :diamond, ms = 4, label = L"I_c^{-}eR_N/\Delta")
plot!(p1, framestyle = :box, size = (640, 480),
          legend = :bottomright, legendfontsize = 11,
          xscale = :log10,
          left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
          right_margin = 4Plots.mm, top_margin = 4Plots.mm,
          tickfontsize = 14, guidefontsize = 18,
          grid = true, gridalpha = 0.18, gridstyle = :dot)
xlabel!(p1, L"T")
ylabel!(p1, L"I_c\,eR_N/\Delta")
savefig(plot!(p1, dpi = 450), "IcTar_" * str1 * ".png")

# (2) diode efficiency vs transparency (expected ~0 at every T: no static diode)
p2 = plot(Tar, etaTar, lc = "#1f5fb4", lw = 2.4, marker = :circle, ms = 4)
hline!(p2, [0.0], lc = :gray, lw = 1.0, alpha = 0.6)
plot!(p2, framestyle = :box, size = (640, 480),
          legend = false,
          xscale = :log10,
          left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
          right_margin = 4Plots.mm, top_margin = 4Plots.mm,
          tickfontsize = 14, guidefontsize = 18,
          grid = true, gridalpha = 0.18, gridstyle = :dot)
xlabel!(p2, L"T")
ylabel!(p2, L"\eta")
savefig(plot!(p2, dpi = 450), "etaTar_" * str1 * ".png")

# (3) normalised CPR I e R_N/Delta for a spread of transparencies (dark=tunnel, bright=T~1)
tidx = unique(round.(Int, range(1, NT, 6)));
cols = palette(:viridis, length(tidx));
p3 = plot(phiar ./ pi, cphiTar[tidx[1],:] .* (RNTar[tidx[1]]/delta), lc = cols[1], lw = 2.0, label = L"T=%$(round(Tar[tidx[1]], sigdigits=2))")
for k in 2:length(tidx)
    gh = tidx[k]
    plot!(p3, phiar ./ pi, cphiTar[gh,:] .* (RNTar[gh]/delta), lc = cols[k], lw = 2.0, label = L"T=%$(round(Tar[gh], sigdigits=2))")
end
hline!(p3, [0.0], lc = :gray, lw = 1.0, alpha = 0.6, label = "")
plot!(p3, framestyle = :box, size = (680, 480),
          legend = :outertopright, legendfontsize = 10,
          left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
          right_margin = 4Plots.mm, top_margin = 4Plots.mm,
          tickfontsize = 14, guidefontsize = 18,
          grid = true, gridalpha = 0.18, gridstyle = :dot,
          xlims = (0, 2), xticks = 0:0.5:2)
xlabel!(p3, L"\phi/\pi")
ylabel!(p3, L"I(\phi)\,eR_N/\Delta")
savefig(plot!(p3, dpi = 450), "cprTar_" * str1 * ".png")
