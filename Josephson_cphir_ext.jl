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
# ---------------------------------------------------------------------------

#energies
mu = 0; delta = 1; zeta = 5; Gamma = 5e-2; T = 0.001;
dw1 = Gamma/5; wmax = 1*zeta; #Converges rapidly as wmax increases over delta.
Nw1 = 2*ceil(Int, wmax/dw1); war1 = -wmax .+ ((0:Nw1-1) .+ 0.5) .* (2*wmax/Nw1); # even-count midpoint sampling: PH-symmetric, no sample on the T=0 step at w=0

#classical-spin impurities (units of Delta): J = (Jx,Jy,Jz) exchange, K potential
#  J=K=0  -> non-magnetic (reproduces 2x the original 2x2 I(phi))
#  collinear YSR:        JL=JR=[0,0,Jz]
#  non-collinear/diode:  rotate JR vs JL, e.g. JR=Jz*[sin(th),0,cos(th)]
JL = [0.0, 0.0, 5.0]; KL = 1.0;
JR = [0.0, 0.0, 3.0]; KR = 0.5;

#Phase
Nphi = 50; phiar = 2*pi*range(0.0, 1.0, Nphi);

#naming
fnum(x) = x isa Integer ? string(x) : replace(string(round(x, sigdigits=4)), "." => "p");   # numeric value -> filename token ('.' -> 'p')
fvec(v) = join(fnum.(v), "-");                                                               # vector value -> components joined by '-'
str1 = "ext_delta$(fnum(delta))_zeta$(fnum(zeta))_Gam$(fnum(Gamma))_JL$(fvec(JL))_KL$(fnum(KL))_JR$(fvec(JR))_KR$(fnum(KR))";

## ----------Current----------
cphif = zeros(Float64, Nphi);
for hi = 1:Nphi
    cphif[hi] = Keldyshsetup_Floquetn_ext.currentPhi_eq_Tfull(war1, zeta, delta, T, Gamma, phiar[hi], JL, KL, JR, KR);
end
Icp = maximum(cphif)              # forward  Ic+
Icm = -minimum(cphif)             # backward Ic-
eta = (Icp - Icm) / (Icp + Icm)   # diode efficiency (0 if reciprocal)
println("Ic+ = $Icp,  Ic- = $Icm,  diode efficiency eta = $eta")

RN = Keldyshsetup_Floquetn_ext.RN_full(22, dw1, zeta, delta, T, Gamma, JL, KL, JR, KR);

## ----------Plots----------
Iphi = cphif .* (RN / delta)               # normalised current  I e R_N / Delta
Icpn =  maximum(Iphi)                       # forward  Ic+ (normalised)
Icmn = -minimum(Iphi)                       # backward Ic- (normalised)

p1 = plot(phiar ./ pi, Iphi, lc = "#1f5fb4", lw = 2.6)
hline!(p1, [0.0],   lc = :gray,     lw = 1.0, alpha = 0.6)                # zero reference
hline!(p1, [ Icpn], lc = :crimson,  lw = 1.2, ls = :dash, alpha = 0.7)    # forward  Ic+
hline!(p1, [-Icmn], lc = :seagreen, lw = 1.2, ls = :dash, alpha = 0.7)    # backward Ic- (at min I)
annotate!(p1, 0.06,  Icpn, text(L"I_c^{+}", 13, :crimson,  :left, :bottom))
annotate!(p1, 0.06, -Icmn, text(L"I_c^{-}", 13, :seagreen, :left, :top))
plot!(p1, framestyle = :box, size = (640, 480),
          legend = false,
          title = latexstring("\\eta_{\\mathrm{diode}} = $(round(eta, digits=4))"), titlefontsize = 16,
          left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
          right_margin = 4Plots.mm, top_margin = 4Plots.mm,
          tickfontsize = 14, guidefontsize = 18,
          grid = true, gridalpha = 0.18, gridstyle = :dot,
          xlims = (0, 2), xticks = 0:0.5:2)
xlabel!(p1, L"\phi/\pi")
ylabel!(p1, L"I(\phi)\,eR_N/\Delta")
savefig(plot!(p1, dpi = 450), "Iphi_" * str1 * ".png")
