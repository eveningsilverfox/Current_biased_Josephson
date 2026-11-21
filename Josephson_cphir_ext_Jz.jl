include("Keldyshsetup_Floquetn_ext.jl")
using .Keldyshsetup_Floquetn_ext

using MKL
using LinearAlgebra
using Statistics
using Plots
using LaTeXStrings
using NLsolve
using JLD


#energies
mu = 0; delta = 1; zeta = 5; Gamma = 5e-2;
dw1 = Gamma/5; wmax = 2*zeta; #Converges rapidly as wmax increases over delta.
Nw1 = 2*ceil(Int, wmax/dw1); war1 = -wmax .+ ((0:Nw1-1) .+ 0.5) .* (2*wmax/Nw1); # even-count midpoint sampling: PH-symmetric, no sample on the T=0 step at w=0

#transparency: fixed (transparency set by T/zeta)
T = 1.0;

#exchange sweep: JL = Jz * nL, NJ points linear in Jz
#(the YSR zero-energy crossing is near Jz ~ 5.5 for KL = 0-1; refine the range around it if needed)
NJ = 200; Jzmin = 0.0; Jzmax = 8.0; Jzar = range(Jzmin, Jzmax, NJ);

#classical-spin impurities (units of Delta): J = (Jx,Jy,Jz) exchange, K potential
#  J=K=0  -> non-magnetic (reproduces 2x the original 2x2 I(phi))
#  collinear YSR:        nL = [0,0,1]
#  non-collinear/diode:  rotate JR vs JL, e.g. JR=Jz*[sin(th),0,cos(th)]
nL = [0.0, 0.0, 1.0]; KL = 1.0;     # direction of JL (swept magnitude Jz), potential scattering
JR = [0.0, 0.0, 0.0]; KR = 0.0;

#Phase
Nphi = 50; phiar = 2*pi*range(0.0, 1.0, Nphi);

# --- Floquet-machinery CPR (currentPhi_eq_Floquet_Tfull) discretisation knobs ---
Omega_cpr = 0.4
# The energy window (Nf_cpr+0.5)*Omega_cpr must cover the band edge sqrt(4 zeta^2 + delta^2) AND, once an on-site
# potential V = |J| + |K| exceeds zeta, the impurity split-off state outside the band at |E| ~ V + zeta^2/V
Vimp_max = max(maximum(abs.(Jzar)) + abs(KL), norm(JR) + abs(KR))           # largest on-site potential on either lead
Emax = Vimp_max > zeta ? max(sqrt(4*zeta^2 + delta^2), Vimp_max + zeta^2/Vimp_max) : sqrt(4*zeta^2 + delta^2)
Nf_cpr = ceil(Int, (Emax + 1.0)/Omega_cpr)                                  # +1.0 margin (Gamma tails, delta shift); 32 for Jz <= 8, KL = 1
dw_cpr = minimum([0.015, Gamma/2.0])

#naming
fnum(x) = x isa Integer ? string(x) : replace(string(round(x, sigdigits=4)), "." => "p");   # numeric value -> filename token ('.' -> 'p')
fvec(v) = join(fnum.(v), "-");                                                               # vector value -> components joined by '-'
str1 = "ext_delta$(fnum(delta))_zeta$(fnum(zeta))_T$(fnum(T))_Gam$(fnum(Gamma))_nL$(fvec(nL))_Jz$(fnum(Jzmin))_$(fnum(Jzmax))_$(NJ)_KL$(fnum(KL))_JR$(fvec(JR))_KR$(fnum(KR))";

## ----------Current vs Jz----------
cphiJz = zeros(Float64, NJ, Nphi);   # CPR I(phi) at each Jz
IcpJz  = zeros(Float64, NJ);         # forward  Ic+
IcmJz  = zeros(Float64, NJ);         # backward Ic-
RNJz   = zeros(Float64, NJ);         # normal-state resistance
EYSRJz = zeros(Float64, NJ);         # left-lead YSR energy (positive in-gap pole)
for gh = 1:NJ
    JL = Jzar[gh] .* nL
    for hi = 1:Nphi
        # cphiJz[gh,hi] = Keldyshsetup_Floquetn_ext.currentPhi_eq_Tfull(war1, zeta, delta, T, Gamma, phiar[hi], JL, KL, JR, KR);
        cphiJz[gh,hi] = Keldyshsetup_Floquetn_ext.currentPhi_eq_Floquet_Tfull(Omega_cpr, Nf_cpr, dw_cpr, zeta, delta, T, Gamma, phiar[hi], JL, KL, JR, KR);
    end
    IcpJz[gh] =  maximum(cphiJz[gh,:]);
    IcmJz[gh] = -minimum(cphiJz[gh,:]);
    RNJz[gh]  = Keldyshsetup_Floquetn_ext.RN_full(22, dw1, zeta, delta, T, Gamma, JL, KL, JR, KR);
    EYSRJz[gh] = Keldyshsetup_Floquetn_ext.ysr_energies_numerical(JL, KL, zeta, delta)[2];
    println("Jz = $(round(Jzar[gh], sigdigits=4)) : eps = $(round(EYSRJz[gh], sigdigits=4)), Ic+ = $(round(IcpJz[gh], sigdigits=4)), Ic- = $(round(IcmJz[gh], sigdigits=4)), RN = $(round(RNJz[gh], sigdigits=4))");
end

# I(phi) = sum_n [ A_n sin(n phi) + B_n cos(n phi) ].
# A1 changes sign (0 -> pi-like) just past the YSR zero-energy crossing; A2 is what survives there.
dphi = 2*pi/(Nphi-1); pint = 1:Nphi-1;
A1Jz = zeros(Float64, NJ);   # sin(phi)  amplitude
B1Jz = zeros(Float64, NJ);   # cos(phi)  amplitude
A2Jz = zeros(Float64, NJ);   # sin(2phi) amplitude
for gh = 1:NJ
    A1Jz[gh] = (dphi/pi) * sum(cphiJz[gh,pint] .* sin.(phiar[pint]));
    B1Jz[gh] = (dphi/pi) * sum(cphiJz[gh,pint] .* cos.(phiar[pint]));
    A2Jz[gh] = (dphi/pi) * sum(cphiJz[gh,pint] .* sin.(2 .* phiar[pint]));
    println("Jz = $(round(Jzar[gh], sigdigits=4)) : A1 = $(round(A1Jz[gh], sigdigits=4)), A2 = $(round(A2Jz[gh], sigdigits=4)), B1 = $(round(B1Jz[gh], sigdigits=4)), |B1/A1| = $(round(abs(B1Jz[gh]/A1Jz[gh]), sigdigits=3)), phi0 = $(round(atan(B1Jz[gh], A1Jz[gh])/pi, sigdigits=3))pi");
end

## ----------Save----------
save("cphirJz_" * str1 * ".jld", "phiar", collect(phiar), "Jzar", collect(Jzar), "T", T, "cphiJz", cphiJz,
     "IcpJz", IcpJz, "IcmJz", IcmJz, "RNJz", RNJz, "EYSRJz", EYSRJz, "A1Jz", A1Jz, "B1Jz", B1Jz, "A2Jz", A2Jz);

## ----------Plots----------
jidx = unique(round.(Int, range(1, NJ, 6)));
cols = palette(:viridis, length(jidx));
p3 = plot(phiar ./ pi, cphiJz[jidx[1],:] .* (RNJz[jidx[1]]/delta), lc = cols[1], lw = 2.0, label = L"J_z=%$(round(Jzar[jidx[1]], sigdigits=3))")
for k in 2:length(jidx)
    gh = jidx[k]
    plot!(p3, phiar ./ pi, cphiJz[gh,:] .* (RNJz[gh]/delta), lc = cols[k], lw = 2.0, label = L"J_z=%$(round(Jzar[gh], sigdigits=3))")
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
savefig(plot!(p3, dpi = 450), "cprJz_" * str1 * ".png")

# signed harmonics so the sign change of A1 at the crossing is visible
# cos(n phi) curves are not plotted: for this collinear setup I(phi) is exactly odd (B1, B2 ~ 1e-16, machine zero).
# A1 zero at 5.6645 / 5.6785 / 5.8090 for T = 0.6 / 1.0 / 2.0
p4 = plot(Jzar, A1Jz .* (RNJz ./ delta), lc = "#1f5fb4", lw = 2.4, label = L"A_1eR_N/\Delta\ \ (\sin\phi)")
plot!(p4, Jzar, A2Jz .* (RNJz ./ delta), lc = :darkorange, lw = 2.4, label = L"A_2eR_N/\Delta\ \ (\sin2\phi)")
hline!(p4, [0.0], lc = :gray, lw = 1.0, alpha = 0.6, label = "")
plot!(p4, framestyle = :box, size = (640, 480),
          legend = :bottomleft, legendfontsize = 11,
          left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
          right_margin = 4Plots.mm, top_margin = 4Plots.mm,
          tickfontsize = 14, guidefontsize = 18,
          grid = true, gridalpha = 0.18, gridstyle = :dot)
xlabel!(p4, L"J_z/\Delta")
ylabel!(p4, L"I\,eR_N/\Delta")
savefig(plot!(p4, dpi = 450), "cprharmJz_" * str1 * ".png")

p5 = plot(Jzar, EYSRJz ./ delta, lc = "#1f5fb4", lw = 2.4, legend = false)
hline!(p5, [0.0], lc = :gray, lw = 1.0, alpha = 0.6)
plot!(p5, framestyle = :box, size = (640, 480),
          left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
          right_margin = 4Plots.mm, top_margin = 4Plots.mm,
          tickfontsize = 14, guidefontsize = 18,
          grid = true, gridalpha = 0.18, gridstyle = :dot)
xlabel!(p5, L"J_z/\Delta")
ylabel!(p5, L"\epsilon_{YSR,L}/\Delta")
savefig(plot!(p5, dpi = 450), "eysrJz_" * str1 * ".png")
