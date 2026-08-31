include("Keldyshsetup_Floquetn_ext.jl")
using .Keldyshsetup_Floquetn_ext

using MKL
using LinearAlgebra
using Statistics
using Plots
using LaTeXStrings
using NLsolve
using JLD
using Printf

# ---------------------------------------------------------------------------
# Voltage-biased single-channel Josephson junction with classical-spin (YSR)
# impurities, 4x4 Nambu(x)spin extension of Josephson_Vbias_Floquetn.jl.
# Uses the exact-Dyson scheme (ws=0) only; the perturbative T2/T4/.. schemes are
# not yet ported to the 4x4 basis.
#
# The phase harmonics `VipI` are IMPOSED (no self-consistency):
#   - pure DC voltage bias:  VipI[2*Nf] = 1   (single harmonic, used below)
#   - microwave (Tien-Gordon) drive V(t)=Vdc+Vac cos(w_mw t): set VipI to the
#     Bessel-function harmonics J_n(Vac/w_mw)  (see the commented hook below).
# ---------------------------------------------------------------------------

#size
Nf = 32;

#energies
mu = 0; delta = 1; zeta = 5; T = 0.001; Gamma = 0.05;
dw0 = minimum([0.015, Gamma/2]);

#classical-spin impurities (units of Delta): J = (Jx,Jy,Jz) exchange, K potential
#  J=K=0  -> non-magnetic (reproduces 2x the original 2x2 result)
#  collinear YSR:      JL=JR=[0,0,Jz]
#  non-collinear/diode: rotate JR relative to JL, e.g. JR=Jz*[sin(th),0,cos(th)]
JL = [0.0, 0.0, 3.0]; KL = 1.0;
JR = [0.0, 0.0, 0.0]; KR = 0.0;

#YSR bound-state energies (in-gap poles of each lead's impurity-dressed surface GF)
EYSR_Ln = Keldyshsetup_Floquetn_ext.ysr_energies_numerical(JL, KL, zeta, delta);
EYSR_Rn = Keldyshsetup_Floquetn_ext.ysr_energies_numerical(JR, KR, zeta, delta);
println("YSR energies E/Δ  | L lead: $(round.(EYSR_Ln./delta, digits=5))")
println("                  | R lead: $(round.(EYSR_Rn./delta, digits=5))")

#voltage
signed_evar = true;
if signed_evar
    Nev1 = 400; evar1 = delta*range(0.24, 3.2, Nev1); evar = [reverse(-evar1); evar1]; Nev = 2*Nev1;
else
    Nev = 400; evar = delta*range(0.24, 3.2, Nev);
end

#Lesser self energy

#Scheme (only ws=0 supported in the 4x4 ext module)
ws = 0;

#naming
fnum(x) = x isa Integer ? string(x) : replace(string(round(x, sigdigits=4)), "." => "p");   # numeric value -> filename token ('.' -> 'p')
fvec(v) = join(fnum.(v), "-");                                                               # vector value -> components joined by '-'
str1 = "Nf$(Nf)_Vbias_ext_delta$(fnum(delta))_zeta$(fnum(zeta))_T$(fnum(T))_Gam$(fnum(Gamma))_V$(fnum(first(evar)))_$(fnum(last(evar)))_$(Nev)_JL$(fvec(JL))_KL$(fnum(KL))_JR$(fvec(JR))_KR$(fnum(KR))";
str2 = "n_" * str1;

## ----------Setup----------
VipI = zeros(ComplexF64, 4*Nf+1);
VipI[2*Nf] = 1;                       # pure DC voltage bias (single harmonic)

# --- Microwave (Tien-Gordon) hook: uncomment and set Vac, n_mw to drive at w_mw=Omega ---
# using SpecialFunctions
# VipI .= 0;
# for n = -Nf:Nf
#     VipI[2*n + 2*Nf+1] = besselj(n, Vac/w_mw);   # phase harmonics of exp(-i phi/2)
# end

## ----------Current----------
If = zeros(ComplexF64, Nev,2*Nf+1,2*Nf+1); Ifa = zeros(ComplexF64, Nev,4*Nf+1);
Iv = zeros(Float64, Nev);

for hi = 1:Nev
    println("evct/Nev = $(hi)/$(Nev)")

    ev = evar[hi]; Omega = ev;
    Nw0 = 2*ceil(Int, abs(Omega)/(2*dw0));                             # even cell count: PH-symmetric midpoint sampling
    war0 = -0.5*abs(Omega) .+ ((0:Nw0-1) .+ 0.5) .* (abs(Omega)/Nw0);  # midpoint rule: no sample on the T=0 occupation step

    If[hi,:,:] = Keldyshsetup_Floquetn_ext.current_Floquet_Tfull(war0, Omega, Nf, zeta, delta, T, Gamma, VipI, JL, KL, JR, KR, hi);

    for kl = -Nf:Nf
        for lm = -Nf:Nf
            Ifa[hi,-(kl-lm)+(2*Nf+1)] = Ifa[hi,-(kl-lm)+(2*Nf+1)] + If[hi,-kl+Nf+1,-lm+Nf+1];
        end
    end

    Iv[hi] = real(sum(diag(If[hi,:,:])));
end

# Centred differences
dVg = evar[2]-evar[1];
dIdv = zeros(Float64, Nev);
for br in (signed_evar ? (1:Nev1, Nev1+1:Nev) : (1:Nev,))
    a, b = first(br), last(br);
    for hi = a+1:b-1
        dIdv[hi] = (Iv[hi+1]-Iv[hi-1]) / (evar[hi+1]-evar[hi-1]);
    end
    dIdv[a] = (-3*Iv[a] + 4*Iv[a+1] - Iv[a+2]) / (2*dVg);   # one-sided, 2nd order
    dIdv[b] = ( 3*Iv[b] - 4*Iv[b-1] + Iv[b-2]) / (2*dVg);
end

## ------------RN--------------
RN = Keldyshsetup_Floquetn_ext.RN_full(Nf, dw0, zeta, delta, T, Gamma, JL, KL, JR, KR);

## ----------Save----------
save("IV_Vbias_" * str2 * ".jld", "evar", collect(evar), "Iv", Iv, "dIdv", dIdv, "RN", RN);

## ----------Threshold guides----------
# Villas et al., PRB 101, 235445 (2020), Fig. 9: with a YSR level at +/-Ey on one
# electrode the subgap thresholds are
#   2D/n      standard MARs (continuum -> continuum)
#   (D+Ey)/n  MARs that start OR end on the YSR level
#   2Ey/n     MARs between YSR states (forbidden in their spin-polarised model)
EyL      = abs(EYSR_Ln[2])/delta;
thr_MAR  = [2/n       for n in 1:3];
thr_YSR1 = [(1+EyL)/n for n in 1:3];
thr_YSR2 = [2*EyL/n   for n in 1:3];

## ----------Plots----------
# YSR energy annotation (numerical, positive in-gap pole, in units of Δ)
ysrLval = round(EYSR_Ln[2]/delta, digits=3); ysrRval = round(EYSR_Rn[2]/delta, digits=3);
ysrttl = latexstring("\\epsilon_{YSR,L}/\\Delta=$(ysrLval),\\quad \\epsilon_{YSR,R}/\\Delta=$(ysrRval)");


if signed_evar
    p2 = plot(evar[1:Nev1]./delta,  (Iv.*RN)[1:Nev1],  lc=:blue, lw=1.5, framestyle=:box, label="")
    plot!(p2, evar[Nev1+1:Nev]./delta, (Iv.*RN)[Nev1+1:Nev], lc=:blue, lw=1.5, label="")
else
    p2 = plot(evar./delta,  (Iv.*RN),  lc=:blue, lw=1.5, framestyle=:box, label="")
end
vline!(p2, thr_MAR,  ls=:dash,    lc=:red,    lw=0.9, label=L"2\Delta/n")
vline!(p2, thr_YSR1, ls=:dot,     lc=:green,  lw=1.1, label=L"(\Delta+\epsilon_{YSR})/n")
vline!(p2, thr_YSR2, ls=:dashdot, lc=:purple, lw=0.9, label=L"2\epsilon_{YSR}/n")
xlabel!(L"eV/\Delta"); ylabel!(L"I eR_N/\Delta")
plot!(p2, legend=:topleft, legendfontsize=16,
      tickfontsize=20, guidefontsize=24, size=(680,480))   # matches sweep_results_Gam0p05 canvas/fonts/margins exactly (no title, no extra top_margin)
savefig(plot!(p2, dpi=450), "IV_Vbias_" * str2 * ".png")

if signed_evar
    p2v = plot(evar[1:Nev1]./delta,  (dIdv.*RN)[1:Nev1],  lc=:blue, lw=1.5, framestyle=:box, label="")
    plot!(p2v, evar[Nev1+1:Nev]./delta, (dIdv.*RN)[Nev1+1:Nev], lc=:blue, lw=1.5, label="")
else
    p2v = plot(evar./delta,  (dIdv.*RN),  lc=:blue, lw=1.5, framestyle=:box, label="")
end
vline!(p2v, thr_MAR,  ls=:dash,    lc=:red,    lw=0.9, label=L"2\Delta/n")
vline!(p2v, thr_YSR1, ls=:dot,     lc=:green,  lw=1.1, label=L"(\Delta+\epsilon_{YSR})/n")
vline!(p2v, thr_YSR2, ls=:dashdot, lc=:purple, lw=0.9, label=L"2\epsilon_{YSR}/n")
xlabel!(L"eV/\Delta"); ylabel!(L"(dI/dV) eR_N/\Delta")
plot!(p2v, legend=:topleft, legendfontsize=16,
      tickfontsize=20, guidefontsize=24, size=(680,480))   # matches sweep_results_Gam0p05 canvas/fonts/margins exactly (no title, no extra top_margin)
savefig(plot!(p2v, dpi=450), "dIdV_Vbias_" * str2 * ".png")

if signed_evar
    Gp = dIdv[Nev1+1:Nev] .* RN;
    Gm = [dIdv[Nev1+1-k] for k in 1:Nev1] .* RN;
    pGpGm = plot(evar1./delta, Gp, lc=:blue, ls=:solid, lw=1.6, framestyle=:box, label="+ve V")
    plot!(pGpGm, evar1./delta, Gm, lc=:blue, ls=:solid, lw=1.6, alpha=0.4, label="-ve V")
    vline!(pGpGm, thr_MAR,  ls=:dash,    lc=:red,    lw=0.9, label=L"2\Delta/n")
    vline!(pGpGm, thr_YSR1, ls=:dot,     lc=:green,  lw=1.1, label=L"(\Delta+\epsilon_{YSR})/n")
    vline!(pGpGm, thr_YSR2, ls=:dashdot, lc=:purple, lw=0.9, label=L"2\epsilon_{YSR}/n")
    xlabel!(pGpGm, L"|eV|/\Delta"); ylabel!(pGpGm, L"(dI/d|V|) eR_N/\Delta")
    xlims!(pGpGm, (0, last(evar1)))
    plot!(pGpGm, legend=:topright, legendfontsize=16,
          tickfontsize=20, guidefontsize=24, size=(700,480))
    savefig(plot!(pGpGm, dpi=450), "GpGm_Vbias_" * str2 * ".png")
end
