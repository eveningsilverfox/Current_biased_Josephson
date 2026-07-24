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
# The AC voltage V(t) (phase harmonics W_m) is solved self-consistently by
# demanding that only DC current flows (I_{2h}=0).
# ---------------------------------------------------------------------------

#size (Nf is the max Floquet support. phisolve starts at Nf_start and grows toward Nf as the
#      phase spectrum widens at low |V|. Raise Nf for deeper low-|V| resolution, at rising cost.)
# Nf_req ≈ 1.2/(V·Γ) # Empirically found
# source	V	Γ	Nf	Nf·V·Γ
# Γ-sweep	0.426	0.10	28	1.19
# Γ-sweep	0.426	0.09	30	1.15
# Γ-sweep	0.426	0.08	34	1.16
# V-sweep	0.240	0.10	48	1.15
# V-sweep	0.333	0.10	34	1.13
# V-sweep	0.519	0.10	24	1.25
Nf = 32;

#energies
mu = 0; delta = 1; zeta = 5; T = 0.6; Gamma = 5e-2;
dw0 = minimum([0.015, Gamma/2.0]);

#classical-spin impurities (units of Delta): J=(Jx,Jy,Jz) exchange, K potential
#  J=K=0  -> non-magnetic (reproduces 2x the original 2x2 self-consistent I-V)
JL = [0.0, 0.0, 1.0]; KL = 1.0;
JR = [0.0, 0.0, 0.0]; KR = 0.0;

#YSR bound-state energies (in-gap poles of each lead's impurity-dressed surface GF)
EYSR_La = Keldyshsetup_Floquetn_ext.ysr_energies_analytical(JL, KL, zeta, delta);
EYSR_Ra = Keldyshsetup_Floquetn_ext.ysr_energies_analytical(JR, KR, zeta, delta);
EYSR_Ln = Keldyshsetup_Floquetn_ext.ysr_energies_numerical(JL, KL, zeta, delta);
EYSR_Rn = Keldyshsetup_Floquetn_ext.ysr_energies_numerical(JR, KR, zeta, delta);
println("YSR energies E/Δ  | L lead: analytical=$(round.(EYSR_La./delta, digits=5)) numerical=$(round.(EYSR_Ln./delta, digits=5))")
println("                  | R lead: analytical=$(round.(EYSR_Ra./delta, digits=5)) numerical=$(round.(EYSR_Rn./delta, digits=5))")

#voltage
signed_evar = true;
if signed_evar
    Nev1 = 160; evar1 = delta*range(0.24, 3.2, Nev1); evar = [reverse(-evar1); evar1]; Nev = 2*Nev1;
else
    Nev = 160; evar = delta*range(0.24, 3.2, Nev);
end

#time (for phase reconstruction)
tmax = 100; dt = 2*pi/(Nf*maximum(evar)); Nt0 = trunc(Int, tmax/dt); tar0 = range(0, tmax, Nt0);

#Lesser self energy

#Scheme (only ws=0 supported in the 4x4 ext module)
ws = 0;

#Adaptive Floquet support (phisolve). The phase spectrum widens as |V| falls, so phisolve starts at
#Nf_start at the largest |V| and, sweeping down, begins each point at the previous converged support
#(non-decreasing) and grows Nf in steps of 2 -- up to the ceiling Nf above -- whenever the solve
#misses tol_accept OR the converged spectrum still has weight at the cutoff (edge/peak > edge_tol).
Nf_start = 12;      # Floquet support at the largest |V| (grows toward low |V|)
edge_tol = 1e-3;    # grow Nf until |W| at the cutoff is below this fraction of the peak

#Current-row equilibration (phisolve scale_current). Row-scale the current-nulling equations
#and their Jacobian rows by RN so all equations are O(Delta). Does NOT move the root; changes
#the actual convergence path. 
scale_current = true;

#naming
fnum(x) = x isa Integer ? string(x) : replace(string(round(x, sigdigits=4)), "." => "p");   # numeric value -> filename token ('.' -> 'p')
fvec(v) = join(fnum.(v), "-");                                                               # vector value -> components joined by '-'
str1 = "Nf$(Nf)_Ibias_ext_delta$(fnum(delta))_zeta$(fnum(zeta))_T$(fnum(T))_Gam$(fnum(Gamma))_V$(fnum(first(evar)))_$(fnum(last(evar)))_$(Nev)_JL$(fvec(JL))_KL$(fnum(KL))_JR$(fvec(JR))_KR$(fnum(KR))";
str2 = "n_" * str1;

## ----------Self-consistent solve----------
if signed_evar
    # Bidirectional solve: split at the sign change and solve each branch
    # independently, each starting from its largest-|V| point (where the simple
    # seed works) so neither continuation has to cross the V=0 jump.
    Nneg = count(<(0), evar);                          # = Nev1
    evarneg = evar[1:Nneg]; evarpos = evar[Nneg+1:end];

    # Positive branch first: it carries the hard low-+V stalls, so solving it up front surfaces the
    # diagnosis quickly (identical to the unsigned run).
    Ivp, Vipp, resp = Keldyshsetup_Floquetn_ext.phisolve(ws, dw0, evarpos, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, itermax = 40, Nf_start = Nf_start, edge_tol = edge_tol, scale_current = scale_current);

    # Negative branch: phisolve sweeps last->first, so pass it reversed to start
    # at the most-negative (largest |V|), then undo the reverse.
    Ivn, Vipn, resn = Keldyshsetup_Floquetn_ext.phisolve(ws, dw0, reverse(evarneg), Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, itermax = 40, Nf_start = Nf_start, edge_tol = edge_tol, scale_current = scale_current);
    Ivn = reverse(Ivn); Vipn = reverse(Vipn, dims=1); resn = reverse(resn);

    Iv = [Ivn; Ivp]; Vipsol = [Vipn; Vipp]; residualarr = [resn; resp];
else
    Iv, Vipsol, residualarr = Keldyshsetup_Floquetn_ext.phisolve(ws, dw0, evar, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, itermax = 40, Nf_start = Nf_start, edge_tol = edge_tol, scale_current = scale_current)
end

dIdv = zeros(Float64, Nev);
if signed_evar
    Nneg = count(<(0), evar);
    # dI/dV per branch (uniform step within each), then concatenate.
    for hi = 1:Nneg-1
        dIdv[hi] = (Iv[hi+1]-Iv[hi]) ./ (evar[2]-evar[1]);
    end
    dIdv[Nneg] = dIdv[Nneg-1] + (dIdv[Nneg-1]-dIdv[Nneg-2]);
    for hi = Nneg+1:Nev-1
        dIdv[hi] = (Iv[hi+1]-Iv[hi]) ./ (evar[2]-evar[1]);
    end
    dIdv[Nev] = dIdv[Nev-1] + (dIdv[Nev-1]-dIdv[Nev-2]);
else
    for hi = 1:Nev-1
        dIdv[hi] = (Iv[hi+1]-Iv[hi]) ./ (evar[2]-evar[1]);
    end
    dIdv[Nev] = dIdv[Nev-1] + (dIdv[Nev-1]-dIdv[Nev-2])
end

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
save("Vipsol_" * str2 * ".jld", "Vipsol", Vipsol, "residualarr", residualarr, "Iv", Iv);   # solved phase harmonics + residuals + I-V
# save("IV_Ibias_" * str2 * ".jld", "Iv", Iv);

## ----------Plots----------

# YSR energy annotation (numerical, positive in-gap pole, in units of Δ)
ysrLval = round(EYSR_Ln[2]/delta, digits=3); ysrRval = round(EYSR_Rn[2]/delta, digits=3);
ysrann1 = text(latexstring("\\epsilon_{YSR,L}/\\Delta=$(ysrLval)"), 11, :left);
ysrann2 = text(latexstring("\\epsilon_{YSR,R}/\\Delta=$(ysrRval)"), 11, :left);

evct = 150;
p0 = plot(tar0/(2*pi/(2*evar[evct])), Vt[evct,:]/(2*pi), lc=:blue, lw=1.5, framestyle = :box)
if evar[evct] < 0
    xlims!(-1, 0); ylims!(-(1+0.2), 0)
else
    xlims!(0, 1); ylims!(0, 1+0.2)
end
xlabel!(L"t/(2\pi/\omega_J)")
ylabel!(L"\phi/(2\pi)")
plot!(legend=:none, titlefontsize=20, tickfontsize=17, guidefontsize = 17, size=(500,400),
      right_margin=3Plots.mm, bottom_margin=3Plots.mm)
savefig(plot!(p0, dpi=450), "Vt_Ibias_" * str2 * ".png")

if signed_evar
    p2 = plot(evar[1:Nev1]./delta,  (Iv.*RN)[1:Nev1],  lc=:blue, lw=1.5, framestyle=:box)
    plot!(p2, evar[Nev1+1:Nev]./delta, (Iv.*RN)[Nev1+1:Nev], lc=:blue, lw=1.5)
else
    p2 = plot(evar./delta,  (Iv.*RN),  lc=:blue, lw=1.5, framestyle=:box)
end
xlabel!(L"eV/\Delta"); ylabel!(L"IeR_N/\Delta")
plot!(legend=:none, titlefontsize=20, tickfontsize=17, guidefontsize=17, size=(500,400))
let xlo = minimum(evar/delta), xhi = maximum(evar/delta), ylo = minimum(Iv .* RN), yhi = maximum(Iv .* RN)
    annotate!(p2, xlo + 0.03*(xhi-xlo), yhi - 0.07*(yhi-ylo), ysrann1)
    annotate!(p2, xlo + 0.03*(xhi-xlo), yhi - 0.16*(yhi-ylo), ysrann2)
end
savefig(plot!(p2, dpi=450), "IV_Ibias_" * str2 * ".png")

if signed_evar
    p2v = plot(evar[1:Nev1]./delta,  (dIdv.*RN)[1:Nev1],  lc=:blue, lw=1.5, framestyle=:box)
    plot!(p2v, evar[Nev1+1:Nev]./delta, (dIdv.*RN)[Nev1+1:Nev], lc=:blue, lw=1.5)
else
    p2v = plot(evar./delta,  (dIdv.*RN),  lc=:blue, lw=1.5, framestyle=:box)
end
xlabel!(L"eV/\Delta"); ylabel!(L"(dI/dV)eR_N/\Delta")
plot!(legend=:none, titlefontsize=20, tickfontsize=17, guidefontsize=17, size=(500,400))
let xlo = minimum(evar/delta), xhi = maximum(evar/delta), ylo = minimum(dIdv .* RN), yhi = maximum(dIdv .* RN)
    annotate!(p2v, xlo + 0.03*(xhi-xlo), yhi - 0.07*(yhi-ylo), ysrann1)
    annotate!(p2v, xlo + 0.03*(xhi-xlo), yhi - 0.16*(yhi-ylo), ysrann2)
end
savefig(plot!(p2v, dpi=450), "dIdV_Ibias_" * str2 * ".png")