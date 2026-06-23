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
Nf = 30;

#energies
mu = 0; delta = 1; zeta = 20; T = 0.1*zeta; Gamma = 0.02;
dw0 = minimum([Gamma/5, 0.01]);

#classical-spin impurities (units of Delta): J = (Jx,Jy,Jz) exchange, K potential
#  J=K=0  -> non-magnetic (reproduces 2x the original 2x2 result)
#  collinear YSR:      JL=JR=[0,0,Jz]
#  non-collinear/diode: rotate JR relative to JL, e.g. JR=Jz*[sin(th),0,cos(th)]
JL = [0.0, 0.0, 0.0]; KL = 0.0;
JR = [0.0, 0.0, 0.0]; KR = 0.0;

#voltage
Nev = 30; evar = delta*range(0.05, 3.25, Nev);

#Lesser self energy

#Scheme (only ws=0 supported in the 4x4 ext module)
ws = 0;

#naming
str1 = "Nf30_Vbias_ext_delta1_zeta20_T0p99zeta_Gam2e-2_V0p05_2p0_30";
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
    Nw0 = trunc(Int, Omega/dw0);
    war0 = -0.5*Omega .+ range(0, (Nw0-1)*Omega/Nw0, Nw0);

    If[hi,:,:] = Keldyshsetup_Floquetn_ext.current_Floquet_Tfull(war0, Omega, Nf, zeta, delta, T, Gamma, VipI, JL, KL, JR, KR, hi);

    for kl = -Nf:Nf
        for lm = -Nf:Nf
            Ifa[hi,-(kl-lm)+(2*Nf+1)] = Ifa[hi,-(kl-lm)+(2*Nf+1)] + If[hi,-kl+Nf+1,-lm+Nf+1];
        end
    end

    Iv[hi] = real(sum(diag(If[hi,:,:])));
end

dIdv = zeros(Float64, Nev);
for hi = 1:Nev-1
    dIdv[hi] = (Iv[hi+1]-Iv[hi]) ./ (evar[2]-evar[1]);
end
dIdv[Nev] = dIdv[Nev-1] + (dIdv[Nev-1]-dIdv[Nev-2])

## ------------RN--------------
RN = Keldyshsetup_Floquetn_ext.RN_full(Nf, dw0, zeta, delta, T, Gamma, JL, KL, JR, KR);

## ----------Plots----------
p2 = plot(evar/delta, Iv * RN, lc=:blue, label=L"I", lw=1.5, framestyle=:box)
vline!([2/1],linestyle=:dash,lc=:red, label="")
vline!([2/2],linestyle=:dash,lc=:red, label="")
vline!([2/3],linestyle=:dash,lc=:red, label="")
vline!([2/4],linestyle=:dash,lc=:red, label="")
vline!([2/5],linestyle=:dash,lc=:red, label="")
xlabel!(L"eV/\Delta"); ylabel!(L"I(V) R_N")
plot!(legend=:none, titlefontsize=20, tickfontsize=17, guidefontsize=17, size=(500,400))
savefig(plot!(p2, dpi=450), "IV_Vbias_" * str2 * ".png")

p2v = plot(evar/delta, dIdv * RN, lc=:blue, label="", lw=1.5, framestyle=:box)
vline!([2/1],linestyle=:dash,lc=:red, label="")
vline!([2/2],linestyle=:dash,lc=:red, label="")
vline!([2/3],linestyle=:dash,lc=:red, label="")
vline!([2/4],linestyle=:dash,lc=:red, label="")
vline!([2/5],linestyle=:dash,lc=:red, label="")
xlabel!(L"eV/\Delta"); ylabel!(L"(dI/dV) R_N")
plot!(legend=:none, titlefontsize=20, tickfontsize=17, guidefontsize=17, size=(500,400))
savefig(plot!(p2v, dpi=450), "dIdV_Vbias_" * str2 * ".png")
