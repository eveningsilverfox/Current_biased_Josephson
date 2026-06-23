include("Keldyshsetup_Floquetn.jl")
using .Keldyshsetup_Floquetn

using MKL
using LinearAlgebra
using Statistics
using Plots
using LaTeXStrings
using NLsolve
using JLD
using Printf


#size
Nf = 30; #Vv max freq = 2*Nf * (ev), or Nf * (2ev), but only even multiples of eV used/solved for. 

#energies
mu = 0; delta = 1; zeta = 20; T = 0.99*zeta; Gamma = 0.02;
dw0 = minimum([Gamma/5, 0.01]);

#voltage
Nev = 30; evar = delta*range(0.05, 2, Nev);

#Lesser self energy

#Scheme
ws = 0; #0:exact, 2:w2, 4:w4

#naming
str1 = "Nf30_Vbias_delta1_zeta20_T0p2zeta_Gam1e-2_V0p05_3p0_100";
if ws == 8
    str2 = "n_w8_" * str1;
elseif ws == 6
    str2 = "n_w6_" * str1;
elseif ws == 4
    str2 = "n_w4_" * str1;
elseif ws == 2
    str2 = "n_w2_" * str1;
elseif ws == 0
    str2 = "n_" * str1;
end

## ----------Setup----------
VipI = zeros(ComplexF64, 4*Nf+1);
VipI[2*Nf] = 1;

## ----------Current----------
If = zeros(ComplexF64, Nev,2*Nf+1,2*Nf+1); Ifa = zeros(ComplexF64, Nev,4*Nf+1);
Iv = zeros(Float64, Nev);
Iwj = zeros(ComplexF64, Nev);
If1 = zeros(ComplexF64, Nev,4*Nf+1);

for hi = 1:Nev
    println("evct/Nev = $(hi)/$(Nev)")
 
    ev = evar[hi]; Omega = ev;
    Nw0 = trunc(Int, Omega/dw0); 
    # war0 = range(0, (Nw0-1)*Omega/Nw0, Nw0);
    war0 = -0.5*Omega .+ range(0, (Nw0-1)*Omega/Nw0, Nw0);

    if ws == 8 
        If[hi,:,:] = Keldyshsetup_Floquetn.current_Floquet_T8(war0, Omega, Nf, zeta, delta, T, Gamma, VipI, hi);
    elseif ws == 6 
        If[hi,:,:] = Keldyshsetup_Floquetn.current_Floquet_T6(war0, Omega, Nf, zeta, delta, T, Gamma, VipI, hi);
    elseif ws == 4 
        If[hi,:,:] = Keldyshsetup_Floquetn.current_Floquet_T4(war0, Omega, Nf, zeta, delta, T, Gamma, VipI, hi);
    elseif ws == 2 
        If[hi,:,:] = Keldyshsetup_Floquetn.current_Floquet_T2(war0, Omega, Nf, zeta, delta, T, Gamma, VipI, hi);
    elseif ws == 0 
        If[hi,:,:] = Keldyshsetup_Floquetn.current_Floquet_Tfull(war0, Omega, Nf, zeta, delta, T, Gamma, VipI, hi);
    end
    
    for kl = -Nf:Nf
        for lm = -Nf:Nf
            Ifa[hi,-(kl-lm)+(2*Nf+1)] = Ifa[hi,-(kl-lm)+(2*Nf+1)] + If[hi,-kl+Nf+1,-lm+Nf+1];
        end
    end

    Iv[hi] = real(sum(diag(If[hi,:,:])));
    Iwj[hi] = Ifa[hi,-2+(2*Nf+1)];
end

dIdv = zeros(Float64, Nev); 
for hi = 1:Nev-1
    dIdv[hi] = (Iv[hi+1]-Iv[hi]) ./ (evar[2]-evar[1]);
end
dIdv[Nev] = dIdv[Nev-1] + (dIdv[Nev-1]-dIdv[Nev-2])


## ------------RN--------------
RN = Keldyshsetup_Floquetn.RN_full(Nf, dw0, zeta, delta, T, Gamma);


## ----------Plots----------

evmin = 1;
p2 = plot(evar[evmin:Nev]/delta, Iv[evmin:Nev] * RN, lc=:blue, label=L"I", lw=1.5, framestyle = :box,titlefontsize=20)
# p2 = plot(evarf/delta, Ivf, lc=:blue, label=L"I", lw=1.5, framestyle = :box,titlefontsize=20)
vline!([2/1],linestyle=:dash,lc=:red)
vline!([2/2],linestyle=:dash,lc=:red)
vline!([2/3],linestyle=:dash,lc=:red)
vline!([2/4],linestyle=:dash,lc=:red)
vline!([2/5],linestyle=:dash,lc=:red)
vline!([2/6],linestyle=:dash,lc=:red, label="")
xlabel!(L"eV")
ylabel!(L"I(V) R_N")
plot!(titlefontsize=20)
plot!(legend=:none, titlefontsize=20, tickfontsize=17, guidefontsize = 17)
# savefig(p2, "IV_" * str2 * ".png")  

evmin = 1;
evmax = Nev;
p2v = plot(evar[evmin:evmax]/delta, dIdv[evmin:evmax] * RN, label="", lc=:blue, lw=1.5, framestyle = :box,titlefontsize=20, legend=:topleft)
vline!([2/1],linestyle=:dash,lc=:red, label="")
vline!([2/2],linestyle=:dash,lc=:red, label="")
vline!([2/3],linestyle=:dash,lc=:red, label="")
vline!([2/4],linestyle=:dash,lc=:red, label="")
vline!([2/5],linestyle=:dash,lc=:red, label="")
vline!([2/6],linestyle=:dash,lc=:red, label="")
xlabel!(L"eV/\Delta")
ylabel!(L"(dI/dV) RN")
plot!(titlefontsize=20)
plot!(legend=:none, titlefontsize=20, tickfontsize=17, guidefontsize = 17)
# savefig(p2v, "dIdV* str2 * ".png")  
