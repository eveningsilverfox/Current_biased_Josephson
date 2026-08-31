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
signed_evar = false;
Nev = 30; evar = delta*range(0.05, 2, Nev);
# signed_evar = true; Nev1 = 30; evar1 = delta*range(0.05, 2, Nev1); evar = [reverse(-evar1); evar1]; Nev = 2*Nev1;

#Lesser self energy

#Scheme
ws = 0; #0:exact, 2:w2, 4:w4

#naming
fnum(x) = x isa Integer ? string(x) : replace(string(round(x, sigdigits=4)), "." => "p");   # numeric value -> filename token ('.' -> 'p')
str1 = "Nf$(Nf)_Vbias_delta$(fnum(delta))_zeta$(fnum(zeta))_T$(fnum(T/zeta))zeta_Gam$(fnum(Gamma))_V$(fnum(first(evar)))_$(fnum(last(evar)))_$(Nev)";
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
    Nw0 = 2*ceil(Int, abs(Omega)/(2*dw0));                             # even cell count: PH-symmetric midpoint sampling
    war0 = -0.5*abs(Omega) .+ ((0:Nw0-1) .+ 0.5) .* (abs(Omega)/Nw0);  # midpoint rule: no sample on the T=0 occupation step

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

# Centred differences, taken branch by branch
dVg = evar[2]-evar[1];
Nbr = signed_evar ? (1:count(<(0), evar), count(<(0), evar)+1:Nev) : (1:Nev,);
dIdv = zeros(Float64, Nev);
for br in Nbr
    a, b = first(br), last(br);
    for hi = a+1:b-1
        dIdv[hi] = (Iv[hi+1]-Iv[hi-1]) / (evar[hi+1]-evar[hi-1]);
    end
    dIdv[a] = (-3*Iv[a] + 4*Iv[a+1] - Iv[a+2]) / (2*dVg);   # one-sided, 2nd order
    dIdv[b] = ( 3*Iv[b] - 4*Iv[b-1] + Iv[b-2]) / (2*dVg);
end


## ------------RN--------------
RN = Keldyshsetup_Floquetn.RN_full(Nf, dw0, zeta, delta, T, Gamma);


## ----------Plots----------
thr_MAR = [2/n for n in 1:6];

evmin = 1;
p2 = plot(evar[evmin:Nev]/delta, Iv[evmin:Nev] * RN, lc=:blue, lw=1.5, framestyle=:box, label="")
vline!(p2, thr_MAR, ls=:dash, lc=:red, lw=0.9, label=L"2\Delta/n")
xlabel!(L"eV/\Delta"); ylabel!(L"I eR_N/\Delta")
plot!(p2, legend=:topleft, legendfontsize=11, tickfontsize=20, guidefontsize=24, size=(680,480))
savefig(plot!(p2, dpi=450), "IV_Vbias_" * str2 * ".png")

evmin = 1;
evmax = Nev;
p2v = plot(evar[evmin:evmax]/delta, dIdv[evmin:evmax] * RN, lc=:blue, lw=1.5, framestyle=:box, label="")
vline!(p2v, thr_MAR, ls=:dash, lc=:red, lw=0.9, label=L"2\Delta/n")
xlabel!(L"eV/\Delta"); ylabel!(L"(dI/dV) eR_N/\Delta")
plot!(p2v, legend=:topleft, legendfontsize=11, tickfontsize=20, guidefontsize=24, size=(680,480))
savefig(plot!(p2v, dpi=450), "dIdV_Vbias_" * str2 * ".png")
