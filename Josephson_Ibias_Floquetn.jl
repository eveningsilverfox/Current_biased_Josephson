include("Keldyshsetup_Floquetn.jl")
using .Keldyshsetup_Floquetn

using MKL
using LinearAlgebra
using Statistics
using Plots
using LaTeXStrings
using NLsolve
using JLD


#size
Nf = 22; #Vv max freq = 2*Nf * (ev), or Nf * (2ev), but only even multiples of eV used/solved for. 

#energies
mu = 0; delta = 1; zeta = 5; T = 0.6; Gamma = 5e-3;
dw0 = minimum([0.015, Gamma/2.0]);

#voltage
Nev = 90; evar = delta*range(0.36, 3.0, Nev);

#time
tmax = 100; dt = 2*pi/(Nf*maximum(evar)); Nt0 = trunc(Int, tmax/dt); tar0 = range(0, tmax, Nt0);
  
#Lesser self energy

#Scheme
ws = 0; #0:exact, 2:w2, 4:w4

#naming
str1 = "Nf22_delta1_zeta5_T0p6_Gam1e-2_V0p36_3p0_90";
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

## ----------Current----------
# Vipsolseed = load("Vipsol_n_w2_Nf22_delta1_zeta5_T0p625_Gam1e-3_V0p38_3p0_80.jld")["Vipsol"]; Nevseed = 80;
Vipsolseed = nothing; Nevseed = nothing;
Iv, Vipsol, residualarr = Keldyshsetup_Floquetn.phisolve(ws, dw0, evar, Nf, zeta, delta, T, Gamma, Vipsolseed, Nevseed)

dIdv = zeros(Float64, Nev); 
for hi = 1:Nev-1
    dIdv[hi] = (Iv[hi+1]-Iv[hi]) ./ (evar[2]-evar[1]);
end
dIdv[Nev] = dIdv[Nev-1] + (dIdv[Nev-1]-dIdv[Nev-2])

# Lower and upper envelop of IV curve
Ivl = zeros(Float64, Nev); Ivu = zeros(Float64, Nev); 
# Lower envelop
Ivl[Nev] = Iv[Nev];
for hi = Nev-1:-1:1
    if Iv[hi]<Ivl[hi+1]
        Ivl[hi] = Iv[hi];
    else
        Ivl[hi] = Ivl[hi+1];
    end
end
# Upper envelop
Ivu[1] = Iv[1];
for hi = 2:Nev
    if Iv[hi]>Ivu[hi-1]
        Ivu[hi] = Iv[hi];
    else
        Ivu[hi] = Ivu[hi-1];
    end
end

Vt = zeros(Float64, Nev,Nt0); 
for hi = Nev:-1:1
    Vt[hi,:] .= real.(Keldyshsetup_Floquetn.Vt(Vipsol[hi,:], Nf, tar0, evar[hi]));
end

Vipsol_complex = zeros(ComplexF64, Nev,4*Nf+1);
for hi = Nev:-1:1
    for kl = 1:2*Nf
        Vipsol_complex[hi,2*kl] = Vipsol[hi,kl] + im*Vipsol[hi,(2*Nf)+kl];
    end
end

## ------------RN--------------
RN = Keldyshsetup_Floquetn.RN_full(Nf, dw0, zeta, delta, T, Gamma);

## ------------Saving----------------
save("Vipsol_" * str2 * ".jld", "Vipsol", Vipsol);
save("IV_Ibias_" * str2 * ".jld", "Iv", Iv);

## ------------Loading----------------
Vipsol = load("Vipsol_" * str2 * ".jld")["Vipsol"];
Iv = load("IV_Ibias_" * str2 * ".jld")["Iv"];

## ------------Werthamer solution to evaluate I^2 and I^4------------
if ws == 2
    Vipsol2 = Vipsol;

    Iv6_2 = zeros(Float64, Nev); dIdv6_2 = zeros(Float64, Nev);
    Iv4_2 = zeros(Float64, Nev); dIdv4_2 = zeros(Float64, Nev);
    Iv2_2 = zeros(Float64, Nev); dIdv2_2 = zeros(Float64, Nev);
    for hi = Nev:-1:1
        println(hi)

        ev = evar[hi]; Omega = ev;
        Nw0 = trunc(Int, Omega/dw0); 
        war0 = range(0, Omega-dw0, Nw0);
        VipI_2 = zeros(ComplexF64, 4*Nf+1);
        for kl = 1:2*Nf
            VipI_2[2*kl] = Vipsol2[hi,kl] + im*Vipsol2[hi,(2*Nf)+kl];
        end

        If6_2 = Keldyshsetup_Floquetn.current_Floquet_T6(war0, Omega, Nf, zeta, delta, T, Gamma, VipI_2, hi);
        If4_2 = Keldyshsetup_Floquetn.current_Floquet_T4(war0, Omega, Nf, zeta, delta, T, Gamma, VipI_2, hi);
        If2_2 = Keldyshsetup_Floquetn.current_Floquet_T2(war0, Omega, Nf, zeta, delta, T, Gamma, VipI_2, hi);
        
        Iv6_2[hi] = real(sum(diag(If6_2)));
        Iv4_2[hi] = real(sum(diag(If4_2)));
        Iv2_2[hi] = real(sum(diag(If2_2)));
    end

    for hi = 1:Nev-1
        dIdv2_2[hi] = (Iv2_2[hi+1]-Iv2_2[hi]) ./ (evar[2]-evar[1]);
        dIdv4_2[hi] = (Iv4_2[hi+1]-Iv4_2[hi]) ./ (evar[2]-evar[1]);
        dIdv6_2[hi] = (Iv6_2[hi+1]-Iv6_2[hi]) ./ (evar[2]-evar[1]);
    end
    dIdv2_2[Nev] = dIdv2_2[Nev-1] + (dIdv2_2[Nev-1]-dIdv2_2[Nev-2]);
    dIdv4_2[Nev] = dIdv4_2[Nev-1] + (dIdv4_2[Nev-1]-dIdv4_2[Nev-2]);
    dIdv6_2[Nev] = dIdv6_2[Nev-1] + (dIdv6_2[Nev-1]-dIdv6_2[Nev-2]);
end



## ----------Plots----------
mlen = 5;
pm4 = heatmap(range(-mlen,+mlen,2*mlen+1), evar,abs.(Vipsol_complex[:,2*Nf+1-mlen:2*Nf+1+mlen]));
xlabel!(L"m [\Omega]")
ylabel!(L"eV/\Delta")
plot!(titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 17, legendtitlefontsize = 17, legend=:topleft)
pm4p = plot(evar,abs.(Vipsol_complex[:,2*Nf+1-1]), label=L"1", framestyle = :box);
plot!(evar,abs.(Vipsol_complex[:,2*Nf+1-3]), label=L"3", framestyle = :box)
plot!(evar,abs.(Vipsol_complex[:,2*Nf+1-5]), label=L"5", framestyle = :box)
plot!(evar,abs.(Vipsol_complex[:,2*Nf+1-7]), label=L"7", framestyle = :box)
ylabel!(L"W_{m}")
xlabel!(L"eV/\Delta")
plot!(legend=:right,titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 17, legendtitlefontsize = 17)
p_finalm4 = plot(pm4, pm4p, layout=(1,2), size=(1100,440), right_margin=4Plots.mm)
display(p_finalm4)
# savefig(pm4, "Vt" * str2 * "_ev$(round(evar[evct]; digits = 3))" * ".png")  

evct = 7;
p0 = plot(tar0/(2*pi/(2*evar[evct])), Vt[evct,:]/(2*pi), framestyle = :box)
xlims!(0,1)
ylims!(0,1+0.2)
xlabel!(L"t/(2\pi/\omega_J)")
ylabel!(L"\phi/(2\pi)")
plot!(titlefontsize=20)
plot!(legend=:topleft, legendtitle=L"T/\zeta", titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 12, legendtitlefontsize = 13)
display(p0)
# savefig(p0, "Vt" * str2 * "_ev$(round(evar[evct]; digits = 3))" * ".png")  

evmin = 1; evmax = 26;
p2 = plot(evar[evmin:Nev]/delta, Iv[evmin:Nev] .* RN, lc=:blue, lw=1.5, framestyle = :box,titlefontsize=20)
# p2 = plot(evarf/delta, Ivf, lc=:blue, label=L"I", lw=1.5, framestyle = :box,titlefontsize=20)
vline!([2/1],linestyle=:dash,lc=:gray, label="")
vline!([2/2],linestyle=:dash,lc=:gray, label="")
vline!([2/3],linestyle=:dash,lc=:gray, label="")
vline!([2/4],linestyle=:dash,lc=:gray, label="")
vline!([2/5],linestyle=:dash,lc=:gray, label="")
xlabel!(L"eV/\Delta")
ylabel!(L"IeR_N/\Delta")
plot!(titlefontsize=20)
plot!(legend=:none, titlefontsize=20, tickfontsize=17, guidefontsize = 17)
savefig(p2, "IV_Ibias_s_" * str2 * ".png")  

evmin = 1;
# evmax = trunc(Int, 0.75*Nev);
evmax = Nev;
p2v = plot(evar[evmin:evmax]/delta, dIdv[evmin:evmax] .* RN, lc=:blue, lw=1.5, framestyle = :box,titlefontsize=20, legend=:topleft)
vline!([2/1],linestyle=:dash,lc=:red, label="")
vline!([2/2],linestyle=:dash,lc=:red, label="")
vline!([2/3],linestyle=:dash,lc=:red, label="")
vline!([2/4],linestyle=:dash,lc=:red, label="")
vline!([2/5],linestyle=:dash,lc=:red, label="")
vline!([2/6],linestyle=:dash,lc=:red)
xlabel!(L"eV/\Delta")
ylabel!(L"(dI/dV)eR_N/\Delta")
plot!(titlefontsize=20)
plot!(legend=:none, titlefontsize=20, tickfontsize=17, guidefontsize = 17)
# savefig(p2v, "dIdV_Ibias1_" * str2 * ".png")  

if ws == 2
    pm2 = plot(evar/delta, Iv2_2, label=L"I^{(4)}(\phi_2)", lc=:blue, lw=1.5, framestyle = :box,titlefontsize=20)
    vline!([2/1],linestyle=:dash,lc=:gray, label="")
    vline!([2/2],linestyle=:dash,lc=:gray, label="")
    vline!([2/3],linestyle=:dash,lc=:gray, label="")
    vline!([2/4],linestyle=:dash,lc=:gray, label="")
    vline!([2/5],linestyle=:dash,lc=:gray, label="")
    xlabel!(L"eV/\Delta")
    ylabel!(L"I/(e\mathcal{T}^2/\hbar)")
    plot!(titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 17, legendtitlefontsize = 17, legend=:topleft)
    pm2p = plot(evar/delta, Iv4_2, label=L"I^{(4)}(\phi_2)", lc=:blue, lw=1.5, framestyle = :box,titlefontsize=20)
    vline!([2/1],linestyle=:dash,lc=:gray, label="")
    vline!([2/2],linestyle=:dash,lc=:gray, label="")
    vline!([2/3],linestyle=:dash,lc=:gray, label="")
    vline!([2/4],linestyle=:dash,lc=:gray, label="")
    vline!([2/5],linestyle=:dash,lc=:gray, label="")
    xlabel!(L"eV/\Delta")
    plot!(titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 17, legendtitlefontsize = 17, legend=:left)
    p_final2 = plot(pm2, pm2p, layout=(1,2), size=(1050,440), right_margin=4Plots.mm)
    # savefig(p_final2, "Fig4a.png")  
end

