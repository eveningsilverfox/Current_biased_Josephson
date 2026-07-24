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
Nf = 16; #Vv max freq = 2*Nf * (ev), or Nf * (2ev), but only even multiples of eV used/solved for. 

#energies
mu = 0; delta = 1; zeta = 5; T = 0.6; Gamma = 5e-2;
dw0 = minimum([0.015, Gamma/2.0]);

#voltage
signed_evar = false;
if signed_evar
    Nev1 = 90; evar1 = delta*range(0.3, 3.2, Nev1); evar = [reverse(-evar1); evar1]; Nev = 2*Nev1;
else
    Nev = 90; evar = delta*range(0.3, 3.2, Nev);
end

#time
tmax = 100; dt = 2*pi/(Nf*maximum(evar)); Nt0 = trunc(Int, tmax/dt); tar0 = range(0, tmax, Nt0);
  
#Lesser self energy

#Scheme
ws = 0; #0:exact, 2:w2, 4:w4

#naming
fnum(x) = x isa Integer ? string(x) : replace(string(round(x, sigdigits=4)), "." => "p");   # numeric value -> filename token ('.' -> 'p')
str1 = "Nf$(Nf)_delta$(fnum(delta))_zeta$(fnum(zeta))_T$(fnum(T))_Gam$(fnum(Gamma))_V$(fnum(first(evar)))_$(fnum(last(evar)))_$(Nev)";
if  ws == 4
    str2 = "n_w4_" * str1;
elseif ws == 2
    str2 = "n_w2_" * str1;
elseif ws == 0
    str2 = "n_" * str1;
end

## ----------Current----------
# Vipsolseed = load("Vipsol_n_w2_Nf22_delta1_zeta5_T0p625_Gam1e-3_V0p38_3p0_80.jld")["Vipsol"]; Nevseed = 80;
Vipsolseed = nothing; Nevseed = nothing;
if signed_evar
    # Bidirectional solve: split at the sign change and solve each branch
    # independently, each starting from its largest-|V| point (where the simple
    # seed works) so neither continuation has to cross the V=0 jump.
    Nneg = count(<(0), evar);                          # = Nev1
    evarneg = evar[1:Nneg]; evarpos = evar[Nneg+1:end];

    # Negative branch: phisolve sweeps last->first, so pass it reversed to start
    # at the most-negative (largest |V|), then undo the reverse.
    Ivn, Vipn, resn = Keldyshsetup_Floquetn.phisolve(ws, dw0, reverse(evarneg), Nf, zeta, delta, T, Gamma, nothing, nothing);
    Ivn = reverse(Ivn); Vipn = reverse(Vipn, dims=1); resn = reverse(resn);

    # Positive branch: identical to the unsigned run.
    Ivp, Vipp, resp = Keldyshsetup_Floquetn.phisolve(ws, dw0, evarpos, Nf, zeta, delta, T, Gamma, Vipsolseed, Nevseed);

    Iv = [Ivn; Ivp]; Vipsol = [Vipn; Vipp]; residualarr = [resn; resp];
else
    Iv, Vipsol, residualarr = Keldyshsetup_Floquetn.phisolve(ws, dw0, evar, Nf, zeta, delta, T, Gamma, Vipsolseed, Nevseed)
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

## ------------Ic--------------
Nphi  = 50
phiar = 2*pi*range(0.0, 1.0, Nphi)
cphi = Keldyshsetup_Floquetn.currentPhi_eq_Tfull(war1, zeta, delta, T, Gamma, ph);
Ic = maximum(cphi)

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
        Nw0 = 2*ceil(Int, abs(Omega)/(2*dw0));                             # even cell count: PH-symmetric midpoint sampling
        war0 = -0.5*abs(Omega) .+ ((0:Nw0-1) .+ 0.5) .* (abs(Omega)/Nw0);  # midpoint BZ (was [0, Omega): half-mode-shifted cutoffs)
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
evct = 7;
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
plot!(p2, [0.0, 0.0], [-Ic, Ic] .* RN, lc=colors[m], lw=2.2, marker=:hline, markersize=6, label="")
xlabel!(L"eV/\Delta")
ylabel!(L"IeR_N/\Delta")
plot!(legend=:none, titlefontsize=20, tickfontsize=17, guidefontsize = 17, size=(500,400))
savefig(plot!(p2, dpi=450), "IV_Ibias_" * str2 * ".png")

if signed_evar
    p2v = plot(evar[1:Nev1]./delta,  (dIdv.*RN)[1:Nev1],  lc=:blue, lw=1.5, framestyle=:box)
    plot!(p2v, evar[Nev1+1:Nev]./delta, (dIdv.*RN)[Nev1+1:Nev], lc=:blue, lw=1.5)
else
    p2v = plot(evar./delta,  (dIdv.*RN),  lc=:blue, lw=1.5, framestyle=:box)
end
vline!([2/6],linestyle=:dash,lc=:red)
xlabel!(L"eV/\Delta")
ylabel!(L"(dI/dV)eR_N/\Delta")
plot!(legend=:none, titlefontsize=20, tickfontsize=17, guidefontsize = 17, size=(500,400))
savefig(plot!(p2v, dpi=450), "dIdV_Ibias_" * str2 * ".png")

if ws == 2
    if signed_evar
        pm2 = plot(evar[1:Nev1]/delta, Iv2_2[1:Nev1], label=L"I^{(4)}(\phi_2)", lc=:blue, lw=1.5, framestyle = :box)
        plot!(pm2, evar[Nev1+1:Nev]/delta, Iv2_2[Nev1+1:Nev], label="", lc=:blue, lw=1.5)
    else
        pm2 = plot(evar/delta, Iv2_2, label=L"I^{(4)}(\phi_2)", lc=:blue, lw=1.5, framestyle = :box)
    end
    xlabel!(L"eV/\Delta")
    ylabel!(L"I/(e\mathcal{T}^2/\hbar)")
    plot!(titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 17, legendtitlefontsize = 17, legend=:topleft)
    if signed_evar
        pm2p = plot(evar[1:Nev1]/delta, Iv4_2[1:Nev1], label=L"I^{(4)}(\phi_2)", lc=:blue, lw=1.5, framestyle = :box)
        plot!(pm2p, evar[Nev1+1:Nev]/delta, Iv4_2[Nev1+1:Nev], label="", lc=:blue, lw=1.5)
    else
        pm2p = plot(evar/delta, Iv4_2, label=L"I^{(4)}(\phi_2)", lc=:blue, lw=1.5, framestyle = :box)
    end
    xlabel!(L"eV/\Delta")
    plot!(titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 17, legendtitlefontsize = 17, legend=:left)
    p_final2 = plot(pm2, pm2p, layout=(1,2), size=(1050,440), right_margin=4Plots.mm)
    # savefig(p_final2, "Fig4a.png")  
end

