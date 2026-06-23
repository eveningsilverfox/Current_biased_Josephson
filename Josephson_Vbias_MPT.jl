include("Keldyshsetup_Floquetn.jl")
using .Keldyshsetup_Floquetn

using MKL
using LinearAlgebra
using Statistics
using Plots
using LaTeXStrings
using NLsolve
using JLD



#energies
mu = 0; delta = 1; zeta = 25; Gamma = 5e-3; T = 0.4;
dw1 = Gamma/5; Nw1 = trunc(Int, 2*zeta/dw1); war1 = range(-zeta, zeta, Nw1);

#voltage
Nev = 300; evar = 1*range(0.2, 2.4, Nev);

#naming
str2 = "Vbiasdirect_Gamma1em3_delta1_zeta10_V260_0p2_2p1";

#Lesser self energy

War0 = 2*pi .* [1+0.6*im 0 0.3-0.5*im];
War17 = 2*pi .* [0.25 0 0.3 0 0.5 0 1 0 -0.2 0 -0.1 0 -0.08 0 -0.06];  #+7, +6, +5, +4, +3, +2, +1, 0, -1, -2, -3, -4, -5, -6, -7
War13 = 2*pi .* [0.5+0.1*im 0 1-0.05*im 0 -0.15+0.1*im 0 -0.12-0.1*im];  #+3, +2, +1, 0, -1, -2, -3
War35 = 2*pi .* [0.3+0.1*im 0 0.5-0.2*im 0 0 0 0 0 -0.2+0.15*im 0 -0.1-0.08*im];  #+5, +4, +3, +2, +1, 0, -1, -2, -3, -4, -5

Idcex = zeros(Float64, Nev);
Idc2 = zeros(Float64, Nev);
Idc2_nn = zeros(Float64, Nev);
Idc2_pair = zeros(Float64, Nev);
Idc4 = zeros(Float64, Nev);
Idc4_nn = zeros(Float64, Nev);
Idc4ab_1 = zeros(Float64, Nev);
Idc4ab_2 = zeros(Float64, Nev);
Idc4ab_3 = zeros(Float64, Nev);
Idc4_pair = zeros(Float64, Nev);
Idc4ab_nn = zeros(Float64, Nev);
Idc4ab_pair = zeros(Float64, Nev);
Idc6 = zeros(Float64, Nev);
Idc6_nn = zeros(Float64, Nev);
Idc6_pair = zeros(Float64, Nev);

Threads.@threads for hi = 1:Nev
    println("evct/Nev = $(hi)/$(Nev)")

    # Idcex[hi] = Keldyshsetup_Floquetn.current_Vbias_Floquet_Tfull(war1, evar[hi], zeta, 0, T, Gamma);
    
    # Idc2[hi] = Keldyshsetup_Floquetn.current_Vbias_MPT_T2(war1, evar[hi], zeta, delta, T, Gamma, War0);
    # Idc2_nn[hi] = Keldyshsetup_Floquetn.current_Vbias_MPT_T2_qp(war1, evar[hi], zeta, delta, T, Gamma, War0);
    # Idc2_pair[hi] = Keldyshsetup_Floquetn.current_Vbias_MPT_T2_pair(war1, evar[hi], zeta, delta, T, Gamma, War0);
    
    # Idc4[hi] = Keldyshsetup_Floquetn.current_Vbias_MPT_T4(war1, evar[hi], zeta, delta, T, Gamma, War0);
    
    # Idc4_nn[hi] = Keldyshsetup_Floquetn.current_Vbias_MPT_T4_qp(war1, evar[hi], zeta, delta, T, Gamma, War0);
    
    # Idc4ab_1[hi] = Keldyshsetup_Floquetn.current_Vbias_MPT_T4_aobo(war1, evar[hi], zeta, delta, T, Gamma, 1, 3, War13);
    # Idc4ab_2[hi] = Keldyshsetup_Floquetn.current_Vbias_MPT_T4_aobo(war1, evar[hi], zeta, delta, T, Gamma, 1, 7, War17);
    # Idc4ab_3[hi] = Keldyshsetup_Floquetn.current_Vbias_MPT_T4_aobo(war1, evar[hi], zeta, delta, T, Gamma, 3, 5, War35);
    
    # Idc4_pair[hi] = Keldyshsetup_Floquetn.current_Vbias_MPT_T4_pair_aobo(war1, evar[hi], zeta, delta, T, Gamma, 1, 1, War0);
    # Idc4ab_pair[hi] = Keldyshsetup_Floquetn.current_Vbias_MPT_T4_pair_aobo(war1, evar[hi], zeta, delta, T, Gamma, 3, 5, War35);

    Idc4ab_nn[hi] = Keldyshsetup_Floquetn.current_Vbias_MPT_T4_qp_aobo(war1, evar[hi], zeta, delta, T, Gamma, 3, 5, War35);
    
    # Idc6[hi] = Keldyshsetup_Floquetn.current_Vbias_MPT_T6(war1, evar[hi], zeta, delta, T, Gamma, War0);
    # Idc6_nn[hi] = Keldyshsetup_Floquetn.current_Vbias_MPT_T6_qp(war1, evar[hi], zeta, delta, T, Gamma, War0);
    # Idc6_pair[hi] = Keldyshsetup_Floquetn.current_Vbias_MPT_T6_pair(war1, evar[hi], zeta, delta, T, Gamma, War0);
end
dIdvex = zeros(Float64, Nev); 
dIdv2 = zeros(Float64, Nev); 
dIdv2_nn = zeros(Float64, Nev); 
dIdv2_pair = zeros(Float64, Nev); 
dIdv4 = zeros(Float64, Nev); 
dIdv4_nn = zeros(Float64, Nev); 
dIdv4ab1 = zeros(Float64, Nev); 
dIdv4ab_1 = zeros(Float64, Nev); 
dIdv4ab_2 = zeros(Float64, Nev); 
dIdv4ab_3 = zeros(Float64, Nev); 
dIdv4_pair = zeros(Float64, Nev); 
dIdv4ab_pair = zeros(Float64, Nev); 
dIdv4ab_nn = zeros(Float64, Nev);  
dIdv6 = zeros(Float64, Nev); 
dIdv6_nn = zeros(Float64, Nev); 
dIdv6_pair = zeros(Float64, Nev); 
for hi = 1:Nev-1
    dIdv2[hi] = (Idc2[hi+1]-Idc2[hi]) ./ (evar[2]-evar[1]);
    dIdvex[hi] = (Idcex[hi+1]-Idcex[hi]) ./ (evar[2]-evar[1]);
    dIdv2_nn[hi] = (Idc2_nn[hi+1]-Idc2_nn[hi]) ./ (evar[2]-evar[1]);
    dIdv2_pair[hi] = (Idc2_pair[hi+1]-Idc2_pair[hi]) ./ (evar[2]-evar[1]);
    dIdv4[hi] = (Idc4[hi+1]-Idc4[hi]) ./ (evar[2]-evar[1]);
    dIdv4_nn[hi] = (Idc4_nn[hi+1]-Idc4_nn[hi]) ./ (evar[2]-evar[1]);
    dIdv4ab_1[hi] = (Idc4ab_1[hi+1]-Idc4ab_1[hi]) ./ (evar[2]-evar[1]);
    dIdv4ab_2[hi] = (Idc4ab_2[hi+1]-Idc4ab_2[hi]) ./ (evar[2]-evar[1]);
    dIdv4ab_3[hi] = (Idc4ab_3[hi+1]-Idc4ab_3[hi]) ./ (evar[2]-evar[1]);
    dIdv4_pair[hi] = (Idc4_pair[hi+1]-Idc4_pair[hi]) ./ (evar[2]-evar[1]);
    dIdv4ab_pair[hi] = (Idc4ab_pair[hi+1]-Idc4ab_pair[hi]) ./ (evar[2]-evar[1]);
    dIdv4ab_nn[hi] = (Idc4ab_nn[hi+1]-Idc4ab_nn[hi]) ./ (evar[2]-evar[1]);
    dIdv6[hi] = (Idc6[hi+1]-Idc6[hi]) ./ (evar[2]-evar[1]);
    dIdv6_nn[hi] = (Idc6_nn[hi+1]-Idc6_nn[hi]) ./ (evar[2]-evar[1]);
    dIdv6_pair[hi] = (Idc6_pair[hi+1]-Idc6_pair[hi]) ./ (evar[2]-evar[1]);
end
dIdv2[Nev] = dIdv2[Nev-1] + (dIdv2[Nev-1]-dIdv2[Nev-2]);
dIdvex[Nev] = dIdvex[Nev-1] + (dIdvex[Nev-1]-dIdvex[Nev-2]);
dIdv2[Nev] = dIdv2[Nev-1] + (dIdv2[Nev-1]-dIdv2[Nev-2]);
dIdv2_nn[Nev] = dIdv2_nn[Nev-1] + (dIdv2_nn[Nev-1]-dIdv2_nn[Nev-2]);
dIdv2_pair[Nev] = dIdv2_pair[Nev-1] + (dIdv2_pair[Nev-1]-dIdv2_pair[Nev-2]);
dIdv4[Nev] = dIdv4[Nev-1] + (dIdv4[Nev-1]-dIdv4[Nev-2]);
dIdv4_nn[Nev] = dIdv4_nn[Nev-1] + (dIdv4_nn[Nev-1]-dIdv4_nn[Nev-2]);
dIdv4ab_1[Nev] = dIdv4ab_1[Nev-1] + (dIdv4ab_1[Nev-1]-dIdv4ab_1[Nev-2]);
dIdv4ab_2[Nev] = dIdv4ab_2[Nev-1] + (dIdv4ab_2[Nev-1]-dIdv4ab_2[Nev-2]);
dIdv4ab_3[Nev] = dIdv4ab_3[Nev-1] + (dIdv4ab_3[Nev-1]-dIdv4ab_3[Nev-2]);
dIdv4_pair[Nev] = dIdv4_pair[Nev-1] + (dIdv4_pair[Nev-1]-dIdv4_pair[Nev-2]);
dIdv4ab_pair[Nev] = dIdv4ab_pair[Nev-1] + (dIdv4ab_pair[Nev-1]-dIdv4ab_pair[Nev-2]);
dIdv4ab_nn[Nev] = dIdv4ab_nn[Nev-1] + (dIdv4ab_nn[Nev-1]-dIdv4ab_nn[Nev-2]);
dIdv6[Nev] = dIdv6[Nev-1] + (dIdv6[Nev-1]-dIdv6[Nev-2]);
dIdv6_pair[Nev] = dIdv6_pair[Nev-1] + (dIdv6_pair[Nev-1]-dIdv6_pair[Nev-2]);

if delta == 0
    GN = (8*pi/(zeta)^2) #1 -> T. T set to 1.
    p2aa = plot(evar, Idc2NN, lc=:blue, label=L"I^{(2)}/\mathcal{T}^2", framestyle = :box)
    plot!(evar, Idcex2NN, lc=:blue, label=L"I^{(2)ex}/\mathcal{T}^2", framestyle = :box)
    scatter!(evar, evar .* GN, lc=:red, label=L"V/R_N", mc=:red, ms=2, ma=1, framestyle = :box)
    xlabel!(L"eV/\Delta")
    ylabel!(L"I(V)")
    plot!(legend=:topleft, legendfontsize=16, titlefontsize=20, tickfontsize=17, guidefontsize = 17)
    display(p2aa)
end


GN =  (4/pi) .* (T ./ zeta) .^ 2 ./ ( ( 1 .+ (T ./ zeta) .^ 2 ) .^ 2 );
RN = 1 ./ GN;

if delta != 0
    p3 = plot(evar ./ (2*delta), dIdv4 ./ T^4, lw=1.25, lc=:green, label=L"4", framestyle = :box)
    plot!(evar ./ (2*delta), dIdv6 ./ T^6, lw=1.25, lc=:blue, label=L"6", framestyle = :box)
    plot!(evar ./ (2*delta), dIdv2 ./ T^2 .* 10, lw=1.25, lc=:red, label=L"2 (\times 10)", framestyle = :box)
    vline!([1/1],linestyle=:dash,lc=:gray, label="")
    vline!([1/2],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/3],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/4],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/5],linestyle=:dash,lc=:gray, lw=1.5, label="")
    xlims!(evar[1]/2, 2.2/2)
    ylims!(-10, 10)
    xlabel!(L"\Omega/2\Delta")
    ylabel!(L"(dI^{(n)}/dV)/(e\mathcal{T}^n)")
    plot!(titlefontsize=21, tickfontsize=18, guidefontsize = 18,  legendfontsize = 14, legendtitlefontsize = 14, legend=:topleft)
    p31 = plot(evar ./ (2*delta), dIdv4ab_1 ./ T^4, lw=1.25, lc=:red, label=L"[1, 3]", framestyle = :box)
    plot!(evar ./ (2*delta), dIdv4ab_2 ./ T^4, lw=1.25, lc=:green, label=L"4[1, 7]", framestyle = :box)
    plot!(evar ./ (2*delta), dIdv4ab_3 ./ T^4, lw=1.25, lc=:blue, label=L"[3, 5]", framestyle = :box)
    vline!([1/1],linestyle=:dash,lc=:gray, label="")
    vline!([1/2],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/3],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/4],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/5],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/6],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/7],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/8],linestyle=:dash,lc=:gray, lw=1.5, label="")
    xlims!(evar[1]/2, 2.2/2)
    ylims!(-1, 1)
    xlabel!(L"\Omega/2\Delta")
    ylabel!(L"(dI^{(4)}/dV)/(e\mathcal{T}^4)")
    plot!(titlefontsize=21, tickfontsize=18, guidefontsize = 18, legendtitle=L"[a, b]", legendfontsize = 14, legendtitlefontsize = 14, legend=:topright)
    p_final3 = plot(p3, p31, layout=(1,2), size=(1100,440), right_margin=4Plots.mm, left_margin=8Plots.mm, bottom_margin=8Plots.mm)
    display(p_final3)
    # savefig(p_final3, "dIdV_" * str2 * ".png")  
    # savefig(p_final3, "Fig6.png")  

    p2p = plot(evar ./ (2*delta), dIdv2_pair ./ T^2, lw=1.25, lc=:blue, label=L"n=2", framestyle = :box)
    plot!(evar ./ (2*delta), dIdv4_pair ./ T^4, lw=1.25, lc=:red, label=L"n=4", framestyle = :box)
    # plot!(evar ./ (2*delta), dIdv6_pair ./ T^6, lw=1.25, lc=:green, label=L"n=6", framestyle = :box)
    vline!([1/1],linestyle=:dash,lc=:gray, label="")
    vline!([1/2],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/3],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/4],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/5],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/6],linestyle=:dash,lc=:gray, lw=1.5, label="")
    xlims!(evar[1]/delta, evar[Nev]/delta)
    xlims!(evar[1]/2, 2.2/2)
    ylims!(-15,15)
    xlabel!(L"\Omega/2\Delta")
    ylabel!(L"(dI_{\mathrm{pair}}^{(2)}/dV)/(e\mathcal{T}^2)")
    plot!(titlefontsize=21, tickfontsize=18, guidefontsize = 18, legendfontsize = 14, legendtitlefontsize = 14, legend=:topleft)
    p2p1 = plot(evar ./ (2*delta), 1e-3 .* dIdv4_pair ./ T^4, lw=1.25, lc=:blue, label=L"[1,1] (\times 10^{-3})", framestyle = :box)
    plot!(evar ./ (2*delta), 1e1 * dIdv4ab_pair ./ T^4, lw=1.25, lc=:green, label=L"[3,5] (\times 10)", framestyle = :box)
    vline!([1/1],linestyle=:dash,lc=:gray, label="")
    vline!([1/2],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/3],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/4],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/5],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/6],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/7],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/8],linestyle=:dash,lc=:gray, lw=1.5, label="")
    xlims!(evar[1]/delta, evar[Nev]/delta)
    xlims!(evar[1]/2, 2.2/2)
    # ylims!(-1, 0.7)
    xlabel!(L"\Omega/2\Delta")
    ylabel!(L"(dI_{\mathrm{pair}}^{(4)}/dV)/(e\mathcal{T}^4)")
    plot!(titlefontsize=21, tickfontsize=18, guidefontsize = 18, legendtitle=L"[a, b]", legendfontsize = 14, legendtitlefontsize = 14, legend=:bottom)
    p_final2p = plot(p2p, p2p1, layout=(1,2), size=(1100,440), right_margin=4Plots.mm, left_margin=8Plots.mm, bottom_margin=8Plots.mm)
    display(p_final2p)
    # savefig(p_final2p, "I_dIdV_24pair" * str2 * ".png")  
    savefig(p_final2p, "IbiasFig5.png")  

    p2p = plot(evar ./ (2*delta), dIdv2_nn ./ T^2, lw=1.25, lc=:blue, label=L"n=2", framestyle = :box)
    plot!(evar ./ (2*delta), dIdv4_nn ./ T^4, lw=1.25, lc=:red, label=L"n=4", framestyle = :box)
    # plot!(evar ./ (2*delta), dIdv6_pair ./ T^6, lw=1.25, lc=:green, label=L"n=6", framestyle = :box)
    vline!([1/1],linestyle=:dash,lc=:gray, label="")
    vline!([1/2],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/3],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/4],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/5],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/6],linestyle=:dash,lc=:gray, lw=1.5, label="")
    xlims!(evar[1]/delta, evar[Nev]/delta)
    xlims!(0.3/2, 2.2/2)
    # ylims!(-30,45)
    xlabel!(L"\Omega/2\Delta")
    ylabel!(L"(dI_{\mathrm{normal}}^{(2)}/dV)/(e\mathcal{T}^2)")
    plot!(titlefontsize=21, tickfontsize=18, guidefontsize = 18, legendfontsize = 14, legendtitlefontsize = 14, legend=:topleft)
    p2p1 = plot(evar ./ (2*delta), 1e-2 .* dIdv4_nn ./ T^4, lw=1.25, lc=:blue, label=L"[1,1] (\times 10^{-3})", framestyle = :box)
    plot!(evar ./ (2*delta), 1e1 * dIdv4ab_nn ./ T^4, lw=1.25, lc=:green, label=L"[3,5]", framestyle = :box)
    vline!([1/1],linestyle=:dash,lc=:gray, label="")
    vline!([1/2],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/3],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/4],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/5],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/6],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/7],linestyle=:dash,lc=:gray, lw=1.5, label="")
    vline!([1/8],linestyle=:dash,lc=:gray, lw=1.5, label="")
    xlims!(evar[1]/delta, evar[Nev]/delta)
    xlims!(0.3/2, 2.2/2)
    ylims!(-3, 1.0)
    xlabel!(L"\Omega/2\Delta")
    ylabel!(L"(dI_{\mathrm{normal}}^{(4)}/dV)/(e\mathcal{T}^4)")
    plot!(titlefontsize=21, tickfontsize=18, guidefontsize = 18, legendtitle=L"[a, b]", legendfontsize = 14, legendtitlefontsize = 14, legend=:bottom)
    p_final2p = plot(p2p, p2p1, layout=(1,2), size=(1100,440), right_margin=4Plots.mm, left_margin=8Plots.mm, bottom_margin=8Plots.mm)
    display(p_final2p)
    # savefig(p_final2p, "I_dIdV_24pair" * str2 * ".png")  
    savefig(p_final2p, "IbiasFig7.png")  


end

