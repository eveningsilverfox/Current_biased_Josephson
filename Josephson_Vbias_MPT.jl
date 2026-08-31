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
dw1 = Gamma/5; Nw1 = 2*ceil(Int, zeta/dw1); war1 = -zeta .+ ((0:Nw1-1) .+ 0.5) .* (2*zeta/Nw1); # even-count midpoint sampling: PH-symmetric, no sample on the T=0 step at w=0

#voltage
signed_evar = false;
Nev = 300; evar = 1*range(0.2, 2.4, Nev);
# signed_evar = true; Nev1 = 300; evar1 = 1*range(0.2, 2.4, Nev1); evar = [reverse(-evar1); evar1]; Nev = 2*Nev1;

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
# Centred differences, taken branch by branch (many channels here, so the stencil
# used by the other drivers is factored into a helper rather than written out once
# per array).
dVg = evar[2]-evar[1];
Nbr = signed_evar ? (1:count(<(0), evar), count(<(0), evar)+1:Nev) : (1:Nev,);
function dIdV_branch(Idc)
    d = zeros(Float64, Nev);
    for br in Nbr
        a, b = first(br), last(br);
        for hi = a+1:b-1
            d[hi] = (Idc[hi+1]-Idc[hi-1]) / (evar[hi+1]-evar[hi-1]);
        end
        d[a] = (-3*Idc[a] + 4*Idc[a+1] - Idc[a+2]) / (2*dVg);   # one-sided, 2nd order
        d[b] = ( 3*Idc[b] - 4*Idc[b-1] + Idc[b-2]) / (2*dVg);
    end
    return d;
end
dIdvex       = dIdV_branch(Idcex);
dIdv2        = dIdV_branch(Idc2);
dIdv2_nn     = dIdV_branch(Idc2_nn);
dIdv2_pair   = dIdV_branch(Idc2_pair);
dIdv4        = dIdV_branch(Idc4);
dIdv4_nn     = dIdV_branch(Idc4_nn);
dIdv4ab_1    = dIdV_branch(Idc4ab_1);
dIdv4ab_2    = dIdV_branch(Idc4ab_2);
dIdv4ab_3    = dIdV_branch(Idc4ab_3);
dIdv4_pair   = dIdV_branch(Idc4_pair);
dIdv4ab_pair = dIdV_branch(Idc4ab_pair);
dIdv4ab_nn   = dIdV_branch(Idc4ab_nn);
dIdv6        = dIdV_branch(Idc6);
dIdv6_nn     = dIdV_branch(Idc6_nn);
dIdv6_pair   = dIdV_branch(Idc6_pair);

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
    savefig(plot!(p_final3, dpi=450), "dIdV_MPT_" * str2 * ".png") 

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
    savefig(plot!(p_final2p, dpi=450), "dIdV_pair_MPT_" * str2 * ".png")    

    p2n = plot(evar ./ (2*delta), dIdv2_nn ./ T^2, lw=1.25, lc=:blue, label=L"n=2", framestyle = :box)
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
    p2n1 = plot(evar ./ (2*delta), 1e-2 .* dIdv4_nn ./ T^4, lw=1.25, lc=:blue, label=L"[1,1] (\times 10^{-3})", framestyle = :box)
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
    p_final2n = plot(p2n, p2n1, layout=(1,2), size=(1100,440), right_margin=4Plots.mm, left_margin=8Plots.mm, bottom_margin=8Plots.mm)
    savefig(plot!(p_final2n, dpi=450), "dIdV_qp_MPT_" * str2 * ".png")  


end

