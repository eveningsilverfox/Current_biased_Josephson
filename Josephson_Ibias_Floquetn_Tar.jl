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
mu = 0; delta = 1; zeta = 5; Gamma = 1e-2;
nT = 5; Tar = range(0.05, 3.5, nT);

#voltage
Nev = 110; evar = range(0.38, 3.5, Nev);

#time
tmax = 100; dt = 2*pi/(Nf*maximum(evar)); Nt0 = trunc(Int, tmax/dt); tar0 = range(0, tmax, Nt0);
  
#Lesser self energy
  
#naming
str2 = "n_Tar_Nf30_delta1_zeta5_Gam1e-2_V0p38_3p5_110"
# str2 = "n_Tar5_0p05_3p5_Nf30_delta1_zeta5_Gam1e-2_V0p38_3p5_110"

## ----------Current----------
If = zeros(ComplexF64, nT,Nev,2*Nf+1,2*Nf+1); Ifa = zeros(ComplexF64, nT,Nev,4*Nf+1);
Iv = zeros(Float64, nT,Nev); Ivl = zeros(Float64, nT,Nev); Ivu = zeros(Float64, nT,Nev); dIdv = zeros(Float64, nT,Nev); 
Vipsol = zeros(Float64, nT,Nev,2*(2*Nf)); #(2*Nf) real + (2*Nf) imag #I_mn, Gr_mn, m,n \in -Nf:+Nf, => V_m m \in -2Nf:2Nf. Only odd multiples of eV kept
Vipseed = zeros(Float64, 2*(2*Nf)); #(2*Nf) real + (2*Nf) imag
residualarr = zeros(Float64, nT,Nev);
Vt = zeros(Float64, nT,Nev,Nt0); 

ftols = 5e-13; xtols = 1e-13; itermax = 80;

for gh = 1:nT
    p0m = plot()
    
    for hi = Nev:-1:1
        println("T iter = ",gh)
        println("ev iter = ",hi)
        
        println("ev iter = ",hi)
    
        ev = evar[hi]; Omega = ev;
        dw0 = 0.015; Nw0 = trunc(Int, Omega/dw0); 
        war0 = range(0, Omega-dw0, Nw0);
        
        if hi == Nev && gh == 1
            Vipseed[Nf] = 1;
        elseif hi == Nev && gh > 1
            Vipseed .= Vipsol[gh-1,hi,:];
        else
            Vipseed .= Vipsol[gh,hi+1,:];
        end

        if hi>trunc(Int,0.6*Nev)
            ftols = 5e-17;
        end

        fvalue = norm(Keldyshsetup_Floquetn.IbiasResidual_Tfull(Vipseed, war0, Omega, Nf, zeta, delta, Tar[gh], Gamma, hi));
        if fvalue < ftols
            Vipsol[gh,hi,:] = Vipseed;
            residualarr[gh,hi] = fvalue;
        else     
            t0 = time()
            # res = nlsolve(x -> Keldyshsetup_Floquetn.IbiasResidual_Tfull(x, war0, Omega, Nf, zeta, delta, Tar[gh], Gamma, gh), Vipseed, show_trace=true, ftol = ftols; xtol = xtols, iterations = itermax);
            res = nlsolve(x -> Keldyshsetup_Floquetn.IbiasResidual_Tfull(x, war0, Omega, Nf, zeta, delta, Tar[gh], Gamma, hi), x -> Keldyshsetup_Floquetn.IbiasJacobian_Tfull(x, war0, Omega, Nf, zeta, delta, Tar[gh], Gamma, hi), Vipseed, show_trace=true, method = :trust_region, ftol = ftols; xtol = xtols, iterations = itermax);
            t1 = time()
            println("time = ",t1-t0)
            Vipsol[gh,hi,:] = res.zero;
            residualarr[gh,hi] = res.residual_norm;
        end
        
        #find current using solution, verify only DC component present
        VipI = zeros(ComplexF64, 4*Nf+1);
        for kl = 1:2*Nf
            VipI[2*kl] = Vipsol[hi,kl] + im*Vipsol[hi,(2*Nf)+kl];
        end

        If[gh,hi,:,:] = Keldyshsetup_Floquetn.current_Floquet_Tfull(war0, Omega, Nf, zeta, delta, Tar[gh], Gamma, VipI, hi);
        for kl = -Nf:Nf
            for lm = -Nf:Nf
                Ifa[hi,-(kl-lm)+(2*Nf+1)] = Ifa[gh,hi,-(kl-lm)+(2*Nf+1)] + If[gh,hi,-kl+Nf+1,-lm+Nf+1];
            end
        end

        Iv[gh,hi] = real(sum(diag(If[gh,hi,:,:])));
        
        scatter!([evar[hi]], [Iv[gh,hi]], framestyle = :box, legend = false)
        xlims!(evar[1], evar[Nev])
        ylims!(0, 1.05*Iv[gh,Nev])
        display(p0m)
    end
    plot() 
end

for gh = 1:nT
    for hi = 1:Nev-1
        dIdv[gh,hi] = (Iv[gh,hi+1]-Iv[gh,hi]) ./ (evar[2]-evar[1]);
    end
    dIdv[gh,Nev] = dIdv[gh,Nev-1] + (dIdv[gh,Nev-1]-dIdv[gh,Nev-2])

    Ivl[gh,Nev] = Iv[gh,Nev];
    for hi = Nev-1:-1:1
        if Iv[gh,hi]<Ivl[gh,hi+1]
            Ivl[gh,hi] = Iv[gh,hi];
        else
            Ivl[gh,hi] = Ivl[gh,hi+1];
        end
    end

    Ivu[gh,1] = Iv[gh,1];
    for hi = 2:Nev
        if Iv[gh,hi]>Ivu[gh,hi-1]
            Ivu[gh,hi] = Iv[gh,hi];
        else
            Ivu[gh,hi] = Ivu[gh,hi-1];
        end
    end

    for hi = Nev:-1:1
        Vt[gh,hi,:] .= real.(Keldyshsetup_Floquetn.Vt(Vipsol[gh,hi,:], Nf, tar0, evar[hi]));
    end

end


## ------------Saving----------------
save("Vipsol_" * str2 * ".jld", "Vipsol", Vipsol);
save("IV_Ibias_" * str2 * ".jld", "Iv", Iv);
# Vipsol = load("Vipsol_" * str2 * ".jld")["Vipsol"];
# Iv = load("IV_Ibias_" * str2 * ".jld")["Iv"];

## ----------Plots----------

evct = 7;
p0 = plot(tar0/(2*pi/(2*evar[evct])), Vt[1,evct,:]/(2*pi), label=L"%$(round(Tar[1]/zeta; digits = 2))", framestyle = :box)
for gh = 2:nT
    plot!(tar0/(2*pi/(2*evar[evct])), Vt[gh,evct,:]/(2*pi), label=L"%$(round(Tar[gh]/zeta; digits = 2))", framestyle = :box)
end
xlims!(0,2)
ylims!(0,2+0.2)
xlabel!(L"t/T_0")
ylabel!(L"\phi/(2\pi)")
plot!(legend=:topleft, legendtitle=L"T/\zeta", titlefontsize=22, tickfontsize=20, guidefontsize = 20, legendfontsize = 20, legendtitlefontsize = 20, left_margin=4Plots.mm, bottom_margin=3Plots.mm, right_margin=7Plots.mm, dpi=600)
savefig(plot!(p0, dpi=450), "Vt" * str2 * "_ev$(round(evar[evct]; digits = 3))" * ".png")  

evmin = 1;
# evmax = trunc(Int, 0.75*Nev);
evmax = Nev;
pct1 = 1; pct2 = 2; pct3 = 3; pct4 = 4; pct5 = 5;
p2ab1 = plot(evar[evmin:evmax]/delta, Iv[pct1,evmin:evmax] ./ Tar[pct1]^2, title=L"\mathcal{T}/\Delta=%$(round(Tar[pct1]/delta; digits = 2))", label="", lc=:black, lw=1.5, framestyle = :box,titlefontsize=20, legend=:topleft)
plot!(evar[evmin:evmax]/delta, Ivl[pct1,evmin:evmax] ./ Tar[pct1]^2, label="", lc=:red, lw=1.5, framestyle = :box,titlefontsize=20, legend=:topleft)
plot!(evar[evmin:evmax]/delta, Ivu[pct1,evmin:evmax] ./ Tar[pct1]^2, label="", lc=:blue, lw=1.5, framestyle = :box,titlefontsize=20, legend=:topleft)
vline!([2/1],linestyle=:dash,lc=:red, label="")
vline!([2/2],linestyle=:dash,lc=:red, label="")
vline!([2/3],linestyle=:dash,lc=:red, label="")
vline!([2/4],linestyle=:dash,lc=:red, label="")
vline!([2/5],linestyle=:dash,lc=:red, label="")
ylims!(0.31,1.1)
xlabel!(L"eV/\Delta")
ylabel!(L"I/(e\mathcal{T}^2/\hbar)")
plot!(titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 12, legendtitlefontsize = 13)
p2ab2 = plot(evar[evmin:evmax]/delta, Iv[pct2,evmin:evmax] ./ Tar[pct2]^2, title=L"\mathcal{T}/\zeta=%$(round(Tar[pct2]/zeta; digits = 2))", label="", lc=:black, lw=1.5, framestyle = :box,titlefontsize=20)
plot!(evar[evmin:evmax]/delta, Ivl[pct2,evmin:evmax] ./ Tar[pct2]^2, label="", lc=:red, lw=1.5, framestyle = :box,titlefontsize=20, legend=:topleft)
plot!(evar[evmin:evmax]/delta, Ivu[pct2,evmin:evmax] ./ Tar[pct2]^2, label="", lc=:blue, lw=1.5, framestyle = :box,titlefontsize=20, legend=:topleft)
vline!([2/1],linestyle=:dash,lc=:red, label="")
vline!([2/2],linestyle=:dash,lc=:red, label="")
vline!([2/3],linestyle=:dash,lc=:red, label="")
vline!([2/4],linestyle=:dash,lc=:red, label="")
vline!([2/5],linestyle=:dash,lc=:red, label="")
ylims!(0.31,1.1)
xlabel!(L"eV/\Delta")
plot!(titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 12, legendtitlefontsize = 13)
p2ab3 = plot(evar[evmin:evmax]/delta, Iv[pct3,evmin:evmax] ./ Tar[pct3]^2, title=L"\mathcal{T}/\zeta=%$(round(Tar[pct3]/zeta; digits = 2))", label="", lc=:black, lw=1.5, framestyle = :box,titlefontsize=20)
plot!(evar[evmin:evmax]/delta, Ivl[pct3,evmin:evmax] ./ Tar[pct3]^2, label="", lc=:red, lw=1.5, framestyle = :box,titlefontsize=20, legend=:topleft)
plot!(evar[evmin:evmax]/delta, Ivu[pct3,evmin:evmax] ./ Tar[pct3]^2, label="", lc=:blue, lw=1.5, framestyle = :box,titlefontsize=20, legend=:topleft)
vline!([2/1],linestyle=:dash,lc=:red, label="")
vline!([2/2],linestyle=:dash,lc=:red, label="")
vline!([2/3],linestyle=:dash,lc=:red, label="")
vline!([2/4],linestyle=:dash,lc=:red, label="")
vline!([2/5],linestyle=:dash,lc=:red, label="")
ylims!(0.31,1.1)
xlabel!(L"eV/\Delta")
plot!(titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 12, legendtitlefontsize = 13)
p2ab4 = plot(evar[evmin:evmax]/delta, Iv[pct4,evmin:evmax] ./ Tar[pct4]^2, title=L"\mathcal{T}/\zeta=%$(round(Tar[pct4]/zeta; digits = 2))", label="", lc=:black, lw=1.5, framestyle = :box,titlefontsize=20)
plot!(evar[evmin:evmax]/delta, Ivl[pct4,evmin:evmax] ./ Tar[pct4]^2, label="", lc=:red, lw=1.5, framestyle = :box,titlefontsize=20, legend=:topleft)
plot!(evar[evmin:evmax]/delta, Ivu[pct4,evmin:evmax] ./ Tar[pct4]^2, label="", lc=:blue, lw=1.5, framestyle = :box,titlefontsize=20, legend=:topleft)
vline!([2/1],linestyle=:dash,lc=:red, label="")
vline!([2/2],linestyle=:dash,lc=:red, label="")
vline!([2/3],linestyle=:dash,lc=:red, label="")
vline!([2/4],linestyle=:dash,lc=:red, label="")
vline!([2/5],linestyle=:dash,lc=:red, label="")
ylims!(0.31,1.1)
xlabel!(L"eV/\Delta")
plot!(titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 12, legendtitlefontsize = 13)
p2ab5 = plot(evar[evmin:evmax]/delta, Iv[pct5,evmin:evmax] ./ Tar[pct5]^2, title=L"\mathcal{T}/\zeta=%$(round(Tar[pct5]/zeta; digits = 2))", label="", lc=:black, lw=1.5, framestyle = :box,titlefontsize=20)
plot!(evar[evmin:evmax]/delta, Ivl[pct5,evmin:evmax] ./ Tar[pct5]^2, label="", lc=:red, lw=1.5, framestyle = :box,titlefontsize=20, legend=:topleft)
plot!(evar[evmin:evmax]/delta, Ivu[pct5,evmin:evmax] ./ Tar[pct5]^2, label="", lc=:blue, lw=1.5, framestyle = :box,titlefontsize=20, legend=:topleft)
vline!([2/1],linestyle=:dash,lc=:red, label="")
vline!([2/2],linestyle=:dash,lc=:red, label="")
vline!([2/3],linestyle=:dash,lc=:red, label="")
vline!([2/4],linestyle=:dash,lc=:red, label="")
vline!([2/5],linestyle=:dash,lc=:red, label="")
ylims!(0.31,1.1)
xlabel!(L"eV/\Delta")
plot!(titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 12, legendtitlefontsize = 13)
p_finala = plot(p2ab1, p2ab2, p2ab3, p2ab4, p2ab5, layout=(1,5), size=(2300,400), right_margin=2Plots.mm)
savefig(plot!(p_finala, dpi=450), "IV_lu_Ibias_" * str2 * ".png")  

evmin = 1;
# evmax = trunc(Int, 0.75*Nev);
evmax = Nev;
pct1 = 1; pct2 = 2; pct3 = 3; pct4 = 4; pct5 = 5;
p2b11 = plot(evar[evmin:evmax]/delta, dIdv[pct1,evmin:evmax] ./ Tar[pct1]^2, title=L"\mathcal{T}/\zeta=%$(round(Tar[pct1]/zeta; digits = 2))", label="", lc=:blue, lw=1.5, framestyle = :box,titlefontsize=20, legend=:topleft)
vline!([2/1],linestyle=:dash,lc=:red, label="")
vline!([2/2],linestyle=:dash,lc=:red, label="")
vline!([2/3],linestyle=:dash,lc=:red, label="")
vline!([2/4],linestyle=:dash,lc=:red, label="")
vline!([2/5],linestyle=:dash,lc=:red, label="")
xlabel!(L"eV/\Delta")
ylabel!(L"(dI/dV)/(e\mathcal{T}^2/\hbar)")
plot!(titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 12, legendtitlefontsize = 13)
p2b21 = plot(evar[evmin:evmax]/delta, dIdv[pct2,evmin:evmax] ./ Tar[pct2]^2, title=L"\mathcal{T}/\zeta=%$(round(Tar[pct2]/zeta; digits = 2))", label="", lc=:blue, lw=1.5, framestyle = :box,titlefontsize=20)
vline!([2/1],linestyle=:dash,lc=:red, label="")
vline!([2/2],linestyle=:dash,lc=:red, label="")
vline!([2/3],linestyle=:dash,lc=:red, label="")
vline!([2/4],linestyle=:dash,lc=:red, label="")
vline!([2/5],linestyle=:dash,lc=:red, label="")
xlabel!(L"eV/\Delta")
plot!(titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 12, legendtitlefontsize = 13)
p2b31 = plot(evar[evmin:evmax]/delta, dIdv[pct3,evmin:evmax] ./ Tar[pct3]^2, title=L"\mathcal{T}/\zeta=%$(round(Tar[pct3]/zeta; digits = 2))", label="", lc=:blue, lw=1.5, framestyle = :box,titlefontsize=20)
vline!([2/1],linestyle=:dash,lc=:red, label="")
vline!([2/2],linestyle=:dash,lc=:red, label="")
vline!([2/3],linestyle=:dash,lc=:red, label="")
vline!([2/4],linestyle=:dash,lc=:red, label="")
vline!([2/5],linestyle=:dash,lc=:red, label="")
xlabel!(L"eV/\Delta")
plot!(titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 12, legendtitlefontsize = 13)
p2b41 = plot(evar[evmin:evmax]/delta, dIdv[pct4,evmin:evmax] ./ Tar[pct4]^2, title=L"\mathcal{T}/\zeta=%$(round(Tar[pct4]/zeta; digits = 2))", label="", lc=:blue, lw=1.5, framestyle = :box,titlefontsize=20)
vline!([2/1],linestyle=:dash,lc=:red, label="")
vline!([2/2],linestyle=:dash,lc=:red, label="")
vline!([2/3],linestyle=:dash,lc=:red, label="")
vline!([2/4],linestyle=:dash,lc=:red, label="")
vline!([2/5],linestyle=:dash,lc=:red, label="")
xlabel!(L"eV/\Delta")
plot!(titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 12, legendtitlefontsize = 13)
p2b51 = plot(evar[evmin:evmax]/delta, dIdv[pct5,evmin:evmax] ./ Tar[pct5]^2, title=L"\mathcal{T}/\zeta=%$(round(Tar[pct5]/zeta; digits = 2))", label="", lc=:blue, lw=1.5, framestyle = :box,titlefontsize=20)
vline!([2/1],linestyle=:dash,lc=:red, label="")
vline!([2/2],linestyle=:dash,lc=:red, label="")
vline!([2/3],linestyle=:dash,lc=:red, label="")
vline!([2/4],linestyle=:dash,lc=:red, label="")
vline!([2/5],linestyle=:dash,lc=:red, label="")
xlabel!(L"eV/\Delta")
plot!(titlefontsize=20, tickfontsize=17, guidefontsize = 17, legendfontsize = 12, legendtitlefontsize = 13)
p_final1 = plot(p2b11, p2b21, p2b31, p2b41, p2b51, layout=(1,5), size=(2300,400), right_margin=2Plots.mm)
savefig(plot!(p_finala, dpi=450), "dIdV_Ibias_" * str2 * ".png")  

