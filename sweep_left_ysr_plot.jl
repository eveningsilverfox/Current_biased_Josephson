include("Keldyshsetup_Floquetn_ext.jl")
using .Keldyshsetup_Floquetn_ext

using MKL
using LinearAlgebra
using Plots
using LaTeXStrings
using JLD

# ---------------------------------------------------------------------------
# Load the left-lead YSR sweep (left lead JL=[0,0,Jz], KL; right lead clean JR=KR=0)
# produced by Josephson_Ibias_Floquetn_ext.jl, and plot vs voltage:
#   (1) current            I e R_N / Delta
#   (2) conductance        (dI/dV) e R_N / Delta
#   (3) asymmetry (V>0)    dI(V)/dV - dI(-V)/dV
#   (4) overlay            G(+V) vs G(-V)
# Two families of figures: one per KL (overlay the 5 Jz), and one per (Jz,KL) combination
# (a single curve -- so the peaks are cleanly resolved).
# Same axis labels / R_N normalisation as the driver.
# ---------------------------------------------------------------------------

sweepdir = joinpath(@__DIR__, "Plots", "I bias", "Exact", "Single series", "sweep_results")

# --- fixed parameters of the sweep (as in the driver that produced the files) ---
Nf    = 24
delta = 1.0
zeta  = 5.0
T     = 0.6
Gamma = 0.05
dw0   = 0.015                 # = min(0.015, Gamma/2)
JR    = [0.0, 0.0, 0.0]
KR    = 0.0
Nev   = 320                   # total bias points (signed sweep)
Nneg  = 160                   # points with V<0  (equals the number with V>0)
Vmax  = 3.2
Vmin  = 0.24                  # smallest |V| in the grid

# --- the two swept parameters ---
jzlist = [1, 2, 3, 4, 5]      # JL = [0,0,Jz]
klvals = [0.0, 0.1, 0.4]      # KL, numeric value
klstr  = ["0p0", "0p1", "0p4"]  # KL as written in the filenames
nJz = length(jzlist)
nKL = length(klvals)

# --- voltage grid, rebuilt exactly as the driver did (signed: -Vmax..-Vmin , Vmin..Vmax) ---
evar1 = collect(range(Vmin, Vmax, Nneg))         # positive half: 0.24 ... 3.2
evar  = zeros(Float64, Nev)
for k in 1:Nneg
    evar[k]        = -evar1[Nneg + 1 - k]        # negative half: -3.2 ... -0.24
    evar[Nneg + k] =  evar1[k]                   # positive half:  0.24 ...  3.2
end
dV = evar[2] - evar[1]                           # uniform step within each branch

# --- storage ---
Iv   = zeros(Float64, nJz, nKL, Nev)             # DC current vs bias, per run
dIdv = zeros(Float64, nJz, nKL, Nev)             # differential conductance, per run
RN   = zeros(Float64, nJz, nKL)                  # normal-state resistance, per run

# --- load each run, compute R_N and dI/dV ---
for i in 1:nJz
    for j in 1:nKL
        JL = [0.0, 0.0, Float64(jzlist[i])]
        KL = klvals[j]
        fname = "Vipsol_n_Nf24_Ibias_ext_delta1_zeta5_T0p6_Gam0p05_V-3p2_3p2_320_JL0p0-0p0-$(jzlist[i])p0_KL$(klstr[j])_JR0p0-0p0-0p0_KR0p0.jld"
        Iv[i, j, :] = load(joinpath(sweepdir, fname), "Iv")

        RN[i, j] = Keldyshsetup_Floquetn_ext.RN_full(Nf, dw0, zeta, delta, T, Gamma, JL, KL, JR, KR)
        println("Jz=$(jzlist[i]) KL=$KL : RN = $(RN[i, j])")

        # dI/dV: forward differences within each branch, endpoints extrapolated (as in the driver)
        for k in 1:Nneg-1
            dIdv[i, j, k] = (Iv[i, j, k+1] - Iv[i, j, k]) / dV
        end
        dIdv[i, j, Nneg] = dIdv[i, j, Nneg-1] + (dIdv[i, j, Nneg-1] - dIdv[i, j, Nneg-2])
        for k in Nneg+1:Nev-1
            dIdv[i, j, k] = (Iv[i, j, k+1] - Iv[i, j, k]) / dV
        end
        dIdv[i, j, Nev] = dIdv[i, j, Nev-1] + (dIdv[i, j, Nev-1] - dIdv[i, j, Nev-2])
    end
end

# --- y-limits: clip the finite-difference spikes at low |V| (marginally-converged points);
#     read the structure above |eV|/Delta ~ 0.7. Widen these if a real peak is cut off. ---
gylim = (-10.0, 15.0)        # dI/dV and G(+V)/G(-V)
aylim = (-6.0, 6.0)          # conductance asymmetry

neg = 1:Nneg                 # indices with V<0
pos = Nneg+1:Nev             # indices with V>0

# --- make the 4 figures for a set of curves --------------------------------------------
# Curve m is run (is[m], js[m]) drawn in colors[m] with legend labels[m]. `base` is the
# LaTeX-math title (e.g. "K_L=0.4"); `tag` is the filename suffix. Reads the data arrays above.
function make_figs(is, js, colors, labels, base, tag)
    ncurve = length(is)
    leg = ncurve == 1 ? false : :outertopright   # a single-curve figure needs no legend (title says it)
    ttl = latexstring(base)

    # (1) current vs voltage
    pIV = plot(framestyle=:box, size=(680,480), legend=leg, legendfontsize=10,
               tickfontsize=13, guidefontsize=16, title=ttl, titlefontsize=16)
    for n in 1:5
        vline!(pIV, [2/n, -2/n], linestyle=:dash, lc=:gray, alpha=0.35, label="")   # MAR 2Delta/n
    end
    for m in 1:ncurve
        i = is[m]; j = js[m]
        # negative and positive branches as separate curves -> nothing drawn across the V=0 gap
        plot!(pIV, evar[neg]./delta, Iv[i, j, neg].*RN[i, j], lc=colors[m], lw=1.6, label="")
        plot!(pIV, evar[pos]./delta, Iv[i, j, pos].*RN[i, j], lc=colors[m], lw=1.6, label=labels[m])
    end
    xlabel!(pIV, L"eV/\Delta"); ylabel!(pIV, L"IeR_N/\Delta")
    savefig(plot!(pIV, dpi=450), joinpath(sweepdir, "IV_left_ysr_$(tag).png"))

    # (2) dI/dV vs voltage
    pG = plot(framestyle=:box, size=(680,480), legend=leg, legendfontsize=10,
              tickfontsize=13, guidefontsize=16, title=ttl, titlefontsize=16)
    for n in 1:5
        vline!(pG, [2/n, -2/n], linestyle=:dash, lc=:gray, alpha=0.35, label="")
    end
    for m in 1:ncurve
        i = is[m]; j = js[m]
        plot!(pG, evar[neg]./delta, dIdv[i, j, neg].*RN[i, j], lc=colors[m], lw=1.6, label="")
        plot!(pG, evar[pos]./delta, dIdv[i, j, pos].*RN[i, j], lc=colors[m], lw=1.6, label=labels[m])
    end
    xlabel!(pG, L"eV/\Delta"); ylabel!(pG, L"(dI/dV)eR_N/\Delta"); ylims!(pG, gylim)
    savefig(plot!(pG, dpi=450), joinpath(sweepdir, "dIdV_left_ysr_$(tag).png"))

    # (3) conductance asymmetry  dI(V)/dV - dI(-V)/dV  for V>0
    pA = plot(framestyle=:box, size=(680,480), legend=leg, legendfontsize=10,
              tickfontsize=13, guidefontsize=16, title=ttl, titlefontsize=16)
    hline!(pA, [0.0], lc=:gray, alpha=0.5, label="")
    for n in 1:5
        vline!(pA, [2/n], linestyle=:dash, lc=:gray, alpha=0.35, label="")
    end
    for m in 1:ncurve
        i = is[m]; j = js[m]
        dG = zeros(Float64, Nneg)
        for k in 1:Nneg
            dG[k] = (dIdv[i, j, Nneg+k] - dIdv[i, j, Nneg+1-k]) * RN[i, j]   # G(+V) - G(-V)
        end
        plot!(pA, evar1./delta, dG, lc=colors[m], lw=1.6, label=labels[m])
    end
    xlabel!(pA, L"eV/\Delta"); ylabel!(pA, L"[\,dI(V)/dV - dI(-V)/dV\,]\,eR_N/\Delta")
    xlims!(pA, (0, Vmax)); ylims!(pA, aylim)
    savefig(plot!(pA, dpi=450), joinpath(sweepdir, "asym_left_ysr_$(tag).png"))

    # (4) overlay G(+V) (solid) vs G(-V) (dashed)
    pO = plot(framestyle=:box, size=(700,480), legend=leg, legendfontsize=10,
              tickfontsize=13, guidefontsize=16, titlefontsize=12,
              title=latexstring(base * ":\\ \\mathrm{solid}\\ G(+V),\\ \\mathrm{dashed}\\ G(-V)"))
    for n in 1:5
        vline!(pO, [2/n], linestyle=:dash, lc=:gray, alpha=0.35, label="")
    end
    for m in 1:ncurve
        i = is[m]; j = js[m]
        Gp = zeros(Float64, Nneg)
        Gm = zeros(Float64, Nneg)
        for k in 1:Nneg
            Gp[k] = dIdv[i, j, Nneg+k]   * RN[i, j]     # G(+V)
            Gm[k] = dIdv[i, j, Nneg+1-k] * RN[i, j]     # G(-V)
        end
        plot!(pO, evar1./delta, Gp, lc=colors[m], ls=:solid, lw=1.6, label=labels[m])
        plot!(pO, evar1./delta, Gm, lc=colors[m], ls=:dash,  lw=1.6, label="")
    end
    xlabel!(pO, L"|eV|/\Delta"); ylabel!(pO, L"(dI/d|V|)eR_N/\Delta")
    xlims!(pO, (0, Vmax)); ylims!(pO, gylim)
    savefig(plot!(pO, dpi=450), joinpath(sweepdir, "GpGm_left_ysr_$(tag).png"))
end

# colour per Jz for the per-KL overlays; single-run figures use one fixed colour
jzcolors = palette(:viridis, nJz)

# --- family A: one set of figures per KL, overlaying the 5 Jz ---
for j in 1:nKL
    labels = [latexstring("J_z=$(jzlist[i])") for i in 1:nJz]
    make_figs(collect(1:nJz), fill(j, nJz), jzcolors, labels, "K_L=$(klvals[j])", "KL$(klstr[j])")
    println("saved KL=$(klvals[j])")
end

# --- family B: one set of figures per (Jz, KL) combination (single curve -> peaks are clearest) ---
for i in 1:nJz
    for j in 1:nKL
        base = "J_z=$(jzlist[i]),\\ K_L=$(klvals[j])"
        tag  = "Jz$(jzlist[i])_KL$(klstr[j])"
        make_figs([i], [j], [:blue], [""], base, tag)
        println("saved Jz=$(jzlist[i]) KL=$(klvals[j])")
    end
end
