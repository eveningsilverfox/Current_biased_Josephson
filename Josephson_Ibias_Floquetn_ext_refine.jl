include("Keldyshsetup_Floquetn_ext.jl")
using .Keldyshsetup_Floquetn_ext

using MKL
using LinearAlgebra
using NLsolve
using JLD

# ---------------------------------------------------------------------------
# Low-bias refinement of a finished run of Josephson_Ibias_Floquetn_ext_env.jl (signed grid).
#
# The old grid has spacing dV. Every old interval [V_k, V_k+1] that starts below |V| = Vmax gets
# R-1 new points inside it (new spacing dV/R), on both branches. Each new point is solved on its
# own, starting from the linear interpolation of the two old solutions next to it (warm start
# from the old .jld file). If that fails, it tries the old solution just above, then just below.
#
# The points do not depend on each other, so they are split into NB batches. Run NB copies of
# this script at the same time (CBJJ_BATCH = 1, ..., NB), each with its own julia threads: they
# share nothing and each writes its own file, so there are no thread races.
# Afterwards, Josephson_Ibias_Floquetn_ext_refine_merge.jl joins the batches with the old run.
# See job_scripts/job_Ibias_ext_refine.sh.
# ---------------------------------------------------------------------------

## ----------Parameters (same as Josephson_Ibias_Floquetn_ext_env.jl)----------
Nf = parse(Int, get(ENV, "CBJJ_NF", "32"));
mu = 0; delta = 1; zeta = 5; T = parse(Float64, get(ENV, "CBJJ_T", "1.0")); Gamma = 5e-2;
dw0 = minimum([0.015, Gamma/2.0]);
JL = [0.0, 0.0, parse(Float64, get(ENV, "CBJJ_JZ", "5.0"))]; KL = parse(Float64, get(ENV, "CBJJ_KL", "1.0"));
JR = [0.0, 0.0, 0.0]; KR = 0.0;

# solver settings of the old run (phisolve with itermax = 40, scale_current = true).
# A point is converged when residual < tol_accept; edge_tol is only a printed warning, as in phisolve.
itermax = 40; ftols = 1e-16; xtols = 1e-13; tol_accept = 1e-12; edge_tol = 1e-3;

## ----------Refinement settings----------
R     = parse(Int,     get(ENV, "CBJJ_R", "4"));        # new spacing = dV/R  (R = 2 or 4)
Vmax  = parse(Float64, get(ENV, "CBJJ_VMAX", "0.43"));  # refine the old intervals that start below |V| = Vmax
batch = parse(Int,     get(ENV, "CBJJ_BATCH", "1"));    # the batch this process solves: 1, ..., NB
NB    = parse(Int,     get(ENV, "CBJJ_NBATCH", "4"));   # number of batches

## ----------Old run----------
fnum(x) = x isa Integer ? string(x) : replace(string(round(x, sigdigits=4)), "." => "p");   # 0.6 -> "0p6"
fvec(v) = join(fnum.(v), "-");                                                               # [0,0,5.0] -> "0p0-0p0-5p0"

Nev1 = 400; evar1 = delta*range(0.24, 3.2, Nev1); evar = [reverse(-evar1); evar1]; Nev = 2*Nev1;   # old grid
str1 = "Nf$(Nf)_Ibias_ext_delta$(fnum(delta))_zeta$(fnum(zeta))_T$(fnum(T))_Gam$(fnum(Gamma))_V$(fnum(first(evar)))_$(fnum(last(evar)))_$(Nev)_JL$(fvec(JL))_KL$(fnum(KL))_JR$(fvec(JR))_KR$(fnum(KR))";

# folder of the old run, relative to this repository (CBJJ_DIR overrides it, e.g. for the KL sweep)
olddir  = get(ENV, "CBJJ_DIR", joinpath("Plots", "I bias", "sweep_T_Jz_KL$(fnum(KL))_Gam$(fnum(Gamma))_Nfconst$(Nf)"));
oldfile = joinpath(olddir, "Vipsol_n_" * str1 * ".jld");
println("old run: ", oldfile)
Vipold = load(oldfile, "Vipsol");        # Nev x 4Nf
resold = load(oldfile, "residualarr");

# Row of the old arrays that holds |V| = evar1[k] on the branch sg (-1 for -V, +1 for +V).
# The old file stores the -V branch in rows 1..Nev1 (from -3.2 up to -0.24) and +V in Nev1+1..Nev.
oldrow(k, sg) = sg < 0 ? Nev1 + 1 - k : Nev1 + k;

## ----------New voltages----------
# For each new point keep: its voltage, the old index k of its lower |V| neighbour, its branch,
# and its position u (0 < u < 1) between the old neighbours evar1[k] and evar1[k+1].
Vnew = Float64[]; knew = Int[]; sgnew = Int[]; unew = Float64[];
for sg in (-1, 1)
    for k in 1:Nev1-1
        if evar1[k] < Vmax
            for j in 1:R-1
                u = j/R
                push!(Vnew, sg*((1-u)*evar1[k] + u*evar1[k+1])); push!(knew, k); push!(sgnew, sg); push!(unew, u)
            end
        end
    end
end
Nnew = length(Vnew);

mine = collect(batch:NB:Nnew);           # every NB-th point, so all batches get a mix of low and high |V|
Nmine = length(mine);
println("refinement R = $R below |V| = $Vmax: $Nnew new points, batch $batch of $NB solves $Nmine of them")
println("threads = ", Threads.nthreads())

## ----------Solve one bias point (the same steps as phisolve in the module)----------
RN = Keldyshsetup_Floquetn_ext.RN_full(Nf, dw0, zeta, delta, T, Gamma, JL, KL, JR, KR);
rowscale = ones(Float64, 4*Nf); rowscale[(2*Nf+1):(4*Nf)] .= RN;     # scale_current = true

function solve_point(ev, seed, ip)
    Omega = ev;
    Nw0 = 2*ceil(Int, abs(Omega)/(2*dw0));
    war0 = -0.5*abs(Omega) .+ ((0:Nw0-1) .+ 0.5) .* (abs(Omega)/Nw0);

    F(x) = rowscale .* Keldyshsetup_Floquetn_ext.IbiasResidual_Tfull(x, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, ip);
    J(x) = rowscale .* Keldyshsetup_Floquetn_ext.IbiasJacobian_Tfull(x, war0, Omega, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, ip);
    res = nlsolve(F, J, seed, show_trace=true, method = :trust_region, ftol = ftols, xtol = xtols, iterations = itermax);
    x = res.zero;

    VipI = zeros(ComplexF64, 4*Nf+1);
    for kl = 1:2*Nf
        VipI[2*kl] = x[kl] + im*x[(2*Nf)+kl];
    end
    Iif = Keldyshsetup_Floquetn_ext.current_Floquet_Tfull(war0, Omega, Nf, zeta, delta, T, Gamma, VipI, JL, KL, JR, KR, ip);
    I = real(sum(diag(Iif)));

    return x, res.residual_norm, Keldyshsetup_Floquetn_ext.edgepeak(x, Nf), I
end

## ----------Solve this batch----------
outfile = joinpath(olddir, "Vipsol_r$(R)_V$(fnum(Vmax))_b$(batch)of$(NB)_" * str1 * ".jld");

Vipsol      = fill(NaN, Nmine, 4*Nf);    # NaN = not solved yet
residualarr = fill(NaN, Nmine);
Iv          = fill(NaN, Nmine);
edgearr     = fill(NaN, Nmine);          # weight at the Floquet cutoff (edgepeak)
distarr     = fill(NaN, Nmine);          # |x - interpolated seed| / |interpolated seed|
seedarr     = zeros(Int, Nmine);         # 1 = interpolated, 2 = old point above, 3 = old point below
seednames   = ["interpolated", "old point above", "old point below"];

for n in 1:Nmine
    p = mine[n];
    ka = oldrow(knew[p], sgnew[p]);      # old row just below in |V|
    kb = oldrow(knew[p]+1, sgnew[p]);    # old row just above in |V|
    seeds = [(1-unew[p]) .* Vipold[ka,:] .+ unew[p] .* Vipold[kb,:],
             Vipold[kb,:],
             Vipold[ka,:]];
    println("== point $n of $Nmine: eV = $(round(Vnew[p], digits=6)), old residuals below/above = $(resold[ka]) / $(resold[kb])")

    for s in 1:3
        t0 = time();
        x, resid, ep, I = solve_point(Vnew[p], seeds[s], p);
        dist = norm(x - seeds[1]) / norm(seeds[1]);
        println("   seed = $(seednames[s]) : residual = $resid, edge/peak = $(round(ep, sigdigits=3)), dist = $(round(dist, sigdigits=3)), time = $(round(time()-t0, digits=1)) s")

        if s == 1 || resid < residualarr[n]       # keep the best try
            Vipsol[n,:] = x; residualarr[n] = resid; Iv[n] = I; edgearr[n] = ep; distarr[n] = dist; seedarr[n] = s;
        end
        if resid < tol_accept
            break
        end
    end

    println(residualarr[n] < tol_accept ? "-- point $n converged" : "-- point $n NOT converged (best residual = $(residualarr[n]))")
    if edgearr[n] >= edge_tol                    # diagnostic only, as in phisolve (old low-V points reach ~3e-3)
        println("-- point $n: edge/peak = $(round(edgearr[n], sigdigits=3)), the Floquet support Nf = $Nf is tight here")
    end
    if distarr[n] > 0.1
        println("-- point $n: far from its old neighbours (dist = $(round(distarr[n], sigdigits=3))) -- check for a branch jump")
    end

    # save after every point, so a killed job keeps what it has done
    save(outfile, "evar", Vnew[mine], "Vipsol", Vipsol, "residualarr", residualarr, "Iv", Iv,
         "edgepeak", edgearr, "dist", distarr, "seed", seedarr);
end

nok = count(residualarr .< tol_accept);
println("-- batch $batch of $NB done: $nok of $Nmine points converged; saved ", outfile)
