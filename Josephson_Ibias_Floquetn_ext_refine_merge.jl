using JLD

# ---------------------------------------------------------------------------
# Join the old run with the batches of Josephson_Ibias_Floquetn_ext_refine.jl.
# Use the same CBJJ_* settings as the refinement run. Only the converged new points are added
# (a missing point just leaves the old spacing there). The result has the same keys as the old
# file plus "evar", the voltage of every row (the grid is no longer uniform):
#     Vipsol_r{R}_V{Vmax}_<same name as the old run>.jld   in the folder of the old run
# ---------------------------------------------------------------------------

Nf = parse(Int, get(ENV, "CBJJ_NF", "32"));
delta = 1; zeta = 5; T = parse(Float64, get(ENV, "CBJJ_T", "1.0")); Gamma = 5e-2;
JL = [0.0, 0.0, parse(Float64, get(ENV, "CBJJ_JZ", "5.0"))]; KL = parse(Float64, get(ENV, "CBJJ_KL", "1.0"));
JR = [0.0, 0.0, 0.0]; KR = 0.0;
R    = parse(Int,     get(ENV, "CBJJ_R", "4"));
Vmax = parse(Float64, get(ENV, "CBJJ_VMAX", "0.43"));
NB   = parse(Int,     get(ENV, "CBJJ_NBATCH", "4"));
tol_accept = 1e-12; edge_tol = 1e-3;

fnum(x) = x isa Integer ? string(x) : replace(string(round(x, sigdigits=4)), "." => "p");
fvec(v) = join(fnum.(v), "-");
Nev1 = 400; evar1 = delta*range(0.24, 3.2, Nev1); evar = [reverse(-evar1); evar1]; Nev = 2*Nev1;   # old grid
str1 = "Nf$(Nf)_Ibias_ext_delta$(fnum(delta))_zeta$(fnum(zeta))_T$(fnum(T))_Gam$(fnum(Gamma))_V$(fnum(first(evar)))_$(fnum(last(evar)))_$(Nev)_JL$(fvec(JL))_KL$(fnum(KL))_JR$(fvec(JR))_KR$(fnum(KR))";
olddir = get(ENV, "CBJJ_DIR", joinpath("Plots", "I bias", "sweep_T_Jz_KL$(fnum(KL))_Gam$(fnum(Gamma))_Nfconst$(Nf)"));

# old run
oldfile = joinpath(olddir, "Vipsol_n_" * str1 * ".jld");
Vall = collect(evar);
Vipall = load(oldfile, "Vipsol"); resall = load(oldfile, "residualarr"); Ivall = load(oldfile, "Iv");

# new points, batch by batch
for b in 1:NB
    f = joinpath(olddir, "Vipsol_r$(R)_V$(fnum(Vmax))_b$(b)of$(NB)_" * str1 * ".jld");
    if !isfile(f)
        println("missing batch $b: ", f)
        continue
    end
    Vb = load(f, "evar"); Vipb = load(f, "Vipsol"); resb = load(f, "residualarr"); Ivb = load(f, "Iv"); epb = load(f, "edgepeak");
    ok = resb .< tol_accept;                                # false also for NaN (not solved)
    println("batch $b: $(count(ok)) of $(length(Vb)) points converged ($(count(ok .& (epb .>= edge_tol))) of them with edge/peak >= $edge_tol)")
    for n in findall(.!ok)
        println("   left out: eV = $(round(Vb[n], digits=6)), residual = $(resb[n])")
    end
    global Vall = [Vall; Vb[ok]];
    global Vipall = [Vipall; Vipb[ok,:]];
    global resall = [resall; resb[ok]];
    global Ivall = [Ivall; Ivb[ok]];
end

# sort by voltage: -3.2 ... -0.24, then 0.24 ... 3.2 (same order as the old file)
order = sortperm(Vall);
Vall = Vall[order]; Vipall = Vipall[order,:]; resall = resall[order]; Ivall = Ivall[order];

outfile = joinpath(olddir, "Vipsol_r$(R)_V$(fnum(Vmax))_" * str1 * ".jld");
save(outfile, "evar", Vall, "Vipsol", Vipall, "residualarr", resall, "Iv", Ivall);
println("saved $(length(Vall)) points (old $(Nev) + new $(length(Vall)-Nev)): ", outfile)
