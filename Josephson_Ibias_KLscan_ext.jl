include("Keldyshsetup_Floquetn_ext.jl")
using .Keldyshsetup_Floquetn_ext

using MKL
using LinearAlgebra
using NLsolve
using JLD
using Printf

# ---------------------------------------------------------------------------
# Single-voltage KL-homotopy scan (diagnostic for the diode convergence failure).
#
# At ONE failing bias point evar[ifail] (the eV = Delta ... Delta+eYSR_L+eYSR_R band),
# ramp KL from 0 to KLtarget in small steps, seeding each stage's phisolve from the
# previous stage's solution (stage 1 seeds from the saved KL=KR=0 run). Discriminates:
#   - every stage converges           -> fine-step homotopy works; chain full sweeps
#     KL: 0 -> ... -> KLtarget, or just rerun the failing eV-window per stage.
#   - some stage KL* fails even after halving the step around it -> fold in KL: the
#     KL!=0 branch is disconnected from the symmetric one at this V; seed tricks
#     cannot cross it -> arc-length continuation (or accept the window as unreachable).
# Cost: one single-point solve per stage (minutes each), not a full 100-point sweep.
# ---------------------------------------------------------------------------

#size
Nf = 24;

#energies
mu = 0; delta = 1; zeta = 5; T = 0.6; Gamma = 2e-2;
dw0 = minimum([0.015, Gamma/2.0]);

#impurities: KL is ramped over KLlist below; everything else fixed
JL = [0.0, 0.0, 4.0];
JR = [0.0, 0.0, 4.0]; KR = 0.0;

#voltage grid of the SAVED KL=KR=0 run (must match its file exactly)
Nev = 100; evar = delta*range(0.24, 3.2, Nev);

#the failing bias point to probe (39 -> eV/Delta ~ 1.38, just below Delta+eL+eR)
ifail = 36;

#KL stages (extend to 1.0 once 0.5 is reached; insert midpoints where a stage fails)
KLlist = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];
tol_accept = 1e-8;   # stage counts as converged below this residual
itermax_scan = 150;  # generous budget: a slow trust-region crawl that WOULD converge must
                     # not be mislabelled FAILED by the production itermax=40 cap

#Scheme (only ws=0 supported in the 4x4 ext module)
ws = 0;

#naming (same scheme as the main driver)
fnum(x) = x isa Integer ? string(x) : replace(string(round(x, sigdigits=4)), "." => "p");
fvec(v) = join(fnum.(v), "-");

## ----------Load the KL=KR=0 seed at this voltage----------
str1_load = "Nf$(Nf)_Ibias_ext_delta$(fnum(delta))_zeta$(fnum(zeta))_T$(fnum(T))_Gam$(fnum(Gamma))_V$(fnum(first(evar)))_$(fnum(last(evar)))_$(Nev)_JL$(fvec(JL))_KL$(fnum(0.0))_JR$(fvec(JR))_KR$(fnum(0.0))";
str2_load = "n_" * str1_load;
Vipsol0 = load("Vipsol_" * str2_load * ".jld", "Vipsol");
@assert size(Vipsol0) == (Nev, 2*(2*Nf)) "loaded Vipsol size $(size(Vipsol0)) != expected ($(Nev), $(2*(2*Nf)))";

evfail = evar[ifail];
println("KL scan at eV/Delta = $(evfail/delta)  (ifail = $ifail)")
println("stages: KL = ", KLlist)

## ----------Ramp KL, chaining seeds----------
seed = reshape(Vipsol0[ifail,:], 1, :);          # 1-row seed matrix for the 1-point evar
Nst = length(KLlist);
reslog = fill(NaN, Nst); Ivlog = fill(NaN, Nst);
Vipstages = zeros(Float64, Nst, 2*(2*Nf));

for st = 1:Nst
    KLs = KLlist[st];
    println("\n===== stage $st/$Nst : KL = $KLs =====")
    EL = Keldyshsetup_Floquetn_ext.ysr_energies_numerical(JL, KLs, zeta, delta);
    println("eYSR_L/Delta = $(round.(EL./delta, digits=5))")

    Iv1, Vip1, res1 = Keldyshsetup_Floquetn_ext.phisolve(ws, dw0, [evfail], Nf, zeta, delta, T, Gamma, JL, KLs, JR, KR, seed, itermax = itermax_scan);
    reslog[st] = res1[1]; Ivlog[st] = Iv1[1]; Vipstages[st,:] = Vip1[1,:];
    @printf("stage KL=%.3f : residual = %.3e   Iv = %.6f\n", KLs, res1[1], Iv1[1])

    if res1[1] < tol_accept
        global seed = Vip1;                      # chain: next stage seeds from this solution
    else
        println("STAGE FAILED at KL = $KLs (seeded from KL = $(st == 1 ? 0.0 : KLlist[st-1])).")
        println("-> insert midpoints in KLlist around KL = $KLs and rerun; if it still fails")
        println("   for arbitrarily small steps, the branch has a fold in KL at this V.")
        break
    end
end

## ----------Summary + save----------
println("\n===== KL-scan summary at eV/Delta = $(evfail/delta) =====")
for st = 1:Nst
    isnan(reslog[st]) && continue
    @printf("KL = %5.3f   residual = %10.3e   Iv = %12.6f   %s\n", KLlist[st], reslog[st], Ivlog[st], reslog[st] < tol_accept ? "ok" : "FAILED")
end

save("KLscan_ev$(fnum(evfail/delta))_" * str2_load * ".jld",
     "KLlist", KLlist, "reslog", reslog, "Ivlog", Ivlog, "Vipstages", Vipstages, "evfail", evfail);
