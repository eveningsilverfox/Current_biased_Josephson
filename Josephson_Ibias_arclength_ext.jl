include("Keldyshsetup_Floquetn_ext.jl")
using .Keldyshsetup_Floquetn_ext

using MKL
using LinearAlgebra
using JLD
using Printf
using Plots

# ---------------------------------------------------------------------------
# Pseudo-arclength continuation of the current-bias self-consistency in eV,
# for the KL != 0 low-bias band where natural (parameter) continuation and the
# K-homotopy rescue both stall.
#
# Why: every backend tried so far (NLsolve/NonlinearSolve trust region, LM) is
# a Newton-family corrector AT FIXED eV. If the odd-harmonic running-solution
# branch FOLDS in eV (turning point, det J = 0), then below the fold there is
# simply no solution continuously connected to the seed: failure of the fixed-V
# solve is structural, not a solver deficiency. The classical fix is to stop
# parameterizing by eV and follow the branch by arclength s: unknowns y = (x, v),
# extended system
#       H(y) = [ F(x, v) ; t . (y - y_pred) ] = 0,
# whose bordered Jacobian [Jx  Fv; t'] stays regular THROUGH the fold. The
# continuation then traverses the turning point and comes back up in v along
# the second (previously unreachable) sheet.
#
# Outcome interpretation:
#   - v marches monotonically below evar[1]: no fold; the sweep failures were
#     basin/seeding artefacts. The arc solutions ARE the missing bias points
#     (saved; reseed the production sweep from them).
#   - the tangent v-component changes sign at v* (FOLD detected): the running
#     2pi-periodic (odd-harmonic) state ceases to exist below v* at this KL --
#     the physical low-bias state is a different attractor (retrapped/period-
#     doubled/chaotic phase dynamics). The stalled sweep points below v* are
#     then genuinely unsolvable in this ansatz; report v* as the branch edge.
#
# Grid note: the modules now use even-count midpoint BZ sampling everywhere,
#     war0(v) = -v/2 + (g + 1/2) v/Nw0,  g = 0..Nw0-1,  Nw0 even,
# so the sample set is exactly particle-hole symmetric, the T=0 occupation step
# at w + m*Omega = 0 falls exactly on a cell BOUNDARY, and no sample sits closer
# than half a cell to a step (every ff sign is stable under small changes of v).
# One continuation hazard remains: phisolve recomputes Nw0 per bias point, so
# its F(x, v) still jumps where Nw0 steps. Here Nw0 is therefore FROZEN at the
# anchor value -- F(x, v) is then smooth in v, as the corrector and the
# finite-difference dF/dv require -- and the frozen grid only gets finer (in
# absolute terms) as v decreases. At the upper anchor it coincides with
# phisolve's grid exactly; below, it differs at O(dw0) discretization error.
# Both anchors are re-polished at fixed v on this grid before marching.
#
# Cost per accepted arc point: 2-4 corrector iterations, each ~ one Jacobian --
# comparable to a few nlsolve iterations of the production sweep.
# ---------------------------------------------------------------------------

#size
Nf = 30;

#energies
mu = 0; delta = 1; zeta = 5; T = 0.6; Gamma = 2e-2;
dw0 = minimum([0.015, Gamma/2.0]);

#impurities of the run to continue (the failing one)
JL = [0.0, 0.0, 4.0]; KL = 0.1;
JR = [0.0, 0.0, 4.0]; KR = 0.0;

#voltage grid of the saved run (must match its file exactly)
Nev = 100; evar = delta*range(0.24, 3.2, Nev);

#continuation controls
tol_ok   = 1e-9;          # a loaded bias point counts as converged below this
istart   = 0;             # lowest converged index to anchor at; 0 -> automatic
ftol_c   = 1e-11;         # corrector convergence on ||F||
h0       = 0.5;           # initial arc step, in units of ||(dx, dv)|| between the anchors
hmin     = 1e-3; hmax = 4.0;   # step bounds (same units)
Nsteps   = 300;           # max accepted arc points
itcmax   = 12;            # max corrector (Newton) iterations per arc point
dvfd     = 1e-6;          # finite-difference step for Fv = dF/dv
v_stop_lo = 0.75*evar[1]; # stop once v drops below this (v_stop_hi is set after the anchors are picked)

#naming (same scheme as the main driver)
fnum(x) = x isa Integer ? string(x) : replace(string(round(x, sigdigits=4)), "." => "p");
fvec(v) = join(fnum.(v), "-");
str1 = "Nf$(Nf)_Ibias_ext_delta$(fnum(delta))_zeta$(fnum(zeta))_T$(fnum(T))_Gam$(fnum(Gamma))_V$(fnum(first(evar)))_$(fnum(last(evar)))_$(Nev)_JL$(fvec(JL))_KL$(fnum(KL))_JR$(fvec(JR))_KR$(fnum(KR))";
str2 = "n_" * str1;

## ----------Load the partially converged run and pick the anchors----------
fname = "Vipsol_" * str2 * ".jld";
Vipsol = load(fname, "Vipsol"); residualarr = load(fname, "residualarr");
@assert size(Vipsol) == (Nev, 2*(2*Nf)) "loaded Vipsol size $(size(Vipsol)) != expected ($(Nev), $(2*(2*Nf)))";

if istart == 0
    istart = findfirst(hi -> residualarr[hi] <= tol_ok && residualarr[hi+1] <= tol_ok, 1:Nev-1);
    istart === nothing && error("no adjacent converged pair found in $fname; lower tol_ok or set istart manually");
end
println("anchors: hi = $(istart+1) (eV/Delta = $(evar[istart+1]/delta)) -> hi = $istart (eV/Delta = $(evar[istart]/delta)), marching DOWN in v")
v_stop_hi = evar[min(istart+3, Nev)];   # stop once the (post-fold) returning sheet climbs past the anchor region

## ----------Frozen frequency grid and residual/Jacobian closures----------
Nw0 = 2*ceil(Int, abs(evar[istart+1])/(2*dw0));                  # frozen and EVEN (see header)
wargrid(v) = -0.5*abs(v) .+ ((0:Nw0-1) .+ 0.5) .* (abs(v)/Nw0);  # midpoint rule: PH-symmetric, step-safe
Fxv(x, v) = real.(Keldyshsetup_Floquetn_ext.IbiasResidual_Tfull(x, wargrid(v), v, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, 0));
Jxv(x, v) = real.(Keldyshsetup_Floquetn_ext.IbiasJacobian_Tfull(x, wargrid(v), v, Nf, zeta, delta, T, Gamma, JL, KL, JR, KR, 0));

function dc_current(x, v)
    VipI = zeros(ComplexF64, 4*Nf+1);
    for kl = 1:2*Nf
        VipI[2*kl] = x[kl] + im*x[(2*Nf)+kl];
    end
    Ifm = Keldyshsetup_Floquetn_ext.current_Floquet_Tfull(wargrid(v), v, Nf, zeta, delta, T, Gamma, VipI, JL, KL, JR, KR, 0);
    return real(sum(diag(Ifm)));
end

# plain Newton at fixed v (seeds are excellent: re-polish of an already-converged point onto the frozen grid)
function newton_fixedv(x0, v; itmax = 15, ftol = ftol_c)
    x = copy(x0); F = Fxv(x, v);
    for it = 1:itmax
        norm(F) < ftol && break
        x = x - ( Jxv(x, v) \ F );
        F = Fxv(x, v);
    end
    return x, norm(F);
end

# one corrector solve of H(y) = [F(x,v); t.(y - ypred)] = 0 (Fv frozen over the corrector)
function correct(t, ypred, n; itc = itcmax, ftol = ftol_c, dv = dvfd)
    yc = copy(ypred);
    F = Fxv(yc[1:n], yc[end]);
    Fv = ( Fxv(yc[1:n], yc[end] + dv) - F ) ./ dv;
    nit = 0;
    while norm(F) > ftol && nit < itc
        nit += 1;
        Jx = Jxv(yc[1:n], yc[end]);
        M = [Jx Fv; transpose(t)];
        dy = M \ [ -F; -dot(t, yc - ypred) ];
        yc = yc + dy;
        F = Fxv(yc[1:n], yc[end]);
    end
    return yc, norm(F), nit;
end

## ----------Pseudo-arclength sweep----------
function arclength_sweep(xhiv, vhi, xlov, vlo)
    n = 2*(2*Nf);
    ya = [xhiv; vhi]; yb = [xlov; vlo];
    t = (yb - ya); sc = norm(t); t = t ./ sc;      # unit tangent, pointing DOWN in v; sc = anchor separation
    y = copy(yb);
    h = h0*sc; hminp = hmin*sc; hmaxp = hmax*sc;   # steps in anchor-separation units

    varc = [y[end]]; resarc = [norm(Fxv(y[1:n], y[end]))];
    Iarc = [dc_current(y[1:n], y[end])];
    tvarc = [t[end]]; Xarc = [copy(y[1:n])];
    folds = Float64[];

    step = 0; rejected = 0;
    while step < Nsteps
        ypred = y + h .* t;
        yc, rn, nit = correct(t, ypred, n);
        if rn > ftol_c || yc[end] <= 0
            h = h/2; rejected += 1;
            @printf("-- step rejected (||F|| = %.2e after %d its); h -> %.3e\n", rn, nit, h)
            h < hminp && (println("-- h < hmin: continuation stuck (sharp fold or genuinely singular point); stopping"); break)
            continue
        end
        step += 1;
        tnew = (yc - y) ./ norm(yc - y);           # secant tangent (orientation follows the walk)
        if sign(tnew[end]) != sign(t[end]) && abs(t[end]) > 1e-12
            push!(folds, min(y[end], yc[end]));    # deepest point bracketing the turn (tighten with smaller hmax)
            println("========== FOLD: tangent dv/ds changed sign; turning point at eV/Delta ~ $(folds[end]/delta) ==========")
            println("   (bracket: [$(min(y[end],yc[end])/delta), $(max(y[end],yc[end])/delta)])")
            println("   no odd-harmonic running solution below the fold on this branch; now tracking the returning sheet")
        end
        y = yc; t = tnew;
        push!(varc, y[end]); push!(resarc, rn); push!(tvarc, t[end]); push!(Xarc, copy(y[1:n]));
        push!(Iarc, dc_current(y[1:n], y[end]));
        @printf("arc %3d : eV/Delta = %.5f   Iv = %+.6e   ||F|| = %.2e   its = %d   h = %.3e   dv/ds = %+.3f\n",
                step, y[end]/delta, Iarc[end], rn, nit, h, t[end])
        nit <= 4 && (h = min(h*1.4, hmaxp));
        nit >= 8 && (h = max(h/1.5, hminp));
        (y[end] < v_stop_lo || y[end] > v_stop_hi) && (println("-- reached v stop window; done"); break)
    end
    println("accepted $step arc points, $rejected rejections, folds at eV/Delta = ", folds./delta)
    return varc, Xarc, Iarc, resarc, tvarc, folds;
end

xhi0, rhi = newton_fixedv(Vector{Float64}(Vipsol[istart+1,:]), evar[istart+1]);
xlo0, rlo = newton_fixedv(Vector{Float64}(Vipsol[istart,:]),   evar[istart]);
println("anchor re-polish on frozen grid: ||F|| = $rhi (hi), $rlo (lo)")
(rhi > tol_ok || rlo > tol_ok) && error("anchor re-polish failed; pick a better-converged istart");

varc, Xarc, Iarc, resarc, tvarc, folds = arclength_sweep(xhi0, evar[istart+1], xlo0, evar[istart]);

## ----------Save----------
Xm = permutedims(reduce(hcat, Xarc));   # (arc points) x 4Nf, same layout as Vipsol rows
save("Arclength_" * str2 * ".jld", "varc", varc, "Xarc", Xm, "Iarc", Iarc,
     "resarc", resarc, "tvarc", tvarc, "folds", folds, "Nw0", Nw0, "istart", istart);
println("saved: Arclength_" * str2 * ".jld")

## ----------Plots----------
p1 = plot(varc/delta, Iarc, marker = :circle, ms = 2.5, lc = :blue, framestyle = :box, legend = :none)
for vf in folds
    vline!(p1, [vf/delta], linestyle = :dash, lc = :red)
end
xlabel!(p1, "eV/Δ"); ylabel!(p1, "I (arc)");
p2 = plot(1:length(varc), varc/delta, marker = :circle, ms = 2.5, lc = :black, framestyle = :box, legend = :none)
xlabel!(p2, "arc point"); ylabel!(p2, "eV/Δ");
savefig(plot(p1, p2, layout = (1,2), size = (1000,400), dpi = 450), "Arclength_" * str2 * ".png")
