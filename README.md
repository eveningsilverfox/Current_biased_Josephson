# Current-biased Josephson junctions — Keldysh–Floquet code

Julia code for the DC current–voltage (I–V) characteristics of a single-channel
Josephson junction of **arbitrary transparency**, computed with a nonequilibrium
**Keldysh Green's-function** formalism and a **Floquet** (harmonic) expansion in
the Josephson frequency.

This repository contains the code used in the publication:

> **Origin of Subharmonic Gap Structure of DC Current-Biased Josephson Junctions**
> *Phys. Rev. B 111, L220504 (2025)* — DOI: [10.1103/zbd5-1cc2](https://journals.aps.org/prb/abstract/10.1103/zbd5-1cc2)
> Preprint: [arXiv:2412.09862](https://arxiv.org/abs/2412.09862)

A 4×4 Nambu⊗spin extension hosting **classical-spin (Yu–Shiba–Rusinov)
impurities** and the Josephson diode effect (following
[arXiv:2602.15213](https://arxiv.org/abs/2602.15213)) is also included; see the
section on the 4×4 (YSR) extension below.

## The physics

For a **voltage-biased** junction the voltage `V(t) = V_dc` is constant and the
subharmonic gap structure (SGS) — the jumps in `dI/dV` at `eV = 2Δ/n` — is well
described by multiple-Andreev-reflection theory. For a **DC current-biased**
junction only the *time-averaged* voltage `⟨V⟩` is fixed; the instantaneous
voltage `V(t)` (equivalently the phase `φ(t)`) is itself a dynamical quantity
that develops AC harmonics and must be solved **self-consistently** by demanding
that no AC current flows. This self-consistency changes the SGS relative to the
voltage-biased case, and the paper identifies previously-missing
**two-quasiparticle tunnelling processes** as the origin of the current-biased
SGS.

The junction is two BCS leads coupled by a single hopping element `T`, in Nambu
(particle–hole) space. The transparency is set by the ratio `T/ζ` (tunnel limit
`T/ζ ≪ 1`, ballistic `T/ζ → 1`). Energies are in units of the gap `Δ`. A
transparency (Werthamer-type) expansion of the current at orders `T²`, `T⁴`, …
is provided to dissect which microscopic processes produce each I–V feature.

A self-contained derivation of every equation solved by the code is in
[`docs/Keldysh_floquet_Vbias_Ibias.pdf`](docs/Keldysh_floquet_Vbias_Ibias.pdf)
(source: `docs/Keldysh_floquet_Vbias_Ibias.tex`).

## Requirements

- **Julia** (run interactively, e.g. from the REPL or VS Code). On Linux it can
  be installed with the official `juliaup` installer:

  ```sh
  curl -fsSL https://install.julialang.org | sh
  ```
- Packages: `MKL`, `LinearAlgebra`, `Statistics`, `Plots`, `LaTeXStrings`,
  `NLsolve`, `JLD`, `OhMyThreads`, `Symbolics`, `Printf`, `SpecialFunctions`.

There is no `Project.toml` checked in — install the packages above into your
Julia environment, e.g.:

```julia
import Pkg
Pkg.add(["MKL","Plots","LaTeXStrings","NLsolve","JLD","OhMyThreads",
         "Symbolics","SpecialFunctions"])
```

The hot loops use `Threads.@threads`; start Julia with multiple threads
(`julia -t auto`) for performance.

## Repository structure

**Core library (the modules)**

| File | Contents |
| --- | --- |
| `Keldyshsetup_Floquetn.jl` | The 2×2 Nambu module: bare/dressed Green's functions, the lesser self-energy, the Keldysh current, the current-bias self-consistency residual + analytic Jacobian, the solver `phisolve`, the normal-state resistance `RN_full`, and the transparency-expansion (`T2`, `T4`, …) and multiparticle-tunnelling routines. |
| `Keldyshsetup_Floquetn_ext.jl` | The 4×4 Nambu⊗spin extension of the above, with classical-spin (YSR) impurities (per-lead `JL,KL,JR,KR`). |

**Driver scripts** (each `include`s a module, sets parameters, runs, plots, saves)

| File | Purpose |
| --- | --- |
| `Josephson_Ibias_Floquetn.jl` | Main current-biased driver — reproduces the paper figures; compares exact vs. perturbative currents. |
| `Josephson_Ibias_Floquetn_Tar.jl` | Current-biased, swept over transparency `T`. |
| `Josephson_Vbias_Floquetn.jl` | Voltage-biased case (no self-consistency). |
| `Josephson_Vbias_MPT.jl` | Voltage-biased multiparticle-tunnelling (real-frequency) currents. |
| `Josephson_cphir.jl` | Equilibrium current–phase relation `I(φ)` and critical current. |
| `Josephson_Ibias_Floquetn_ext.jl` | Current-biased driver for the 4×4 (YSR) module. |
| `Josephson_Vbias_Floquetn_ext.jl` | Voltage-biased driver for the 4×4 (YSR) module. |

**Other**

- `docs/` — the technical notes (PDF + LaTeX source) deriving the formalism.
- `Plots/` — saved figures (`.png`) and cached solutions (`.jld`).
- `job_scripts/` — cluster submission scripts.

## Usage

The code leverages multi-threading. Start Julia with 
```
julia -t num_threads
```
and run the driver scripts interactively in the Julia REPL as
```julia
include("Josephson_Ibias_Floquetn.jl")   # current-biased I–V (main result)
```

Or, you may run it directly as
```
julia -t num_threads Josephson_Ibias_Floquetn.jl
```

Each driver sets its parameters at the top (`Nf` Floquet harmonics, `delta`,
`zeta`, `T`, `Gamma`, the bias array `evar`, and the scheme flag `ws` — `ws=0`
exact Dyson, `ws=2,4,…` truncated transparency orders), runs the solve/sweep,
and produces the I–V and `dI/dV` plots. Heavy runs cache to `.jld` files in
`Plots/`; the naming strings (`str1`/`str2`) encode the full parameter set, so
re-loading a cached result requires matching them.

## The 4×4 Yu–Shiba–Rusinov extension

`Keldyshsetup_Floquetn_ext.jl` extends the formalism to the full Nambu⊗spin
basis `(c↑, c↓, c↑†, c↓†)`, so that **classical-spin (magnetic) impurities** can
be placed on each lead via a local Dyson dressing of the surface Green's
function. Each lead carries its own exchange and potential scattering terms, passed 
as `JL, KL, JR, KR`. he corresponding derivation is Section 10 of the technical notes in `docs/`.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{SubharmonicSGS,
  author  = {Aritra Lahiri and Sang-Jun Choi and Bj\"orn Trauzettel},
  title   = {Origin of Subharmonic Gap Structure of DC Current-Biased Josephson Junctions},
  journal = {Phys. Rev. B},
  volume  = {111},
  pages   = {L220504},
  year    = {2025},
  doi     = {10.1103/zbd5-1cc2},
  url     = {https://journals.aps.org/prb/abstract/10.1103/zbd5-1cc2},
  note    = {Preprint: arXiv:2412.09862},
}
```
