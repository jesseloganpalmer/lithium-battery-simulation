# Sanders LTO: secondary particle size and rate performance in PyBaMM

A PyBaMM study of how **secondary (micro)particle size** controls the rate
performance and volumetric energy density of lithium titanate electrodes,
calibrated against and validated on:

> K. A. Sanders, R. Mizuta, H. J. Tan, J. E. Trevelyan, M. F. L. De Volder,
> *Microdroplet Templating of Uniform Nanostructured Battery Microparticles
> with Scalable Membrane Emulsification*, **ACS Nano** 2026, 20, 7555-7568.
> <https://doi.org/10.1021/acsnano.5c17777> (CC-BY 4.0)

## Quick start

```bash
.venv/bin/python Sanders_LTO/run_all.py
```

Roughly four minutes. Everything lands in `Sanders_LTO/results/`.

Individual stages, in dependency order:

| Script | What it does |
|---|---|
| `plot_experimental.py` | Replots Sanders' rate-test step figures (6d/6e, 7b/7c, S12, S14) from the tabulated SI data |
| `calibrate.py` | Fits the model to Tables S8/S10 and writes `calibration.json` |
| `plot_calibration.py` | Model-vs-experiment overlay and the fitted diffusion-domain radii |
| `run_design_sweep.py` | Forward sweep over secondary diameter x C-rate x intra-particle conductivity |
| `plot_design_map.py` | The predicted design map |
| `run_primary_sweep.py` | Sweeps the *primary* crystallite size inside a fixed 8 µm secondary particle |
| `plot_primary_sweep.py` | The primary-size figure |
| `run_scale_map.py` | Joint sweep of *both* length scales, with and without an intra-particle carbon budget |
| `plot_scale_map.py` | The joint design map |

## The modelling problem, and why the obvious approach fails

Every sample in the paper is built from **the same** size-selected LTO
nanopowder (172 nm Z-average, Figure S1). What membrane emulsification changes
is the *secondary* particle: the droplet-templated assembly of those
nanocrystals, 2-9 µm across.

So the naive model - "sweep `Positive particle radius [m]`" - is not just
imprecise, it gets the paper's central result backwards. It predicts the
unstructured nanopowder should have the *best* rate performance, because it has
the shortest diffusion length. Sanders et al. measure the worst by a wide
margin: 53 mAh/g at 20C versus 148 mAh/g for the 8 µm microparticles.

Two further things turned out to be true when tested numerically, and both
shape the model:

- With Sanders' loadings (~1.2 mg/cm², ~30 µm coatings, 82-86 % porosity) and
  the 0.55 V of headroom below LTO's 1.55 V plateau, **the measured EIS
  resistances are far too small to cause any rate fade at all**. Feeding
  Tables S6 and S10 into a DFN gives a flat 171 mAh/g from 0.1C to 20C for
  every electrode. Contact resistance is not the mechanism, and neither is
  electrode-scale electronic conduction.
- **Solid-state diffusion is the only parameter with leverage**, and the
  length scale it acts over is what secondary structuring actually changes.

## What the model says

The secondary particle enters through three mechanisms rather than through the
particle radius:

1. **The effective lithium diffusion domain.** A droplet-templated secondary
   particle has an open, electrolyte-filled internal pore network, so every
   primary nanocrystal is wetted and the diffusion length is the *primary*
   radius - independent of how big the secondary particle is. The bare
   nanopowder instead forms dense, unwetted aggregates (the paper documents
   this directly: hard aggregates that "could not easily be dispersed",
   gel-like slurries, and a *lower* BET area, 24.1 m²/g, than the templated
   microparticles at 28-29 m²/g), so its diffusion domain is the aggregate.
2. **Intra-particle transport.** Inside a secondary particle, electrons cross
   the pyrolysed-carbon or CNT network and Li+ crosses the intra-particle
   electrolyte. For uniform generation in a sphere both drops scale as
   `R_secondary²`, which is what eventually caps useful particle size.
3. **Electrode packing.** Bigger, more uniform particles pack denser
   (Table S6: 0.39 -> 0.50 g/cm³), so a given areal loading makes a thinner
   coating and a higher volumetric energy density.

Everything else is read from the paper: loadings, thicknesses, porosities
(Table S6), series and charge-transfer resistances (Table S10), electrode
composition (Table S7), primary size (Figure S1), the 1.0-2.5 V window and
25 °C (Methods). The charge-transfer resistance is converted into an
exchange-current density using the model's own surface area, so kinetics are
set by measurement rather than by a free parameter.

### What is fitted, and against what

Three numbers, each fitted against a different part of the data:

| Fitted | Value | Fitted against |
|---|---|---|
| Accessible stoichiometry span | 0.964 | The 0.1C capacity only, which carries no rate information |
| LTO solid-state diffusivity | 2.4 x 10⁻¹⁷ m² s⁻¹ | The two droplet-templated pure-LTO electrodes |
| Effective diffusion domain radius | one per electrode | That electrode's own rate curve |

### The result worth looking at

The diffusion-domain radius is fitted **per electrode**, over a search range
spanning 40 nm to 3 µm, using nothing but the rate curve. For every
droplet-templated sample it comes back at the independently measured primary
crystallite radius:

| Electrode | Secondary diameter | Fitted domain radius | vs 86 nm primary |
|---|---|---|---|
| Bulk-emulsified LTO | 2.3 µm | 85 nm | 0.99x |
| Membrane-emulsified LTO | 8.0 µm | 88 nm | 1.02x |
| Membrane LTO/oxCNT | 9.1 µm | 89 nm | 1.04x |
| Membrane LTO/CNT/PVP | 9.3 µm | 111 nm | 1.29x |
| Commercial spray dried | ~7 µm (D50) | 134 nm | 1.56x |
| Unstructured LTO + CNT | none | 127 nm | 1.48x |
| Unstructured LTO + oxCNT | none | 216 nm | 2.51x |
| **Unstructured nanopowder** | none | **246 nm** | **2.86x** |

Three different secondary sizes, spanning a factor of four, all recover the
same 85-89 nm. That is the quantitative statement of the paper's mechanism:
**droplet templating decouples the diffusion length from the particle size**,
which is exactly what a spray-dried or dry-aggregated powder fails to do. The
unstructured nanopowder behaves as ~0.5 µm dense aggregates; adding pristine
CNTs to the powder mix partly relieves that (127 nm), which is why it is the
only unstructured electrode that survives 20C in Sanders' Figure 7b.

RMSE against the eight tabulated rate points is 0.9-7.3 mAh/g on capacities
of 53-168 mAh/g.

### The predicted design map

Running the calibrated model forward over sizes Sanders did not make gives an
optimum, and the optimum is set by how conductive the carbon network *inside*
the particle is - which is precisely the knob the paper turns by co-assembling
CNTs with the LTO:

| Intra-particle conductivity | Optimal diameter | Energy density at 20C |
|---|---|---|
| 10⁻⁴ S m⁻¹ | 8 µm | 74 Wh/L |
| 10⁻³ S m⁻¹ (pyrolysed carbon only) | 12 µm | 81 Wh/L |
| 10⁻² S m⁻¹ (CNT co-assembled) | 20 µm | 83 Wh/L |

against 24 Wh/L for the unstructured baseline. Sanders' 8 µm sits at or just
below the optimum for a plain pyrolysed-carbon particle, and the model says CNT
co-assembly is what would let the particles get usefully bigger.

### Primary size at a fixed 8 µm secondary particle

Because the calibration shows the templated secondary particle diffuses over
its *primary* crystallites, the primary radius is the length that actually sets
the rate limit - and droplet templating can change it independently, by
emulsifying a different precursor powder. Holding the secondary at the 8.0 µm
Sanders et al. made:

| Primary diameter | `r²/D` | Capacity at 20C |
|---|---|---|
| 50 nm | 26 s | 165 mAh/g |
| 120 nm | 147 s | 157 mAh/g |
| **172 nm (Sanders' powder)** | **303 s** | **146 mAh/g** (148 measured) |
| 400 nm | 1637 s | 80 mAh/g |
| 1000 nm | 10 200 s | 25 mAh/g |

The knee sits where `r²/D` crosses the discharge duration (180 s at 20C), and
20C performance saturates within 2 % of its ceiling below about **50 nm** -
past that, diffusion is no longer the limit and there is nothing left to win.
Sanders' 172 nm powder sits just past the knee, so the model says a finer
precursor is the single biggest remaining lever at 20C, worth roughly
+19 mAh/g.

Two bounds are worth stating with that. Above ~500 nm the dispersed solid stops
being "more than an order of magnitude smaller than the membrane pores", which
the SI flags as where blockages become a risk. Below, the model sees no
penalty for going finer, but it only simulates a single discharge: the higher
surface area of a finer powder is exactly what the paper associates with
SEI-like film formation and gassing on LTO, and that is a degradation
mechanism this model does not contain.

What is held fixed for this sweep, and why: electrode packing (set by the
secondary particle, which is not changing); intra-particle porosity (random
packing of monodisperse spheres has a size-independent void fraction, so what
changes is pore *size*, not pore *fraction*); and the intrinsic
exchange-current density. That last one matters - the measured 20.5 Ω
charge-transfer resistance is a *cell-level* number, and halving the primary
radius doubles the interfacial area, so holding the resistance fixed would
have quietly halved the intrinsic rate constant and penalised fine powders for
no physical reason. It is rescaled in proportion to the radius instead.

### The joint map: both scales at once

`run_scale_map.py` sweeps primary and secondary together. It is run twice,
because the answer turns on one assumption the paper cannot constrain:

- **Scales independent** - the model exactly as calibrated. Nothing in it
  couples the two: the primary sets diffusion length and interfacial area, the
  secondary sets packing and intra-particle transport. The map is therefore
  separable, and the best primary is simply the smallest one simulated.
- **Fixed carbon budget** - a secondary particle carries a fixed carbon *weight*
  fraction (2.2 wt % by TGA for the plain microparticles), so finer
  crystallites spread it over more internal surface and the intra-particle
  network thins out. This is plausible but **unmeasured**, and it is the only
  thing in the model that puts a floor under how fine the primary should be.

| | Optimum | Energy density | Sanders' 172 nm x 8 µm |
|---|---|---|---|
| Independent, 5C | 30 nm x 20 µm | 106 Wh/L | 96 (90 %) |
| Independent, 20C | 30 nm x 12 µm | 96 Wh/L | 78 (81 %) |
| Carbon-limited, 5C | 80 nm x 20 µm | 103 Wh/L | 96 (93 %) |
| Carbon-limited, 20C | 50 nm x 12 µm | 91 Wh/L | 78 (86 %) |

Three things survive both assumptions, and only those should be quoted:

1. **The secondary optimum is 12-20 µm at every primary size and both rates** -
   larger than the 8 µm Sanders made, and the ridge is broad rather than
   peaked, so 8-20 µm is all much the same.
2. **The primary axis matters far more than the secondary axis, and more so
   the harder you push.** At 20C, going from 600 nm to 50 nm primaries (at a
   fixed 8 µm secondary) changes energy density by 4.0x, while going from 2 µm
   to 20 µm secondaries changes it by 1.2x. At 5C the same comparison is 1.6x
   against 1.2x - the secondary particle is what buys you packing density, the
   primary is what buys you rate.
3. **Sanders' 172 nm x 8 µm sits at 81-93 % of the model optimum**, and the
   shortfall is almost entirely in the primary direction.

Where the assumption bites: without coupling the primary optimum runs off the
bottom of the grid (smaller is always better); with it, an interior optimum
appears at 50-80 nm. So "go finer" is robust; "stop at 50 nm" is not.

## Caveats, stated plainly

- **Volumetric energy densities are on an electrode-volume basis** (areal
  loading / cast density) and come out about 1.6x below the paper's tabulated
  Wh/L. The paper's Wh/L could not be reproduced from its own tabulated
  capacities, loadings and thicknesses on any basis tried here, and the ratio
  is not constant across its electrodes (1.55-1.74x), so the volume basis
  differs somewhere. **Compare trends and ratios, not absolute Wh/L.** The
  measured 20C ratio of structured to unstructured is 3.3x; the model gives
  3.3x on its own basis.
- **The intra-particle conductivity is the weakest input.** The paper measures
  electrode-level sheet resistance, not conductivity inside one microparticle,
  so it is swept over two decades rather than asserted. It sets *where* the
  optimum in the design map falls, so that curve is a shape, not a number.
- **The LTO OCP is a shape model** - a flat 1.55 V plateau with two
  exponential end walls - not a fit to a digitised OCV curve.
- **The two-phase transformation is modelled as Fickian diffusion** with an
  effective diffusivity, as is standard for LTO but not strictly correct.
- **The intra-particle transport term is a lumped series resistance** derived
  from the volume-averaged drop in a uniformly reacting sphere, not a resolved
  second particle scale.
- **The carbon-budget coupling in the joint map is an assumption, not a
  measurement.** The paper reports electrode-level sheet resistance, not
  conductivity inside one microparticle, and uses a single primary size
  throughout, so nothing constrains it. Any interior optimum in the primary
  direction is a consequence of it.
- **The primary-size sweep assumes a size-independent diffusivity**, fitted at
  172 nm. For a two-phase material whose kinetics are transformation-limited
  that is an approximation, and it is the main reason to read the fine end of
  that sweep as a trend rather than a number.
- **The final 0.1C recovery block** in every step plot is reconstructed from
  the initial 0.1C value; Tables S8/S9 stop at 20C. It is labelled as such
  wherever it appears.
- **Table S9 names one electrode "Unstructured LTO/CNT/PVP"** where Table S8
  calls it "Unstructured LTO/CNT". Table S7 lists only one such electrode, so
  they are treated as the same sample.

## Files

```
sanders_data.py       every number transcribed from the paper, with its table cited
lto_model.py          the LTO OCP, the hierarchical structure physics, the PyBaMM builder
calibrate.py          the three-parameter calibration
run_design_sweep.py   the forward sweep over secondary size
run_primary_sweep.py  the forward sweep over primary size at a fixed 8 um secondary
run_scale_map.py      the joint primary x secondary sweep
plot_style.py         shared figure styling and step-plot construction
plot_experimental.py  replots of the paper's step figures
plot_calibration.py   model vs experiment
plot_design_map.py    the design map
plot_primary_sweep.py the primary-size figure
plot_scale_map.py     the joint design map
run_all.py            all of the above, in order
```

Colours come from a categorical palette whose first three slots are documented
as separable under colour-vision deficiency for all pairs, so no panel uses
more than three hues for identity; where four series are needed, two hues are
crossed with a line style, and every series is named in a legend.
