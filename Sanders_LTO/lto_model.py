"""A hierarchical PyBaMM model of LTO secondary-particle size and rate.

The model is a PyBaMM DFN half cell (LTO working electrode vs Li metal) that
represents the *two length scales* Sanders et al. control independently:

* the **primary** LTO nanocrystal (172 nm Z-average, Figure S1) - identical in
  every sample in the paper, because every sample is templated from the same
  size-selected nanopowder;
* the **secondary** microparticle (0 - 30 um) - the droplet-templated assembly
  of those nanocrystals, which is what membrane emulsification actually sets.

Because the primary size is fixed, a naive "particle radius = the thing being
swept" model gets the paper's central result exactly backwards: it predicts
that the unstructured nanopowder should have the *best* rate performance,
whereas Sanders et al. measure the worst by a wide margin (53 vs 148 mAh/g at
20C, Table S8).  Secondary structuring therefore has to enter the model
through the mechanisms it genuinely changes:

1. **Effective lithium diffusion domain.**  In a droplet-templated secondary
   particle the internal pore network is open and electrolyte-filled, so every
   primary nanocrystal is wetted and the solid-state diffusion length is the
   *primary* radius, independent of secondary size.  Uncontrolled aggregation
   of the bare nanopowder (which the paper documents directly: hard aggregates
   that "could not easily be dispersed", gel-like slurries, a *lower* BET area
   than the templated microparticles) produces dense aggregates that
   electrolyte cannot penetrate, so the diffusion domain becomes the aggregate,
   not the crystallite.  This is the single structure-sensitive parameter the
   model fits to data (see :mod:`calibrate`).
2. **Intra-particle transport.**  Inside a porous secondary particle, electrons
   must reach each primary through the pyrolysed-carbon / CNT network and Li+
   through the intra-particle electrolyte.  Both give a drop that scales as
   ``R_secondary ** 2``, so they set an upper bound on useful secondary size.
3. **Electrode packing.**  Uniform microparticles pack far more densely than
   nanopowder (Table S6: 0.50 vs 0.39 g/cm3), which for a fixed areal loading
   makes a thinner electrode and a higher volumetric energy density.

Everything else - loadings, thicknesses, porosities, series resistances,
charge-transfer resistances - is read from Sanders' own measured tables in
:mod:`sanders_data` rather than tuned.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pybamm
import sanders_data as sd

if TYPE_CHECKING:
    from collections.abc import Sequence

FARADAY = 96485.33212
GAS_CONSTANT = 8.314462618

RESULTS_DIR = Path(__file__).parent / "results"
CALIBRATION_FILE = RESULTS_DIR / "calibration.json"

DISC_AREA_M2 = np.pi * (sd.COIN_CELL_DISC_DIAMETER_M / 2) ** 2

#: Molar mass of Li4Ti5O12 [kg/mol].
LTO_MOLAR_MASS_KG_MOL = 0.45909
#: Lithium inserted per formula unit on the 1.55 V plateau.
LTO_LI_PER_FORMULA_UNIT = 3.0

#: Maximum lithium concentration in the LTO solid phase [mol/m3].
#: rho * n_Li / M = 3430 * 3 / 0.45909, which reproduces 175 mAh/g exactly.
LTO_MAX_CONCENTRATION_MOL_M3 = (
    sd.LTO_CRYSTAL_DENSITY_KG_M3
    * LTO_LI_PER_FORMULA_UNIT
    / LTO_MOLAR_MASS_KG_MOL
)


def lto_open_circuit_potential(sto: Any) -> Any:
    """Open-circuit potential of spinel LTO vs Li/Li+ [V].

    LTO is a two-phase insertion material: the spinel/rock-salt transformation
    pins the potential at a famously flat ~1.55 V plateau across almost the
    whole composition range, with steep excursions only as the end members are
    approached.  This analytic form encodes that shape - a constant plateau
    plus two exponential end walls - and reproduces the flat plateaus and sharp
    terminations of the measured discharge curves in Figure 6a-c.  It is a
    shape model, not a fit to a digitised OCV curve.

    Args:
        sto: Stoichiometry of the working electrode, 0 (fully delithiated,
            Li4Ti5O12) to 1 (fully lithiated, Li7Ti5O12).  May be a float,
            an array, or a PyBaMM symbol.

    Returns:
        The open-circuit potential in volts.

    Examples:
        >>> round(float(lto_open_circuit_potential(0.5).evaluate()), 3)
        1.55
    """
    return (
        1.55
        + 0.80 * pybamm.exp(-60.0 * sto)
        - 0.70 * pybamm.exp(-60.0 * (1.0 - sto))
        - 0.02 * (sto - 0.5)
    )


@dataclass(frozen=True)
class Calibration:
    """The handful of quantities that are fitted rather than measured.

    Attributes:
        lto_diffusivity_m2_s: Effective solid-state Li diffusivity in the LTO
            primary crystallites.  Fitted once, globally, against the
            droplet-templated electrodes (whose diffusion domain is known to be
            the measured primary radius).  Literature values for LTO span
            roughly 1e-16 to 1e-18 m2/s.
        accessible_stoichiometry_span: Fraction of the theoretical 175 mAh/g
            that is electrochemically accessible within the 1.0-2.5 V window.
            Set so the model's 0.1C capacity matches the measured 0.1C
            capacity; it absorbs the small inactive/irreversible fraction that
            every real electrode has.
        aggregate_radius_m: Fitted effective diffusion-domain radius of the
            *unstructured* nanopowder, i.e. how big its uncontrolled dry
            aggregates behave electrochemically.  Compare against the primary
            radius of 86 nm.
    """

    lto_diffusivity_m2_s: float = 2.2e-17
    accessible_stoichiometry_span: float = 0.964
    aggregate_radius_m: float = 2.5e-7

    @classmethod
    def load(cls, path: Path = CALIBRATION_FILE) -> Calibration:
        """Load a calibration written by :mod:`calibrate`, or the defaults.

        Args:
            path: JSON file to read.

        Returns:
            The stored calibration, or the module defaults if no file exists.
        """
        if not path.exists():
            return cls()
        stored = json.loads(path.read_text())
        fields = {k: stored[k] for k in cls().__dict__ if k in stored}
        return cls(**fields)

    def save(self, path: Path = CALIBRATION_FILE) -> None:
        """Write this calibration to disk as JSON.

        Args:
            path: Destination file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


@dataclass(frozen=True)
class Structure:
    """The structural description of one electrode, fed to the model.

    Attributes:
        primary_radius_m: Radius of the LTO primary nanocrystals.
        diffusion_radius_m: Effective solid-state lithium diffusion domain
            radius.  Equal to ``primary_radius_m`` for a templated, porous,
            electrolyte-filled secondary particle; larger for an uncontrolled
            dry aggregate.
        secondary_diameter_m: Secondary particle diameter (0 for unstructured
            material, which has no secondary particle to speak of).
        intra_particle_porosity: Pore fraction inside a secondary particle.
        intra_particle_conductivity_s_m: Effective electronic conductivity of
            the carbon network *inside* a secondary particle.
        thickness_m: Cast electrode thickness.
        active_loading_kg_m2: Active-material areal loading.
        porosity: Inter-particle (electrode-scale) porosity.
        series_resistance_ohm: Measured ohmic + contact resistance of the cell.
        charge_transfer_resistance_ohm: Measured mid-frequency EIS resistance,
            used to set the exchange-current density.
    """

    primary_radius_m: float
    diffusion_radius_m: float
    secondary_diameter_m: float
    intra_particle_porosity: float
    intra_particle_conductivity_s_m: float
    thickness_m: float
    active_loading_kg_m2: float
    porosity: float
    series_resistance_ohm: float
    charge_transfer_resistance_ohm: float


# --------------------------------------------------------------------------
# Structure -> parameter physics
# --------------------------------------------------------------------------

#: Default electronic conductivity of the intra-particle carbon network [S/m].
#: A pyrolysed-surfactant carbon skeleton at the ~2.2 wt % measured by TGA for
#: the plain LTO microparticles.  This is an order-of-magnitude model input,
#: not a measurement, so :mod:`run_design_sweep` sweeps it over two decades.
DEFAULT_INTRA_PARTICLE_CONDUCTIVITY_S_M = 1e-3

#: Intra-particle porosity of a droplet-templated microparticle.  Estimated
#: from the measured tap density (0.78 g/cm3, Table S13) against the LTO
#: crystal density and a typical random-packing inter-particle void fraction.
DEFAULT_INTRA_PARTICLE_POROSITY = 0.5

#: Pore-network percolation threshold, below which electrolyte no longer
#: penetrates a secondary particle and the diffusion domain collapses onto the
#: whole aggregate rather than the individual primary crystallite.
PORE_PERCOLATION_THRESHOLD = 0.12
PORE_PERCOLATION_WIDTH = 0.13


def electrolyte_accessible_fraction(intra_particle_porosity: float) -> float:
    """Fraction of a secondary particle's interior wetted by electrolyte.

    A porous, templated microparticle is fully wetted; a compacted dry
    aggregate is not wetted at all.  The transition is modelled as a
    percolation-like ramp in intra-particle porosity.

    Args:
        intra_particle_porosity: Pore fraction inside the secondary particle.

    Returns:
        A value in [0, 1]: 1 when the internal pore network percolates.

    Examples:
        >>> electrolyte_accessible_fraction(0.5)
        1.0
        >>> electrolyte_accessible_fraction(0.05)
        0.0
    """
    ramp = (
        intra_particle_porosity - PORE_PERCOLATION_THRESHOLD
    ) / PORE_PERCOLATION_WIDTH
    return float(np.clip(ramp, 0.0, 1.0))


def effective_diffusion_radius(
    *,
    primary_radius_m: float,
    secondary_diameter_m: float,
    intra_particle_porosity: float,
) -> float:
    """Effective solid-state lithium diffusion domain radius.

    Interpolates between the two physical limits: a fully wetted porous
    agglomerate diffuses over its primary crystallites, while a non-wetted
    dense aggregate must diffuse across the whole aggregate.

    Args:
        primary_radius_m: Radius of the primary nanocrystals.
        secondary_diameter_m: Diameter of the secondary particle or aggregate.
        intra_particle_porosity: Pore fraction inside the secondary particle.

    Returns:
        The effective diffusion domain radius in metres.

    Examples:
        >>> round(
        ...     effective_diffusion_radius(
        ...         primary_radius_m=86e-9,
        ...         secondary_diameter_m=8e-6,
        ...         intra_particle_porosity=0.5,
        ...     )
        ...     * 1e9
        ... )
        86
    """
    wetted = electrolyte_accessible_fraction(intra_particle_porosity)
    secondary_radius_m = max(secondary_diameter_m / 2.0, primary_radius_m)
    return primary_radius_m * wetted + secondary_radius_m * (1.0 - wetted)


def intra_particle_resistance_ohm(  # noqa: PLR0913
    *,
    secondary_diameter_m: float,
    intra_particle_porosity: float,
    intra_particle_conductivity_s_m: float,
    electrode_thickness_m: float,
    electrode_porosity: float,
    electrolyte_conductivity_s_m: float = 1.0,
    disc_area_m2: float = DISC_AREA_M2,
) -> float:
    """Lumped cell resistance from transport *inside* secondary particles.

    Reaction current is generated uniformly through a secondary particle and
    has to be collected at its surface, electronically through the internal
    carbon network and ionically through the internal electrolyte.  For a
    sphere with uniform generation the volume-averaged potential drop is
    ``i_v R**2 / (15 sigma)``, which converts to a cell-level series
    resistance that grows as the square of the secondary radius.  It is
    negligible at Sanders' 8-10 um particles and is what ultimately caps how
    large a useful secondary particle can be.

    Args:
        secondary_diameter_m: Secondary particle diameter.
        intra_particle_porosity: Pore fraction inside the secondary particle.
        intra_particle_conductivity_s_m: Electronic conductivity of the
            intra-particle carbon network.
        electrode_thickness_m: Cast electrode thickness.
        electrode_porosity: Inter-particle electrode porosity.
        electrolyte_conductivity_s_m: Bulk electrolyte conductivity.
        disc_area_m2: Geometric electrode area.

    Returns:
        The equivalent series resistance in ohms.

    Examples:
        >>> round(
        ...     intra_particle_resistance_ohm(
        ...         secondary_diameter_m=8e-6,
        ...         intra_particle_porosity=0.5,
        ...         intra_particle_conductivity_s_m=1e-3,
        ...         electrode_thickness_m=30.7e-6,
        ...         electrode_porosity=0.82,
        ...     ),
        ...     2,
        ... )
        1.1
    """
    if secondary_diameter_m <= 0.0:
        return 0.0
    secondary_radius_m = secondary_diameter_m / 2.0
    agglomerate_volume_fraction = max(1.0 - electrode_porosity, 1e-3)
    # An insertion reaction needs an electron *and* a Li+ delivered to the
    # same site, so the electronic drop through the intra-particle carbon and
    # the ionic drop through the intra-particle electrolyte add: the two
    # resistances are in series, i.e. their reciprocal conductivities sum.
    ionic = electrolyte_conductivity_s_m * intra_particle_porosity**1.5
    inverse_conductivity = 1.0 / intra_particle_conductivity_s_m + 1.0 / ionic
    return (
        secondary_radius_m**2
        * inverse_conductivity
        / (
            15.0
            * disc_area_m2
            * electrode_thickness_m
            * agglomerate_volume_fraction
        )
    )


def fit_saturating_property(
    diameters_um: Sequence[float],
    values: Sequence[float],
) -> tuple[float, float, float]:
    """Fit an electrode property against secondary particle diameter.

    Several of Sanders' measured electrode properties change monotonically
    and then saturate as loose nanopowder is replaced by progressively larger
    secondary particles: the cast electrode densifies (Table S6, 0.39 ->
    0.50 g/cm3) while the ohmic and charge-transfer resistances fall
    (Table S10).  All are described by the same saturating form,

    ``v(d) = v_inf - (v_inf - v_0) * exp(-d / d_scale)``,

    where ``v_0`` is the unstructured (``d = 0``) value.  With ``v_0`` pinned
    to the measured unstructured electrode, only the amplitude and length
    scale are fitted, from two or three structured points.

    Args:
        diameters_um: Secondary particle diameters, including 0 for the
            unstructured nanopowder.
        values: The corresponding measured property values.

    Returns:
        A tuple ``(v_0, v_inf, d_scale)``, the last in micrometres.

    Examples:
        >>> v0, vinf, scale = fit_saturating_property(
        ...     [0.0, 2.3, 8.0], [0.39, 0.44, 0.50]
        ... )
        >>> round(v0, 2), round(vinf, 2), round(scale, 1)
        (0.39, 0.53, 5.2)
    """
    diameters = np.asarray(diameters_um, dtype=float)
    measured = np.asarray(values, dtype=float)
    value_0 = float(measured[diameters == 0.0].mean())

    structured = diameters > 0.0
    best: tuple[float, float, float] = (value_0, value_0, 1.0)
    best_cost = np.inf
    for d_scale in np.linspace(0.3, 40.0, 800):
        basis = 1.0 - np.exp(-diameters[structured] / d_scale)
        target = measured[structured] - value_0
        amplitude = float(basis @ target / (basis @ basis))
        cost = float(np.sum((amplitude * basis - target) ** 2))
        if cost < best_cost:
            best_cost = cost
            best = (value_0, value_0 + amplitude, float(d_scale))
    return best


def evaluate_saturating_property(
    diameter_um: float,
    model: tuple[float, float, float],
) -> float:
    """Evaluate a correlation fitted by :func:`fit_saturating_property`.

    Args:
        diameter_um: Secondary particle diameter in micrometres.
        model: ``(v_0, v_inf, d_scale)``.

    Returns:
        The interpolated property value.

    Examples:
        >>> round(evaluate_saturating_property(0.0, (0.39, 0.53, 5.2)), 3)
        0.39
    """
    value_0, value_inf, d_scale = model
    return value_inf - (value_inf - value_0) * float(
        np.exp(-diameter_um / d_scale)
    )


def rescale_primary_radius(
    structure: Structure,
    new_primary_radius_m: float,
) -> Structure:
    """Change a templated particle's primary size at fixed intrinsic kinetics.

    Two things have to move together when the primary crystallite is resized
    inside a fixed secondary particle.

    The diffusion domain follows the primary radius, because a templated,
    electrolyte-filled secondary particle diffuses over its crystallites -
    which is what the calibration recovered independently for every templated
    sample in the paper.

    The measured charge-transfer resistance must be rescaled.  Sanders' EIS
    gives a *cell-level* resistance for an electrode of a known primary size;
    the intrinsic quantity behind it is the exchange-current density per unit
    interfacial area.  Halving the primary radius doubles the interfacial area
    at fixed loading, so the same intrinsic kinetics would show half the
    charge-transfer resistance.  Since
    :func:`build_parameter_values` back-computes the rate constant as
    ``k ~ r / R_ct``, holding ``R_ct`` fixed while shrinking ``r`` would
    quietly halve the intrinsic rate constant and penalise small crystallites
    for no physical reason.  Scaling ``R_ct`` in proportion to the radius
    keeps ``k`` - the thing that is actually a material property - constant.

    Args:
        structure: A templated electrode structure to start from.
        new_primary_radius_m: The primary crystallite radius to move to.

    Returns:
        A new structure with the primary radius, diffusion domain and
        charge-transfer resistance all consistent.

    Examples:
        >>> base = Structure(
        ...     primary_radius_m=86e-9,
        ...     diffusion_radius_m=86e-9,
        ...     secondary_diameter_m=8e-6,
        ...     intra_particle_porosity=0.5,
        ...     intra_particle_conductivity_s_m=1e-3,
        ...     thickness_m=30.7e-6,
        ...     active_loading_kg_m2=1.22e-2,
        ...     porosity=0.82,
        ...     series_resistance_ohm=8.84,
        ...     charge_transfer_resistance_ohm=20.5,
        ... )
        >>> halved = rescale_primary_radius(base, 43e-9)
        >>> round(halved.charge_transfer_resistance_ohm, 2)
        10.25
    """
    scale = new_primary_radius_m / structure.primary_radius_m
    return dataclasses.replace(
        structure,
        primary_radius_m=new_primary_radius_m,
        diffusion_radius_m=new_primary_radius_m,
        charge_transfer_resistance_ohm=(
            structure.charge_transfer_resistance_ohm * scale
        ),
    )


def carbon_limited_intra_particle_conductivity(
    primary_radius_m: float,
    *,
    reference_conductivity_s_m: float = (
        DEFAULT_INTRA_PARTICLE_CONDUCTIVITY_S_M
    ),
    reference_primary_radius_m: float = 86e-9,
) -> float:
    """Intra-particle conductivity when the carbon budget is held fixed.

    This is the one place where the two length scales genuinely couple, and it
    is switched on explicitly rather than assumed.

    A droplet-templated microparticle carries a fixed *weight* fraction of
    carbon (2.2 wt % of pyrolysed surfactant for the plain LTO particles, by
    TGA; 4.7 and 9.2 wt % for the CNT composites).  That carbon has to coat an
    internal surface area that scales as ``1 / r_primary``, so at fixed carbon
    mass the coating thickness - and with it the conductance of the percolating
    network - scales roughly as ``r_primary``.  Halving the crystallite size
    therefore buys a 4x shorter diffusion time but spreads the same carbon
    twice as thin.

    **This scaling is not measured.**  Sanders et al. report electrode-level
    sheet resistance, not conductivity inside a single microparticle, and they
    use one primary size throughout, so nothing in the paper constrains it.
    Linear scaling is the simplest defensible form.  Treat any interior
    optimum it produces in the primary direction as a consequence of this
    assumption, not as a result - which is why
    :mod:`run_scale_map` maps the design space both with it and without it.

    Args:
        primary_radius_m: Primary crystallite radius.
        reference_conductivity_s_m: Conductivity at the reference radius.
        reference_primary_radius_m: Radius at which the reference conductivity
            applies, by default Sanders' 86 nm.

    Returns:
        The scaled intra-particle electronic conductivity in S/m.

    Examples:
        >>> round(carbon_limited_intra_particle_conductivity(43e-9), 8)
        0.0005
        >>> round(carbon_limited_intra_particle_conductivity(86e-9), 8)
        0.001
    """
    return (
        reference_conductivity_s_m
        * primary_radius_m
        / reference_primary_radius_m
    )


def diffusion_time_constant_s(
    radius_m: float, diffusivity_m2_s: float
) -> float:
    """Characteristic solid-state diffusion time of a sphere.

    Args:
        radius_m: Particle radius.
        diffusivity_m2_s: Solid-state diffusivity.

    Returns:
        ``r**2 / D`` in seconds, the timescale to compare against the
        discharge duration ``3600 / C``.

    Examples:
        >>> round(diffusion_time_constant_s(86e-9, 2.44e-17))
        303
    """
    return radius_m**2 / diffusivity_m2_s


# --------------------------------------------------------------------------
# Building the PyBaMM half cell
# --------------------------------------------------------------------------


def structure_from_electrode(
    key: str,
    calibration: Calibration,
    *,
    intra_particle_conductivity_s_m: float = (
        DEFAULT_INTRA_PARTICLE_CONDUCTIVITY_S_M
    ),
) -> Structure:
    """Build a :class:`Structure` from one of Sanders' measured electrodes.

    Args:
        key: Key into :data:`sanders_data.ELECTRODES`.
        calibration: Fitted quantities (diffusivity, aggregate radius).
        intra_particle_conductivity_s_m: Intra-particle carbon conductivity.

    Returns:
        The structural description of that electrode.
    """
    electrode = sd.ELECTRODES[key]
    primary_radius_m = sd.PRIMARY_PARTICLE_Z_AVERAGE_M / 2.0
    is_structured = electrode.secondary_diameter_um > 0.0

    if is_structured:
        intra_porosity = DEFAULT_INTRA_PARTICLE_POROSITY
        diffusion_radius_m = primary_radius_m
    else:
        # Unstructured nanopowder: dense, non-wetted aggregates.  Their size
        # is the one structure-sensitive quantity the model fits.
        intra_porosity = 0.0
        diffusion_radius_m = calibration.aggregate_radius_m

    # Where the paper reports no EIS fit for an electrode, fall back to the
    # values of the closest electrode it did fit (structured vs unstructured).
    fallback = "uniform" if is_structured else "nano"
    reference = sd.ELECTRODES[fallback]
    r0 = electrode.eis_r0_ohm or reference.eis_r0_ohm
    r1 = electrode.eis_r1_ohm or reference.eis_r1_ohm
    r3 = electrode.eis_r3_ohm or reference.eis_r3_ohm

    return Structure(
        primary_radius_m=primary_radius_m,
        diffusion_radius_m=diffusion_radius_m,
        secondary_diameter_m=electrode.secondary_diameter_um * 1e-6,
        intra_particle_porosity=intra_porosity,
        intra_particle_conductivity_s_m=intra_particle_conductivity_s_m,
        thickness_m=electrode.thickness_um * 1e-6,
        active_loading_kg_m2=electrode.active_loading_mg_cm2 * 1e-2,
        porosity=electrode.porosity,
        series_resistance_ohm=float(r0) + float(r1),
        charge_transfer_resistance_ohm=float(r3),
    )


def build_parameter_values(
    structure: Structure,
    calibration: Calibration,
) -> pybamm.ParameterValues:
    """Assemble the LTO-vs-Li half-cell parameter set.

    An existing lithium-metal half-cell parameter set supplies the electrolyte,
    separator and lithium counter-electrode physics; every working-electrode
    entry is then replaced with LTO values derived from ``structure``.

    Args:
        structure: Structural description of the electrode.
        calibration: Fitted diffusivity and accessible stoichiometry span.

    Returns:
        A PyBaMM parameter set ready to solve.
    """
    values = pybamm.ParameterValues("OKane2022_graphite_SiOx_halfcell")

    # Coin-cell geometry: PyBaMM's 1D models use height x width as the area.
    values["Electrode height [m]"] = DISC_AREA_M2
    values["Electrode width [m]"] = 1.0
    values["Number of electrodes connected in parallel to make a cell"] = 1.0
    values["Ambient temperature [K]"] = sd.CELL_TEMPERATURE_K
    values["Initial temperature [K]"] = sd.CELL_TEMPERATURE_K
    values["Reference temperature [K]"] = sd.CELL_TEMPERATURE_K
    values["Lower voltage cut-off [V]"] = sd.VOLTAGE_CUT_OFF_LOW_V
    values["Upper voltage cut-off [V]"] = sd.VOLTAGE_CUT_OFF_HIGH_V
    values["Separator thickness [m]"] = 25e-6  # Celgard 2325, Methods

    # Electrode geometry from the measured loading, thickness and porosity.
    active_volume_fraction = structure.active_loading_kg_m2 / (
        sd.LTO_CRYSTAL_DENSITY_KG_M3 * structure.thickness_m
    )
    values["Positive electrode thickness [m]"] = structure.thickness_m
    values["Positive electrode active material volume fraction"] = (
        active_volume_fraction
    )
    values["Positive electrode porosity"] = structure.porosity
    values["Positive electrode density [kg.m-3]"] = sd.LTO_CRYSTAL_DENSITY_KG_M3

    # LTO thermodynamics and solid-state transport.
    values["Positive electrode OCP [V]"] = lto_open_circuit_potential
    values["Positive electrode OCP entropic change [V.K-1]"] = 0.0
    values["Maximum concentration in positive electrode [mol.m-3]"] = (
        LTO_MAX_CONCENTRATION_MOL_M3
    )
    initial_stoichiometry = 1.0 - calibration.accessible_stoichiometry_span
    values["Initial concentration in positive electrode [mol.m-3]"] = (
        initial_stoichiometry * LTO_MAX_CONCENTRATION_MOL_M3
    )
    values["Positive particle radius [m]"] = structure.diffusion_radius_m
    values["Positive particle diffusivity [m2.s-1]"] = (
        calibration.lto_diffusivity_m2_s
    )
    values["Positive electrode conductivity [S.m-1]"] = 100.0

    # Interfacial kinetics, set so that the linearised charge-transfer
    # resistance of the model equals the measured EIS value at 50 % SOC.
    specific_surface_area = (
        3.0 * active_volume_fraction / (structure.diffusion_radius_m)
    )
    reaction_area_m2 = (
        specific_surface_area * structure.thickness_m * DISC_AREA_M2
    )
    exchange_current_at_half_soc = (
        GAS_CONSTANT
        * sd.CELL_TEMPERATURE_K
        / (
            FARADAY
            * structure.charge_transfer_resistance_ohm
            * reaction_area_m2
        )
    )
    half_max = 0.5 * LTO_MAX_CONCENTRATION_MOL_M3
    reference_electrolyte_concentration = 1000.0
    rate_constant = exchange_current_at_half_soc / (
        reference_electrolyte_concentration**0.5 * half_max
    )

    def exchange_current_density(
        c_e: Any,
        c_s_surf: Any,
        c_s_max: Any,
        temperature: Any,  # noqa: ARG001
    ) -> Any:
        """Symmetric Butler-Volmer exchange-current density [A/m2]."""
        return (
            rate_constant
            * c_e**0.5
            * c_s_surf**0.5
            * (c_s_max - c_s_surf) ** 0.5
        )

    values["Positive electrode exchange-current density [A.m-2]"] = (
        exchange_current_density
    )
    values["Positive electrode charge transfer coefficient"] = 0.5

    # Measured ohmic/contact resistance plus the derived intra-particle term.
    values["Contact resistance [Ohm]"] = (
        structure.series_resistance_ohm
        + intra_particle_resistance_ohm(
            secondary_diameter_m=structure.secondary_diameter_m,
            intra_particle_porosity=structure.intra_particle_porosity,
            intra_particle_conductivity_s_m=(
                structure.intra_particle_conductivity_s_m
            ),
            electrode_thickness_m=structure.thickness_m,
            electrode_porosity=structure.porosity,
        )
    )
    return values


def active_mass_kg(structure: Structure) -> float:
    """Active-material mass on one 15 mm disc.

    Args:
        structure: Structural description of the electrode.

    Returns:
        Mass of LTO in kilograms.
    """
    return structure.active_loading_kg_m2 * DISC_AREA_M2


def simulate_discharge(
    structure: Structure,
    calibration: Calibration,
    c_rate: float,
) -> dict[str, float]:
    """Solve one constant-current discharge and summarise it.

    Args:
        structure: Structural description of the electrode.
        calibration: Fitted model quantities.
        c_rate: Discharge rate, as a multiple of the 175 mAh/g theoretical
            capacity (which is how Sanders et al. define C).

    Returns:
        A dictionary with the delivered specific capacity [mAh/g], mean
        discharge voltage [V], volumetric energy density [Wh/L] on an
        electrode-volume basis, and the areal capacity [mAh/cm2].  All values
        are zero if the cell cannot sustain the current at all.
    """
    values = build_parameter_values(structure, calibration)
    mass_kg = active_mass_kg(structure)
    theoretical_capacity_ah = (
        sd.LTO_THEORETICAL_CAPACITY_MAH_G * 1e-3 * mass_kg * 1e3
    )
    current_a = c_rate * theoretical_capacity_ah
    values["Current function [A]"] = current_a
    values["Nominal cell capacity [A.h]"] = theoretical_capacity_ah

    model = pybamm.lithium_ion.DFN(
        options={
            "working electrode": "positive",
            "contact resistance": "true",
        },
    )
    simulation = pybamm.Simulation(model, parameter_values=values)
    max_time_s = 1.3 * 3600.0 / c_rate
    try:
        solution = simulation.solve([0.0, max_time_s])
    except pybamm.SolverError:
        # The IR drop exceeds the available voltage window at t = 0: the cell
        # genuinely cannot pass this current.  That is a result, not a bug.
        return {
            "specific_capacity_mah_g": 0.0,
            "mean_voltage_v": 0.0,
            "energy_density_wh_l": 0.0,
            "areal_capacity_mah_cm2": 0.0,
        }

    time_h = solution["Time [h]"].entries
    voltage_v = solution["Voltage [V]"].entries
    capacity_ah = current_a * float(time_h[-1])
    energy_wh = float(np.trapezoid(voltage_v * current_a, time_h))

    electrode_volume_l = structure.thickness_m * DISC_AREA_M2 * 1e3
    return {
        "specific_capacity_mah_g": capacity_ah * 1e3 / (mass_kg * 1e3),
        "mean_voltage_v": energy_wh / capacity_ah if capacity_ah else 0.0,
        "energy_density_wh_l": energy_wh / electrode_volume_l,
        "areal_capacity_mah_cm2": capacity_ah * 1e3 / (DISC_AREA_M2 * 1e4),
    }


def rate_curve(
    structure: Structure,
    calibration: Calibration,
    c_rates: Sequence[float] = sd.RATE_TEST_C_RATES,
) -> dict[str, np.ndarray]:
    """Simulate a full rate test.

    Args:
        structure: Structural description of the electrode.
        calibration: Fitted model quantities.
        c_rates: C-rates to simulate, in order.

    Returns:
        A dictionary of arrays keyed as in :func:`simulate_discharge`, plus
        ``"c_rate"``.
    """
    records = [simulate_discharge(structure, calibration, c) for c in c_rates]
    keys = records[0].keys()
    out = {k: np.array([r[k] for r in records]) for k in keys}
    out["c_rate"] = np.asarray(c_rates, dtype=float)
    return out
