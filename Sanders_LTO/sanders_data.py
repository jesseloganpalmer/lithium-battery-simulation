"""Digitised experimental data from Sanders et al., ACS Nano 2026, 20, 7555.

"Microdroplet Templating of Uniform Nanostructured Battery Microparticles
with Scalable Membrane Emulsification"
https://doi.org/10.1021/acsnano.5c17777  (CC-BY 4.0)

Every number in this module is transcribed from the paper or its Supporting
Information; nothing here is fitted, smoothed or invented.  The source table
or figure is named in the comment above each block so any value can be traced
back.  Values that are *derived* (rather than transcribed) live in
:mod:`lto_model`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Cell / protocol constants (Methods, main text)
# ---------------------------------------------------------------------------

COIN_CELL_DISC_DIAMETER_M = 15e-3
"""Punched electrode disc diameter for the rate-test cells (Methods)."""

VOLTAGE_CUT_OFF_LOW_V = 1.0
VOLTAGE_CUT_OFF_HIGH_V = 2.5
"""Cut-off window vs Li/Li+ used for every cycling protocol (Methods)."""

CELL_TEMPERATURE_K = 298.15
"""Cycling chamber temperature, 25 degrees C (Methods)."""

LTO_THEORETICAL_CAPACITY_MAH_G = 175.0
"""Li4Ti5O12 + 3 Li -> Li7Ti5O12, the basis of every C-rate in the paper."""

LTO_CRYSTAL_DENSITY_KG_M3 = 3430.0
"""Spinel Li4Ti5O12 crystallographic density (literature value)."""

ACTIVE_MASS_FRACTION = 0.80
"""80:10:10 active:carbon:binder by weight for all cast electrodes."""

# Figure 1b / Figure S1: DLS of the size-selected LTO nanopowder that is the
# primary building block of *every* sample in the study.
PRIMARY_PARTICLE_Z_AVERAGE_M = 172e-9
PRIMARY_PARTICLE_DISPERSITY = 0.145

RATE_TEST_C_RATES = (0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
"""C-rates of the rate test, in the order applied (Figures 6d/6e, 7b/7c)."""

CYCLES_PER_RATE_BLOCK = 5
"""Table S8 caption: five cycles at each rate, averaged over three cells."""


# ---------------------------------------------------------------------------
# Electrode-level structure and morphology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Electrode:
    """One cast electrode as characterised in the Supporting Information.

    Attributes:
        label: Human-readable name used in figures.
        secondary_diameter_um: Mean secondary (micro)particle diameter.  ``0``
            means the material has no secondary structure at all, i.e. the
            as-received nanopowder cast directly.
        secondary_cv: Coefficient of variation of the secondary particle
            diameter.  ``None`` where the paper does not report one.
        thickness_um: Table S6 average cast thickness.
        active_loading_mg_cm2: Table S6 average active-material areal loading.
        areal_loading_mg_cm2: Table S6 average total (incl. C + binder) areal
            loading.
        electrode_density_g_cm3: Table S6 average cast-electrode density.
        porosity: Table S6 estimated (calculated) electrode porosity.
        tap_density_g_cm3: Figure 5f / Table S13 powder tap density.
        sheet_resistance_ohm_sq: Table S11 four-point-probe sheet resistance.
        sheet_resistance_sd: Standard deviation of the above (Table S11).
        eis_r0_ohm: Table S10 fitted series (ohmic) resistance.
        eis_r1_ohm: Table S10 fitted high-frequency RC resistance.
        eis_r3_ohm: Table S10 fitted mid-frequency (charge-transfer) resistance.
        capacity_retention_pct: Table S12 retention after 500 cycles at 1C.
        capacity_retention_sd: Standard deviation of the above.
    """

    label: str
    secondary_diameter_um: float
    secondary_cv: float | None
    thickness_um: float
    active_loading_mg_cm2: float
    areal_loading_mg_cm2: float
    electrode_density_g_cm3: float
    porosity: float
    tap_density_g_cm3: float | None
    sheet_resistance_ohm_sq: float | None
    sheet_resistance_sd: float | None
    eis_r0_ohm: float | None
    eis_r1_ohm: float | None
    eis_r3_ohm: float | None
    capacity_retention_pct: float
    capacity_retention_sd: float


# Tables S6, S10, S11, S12, S13 and Figures 3, 4, 5f.
# Secondary diameters: Figure 3b/3d and Figure 4b/4d (n = 500 each).
#   bulk (polydisperse) 2.3 +/- 1.5 um (CV 67 %)
#   membrane (uniform)  8.0 +/- 1.4 um (CV 17 %)
#   oxCNT/LTO           9.1 +/- 1.3 um
#   CNT-PVP/LTO         9.3 +/- 1.2 um
# The commercial spray-dried benchmark is quoted by D50 (Table S13), not by a
# mean +/- sd, so its CV is left as None.
ELECTRODES: dict[str, Electrode] = {
    "nano": Electrode(
        label="Unstructured LTO nanoparticles",
        secondary_diameter_um=0.0,
        secondary_cv=None,
        thickness_um=35.7,
        active_loading_mg_cm2=1.11,
        areal_loading_mg_cm2=1.39,
        electrode_density_g_cm3=0.39,
        porosity=0.86,
        tap_density_g_cm3=0.47,  # after the 5 min grinding step (Figure 5f)
        sheet_resistance_ohm_sq=71.5,  # mean of the two sections in Table S11
        sheet_resistance_sd=78.7,  # mean of the two reported sds
        eis_r0_ohm=7.82,
        eis_r1_ohm=11.3,
        eis_r3_ohm=29.5,
        capacity_retention_pct=96.48,
        capacity_retention_sd=1.10,
    ),
    "polydisperse": Electrode(
        label="Bulk-emulsified LTO microparticles (2.3 um, CV 67 %)",
        secondary_diameter_um=2.3,
        secondary_cv=0.67,
        thickness_um=29.6,
        active_loading_mg_cm2=1.04,
        areal_loading_mg_cm2=1.30,
        electrode_density_g_cm3=0.44,
        porosity=0.85,
        tap_density_g_cm3=0.63,
        sheet_resistance_ohm_sq=12.1,
        sheet_resistance_sd=3.0,
        eis_r0_ohm=4.13,
        eis_r1_ohm=8.63,
        eis_r3_ohm=19.6,
        capacity_retention_pct=97.83,
        capacity_retention_sd=0.22,
    ),
    "uniform": Electrode(
        label="Membrane-emulsified LTO microparticles (8.0 um, CV 17 %)",
        secondary_diameter_um=8.0,
        secondary_cv=0.17,
        thickness_um=30.7,
        active_loading_mg_cm2=1.22,
        areal_loading_mg_cm2=1.53,
        electrode_density_g_cm3=0.50,
        porosity=0.82,
        tap_density_g_cm3=0.78,
        sheet_resistance_ohm_sq=15.2,
        sheet_resistance_sd=7.7,
        eis_r0_ohm=3.85,
        eis_r1_ohm=4.99,
        eis_r3_ohm=20.5,
        capacity_retention_pct=97.20,
        capacity_retention_sd=0.28,
    ),
    "oxcnt_mp": Electrode(
        label="Membrane-emulsified LTO/oxCNT microparticles (9.1 um)",
        secondary_diameter_um=9.1,
        secondary_cv=0.14,
        thickness_um=31.5,
        active_loading_mg_cm2=1.27,
        areal_loading_mg_cm2=1.59,
        electrode_density_g_cm3=0.50,
        porosity=0.82,
        tap_density_g_cm3=0.67,
        sheet_resistance_ohm_sq=14.6,
        sheet_resistance_sd=6.9,
        eis_r0_ohm=None,
        eis_r1_ohm=None,
        eis_r3_ohm=None,
        capacity_retention_pct=99.09,
        capacity_retention_sd=1.41,
    ),
    "cnt_pvp_mp": Electrode(
        label="Membrane-emulsified LTO/CNT/PVP microparticles (9.3 um)",
        secondary_diameter_um=9.3,
        secondary_cv=0.13,
        thickness_um=29.0,
        active_loading_mg_cm2=1.04,
        areal_loading_mg_cm2=1.40,
        electrode_density_g_cm3=0.48,
        porosity=0.82,
        tap_density_g_cm3=0.49,
        sheet_resistance_ohm_sq=45.7,
        sheet_resistance_sd=14.5,
        eis_r0_ohm=None,
        eis_r1_ohm=None,
        eis_r3_ohm=None,
        capacity_retention_pct=98.52,
        capacity_retention_sd=1.35,
    ),
    "nano_oxcnt": Electrode(
        label="Unstructured LTO + oxCNT powder mixture",
        secondary_diameter_um=0.0,
        secondary_cv=None,
        thickness_um=43.4,
        active_loading_mg_cm2=1.00,
        areal_loading_mg_cm2=1.25,
        electrode_density_g_cm3=0.29,
        porosity=0.89,
        tap_density_g_cm3=0.43,
        sheet_resistance_ohm_sq=35.3,
        sheet_resistance_sd=17.2,
        eis_r0_ohm=None,
        eis_r1_ohm=None,
        eis_r3_ohm=None,
        capacity_retention_pct=96.16,
        capacity_retention_sd=3.21,
    ),
    "nano_cnt": Electrode(
        label="Unstructured LTO + CNT powder mixture",
        secondary_diameter_um=0.0,
        secondary_cv=None,
        thickness_um=44.4,
        active_loading_mg_cm2=1.05,
        areal_loading_mg_cm2=1.41,
        electrode_density_g_cm3=0.32,
        porosity=0.88,
        tap_density_g_cm3=0.30,
        sheet_resistance_ohm_sq=77.1,
        sheet_resistance_sd=12.1,
        eis_r0_ohm=None,
        eis_r1_ohm=None,
        eis_r3_ohm=None,
        capacity_retention_pct=97.31,
        capacity_retention_sd=1.70,
    ),
    "commercial": Electrode(
        label="Commercial spray-dried LTO microparticles (BTR, D50 4-10 um)",
        secondary_diameter_um=7.0,  # midpoint of the quoted D50 range, Table S13
        secondary_cv=None,
        thickness_um=29.2,
        active_loading_mg_cm2=1.27,
        areal_loading_mg_cm2=1.58,
        electrode_density_g_cm3=0.54,
        porosity=0.81,
        tap_density_g_cm3=0.83,
        sheet_resistance_ohm_sq=None,
        sheet_resistance_sd=None,
        eis_r0_ohm=None,
        eis_r1_ohm=None,
        eis_r3_ohm=None,
        capacity_retention_pct=99.19,
        capacity_retention_sd=0.08,
    ),
}


# ---------------------------------------------------------------------------
# Table S8: average specific capacity [mAh/g] vs C-rate
# ---------------------------------------------------------------------------
# Each value is the mean over three half cells x five cycles (15 cycles).
SPECIFIC_CAPACITY_MAH_G: dict[str, tuple[float, ...]] = {
    "nano": (165.9, 159.6, 154.9, 147.3, 140.9, 129.5, 103.1, 53.2),
    "polydisperse": (167.1, 165.5, 164.4, 162.4, 161.1, 159.3, 156.1, 149.7),
    "uniform": (167.9, 166.4, 165.0, 162.6, 161.2, 159.2, 155.4, 148.2),
    "oxcnt_mp": (165.5, 164.0, 162.6, 160.7, 159.7, 158.4, 155.5, 147.5),
    "cnt_pvp_mp": (168.4, 166.8, 165.1, 162.5, 160.5, 157.7, 151.6, 134.5),
    "nano_oxcnt": (165.6, 160.6, 156.0, 148.6, 142.8, 133.3, 114.1, 73.3),
    "nano_cnt": (166.6, 163.5, 159.9, 154.9, 151.2, 146.0, 139.0, 131.9),
    "commercial": (166.8, 166.2, 165.3, 163.5, 161.5, 156.6, 144.2, 118.4),
}

SPECIFIC_CAPACITY_SD: dict[str, tuple[float, ...]] = {
    "nano": (1.3, 2.3, 3.7, 5.2, 6.2, 7.5, 10.5, 10.5),
    "polydisperse": (0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.5),
    "uniform": (0.3, 0.4, 0.6, 1.0, 1.5, 2.0, 3.1, 4.7),
    "oxcnt_mp": (0.6, 0.6, 0.7, 0.7, 0.6, 0.6, 0.6, 3.0),
    "cnt_pvp_mp": (1.1, 1.4, 2.1, 2.7, 3.2, 3.7, 5.9, 12.1),
    "nano_oxcnt": (0.6, 0.5, 0.7, 0.9, 1.0, 1.0, 1.9, 6.7),
    "nano_cnt": (1.0, 1.1, 1.2, 1.3, 1.3, 0.9, 1.7, 2.4),
    "commercial": (0.5, 0.5, 0.5, 0.6, 0.6, 0.5, 0.6, 7.5),
}

# ---------------------------------------------------------------------------
# Table S9: average volumetric energy density [Wh/L] vs C-rate
# ---------------------------------------------------------------------------
# NOTE ON NOMENCLATURE: Table S9 labels the last unstructured composite row
# "Unstructured LTO/CNT/PVP" where Table S8 calls the same electrode
# "Unstructured LTO/CNT".  Table S7 lists only one unstructured pristine-CNT
# electrode, so the two rows are the same sample and the S8 name is used here.
ENERGY_DENSITY_WH_L: dict[str, tuple[float, ...]] = {
    "nano": (137.3, 132.2, 128.5, 122.4, 117.1, 107.7, 86.2, 44.8),
    "polydisperse": (149.4, 148.0, 147.0, 145.2, 144.0, 142.4, 139.5, 133.7),
    "uniform": (169.7, 168.2, 166.9, 165.2, 164.3, 162.7, 158.7, 146.5),
    "oxcnt_mp": (156.7, 155.3, 154.1, 152.5, 151.7, 150.5, 147.8, 140.4),
    "cnt_pvp_mp": (155.8, 154.4, 152.9, 150.6, 148.8, 146.4, 140.8, 124.9),
    "nano_oxcnt": (96.0, 93.1, 90.4, 86.1, 82.8, 77.3, 66.1, 42.4),
    "nano_cnt": (99.5, 97.7, 95.6, 92.8, 90.7, 88.0, 84.0, 79.6),
    "commercial": (171.8, 171.2, 170.4, 168.7, 166.6, 161.6, 148.7, 122.0),
}

ENERGY_DENSITY_SD: dict[str, tuple[float, ...]] = {
    "nano": (12.7, 14.5, 15.4, 16.0, 16.3, 16.3, 16.5, 12.2),
    "polydisperse": (10.9, 10.5, 10.4, 10.2, 10.1, 9.9, 9.7, 9.4),
    "uniform": (8.8, 8.8, 8.9, 8.8, 8.7, 8.9, 10.1, 16.5),
    "oxcnt_mp": (27.3, 27.2, 27.1, 26.9, 26.6, 26.3, 25.8, 25.0),
    "cnt_pvp_mp": (2.2, 2.6, 3.3, 3.9, 4.3, 4.8, 6.8, 12.1),
    "nano_oxcnt": (3.5, 3.3, 3.0, 2.7, 2.5, 2.4, 1.9, 3.3),
    "nano_cnt": (3.0, 3.0, 3.1, 3.0, 2.9, 1.4, 3.3, 3.5),
    "commercial": (5.3, 5.3, 5.0, 5.1, 5.2, 5.0, 4.2, 5.6),
}

# ---------------------------------------------------------------------------
# Table S7: carbon content of the cast electrodes [wt %]
# ---------------------------------------------------------------------------
# "cnt" is the carbon that sits *inside* the secondary particles for the
# structured composites, and is loose powder for the unstructured mixtures.
ELECTRODE_CARBON_WT_PCT: dict[str, dict[str, float]] = {
    "nano": {"lto": 80, "pvdf": 10, "super_p": 10, "cnt": 0.0},
    "polydisperse": {"lto": 80, "pvdf": 10, "super_p": 10, "cnt": 0.0},
    "uniform": {"lto": 80, "pvdf": 10, "super_p": 10, "cnt": 0.0},
    "oxcnt_mp": {"lto": 76, "pvdf": 9.9, "super_p": 10.0, "cnt": 3.8},
    "cnt_pvp_mp": {"lto": 74, "pvdf": 9.5, "super_p": 9.2, "cnt": 7.5},
    "nano_oxcnt": {"lto": 75, "pvdf": 11, "super_p": 9.9, "cnt": 3.9},
    "nano_cnt": {"lto": 75, "pvdf": 9.4, "super_p": 9.3, "cnt": 6.8},
    "commercial": {"lto": 80, "pvdf": 10, "super_p": 10, "cnt": 0.0},
}

# Main text: TGA-measured carbon *within* the microparticles themselves.
INTRA_PARTICLE_CARBON_WT_PCT: dict[str, float] = {
    "uniform": 2.2,  # pyrolysed surfactant residue only
    "oxcnt_mp": 4.7,
    "cnt_pvp_mp": 9.2,
}

# Main text: BET specific surface areas [m2/g].
BET_SURFACE_AREA_M2_G: dict[str, float] = {
    "nano": 24.1,
    "polydisperse": 29.2,
    "uniform": 28.4,
}

# ---------------------------------------------------------------------------
# Tables S14 / S15: the thick, high-loading electrode (Figure S14)
# ---------------------------------------------------------------------------
HIGH_LOADING_C_RATES = (0.1, 0.5, 1.0, 3.0)
HIGH_LOADING = {
    "thickness_um": 245.0,
    "thickness_sd_um": 16.0,
    "active_loading_mg_cm2": 15.5,
    "areal_loading_mg_cm2": 19.4,
    "electrode_density_g_cm3": 0.39,
    "porosity": 0.72,
    "specific_capacity_mah_g": (139.1, 136.9, 130.5, 76.4),
    "specific_capacity_sd": (0.2, 0.1, 1.8, 14.4),
    "energy_density_wh_l": (220.0, 216.4, 206.3, 121.4),
    "energy_density_sd": (12.1, 11.8, 12.0, 26.7),
}

# Figure 5f: tap densities of the loose powders [g/cm3], for the packing story.
TAP_DENSITY_G_CM3: dict[str, float] = {
    "nLTO (as received)": 0.32,
    "nLTO (ground)": 0.47,
    "nLTO/oxCNT (ground)": 0.43,
    "nLTO/CNT (ground)": 0.30,
    "Polydisperse LTO MPs": 0.63,
    "Membrane LTO MPs": 0.78,
    "LTO/oxCNT MPs": 0.67,
    "LTO/CNT/PVP MPs": 0.49,
}


def c_rate_index(c_rate: float) -> int:
    """Return the index of ``c_rate`` within :data:`RATE_TEST_C_RATES`.

    Args:
        c_rate: One of the eight tabulated rate-test C-rates.

    Returns:
        The position of that C-rate in the rate-test sequence.

    Raises:
        ValueError: If ``c_rate`` is not one of the tabulated rates.

    Examples:
        >>> c_rate_index(5.0)
        5
    """
    try:
        return RATE_TEST_C_RATES.index(c_rate)
    except ValueError as exc:  # pragma: no cover - defensive
        msg = f"{c_rate}C is not one of the tabulated rates"
        raise ValueError(msg) from exc
