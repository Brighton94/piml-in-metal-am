import numpy as np
import pandas as pd

# Constants
DEFAULT_ALPHA = 1e-5  # Thermal expansion coefficient (1/K)
DEFAULT_PLUME_ALPHA = 0.15  # Absorption coefficient for Beer–Lambert (1/mm, empirical)
CHAMBER_AREA_CM2 = 100  # Approx. chamber area in cm² (can be adjusted)
MELT_TEMP_K = 2000  # Approx. melt pool peak temperature (K)
AMBIENT_TEMP_K = 300  # Ambient reference temp (K)


def compute_energy_density(
    power: float, speed: float, hatch_spacing: float, layer_thickness: float
) -> float:
    """Compute volumetric energy density in J/mm³."""
    return power / (speed * hatch_spacing * layer_thickness)


def compute_attenuation_factor(
    energy_density: float, plume_thickness_mm: float, alpha: float = DEFAULT_PLUME_ALPHA
) -> float:
    """Compute laser beam attenuation factor via Beer–Lambert Law."""
    return np.exp(-alpha * plume_thickness_mm)


def compute_gas_velocity(
    flow_rate_lpm: float, area_cm2: float = CHAMBER_AREA_CM2
) -> float:
    """Estimate gas velocity (cm/s) from flow rate in liters per minute."""
    area_m2 = area_cm2 / 1e4  # cm² to m²
    flow_rate_m3s = flow_rate_lpm / 60000  # L/min to m³/s
    velocity_m_s = flow_rate_m3s / area_m2
    return velocity_m_s * 100  # convert to cm/s


def compute_thermal_strain(delta_temp: float, alpha: float = DEFAULT_ALPHA) -> float:
    """Compute thermal strain from temperature gradient."""
    return alpha * delta_temp


def compute_recoil_pressure_proxy(
    energy_density: float, scaling_factor: float = 1e-3
) -> float:
    """Proxy for recoil pressure based on exponential scaling of energy density."""
    return np.exp(scaling_factor * energy_density)


def add_physics_features(
    df: pd.DataFrame,
    build_plate_temp_col: str = "build_plate_temperature",
    chamber_temp_col: str = "top_chamber_temperature",
    flow_rate_col: str = "actual_ventilator_flow_rate",
    layer_time_col: str = "layer_times",
    energy_density_col: str = "energy_density",
    plume_thickness_mm: float = 1.5,
) -> pd.DataFrame:
    """Given a DataFrame with temperature, flow rate, and energy density columns.

    Return the same DataFrame with added physics-informed heuristic features.
    """
    df = df.copy()

    # ΔT between build plate and ambient/top chamber
    delta_temp = df[build_plate_temp_col] - df[chamber_temp_col]
    df["thermal_strain"] = compute_thermal_strain(delta_temp)

    # Attenuation factor from Beer–Lambert
    df["attenuation_factor"] = compute_attenuation_factor(
        df[energy_density_col], plume_thickness_mm
    )

    # Gas velocity (assuming constant chamber cross-section)
    df["gas_velocity"] = compute_gas_velocity(df[flow_rate_col])

    # Recoil pressure proxy from melt pool energy
    df["recoil_pressure_proxy"] = compute_recoil_pressure_proxy(df[energy_density_col])

    return df
