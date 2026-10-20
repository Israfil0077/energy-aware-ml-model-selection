"""Energy / CO2 measurement with CodeCarbon, shared by Phase 11 (same method as Phase 5).

CO2 = measured energy x the grid carbon intensity CodeCarbon's own data table gives for the
detected location. See notebook 05 for why we don't use CodeCarbon's built-in emissions total.
"""
import logging
import warnings

import config as C
import utils as U

logging.getLogger("codecarbon").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="codecarbon")


def grid_carbon_intensity(d):
    """kg CO2 per kWh for the location in CodeCarbon's final_emissions_data `d`."""
    from codecarbon.core.emissions import Emissions
    from codecarbon.core.units import Energy
    from codecarbon.external.geography import CloudMetadata, GeoMetadata
    from codecarbon.input import DataSource
    e = Emissions(DataSource())
    one_kwh = Energy.from_energy(kWh=1)
    if d.cloud_provider and d.cloud_region:
        try:
            return float(e.get_cloud_emissions(one_kwh, CloudMetadata(provider=d.cloud_provider, region=d.cloud_region))), \
                   f"cloud {d.cloud_provider}/{d.cloud_region}"
        except Exception:
            pass
    geo = GeoMetadata(country_iso_code=d.country_iso_code, country_name=d.country_name, region=d.region)
    return float(e.get_country_emissions(one_kwh, geo)), f"country {d.country_iso_code} ({d.region})"


_SESSION = {}   # grid intensity fixed at the first measurement of the session


def measure(fn, repeats: int, project: str, raw_file: str = "phase11_codecarbon_raw.csv"):
    """Run fn() `repeats` times under one CodeCarbon tracker; return per-call figures and fn's last result.

    CodeCarbon re-detects the location (by IP) for every tracker, and in testing one lookup
    out of 22 came back as a different country. The machine does not move during a session,
    so the location and grid intensity from the FIRST measurement are used for all of them;
    a later lookup that disagrees is recorded in 'location_mismatch' rather than used.
    """
    from codecarbon import EmissionsTracker
    tracker = EmissionsTracker(project_name=project, measure_power_secs=1, output_dir=str(C.LOGS_DIR),
                               output_file=raw_file, save_to_file=True, log_level="error")
    tracker.start()
    with U.Timer() as t:
        for _ in range(repeats):
            result = fn()
    tracker.stop()
    d = tracker.final_emissions_data
    detected_intensity, detected_source = grid_carbon_intensity(d)
    if not _SESSION:
        _SESSION.update(intensity=detected_intensity, source=detected_source,
                        country=d.country_name, region=d.region)
    mismatch = detected_source != _SESSION["source"]
    if mismatch:
        logging.getLogger("energy").warning(
            f"{project}: location lookup returned {detected_source}; keeping session location {_SESSION['source']}")
    energy = d.energy_consumed / repeats
    return {
        "seconds": t.seconds / repeats, "energy_kwh": energy, "co2_kg": energy * _SESSION["intensity"],
        "co2_codecarbon_raw_kg": d.emissions / repeats, "repeats": repeats,
        "carbon_intensity_kg_per_kwh": _SESSION["intensity"], "carbon_intensity_source": _SESSION["source"],
        "country": _SESSION["country"], "region": _SESSION["region"], "cpu_model": d.cpu_model,
        "location_mismatch": detected_source if mismatch else "",
    }, result
