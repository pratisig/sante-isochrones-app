"""
projection.py — Utilitaires de projection UTM automatique

Détecte la zone UTM optimale depuis les coordonnées WGS84,
propose une liste de projections UTM par pays/région d'Afrique,
et fournit des fonctions de reprojection pour GeoDataFrames.
"""
import math
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

# ─────────────────────────────────────────────────────────────────
# Table des zones UTM d'Afrique (et pays voisins fréquents)
# EPSG code, zone UTM, pays/région couverts
# ─────────────────────────────────────────────────────────────────
AFRICA_UTM_ZONES = [
    # Afrique de l'Ouest
    {"epsg": 32628, "name": "UTM 28N — Cabo Verde, Sénégal Ouest",           "lon_min": -18, "lon_max": -12, "lat_min":  0, "lat_max": 25},
    {"epsg": 32629, "name": "UTM 29N — Sénégal, Gambie, Guinée-Bissau",      "lon_min": -12, "lon_max":  -6, "lat_min":  0, "lat_max": 25},
    {"epsg": 32630, "name": "UTM 30N — Burkina Faso Ouest, Ghana, Côte d'Ivoire", "lon_min": -6, "lon_max": 0, "lat_min": 0, "lat_max": 25},
    {"epsg": 32631, "name": "UTM 31N — Burkina Faso Est, Niger Ouest, Bénin, Togo", "lon_min": 0, "lon_max": 6, "lat_min": 0, "lat_max": 25},
    {"epsg": 32632, "name": "UTM 32N — Niger, Nigéria, Cameroun Nord",        "lon_min":  6,  "lon_max": 12, "lat_min":  0, "lat_max": 25},
    {"epsg": 32633, "name": "UTM 33N — Tchad, Cameroun Est, RCA",              "lon_min": 12,  "lon_max": 18, "lat_min":  0, "lat_max": 25},
    {"epsg": 32634, "name": "UTM 34N — Soudan Ouest, Libye Est",               "lon_min": 18,  "lon_max": 24, "lat_min":  0, "lat_max": 25},
    {"epsg": 32635, "name": "UTM 35N — Soudan Est, Éthiopie Ouest",            "lon_min": 24,  "lon_max": 30, "lat_min":  0, "lat_max": 25},
    {"epsg": 32636, "name": "UTM 36N — Éthiopie, Kenya Ouest",                 "lon_min": 30,  "lon_max": 36, "lat_min":  0, "lat_max": 25},
    {"epsg": 32637, "name": "UTM 37N — Somalie, Kenya Est",                    "lon_min": 36,  "lon_max": 42, "lat_min":  0, "lat_max": 25},
    # Afrique du Nord
    {"epsg": 32630, "name": "UTM 30N — Maroc Est, Algérie Ouest",              "lon_min": -6,  "lon_max":  0, "lat_min": 25, "lat_max": 40},
    {"epsg": 32631, "name": "UTM 31N — Algérie Centre, Tunisie",               "lon_min":  0,  "lon_max":  6, "lat_min": 25, "lat_max": 40},
    {"epsg": 32632, "name": "UTM 32N — Libye Ouest, Tunisie Est",              "lon_min":  6,  "lon_max": 12, "lat_min": 25, "lat_max": 40},
    {"epsg": 32633, "name": "UTM 33N — Libye Est, Égypte Ouest",               "lon_min": 12,  "lon_max": 18, "lat_min": 25, "lat_max": 40},
    {"epsg": 32634, "name": "UTM 34N — Égypte",                                "lon_min": 18,  "lon_max": 24, "lat_min": 25, "lat_max": 40},
    {"epsg": 32635, "name": "UTM 35N — Égypte Est",                            "lon_min": 24,  "lon_max": 30, "lat_min": 25, "lat_max": 40},
    # Afrique Centrale
    {"epsg": 32632, "name": "UTM 32N — Nigéria Sud, Cameroun",                 "lon_min":  6,  "lon_max": 12, "lat_min": -5, "lat_max":  0},
    {"epsg": 32633, "name": "UTM 33N — RCA, Congo Nord",                       "lon_min": 12,  "lon_max": 18, "lat_min": -5, "lat_max":  0},
    {"epsg": 32634, "name": "UTM 34N — RDC Ouest",                             "lon_min": 18,  "lon_max": 24, "lat_min": -5, "lat_max":  0},
    {"epsg": 32635, "name": "UTM 35N — RDC Centre",                            "lon_min": 24,  "lon_max": 30, "lat_min": -5, "lat_max":  0},
    # Afrique de l'Est (Sud de l'équateur)
    {"epsg": 32736, "name": "UTM 36S — Tanzanie, Zambie, Malawi",              "lon_min": 30,  "lon_max": 36, "lat_min":-25, "lat_max":  0},
    {"epsg": 32737, "name": "UTM 37S — Mozambique, Madagascar Nord",           "lon_min": 36,  "lon_max": 42, "lat_min":-25, "lat_max":  0},
    # Afrique du Sud
    {"epsg": 32733, "name": "UTM 33S — Namibie, Botswana, Zimbabwe",           "lon_min": 12,  "lon_max": 18, "lat_min":-35, "lat_max":-25},
    {"epsg": 32734, "name": "UTM 34S — Zimbabwe, Mozambique Sud",              "lon_min": 18,  "lon_max": 24, "lat_min":-35, "lat_max":-25},
    {"epsg": 32735, "name": "UTM 35S — Afrique du Sud Est",                    "lon_min": 24,  "lon_max": 30, "lat_min":-35, "lat_max":-25},
    {"epsg": 32736, "name": "UTM 36S — Afrique du Sud, Mozambique",            "lon_min": 30,  "lon_max": 36, "lat_min":-35, "lat_max":-25},
    # Îles
    {"epsg": 32738, "name": "UTM 38S — Réunion, Maurice, Madagascar Sud",     "lon_min": 42,  "lon_max": 48, "lat_min":-30, "lat_max":  0},
    {"epsg": 32632, "name": "UTM 32N — Maroc (zone standard)",                 "lon_min": -8,  "lon_max":  0, "lat_min": 28, "lat_max": 36},
]

# Projections nationales recommandées par pays
COUNTRY_PROJECTIONS = {
    "Burkina Faso":      [{"epsg": 32630, "name": "UTM 30N (Ouest BF)"},   {"epsg": 32631, "name": "UTM 31N (Est BF)"}],
    "Sénégal":           [{"epsg": 32628, "name": "UTM 28N (Ouest SN)"},   {"epsg": 32629, "name": "UTM 29N (Est SN)"}],
    "Mali":              [{"epsg": 32629, "name": "UTM 29N (Ouest ML)"},   {"epsg": 32630, "name": "UTM 30N (Centre ML)"}, {"epsg": 32631, "name": "UTM 31N (Est ML)"}],
    "Niger":             [{"epsg": 32631, "name": "UTM 31N (Ouest NE)"},   {"epsg": 32632, "name": "UTM 32N (Est NE)"}],
    "Côte d'Ivoire":     [{"epsg": 32630, "name": "UTM 30N (CI)"}],
    "Ghana":             [{"epsg": 32630, "name": "UTM 30N (GH Ouest)"},   {"epsg": 32631, "name": "UTM 31N (GH Est)"}],
    "Cameroun":          [{"epsg": 32632, "name": "UTM 32N (CM Ouest)"},   {"epsg": 32633, "name": "UTM 33N (CM Est)"}],
    "Nigeria":           [{"epsg": 32631, "name": "UTM 31N (NG Ouest)"},   {"epsg": 32632, "name": "UTM 32N (NG Est)"}],
    "Tchad":             [{"epsg": 32633, "name": "UTM 33N (TD Ouest)"},   {"epsg": 32634, "name": "UTM 34N (TD Est)"}],
    "Guinée":            [{"epsg": 32629, "name": "UTM 29N (GN)"}],
    "Bénin":             [{"epsg": 32631, "name": "UTM 31N (BJ)"}],
    "Togo":              [{"epsg": 32631, "name": "UTM 31N (TG)"}],
    "Maroc":             [{"epsg": 32629, "name": "UTM 29N (MA Ouest)"},   {"epsg": 32630, "name": "UTM 30N (MA Est)"}],
    "Algérie":           [{"epsg": 32630, "name": "UTM 30N (DZ Ouest)"},   {"epsg": 32631, "name": "UTM 31N (DZ Est)"}],
    "Tunisie":           [{"epsg": 32632, "name": "UTM 32N (TN)"}],
    "Égypte":            [{"epsg": 32636, "name": "UTM 36N (EG Ouest)"},   {"epsg": 32637, "name": "UTM 37N (EG Est)"}],
    "Kenya":             [{"epsg": 32636, "name": "UTM 36N (KE Ouest)"},   {"epsg": 32637, "name": "UTM 37N (KE Est)"}],
    "Éthiopie":          [{"epsg": 32637, "name": "UTM 37N (ET Ouest)"},   {"epsg": 32638, "name": "UTM 38N (ET Est)"}],
    "Tanzanie":          [{"epsg": 32736, "name": "UTM 36S (TZ Ouest)"},   {"epsg": 32737, "name": "UTM 37S (TZ Est)"}],
    "Mozambique":        [{"epsg": 32736, "name": "UTM 36S (MZ Ouest)"},   {"epsg": 32737, "name": "UTM 37S (MZ Est)"}],
    "Madagascar":        [{"epsg": 32738, "name": "UTM 38S (MG)"}],
    "Afrique du Sud":    [{"epsg": 32734, "name": "UTM 34S (ZA Ouest)"},   {"epsg": 32735, "name": "UTM 35S (ZA Centre)"}, {"epsg": 32736, "name": "UTM 36S (ZA Est)"}],
    "RDC":               [{"epsg": 32633, "name": "UTM 33N (CD Nord)"},    {"epsg": 32634, "name": "UTM 34S (CD Centre)"}, {"epsg": 32735, "name": "UTM 35S (CD Sud)"}],
    "Angola":            [{"epsg": 32733, "name": "UTM 33S (AO Ouest)"},   {"epsg": 32734, "name": "UTM 34S (AO Est)"}],
    "Zambie":            [{"epsg": 32735, "name": "UTM 35S (ZM Ouest)"},   {"epsg": 32736, "name": "UTM 36S (ZM Est)"}],
    "Zimbabwe":          [{"epsg": 32735, "name": "UTM 35S (ZW Ouest)"},   {"epsg": 32736, "name": "UTM 36S (ZW Est)"}],
    "Namibie":           [{"epsg": 32733, "name": "UTM 33S (NA)"}],
    "Botswana":          [{"epsg": 32734, "name": "UTM 34S (BW)"}],
    "Rwanda":            [{"epsg": 32735, "name": "UTM 35S (RW)"}],
    "Burundi":           [{"epsg": 32735, "name": "UTM 35S (BI)"}],
    "Ouganda":           [{"epsg": 32636, "name": "UTM 36N (UG)"}],
    "Somalie":           [{"epsg": 32637, "name": "UTM 37N (SO)"},          {"epsg": 32638, "name": "UTM 38N (SO Est)"}],
    "Soudan":            [{"epsg": 32635, "name": "UTM 35N (SD Ouest)"},    {"epsg": 32636, "name": "UTM 36N (SD Est)"}],
    "Soudan du Sud":     [{"epsg": 32635, "name": "UTM 35N (SS)"},          {"epsg": 32636, "name": "UTM 36N (SS Est)"}],
    "Libye":             [{"epsg": 32632, "name": "UTM 32N (LY Ouest)"},    {"epsg": 32633, "name": "UTM 33N (LY Centre)"}, {"epsg": 32634, "name": "UTM 34N (LY Est)"}],
    "Mauritanie":        [{"epsg": 32628, "name": "UTM 28N (MR Ouest)"},    {"epsg": 32629, "name": "UTM 29N (MR Est)"}],
    "Guinée-Bissau":     [{"epsg": 32628, "name": "UTM 28N (GW)"}],
    "Sierra Leone":      [{"epsg": 32629, "name": "UTM 29N (SL)"}],
    "Liberia":           [{"epsg": 32629, "name": "UTM 29N (LR)"}],
    "Gambie":            [{"epsg": 32628, "name": "UTM 28N (GM)"}],
    "Djibouti":          [{"epsg": 32638, "name": "UTM 38N (DJ)"}],
    "Érythrée":          [{"epsg": 32637, "name": "UTM 37N (ER)"}],
    "Malawi":            [{"epsg": 32736, "name": "UTM 36S (MW)"}],
    "Lesotho":           [{"epsg": 32735, "name": "UTM 35S (LS)"}],
    "Eswatini":          [{"epsg": 32736, "name": "UTM 36S (SZ)"}],
    "Gabon":             [{"epsg": 32632, "name": "UTM 32S (GA)"}],
    "Congo":             [{"epsg": 32633, "name": "UTM 33S (CG)"}],
    "Guinée Équatoriale":[{"epsg": 32632, "name": "UTM 32N (GQ)"}],
    "São Tomé-et-Príncipe": [{"epsg": 32632, "name": "UTM 32N (ST)"}],
    "Cap-Vert":          [{"epsg": 32626, "name": "UTM 26N (CV)"}],
    "Comores":           [{"epsg": 32738, "name": "UTM 38S (KM)"}],
    "Maurice":           [{"epsg": 32740, "name": "UTM 40S (MU)"}],
    "Réunion":           [{"epsg": 32740, "name": "UTM 40S (RE)"}],
    "Seychelles":        [{"epsg": 32739, "name": "UTM 39S (SC)"}],
}


# ─────────────────────────────────────────────────────────────────
# Fonctions de détection automatique
# ─────────────────────────────────────────────────────────────────

def auto_utm_epsg(lon: float, lat: float) -> int:
    """
    Calcule automatiquement l'EPSG UTM optimal pour une coordonnée WGS84.
    Formule standard : zone = floor((lon + 180) / 6) + 1
    Hémisphère Sud (lat < 0) : EPSG 327xx, Nord : EPSG 326xx
    """
    zone = int(math.floor((lon + 180) / 6)) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


def auto_utm_epsg_from_gdf(gdf: gpd.GeoDataFrame) -> int:
    """
    Détecte l'EPSG UTM optimal à partir d'un GeoDataFrame.
    Utilise le centroïde de l'enveloppe totale.
    """
    # Reprojeter en WGS84 si nécessaire pour obtenir lon/lat
    gdf_wgs = gdf.to_crs(4326) if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf
    bounds = gdf_wgs.total_bounds  # (minx, miny, maxx, maxy)
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    return auto_utm_epsg(center_lon, center_lat)


def utm_name_from_epsg(epsg: int) -> str:
    """Retourne un nom lisible pour un EPSG UTM."""
    zone = epsg % 100
    if 32601 <= epsg <= 32660:
        hemi = "N"
    elif 32701 <= epsg <= 32760:
        hemi = "S"
    else:
        return f"EPSG:{epsg}"
    return f"UTM {zone}{hemi} (EPSG:{epsg})"


def suggest_utm_for_gdf(gdf: gpd.GeoDataFrame) -> dict:
    """
    Retourne un dict avec :
    - 'auto_epsg'   : EPSG détecté automatiquement
    - 'auto_name'   : nom lisible
    - 'is_projected': True si la couche est déjà projetée
    - 'current_crs' : CRS actuel
    - 'center_lon', 'center_lat' : centre géographique
    """
    is_proj = gdf.crs is not None and gdf.crs.is_projected
    current = str(gdf.crs) if gdf.crs else "Inconnu"

    gdf_wgs = gdf.to_crs(4326) if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf
    bounds = gdf_wgs.total_bounds
    c_lon = (bounds[0] + bounds[2]) / 2
    c_lat = (bounds[1] + bounds[3]) / 2

    epsg = auto_utm_epsg(c_lon, c_lat)
    return {
        "auto_epsg":    epsg,
        "auto_name":    utm_name_from_epsg(epsg),
        "is_projected": is_proj,
        "current_crs":  current,
        "center_lon":   round(c_lon, 4),
        "center_lat":   round(c_lat, 4),
    }


def reproject_to_utm(gdf: gpd.GeoDataFrame, epsg: int = None) -> gpd.GeoDataFrame:
    """
    Reprojette un GeoDataFrame en UTM.
    Si epsg est None, détecte automatiquement la zone UTM optimale.
    """
    if epsg is None:
        epsg = auto_utm_epsg_from_gdf(gdf)
    return gdf.to_crs(epsg)


def compute_area_km2(geometry, source_crs=4326, utm_epsg: int = None) -> float:
    """
    Calcule l'aire en km² d'une géométrie shapely.
    Reprojette automatiquement en UTM si nécessaire.
    """
    import geopandas as gpd
    gs = gpd.GeoSeries([geometry], crs=source_crs)
    if utm_epsg is None:
        bounds = gs.to_crs(4326).total_bounds
        c_lon = (bounds[0] + bounds[2]) / 2
        c_lat = (bounds[1] + bounds[3]) / 2
        utm_epsg = auto_utm_epsg(c_lon, c_lat)
    return round(gs.to_crs(utm_epsg).area.values[0] / 1e6, 4)


def get_all_utm_options() -> list:
    """
    Retourne la liste complète des zones UTM disponibles
    pour l'Afrique, sans doublons EPSG.
    Chaque élément : {"epsg": int, "name": str}
    """
    seen = set()
    result = []
    for z in AFRICA_UTM_ZONES:
        if z["epsg"] not in seen:
            seen.add(z["epsg"])
            result.append({"epsg": z["epsg"], "name": z["name"]})
    # Trier par EPSG
    result.sort(key=lambda x: x["epsg"])
    return result
