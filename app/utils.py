"""
utils.py — Fonctions utilitaires pour le calcul d'isochrones
Moteurs : OSMnx (local), ORS (API), OSRM (public), Valhalla (public), GraphHopper (API gratuite)
"""
import numpy as np
import networkx as nx
import osmnx as ox
import requests
import json
import time
from shapely.geometry import Point, shape
from shapely.ops import unary_union
import alphashape
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────
# 1. OSMnx local — avec support colonnes Tps_min / Vit_kmh
# ─────────────────────────────────────────────────────────────────
def compute_osmnx_isochrone(lon: float, lat: float, minutes: int,
                            mode: str, method: str, alpha: float,
                            gdf_roads=None):
    """
    Calcule un isochrone localement via OSMnx + NetworkX.
    Si gdf_roads est fourni (couche OSM avec Tps_min/Vit_kmh/maxspeed),
    les vitesses sont lues depuis les attributs de la couche.
    """
    DEFAULT_SPEEDS = {"walk": 4.5, "bike": 15.0, "drive": 40.0}
    spd = DEFAULT_SPEEDS.get(mode, 4.5)
    dist = (minutes / 60) * spd * 1000 + 600

    G = ox.graph_from_point((lat, lon), dist=dist, network_type=mode, simplify=True)
    center = ox.nearest_nodes(G, lon, lat)

    # Dictionnaire osm_id -> vitesse depuis la couche OSM chargée
    road_speeds = {}
    if gdf_roads is not None:
        for _, row in gdf_roads.iterrows():
            oid = row.get("osm_id")
            if oid is None:
                continue
            # Priorité : Vit_kmh > maxspeed > défaut
            v = None
            if "Vit_kmh" in row and row["Vit_kmh"] and not np.isnan(float(row["Vit_kmh"])):
                v = float(row["Vit_kmh"])
            elif "maxspeed" in row and row["maxspeed"]:
                try:
                    v = float(str(row["maxspeed"]).split()[0])
                except (ValueError, TypeError):
                    pass
            if v:
                road_speeds[str(oid)] = v

    for u, v_node, d in G.edges(data=True):
        # Cherche la vitesse dans la couche OSM via osmid
        edge_speed = spd
        osmid = str(d.get("osmid", ""))
        if osmid in road_speeds:
            edge_speed = road_speeds[osmid]
        else:
            ms = d.get("maxspeed")
            if isinstance(ms, list):
                ms = ms[0]
            if ms:
                try:
                    edge_speed = float(str(ms).split()[0])
                except (ValueError, TypeError):
                    pass
        length = d.get("length", 1)
        d["travel_time"] = length / (max(edge_speed, 1) * 1000 / 3600)

    sub = nx.ego_graph(G, center, radius=minutes * 60, distance="travel_time")
    pts = [Point(G.nodes[n]["x"], G.nodes[n]["y"]) for n in sub.nodes()]

    if len(pts) < 4:
        return Point(lon, lat).buffer(0.01)

    coords = np.array([(p.x, p.y) for p in pts])

    if method == "Alpha Shape (recommandé)":
        try:
            poly = alphashape.alphashape(coords, alpha)
            if poly and not poly.is_empty:
                return poly
        except Exception:
            pass
        return unary_union(pts).convex_hull
    elif method == "Convex Hull":
        return unary_union(pts).convex_hull
    else:
        return unary_union([p.buffer(0.002) for p in pts])


# ─────────────────────────────────────────────────────────────────
# 2. OpenRouteService
# ─────────────────────────────────────────────────────────────────
def isochrone_ors(lon: float, lat: float, minutes: int, profile: str, key: str):
    url = f"https://api.openrouteservice.org/v2/isochrones/{profile}"
    resp = requests.post(url, json={
        "locations": [[lon, lat]],
        "range": [minutes * 60],
        "smoothing": 5,
        "attributes": ["area"]
    }, headers={"Authorization": key, "Content-Type": "application/json"}, timeout=20)
    if resp.status_code == 200:
        data = resp.json()
        if "features" in data and data["features"]:
            return shape(data["features"][0]["geometry"])
    raise Exception(f"ORS {resp.status_code}: {resp.text[:300]}")


# ─────────────────────────────────────────────────────────────────
# 3. OSRM Public
# ─────────────────────────────────────────────────────────────────
def isochrone_osrm(lon: float, lat: float, minutes: int, profile: str):
    speeds = {"foot": 4.5, "bike": 15.0, "car": 50.0}
    spd = speeds.get(profile, 50.0)
    R = (minutes / 60) * spd * 1000
    N = 36
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    Re = 6371000
    dlats = [np.degrees(R / Re * np.cos(a)) for a in angles]
    dlons = [np.degrees(R / Re * np.sin(a) / np.cos(np.radians(lat))) for a in angles]
    dests = [(lon + dlons[i], lat + dlats[i]) for i in range(N)]
    coords_str = f"{lon},{lat};" + ";".join([f"{d[0]},{d[1]}" for d in dests])
    url = (f"http://router.project-osrm.org/table/v1/{profile}/{coords_str}"
           f"?sources=0&destinations={';'.join(str(i+1) for i in range(N))}&annotations=duration")
    resp = requests.get(url, timeout=25)
    if resp.status_code != 200:
        raise Exception(f"OSRM {resp.status_code}")
    durations = resp.json()["durations"][0]
    target = minutes * 60
    pts = []
    for i, dur in enumerate(durations):
        if dur is None:
            dur = target * 2
        ratio = min(1.0, target / dur) if dur > 0 else 1.0
        pts.append(Point(lon + dlons[i] * ratio, lat + dlats[i] * ratio))
    return unary_union(pts).convex_hull if pts else Point(lon, lat).buffer(0.01)


# ─────────────────────────────────────────────────────────────────
# 4. Valhalla Public
# ─────────────────────────────────────────────────────────────────
def isochrone_valhalla(lon: float, lat: float, minutes: int, costing: str):
    url = "https://valhalla1.openstreetmap.de/isochrone"
    body = {
        "locations": [{"lon": lon, "lat": lat}],
        "costing": costing,
        "contours": [{"time": minutes, "color": "ff0000"}],
        "polygons": True,
        "generalize": 50
    }
    resp = requests.get(url, params={"json": json.dumps(body)}, timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        if "features" in data and data["features"]:
            return shape(data["features"][0]["geometry"])
    raise Exception(f"Valhalla {resp.status_code}: {resp.text[:300]}")


# ─────────────────────────────────────────────────────────────────
# 5. GraphHopper (API gratuite — 500 req/jour sans clé, ou avec clé)
# ─────────────────────────────────────────────────────────────────
def isochrone_graphhopper(lon: float, lat: float, minutes: int,
                          profile: str, key: str = ""):
    """
    GraphHopper Isochrones API.
    Sans clé : 500 req/jour sur graphhopper.com/api/1/
    Profiles : car | bike | foot | motorcycle | hike | mtb | racingbike
    """
    url = "https://graphhopper.com/api/1/isochrone"
    params = {
        "point": f"{lat},{lon}",
        "time_limit": minutes * 60,
        "vehicle": profile,
        "buckets": 1,
        "type": "json",
        "debug": "false"
    }
    if key:
        params["key"] = key
    resp = requests.get(url, params=params, timeout=25)
    if resp.status_code == 200:
        data = resp.json()
        polygons = data.get("polygons", [])
        if polygons:
            return shape(polygons[0]["geometry"])
    raise Exception(f"GraphHopper {resp.status_code}: {resp.text[:300]}")


# ─────────────────────────────────────────────────────────────────
# Dispatcher central
# ─────────────────────────────────────────────────────────────────
def compute_isochrone(engine: str, lon: float, lat: float, minutes: int,
                      mode_api: str, mode_osm: str,
                      iso_method: str, alpha: float,
                      ors_key: str = "", gh_key: str = "",
                      gdf_roads=None):
    """
    Point d'entrée unique. Retourne un shapely Polygon/MultiPolygon.
    """
    if "OSM local" in engine:
        return compute_osmnx_isochrone(lon, lat, minutes, mode_osm,
                                       iso_method, alpha, gdf_roads)
    elif "ORS" in engine:
        return isochrone_ors(lon, lat, minutes, mode_api, ors_key)
    elif "OSRM" in engine:
        return isochrone_osrm(lon, lat, minutes, mode_api)
    elif "Valhalla" in engine:
        return isochrone_valhalla(lon, lat, minutes, mode_api)
    elif "GraphHopper" in engine:
        return isochrone_graphhopper(lon, lat, minutes, mode_api, gh_key)
    else:
        raise ValueError(f"Moteur inconnu: {engine}")


# ─────────────────────────────────────────────────────────────────
# Métadonnées moteurs (pour l'UI)
# ─────────────────────────────────────────────────────────────────
def engines_metadata() -> dict:
    return {
        "OSM local (OSMnx + NetworkX) — Gratuit": {
            "badge_class": "badge-local", "badge_label": "LOCAL",
            "description": "100% local. Lit les colonnes Tps_min, Vit_kmh, maxspeed de votre couche OSM.",
            "needs_key": False,
            "modes": {
                "🚶 Marche": ("walk", "walk"),
                "🚗 Véhicule": ("drive", "drive"),
                "🚲 Vélo": ("bike", "bike"),
            },
        },
        "OpenRouteService (ORS) — Clé API": {
            "badge_class": "badge-api", "badge_label": "API",
            "description": "Très précis. Clé gratuite sur openrouteservice.org (500 req/jour).",
            "needs_key": True, "key_name": "ORS_API_KEY",
            "key_url": "https://openrouteservice.org/dev/#/signup",
            "modes": {
                "🚶 Marche": ("foot-walking", "walk"),
                "🚗 Voiture": ("driving-car", "drive"),
                "🚲 Vélo": ("cycling-regular", "bike"),
                "🛵 HGV": ("driving-hgv", "drive"),
            },
        },
        "OSRM Public — Gratuit": {
            "badge_class": "badge-free", "badge_label": "GRATUIT",
            "description": "Sans clé. Isochrones approx. via matrice de durées OSRM.",
            "needs_key": False,
            "modes": {
                "🚗 Voiture": ("car", "drive"),
                "🚲 Vélo": ("bike", "bike"),
                "🚶 Marche": ("foot", "walk"),
            },
        },
        "Valhalla Public — Gratuit": {
            "badge_class": "badge-free", "badge_label": "GRATUIT",
            "description": "valhalla.openstreetmap.de — polygones précis, 4 modes, sans clé.",
            "needs_key": False,
            "modes": {
                "🚶 Marche": ("pedestrian", "walk"),
                "🚗 Voiture": ("auto", "drive"),
                "🚲 Vélo": ("bicycle", "bike"),
                "🛵 Moto": ("motorcycle", "drive"),
            },
        },
        "GraphHopper — Gratuit (500 req/j)": {
            "badge_class": "badge-free", "badge_label": "GRATUIT",
            "description": "graphhopper.com — polygones précis, sans clé (500 req/j). Clé optionnelle pour plus.",
            "needs_key": False, "key_name": "GH_API_KEY",
            "key_url": "https://www.graphhopper.com/dashboard/",
            "modes": {
                "🚗 Voiture": ("car", "drive"),
                "🚶 Marche": ("foot", "walk"),
                "🚲 Vélo": ("bike", "bike"),
                "🛵 Moto": ("motorcycle", "drive"),
                "🥾 Randonnée": ("hike", "walk"),
            },
        },
    }
