"""
utils.py — Fonctions utilitaires pour le calcul d'isochrones
Moteurs : OSMnx (local), ORS (API), OSRM (public), Valhalla (public), GraphHopper (API)
"""
import os
import tempfile
import numpy as np
import networkx as nx
import osmnx as ox
import requests
import json
import time
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.ops import unary_union
import geopandas as gpd
import alphashape
import warnings
warnings.filterwarnings("ignore")

from projection import auto_utm_epsg, auto_utm_epsg_from_gdf, reproject_to_utm, compute_area_km2

# Forcer pyogrio comme moteur GDAL
gpd.options.io_engine = "pyogrio"


# ─────────────────────────────────────────────────────────────────
# UTILITAIRE : lecture robuste
# ─────────────────────────────────────────────────────────────────
def read_geodata(source) -> gpd.GeoDataFrame:
    """
    Lit une source géospatiale de manière robuste.
    Retourne un GeoDataFrame avec le CRS d'origine préservé.
    """
    driver_map = {
        ".shp":     "ESRI Shapefile",
        ".gpkg":    "GPKG",
        ".geojson": "GeoJSON",
        ".json":    "GeoJSON",
        ".kml":     "KML",
    }

    if isinstance(source, (str, os.PathLike)):
        path = os.path.abspath(str(source))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        ext = os.path.splitext(path)[1].lower()
        try:
            return gpd.read_file(path, engine="pyogrio")
        except Exception:
            driver = driver_map.get(ext)
            if driver:
                try:
                    return gpd.read_file(path, engine="pyogrio", driver=driver)
                except Exception:
                    pass
            raise

    filename = getattr(source, "name", "file.geojson")
    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        ext = ".geojson"

    if hasattr(source, "read"):
        raw = source.read()
    elif hasattr(source, "getvalue"):
        raw = source.getvalue()
    else:
        raise TypeError(f"Type de source non supporté : {type(source)}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        try:
            return gpd.read_file(tmp_path, engine="pyogrio")
        except Exception:
            pass
        driver = driver_map.get(ext)
        if driver:
            try:
                return gpd.read_file(tmp_path, engine="pyogrio", driver=driver)
            except Exception:
                pass
        return gpd.read_file(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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
    # Vitesses par défaut (km/h) selon le mode
    DEFAULT_SPEEDS = {"walk": 4.5, "bike": 15.0, "drive": 40.0}
    spd = DEFAULT_SPEEDS.get(mode, 4.5)

    # Rayon de téléchargement adaptatif : distance max atteignable + marge 20%
    # On majore de 50% pour les modes rapides afin de ne pas tronquer le graphe
    dist = int((minutes / 60) * spd * 1000 * 1.5) + 500
    dist = max(dist, 800)   # minimum 800 m pour les petits intervalles
    dist = min(dist, 80000) # maximum 80 km pour éviter les téléchargements géants

    G = ox.graph_from_point((lat, lon), dist=dist, network_type=mode, simplify=True)
    center = ox.nearest_nodes(G, lon, lat)

    # Assurer que gdf_roads est en WGS84
    if gdf_roads is not None and hasattr(gdf_roads, "crs") and gdf_roads.crs:
        if gdf_roads.crs.to_epsg() != 4326:
            gdf_roads = gdf_roads.to_crs(4326)

    # Dictionnaire osm_id -> vitesse depuis la couche OSM chargée
    road_speeds = {}
    if gdf_roads is not None:
        for _, row in gdf_roads.iterrows():
            oid = row.get("osm_id")
            if oid is None:
                continue
            v = None
            for col in ("Vit_kmh", "vit_kmh", "vitesse_kmh"):
                val = row.get(col)
                if val is not None:
                    try:
                        fval = float(val)
                        if not np.isnan(fval) and fval > 0:
                            v = fval
                            break
                    except (ValueError, TypeError):
                        pass
            if v is None:
                ms = row.get("maxspeed") or row.get("MAXSPEED")
                if ms:
                    try:
                        v = float(str(ms).split()[0])
                    except (ValueError, TypeError):
                        pass
            if v:
                road_speeds[str(oid)] = v

    # Calculer travel_time sur chaque arête
    for u, v_node, d in G.edges(data=True):
        edge_speed = spd
        osmid = str(d.get("osmid", ""))
        # Priorité 1 : osmid dans la couche locale
        if osmid in road_speeds:
            edge_speed = road_speeds[osmid]
        else:
            # Priorité 2 : attribut maxspeed de l'arête OSM
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

    # Priorité 3 : remplacer travel_time par Tps_min de la couche si disponible
    if gdf_roads is not None and "Tps_min" in gdf_roads.columns:
        tps_map = {}
        for _, row in gdf_roads.iterrows():
            oid = row.get("osm_id")
            tps = row.get("Tps_min")
            if oid is not None and tps is not None:
                try:
                    tps_map[str(oid)] = float(tps) * 60  # minutes → secondes
                except (ValueError, TypeError):
                    pass
        if tps_map:
            for u, v_node, d in G.edges(data=True):
                osmid = str(d.get("osmid", ""))
                if osmid in tps_map:
                    d["travel_time"] = tps_map[osmid]

    # Sous-graphe accessible en 'minutes' secondes depuis le centre
    sub = nx.ego_graph(G, center, radius=minutes * 60, distance="travel_time")
    pts = [Point(G.nodes[n]["x"], G.nodes[n]["y"]) for n in sub.nodes()]

    if len(pts) < 4:
        # Fallback : buffer circulaire
        return Point(lon, lat).buffer(0.005)

    coords = np.array([(p.x, p.y) for p in pts])

    if method == "Alpha Shape (recommandé)":
        try:
            poly = alphashape.alphashape(coords, alpha)
            if poly and not poly.is_empty:
                return poly
        except Exception:
            pass
        # Fallback si alpha shape échoue
        return unary_union(pts).convex_hull
    elif method == "Convex Hull":
        return unary_union(pts).convex_hull
    else:  # Buffer sur noeuds
        # Rayon de buffer proportionnel à la densité du réseau
        buf_deg = max(0.0008, (minutes / 60) * spd * 1000 / 111000 * 0.05)
        return unary_union([p.buffer(buf_deg) for p in pts])


# ─────────────────────────────────────────────────────────────────
# 2. OpenRouteService
# ─────────────────────────────────────────────────────────────────
def isochrone_ors(lon: float, lat: float, minutes: int, profile: str, key: str):
    if not key or not key.strip():
        raise Exception("Clé API ORS manquante. Obtenez-en une sur openrouteservice.org")
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
        raise Exception("ORS : réponse vide (aucune feature retournée)")
    raise Exception(f"ORS {resp.status_code}: {resp.text[:300]}")


# ─────────────────────────────────────────────────────────────────
# 3. OSRM Public — matrice de durées réelle (36 points autour)
#    Méthode : on place N points sur un cercle de rayon adaptatif,
#    on snappe chacun sur le réseau via /nearest, puis on interroge
#    /table pour obtenir les durées réelles depuis le point central.
#    On redimensionne chaque vecteur selon le ratio durée cible / durée réelle.
# ─────────────────────────────────────────────────────────────────
def isochrone_osrm(lon: float, lat: float, minutes: int, profile: str):
    """
    Isochrone OSRM basé sur la vraie matrice de durées réseau.
    Utilise le serveur public router.project-osrm.org.
    """
    speeds = {"foot": 4.5, "bike": 15.0, "car": 50.0}
    spd = speeds.get(profile, 50.0)

    # Rayon initial basé sur la vitesse — point de départ du cercle d'exploration
    R = (minutes / 60) * spd * 1000  # mètres
    N = 36  # points autour
    Re = 6371000.0  # rayon terrestre en mètres

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    dlats = [np.degrees(R / Re * np.cos(a)) for a in angles]
    dlons = [np.degrees(R / Re * np.sin(a) / np.cos(np.radians(lat))) for a in angles]
    dests = [(lon + dlons[i], lat + dlats[i]) for i in range(N)]

    # Construire la chaîne de coordonnées : source (index 0) + destinations
    coords_str = f"{lon},{lat};" + ";".join(f"{d[0]:.6f},{d[1]:.6f}" for d in dests)
    dest_indices = ";".join(str(i + 1) for i in range(N))

    url = (f"http://router.project-osrm.org/table/v1/{profile}/{coords_str}"
           f"?sources=0&destinations={dest_indices}&annotations=duration")

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise Exception(f"OSRM table API : {e}")

    if data.get("code") != "Ok":
        raise Exception(f"OSRM erreur : {data.get('message', data.get('code', 'inconnu'))}")

    durations = data["durations"][0]
    target_sec = minutes * 60
    pts = []

    for i, dur in enumerate(durations):
        if dur is None or dur == 0:
            # Destination inaccessible ou co-localisée : on garde le point du cercle
            pts.append(Point(dests[i][0], dests[i][1]))
            continue
        # Ratio : si dur > target, on rapproche le point proportionnellement
        ratio = min(1.0, target_sec / dur)
        new_lon = lon + dlons[i] * ratio
        new_lat = lat + dlats[i] * ratio
        pts.append(Point(new_lon, new_lat))

    if len(pts) < 3:
        return Point(lon, lat).buffer(0.005)

    # Alpha shape pour un contour plus naturel que le convex hull
    coords = np.array([(p.x, p.y) for p in pts])
    try:
        poly = alphashape.alphashape(coords, 0.3)
        if poly and not poly.is_empty:
            return poly
    except Exception:
        pass
    return unary_union(pts).convex_hull


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
    try:
        resp = requests.get(url, params={"json": json.dumps(body)}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise Exception(f"Valhalla réseau : {e}")

    if "features" in data and data["features"]:
        return shape(data["features"][0]["geometry"])
    raise Exception(f"Valhalla : réponse vide — {data.get('error', 'inconnu')}")


# ─────────────────────────────────────────────────────────────────
# 5. GraphHopper
#    Depuis 2024, l'API publique sans clé requiert une inscription.
#    On tente avec clé si fournie, sinon on lève une erreur claire.
# ─────────────────────────────────────────────────────────────────
def isochrone_graphhopper(lon: float, lat: float, minutes: int,
                          profile: str, key: str = ""):
    url = "https://graphhopper.com/api/1/isochrone"
    params = {
        "point": f"{lat},{lon}",
        "time_limit": minutes * 60,
        "vehicle": profile,
        "buckets": 1,
        "type": "json",
    }
    if key and key.strip():
        params["key"] = key.strip()
    else:
        # Sans clé, GraphHopper retourne 401 depuis 2024
        # On tente quand même (peut fonctionner sur certains endpoints publics)
        params["key"] = ""  # header vide

    try:
        resp = requests.get(url, params=params, timeout=25)
    except Exception as e:
        raise Exception(f"GraphHopper réseau : {e}")

    if resp.status_code == 401:
        raise Exception(
            "GraphHopper requiert une clé API. "
            "Créez un compte gratuit sur graphhopper.com/dashboard (500 req/j)."
        )
    if resp.status_code != 200:
        raise Exception(f"GraphHopper {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    polygons = data.get("polygons", [])
    if polygons:
        return shape(polygons[0]["geometry"])
    raise Exception("GraphHopper : aucun polygone retourné")


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
    gdf_roads peut être un GeoDataFrame déjà chargé ou un chemin/fichier.
    """
    # Charger gdf_roads si c'est un chemin ou un fichier uploadé
    if gdf_roads is not None and not hasattr(gdf_roads, "geometry"):
        gdf_roads = read_geodata(gdf_roads)

    # Reprojeter en WGS84 pour les moteurs API et OSMnx
    if gdf_roads is not None and hasattr(gdf_roads, "crs") and gdf_roads.crs:
        if gdf_roads.crs.to_epsg() != 4326:
            gdf_roads = gdf_roads.to_crs(4326)

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
                "🚶 Marche":   ("walk",  "walk"),
                "🚗 Véhicule": ("drive", "drive"),
                "🚲 Vélo":     ("bike",  "bike"),
            },
        },
        "OpenRouteService (ORS) — Clé API": {
            "badge_class": "badge-api", "badge_label": "API",
            "description": "Très précis. Clé gratuite sur openrouteservice.org (500 req/jour).",
            "needs_key": True, "key_name": "ORS_API_KEY",
            "key_url": "https://openrouteservice.org/dev/#/signup",
            "modes": {
                "🚶 Marche":  ("foot-walking",   "walk"),
                "🚗 Voiture": ("driving-car",    "drive"),
                "🚲 Vélo":    ("cycling-regular", "bike"),
                "🛵 HGV":     ("driving-hgv",    "drive"),
            },
        },
        "OSRM Public — Gratuit": {
            "badge_class": "badge-free", "badge_label": "GRATUIT",
            "description": "Sans clé. Isochrones via matrice de durées réseau réelle (36 points).",
            "needs_key": False,
            "modes": {
                "🚗 Voiture": ("car",  "drive"),
                "🚲 Vélo":    ("bike", "bike"),
                "🚶 Marche":  ("foot", "walk"),
            },
        },
        "Valhalla Public — Gratuit": {
            "badge_class": "badge-free", "badge_label": "GRATUIT",
            "description": "valhalla.openstreetmap.de — polygones précis, 4 modes, sans clé.",
            "needs_key": False,
            "modes": {
                "🚶 Marche":  ("pedestrian", "walk"),
                "🚗 Voiture": ("auto",       "drive"),
                "🚲 Vélo":    ("bicycle",    "bike"),
                "🛵 Moto":    ("motorcycle", "drive"),
            },
        },
        "GraphHopper — Clé API (gratuit 500 req/j)": {
            "badge_class": "badge-api", "badge_label": "API",
            "description": "graphhopper.com — polygones précis. Clé requise depuis 2024 (gratuit : 500 req/j).",
            "needs_key": True, "key_name": "GH_API_KEY",
            "key_url": "https://www.graphhopper.com/dashboard/",
            "modes": {
                "🚗 Voiture":    ("car",          "drive"),
                "🚶 Marche":     ("foot",         "walk"),
                "🚲 Vélo":       ("bike",         "bike"),
                "🛵 Moto":       ("motorcycle",   "drive"),
                "🥾 Randonnée":  ("hike",         "walk"),
            },
        },
    }
