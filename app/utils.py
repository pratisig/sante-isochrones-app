"""
utils.py — Fonctions utilitaires pour le calcul d'isochrones
Moteurs : OSMnx (local), ORS (API), OSRM (public), Valhalla (public)
"""
import numpy as np
import networkx as nx
import osmnx as ox
from shapely.geometry import Point
from shapely.ops import unary_union
import alphashape
import warnings
warnings.filterwarnings("ignore")


def compute_osmnx_isochrone(lon: float, lat: float, minutes: int,
                            mode: str, method: str, alpha: float):
    """
    Calcule un isochrone localement via OSMnx + NetworkX.

    Paramètres
    ----------
    lon, lat   : coordonnées WGS84 du point d'origine
    minutes    : seuil de temps en minutes
    mode       : 'walk' | 'bike' | 'drive'
    method     : 'Alpha Shape (recommandé)' | 'Convex Hull' | 'Buffer sur noeuds'
    alpha      : paramètre alpha pour Alpha Shape (0.1 – 1.0)

    Retourne
    --------
    shapely.geometry.Polygon ou MultiPolygon
    """
    speeds = {"walk": 4.5, "bike": 15.0, "drive": 40.0}
    spd = speeds.get(mode, 4.5)
    dist = (minutes / 60) * spd * 1000  # rayon de téléchargement en mètres

    # Téléchargement du graphe OSM autour du point
    G = ox.graph_from_point(
        (lat, lon),
        dist=dist + 600,
        network_type=mode,
        simplify=True
    )
    center = ox.nearest_nodes(G, lon, lat)

    # Calcul du temps de parcours sur chaque arête
    for u, v, d in G.edges(data=True):
        ms = d.get("maxspeed")
        if isinstance(ms, list):
            ms = ms[0]
        edge_speed = spd
        if ms:
            try:
                edge_speed = float(str(ms).split()[0])
            except (ValueError, TypeError):
                pass
        length = d.get("length", 1)
        d["travel_time"] = length / (edge_speed * 1000 / 3600)

    # Sous-graphe atteignable dans le temps imparti
    sub = nx.ego_graph(
        G, center,
        radius=minutes * 60,
        distance="travel_time"
    )
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

    else:  # Buffer sur noeuds
        return unary_union([p.buffer(0.002) for p in pts])


def compare_engines_info() -> dict:
    """
    Retourne les métadonnées des moteurs disponibles pour l'affichage UI.
    """
    return {
        "OSM local (OSMnx + NetworkX) — Gratuit": {
            "badge_class": "badge-local",
            "badge_label": "LOCAL",
            "description": "Calcul 100% local avec routes OSM. Colonnes utilisées : Tps_min, Vit_kmh, maxspeed.",
            "modes": {
                "🚶 Marche": ("walk", "walk"),
                "🚗 Véhicule": ("drive", "drive"),
                "🚲 Vélo": ("bike", "bike"),
            },
        },
        "OpenRouteService (ORS) — Clé API": {
            "badge_class": "badge-api",
            "badge_label": "API",
            "description": "Très précis. Clé gratuite sur openrouteservice.org (500 req/jour).",
            "modes": {
                "🚶 Marche": ("foot-walking", "walk"),
                "🚗 Voiture": ("driving-car", "drive"),
                "🚲 Vélo": ("cycling-regular", "bike"),
                "🛵 HGV": ("driving-hgv", "drive"),
            },
        },
        "OSRM Public — Gratuit": {
            "badge_class": "badge-free",
            "badge_label": "GRATUIT",
            "description": "router.project-osrm.org — sans clé. Isochrones approx. par matrice de durées.",
            "modes": {
                "🚗 Voiture": ("car", "drive"),
                "🚲 Vélo": ("bike", "bike"),
                "🚶 Marche": ("foot", "walk"),
            },
        },
        "Valhalla Public — Gratuit": {
            "badge_class": "badge-free",
            "badge_label": "GRATUIT",
            "description": "valhalla.openstreetmap.de — polygones natifs précis, 4 modes, sans clé.",
            "modes": {
                "🚶 Marche": ("pedestrian", "walk"),
                "🚗 Voiture": ("auto", "drive"),
                "🚲 Vélo": ("bicycle", "bike"),
                "🛵 Moto": ("motorcycle", "drive"),
            },
        },
    }
