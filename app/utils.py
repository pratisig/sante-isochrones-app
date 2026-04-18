"""
utils.py — Fonctions utilitaires pour le calcul d'isochrones et d'itinéraires
Moteurs : OSMnx (local), OSM Pur Tps_min, ORS (API), OSRM (public), Valhalla (public), GraphHopper (API)
"""
import os
import math
import tempfile
import numpy as np
import networkx as nx
import osmnx as ox
import requests
import json
import time
from shapely.geometry import Point, Polygon, MultiPolygon, shape, mapping, LineString
from shapely.ops import unary_union
from shapely.validation import make_valid
import geopandas as gpd
import alphashape
import warnings
warnings.filterwarnings("ignore")

from projection import auto_utm_epsg, auto_utm_epsg_from_gdf, reproject_to_utm, compute_area_km2

gpd.options.io_engine = "pyogrio"


def read_geodata(source) -> gpd.GeoDataFrame:
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


def clean_geometry(geom):
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    return geom if not geom.is_empty else None


def build_polygon_from_points(pts: list, method: str, alpha: float):
    if len(pts) < 3:
        return None
    coords = np.array([(p.x, p.y) for p in pts])

    if method == "Alpha Shape (recommandé)":
        try:
            poly = alphashape.alphashape(coords, alpha)
            poly = clean_geometry(poly)
            if poly and not poly.is_empty:
                return poly
        except Exception:
            pass
        hull = unary_union(pts).convex_hull
        return clean_geometry(hull)

    elif method == "Convex Hull":
        hull = unary_union(pts).convex_hull
        return clean_geometry(hull)

    elif method == "Buffer sur noeuds":
        if len(coords) > 0:
            from scipy.spatial.distance import cdist
            try:
                dists = cdist(coords[:min(50, len(coords))], coords[:min(50, len(coords))])
                np.fill_diagonal(dists, np.inf)
                median_dist = np.median(dists.min(axis=1))
                buf_deg = max(0.0005, min(median_dist * 2, 0.01))
            except Exception:
                buf_deg = 0.001
        else:
            buf_deg = 0.001
        merged = unary_union([p.buffer(buf_deg) for p in pts])
        return clean_geometry(merged)

    else:
        hull = unary_union(pts).convex_hull
        return clean_geometry(hull)


def _extract_linestring_from_geojson(data):
    routes = data.get("routes", [])
    if not routes:
        return None
    geom = routes[0].get("geometry")
    if isinstance(geom, dict):
        return clean_geometry(shape(geom))
    if isinstance(geom, str):
        try:
            import polyline
            coords = polyline.decode(geom)
            return LineString([(lon, lat) for lat, lon in coords])
        except Exception:
            return None
    return None


def _route_summary(geometry, distance_m=None, duration_s=None, engine=""):
    return {
        "geometry": clean_geometry(geometry),
        "distance_km": round((distance_m or 0) / 1000, 3) if distance_m is not None else None,
        "duration_min": round((duration_s or 0) / 60, 2) if duration_s is not None else None,
        "engine": engine,
    }


def compute_osmnx_isochrone(lon: float, lat: float, minutes: int,
                            mode: str, method: str, alpha: float,
                            gdf_roads=None):
    DEFAULT_SPEEDS = {"walk": 4.5, "bike": 15.0, "drive": 40.0}
    spd = DEFAULT_SPEEDS.get(mode, 4.5)

    dist = int((minutes / 60) * spd * 1000 * 1.6) + 600
    dist = max(dist, 1000)
    dist = min(dist, 80000)

    G = ox.graph_from_point((lat, lon), dist=dist, network_type=mode, simplify=True)
    center = ox.nearest_nodes(G, lon, lat)

    if gdf_roads is not None and hasattr(gdf_roads, "crs") and gdf_roads.crs:
        if gdf_roads.crs.to_epsg() != 4326:
            gdf_roads = gdf_roads.to_crs(4326)

    road_speeds = {}
    road_times  = {}

    if gdf_roads is not None:
        for _, row in gdf_roads.iterrows():
            oid = row.get("osm_id") or row.get("OSM_ID")
            if oid is None:
                continue
            oid = str(oid)

            v = None
            for col in ("Vit_kmh", "vit_kmh", "vitesse_kmh", "speed_kmh"):
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
                road_speeds[oid] = v

            for col in ("Tps_min", "tps_min", "time_min", "duree_min"):
                val = row.get(col)
                if val is not None:
                    try:
                        fval = float(val)
                        if not np.isnan(fval) and fval > 0:
                            road_times[oid] = fval * 60
                            break
                    except (ValueError, TypeError):
                        pass

    for _, _, d in G.edges(data=True):
        osmid = str(d.get("osmid", ""))
        if osmid in road_times:
            d["travel_time"] = road_times[osmid]
            continue

        edge_speed = road_speeds.get(osmid, spd)
        if osmid not in road_speeds:
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

    target_sec = minutes * 60
    sub = nx.ego_graph(G, center, radius=target_sec, distance="travel_time")
    pts = [Point(G.nodes[n]["x"], G.nodes[n]["y"]) for n in sub.nodes()]

    if len(pts) < 4:
        return Point(lon, lat).buffer(0.005)

    return build_polygon_from_points(pts, method, alpha)


def compute_osm_pur_isochrone(lon: float, lat: float, minutes: int,
                              gdf_roads: gpd.GeoDataFrame,
                              method: str, alpha: float):
    if gdf_roads is None or len(gdf_roads) == 0:
        raise ValueError("La couche de routes OSM est requise pour le moteur OSM Pur.")

    if gdf_roads.crs and gdf_roads.crs.to_epsg() != 4326:
        gdf_roads = gdf_roads.to_crs(4326)

    has_tps = "Tps_min" in gdf_roads.columns
    has_vit = "Vit_kmh" in gdf_roads.columns
    has_max = "maxspeed" in gdf_roads.columns

    G = nx.Graph()
    node_counter = 0
    node_map = {}

    def get_or_create_node(x, y):
        nonlocal node_counter
        key = (round(x, 6), round(y, 6))
        if key not in node_map:
            node_map[key] = node_counter
            G.add_node(node_counter, x=x, y=y)
            node_counter += 1
        return node_map[key]

    for _, row in gdf_roads.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "LineString":
            segments = [(geom.coords[i], geom.coords[i + 1]) for i in range(len(geom.coords) - 1)]
        elif geom.geom_type == "MultiLineString":
            segments = []
            for part in geom.geoms:
                for i in range(len(part.coords) - 1):
                    segments.append((part.coords[i], part.coords[i + 1]))
        else:
            continue

        travel_time_sec = None
        if has_tps:
            tps = row.get("Tps_min")
            if tps is not None:
                try:
                    fval = float(tps)
                    if not np.isnan(fval) and fval > 0:
                        travel_time_sec = fval * 60
                except (ValueError, TypeError):
                    pass

        if travel_time_sec is None:
            speed = 30.0
            if has_vit:
                v = row.get("Vit_kmh")
                if v is not None:
                    try:
                        fval = float(v)
                        if not np.isnan(fval) and fval > 0:
                            speed = fval
                    except (ValueError, TypeError):
                        pass
            elif has_max:
                ms = row.get("maxspeed")
                if ms:
                    try:
                        speed = float(str(ms).split()[0])
                    except (ValueError, TypeError):
                        pass

            gs = gpd.GeoSeries([geom], crs=4326)
            try:
                epsg = auto_utm_epsg(lon, lat)
                length_m = gs.to_crs(epsg).length.values[0]
            except Exception:
                length_m = geom.length * 111000
            travel_time_sec = length_m / (max(speed, 1) * 1000 / 3600)

        time_per_seg = travel_time_sec / max(len(segments), 1)

        for (x1, y1), (x2, y2) in segments:
            u = get_or_create_node(x1, y1)
            v = get_or_create_node(x2, y2)
            if not G.has_edge(u, v):
                G.add_edge(u, v, travel_time=time_per_seg)
            else:
                existing = G[u][v].get("travel_time", time_per_seg)
                G[u][v]["travel_time"] = min(existing, time_per_seg)

    if G.number_of_nodes() == 0:
        raise ValueError("Le graphe OSM local est vide. Vérifiez la couche de routes.")

    nodes_xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in G.nodes()])
    node_ids = list(G.nodes())
    dists = np.sqrt((nodes_xy[:, 0] - lon) ** 2 + (nodes_xy[:, 1] - lat) ** 2)
    source = node_ids[np.argmin(dists)]

    lengths = nx.single_source_dijkstra_path_length(G, source, cutoff=minutes * 60, weight="travel_time")
    accessible_nodes = list(lengths.keys())
    if len(accessible_nodes) < 4:
        return Point(lon, lat).buffer(0.005)

    pts = [Point(G.nodes[n]["x"], G.nodes[n]["y"]) for n in accessible_nodes]
    return build_polygon_from_points(pts, method, alpha)


def isochrone_ors(lon: float, lat: float, minutes: int, profile: str, key: str):
    if not key or not key.strip():
        raise Exception("Clé API ORS manquante. Obtenez-en une sur openrouteservice.org")
    url = f"https://api.openrouteservice.org/v2/isochrones/{profile}"
    resp = requests.post(url, json={
        "locations": [[lon, lat]],
        "range": [minutes * 60],
        "smoothing": 5,
        "attributes": ["area"]
    }, headers={"Authorization": key, "Content-Type": "application/json"}, timeout=25)
    if resp.status_code == 200:
        data = resp.json()
        if "features" in data and data["features"]:
            return clean_geometry(shape(data["features"][0]["geometry"]))
        raise Exception("ORS : réponse vide")
    raise Exception(f"ORS {resp.status_code}: {resp.text[:300]}")


def isochrone_osrm(lon: float, lat: float, minutes: int, profile: str):
    speeds = {"foot": 4.5, "bike": 15.0, "car": 50.0}
    spd = speeds.get(profile, 50.0)
    r_base = (minutes / 60) * spd * 1000
    re = 6371000.0
    radii = [r_base * 0.8, r_base * 1.2]
    n = 36
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    dests = []
    dlats_all = []
    dlons_all = []
    for r in radii:
        dlats = [np.degrees(r / re * np.cos(a)) for a in angles]
        dlons = [np.degrees(r / re * np.sin(a) / np.cos(np.radians(lat))) for a in angles]
        for i in range(n):
            dests.append((lon + dlons[i], lat + dlats[i]))
            dlats_all.append(dlats[i])
            dlons_all.append(dlons[i])

    coords_str = f"{lon},{lat};" + ";".join(f"{d[0]:.6f},{d[1]:.6f}" for d in dests)
    dest_indices = ";".join(str(i + 1) for i in range(len(dests)))
    url = f"http://router.project-osrm.org/table/v1/{profile}/{coords_str}?sources=0&destinations={dest_indices}&annotations=duration"

    resp = requests.get(url, timeout=35)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok":
        raise Exception(f"OSRM erreur : {data.get('message', data.get('code', 'inconnu'))}")

    durations = data["durations"][0]
    target_sec = minutes * 60
    pts = []
    for i, dur in enumerate(durations):
        if dur is None or dur == 0:
            pts.append(Point(dests[i][0], dests[i][1]))
            continue
        ratio = min(1.0, target_sec / dur)
        pts.append(Point(lon + dlons_all[i] * ratio, lat + dlats_all[i] * ratio))

    if len(pts) < 3:
        return Point(lon, lat).buffer(0.005)
    coords = np.array([(p.x, p.y) for p in pts])
    try:
        poly = alphashape.alphashape(coords, 0.25)
        poly = clean_geometry(poly)
        if poly:
            return poly
    except Exception:
        pass
    return clean_geometry(unary_union(pts).convex_hull)


def isochrone_valhalla(lon: float, lat: float, minutes: int, costing: str):
    url = "https://valhalla1.openstreetmap.de/isochrone"
    body = {
        "locations": [{"lon": lon, "lat": lat}],
        "costing": costing,
        "contours": [{"time": minutes, "color": "ff0000"}],
        "polygons": True,
        "generalize": 30
    }
    resp = requests.get(url, params={"json": json.dumps(body)}, timeout=35)
    resp.raise_for_status()
    data = resp.json()
    if "features" in data and data["features"]:
        return clean_geometry(shape(data["features"][0]["geometry"]))
    raise Exception(f"Valhalla : réponse vide — {data.get('error', 'inconnu')}")


def isochrone_graphhopper(lon: float, lat: float, minutes: int, profile: str, key: str = ""):
    url = "https://graphhopper.com/api/1/isochrone"
    params = {
        "point": f"{lat},{lon}",
        "time_limit": minutes * 60,
        "vehicle": profile,
        "buckets": 1,
        "type": "json",
        "key": key.strip() if key else "",
    }
    resp = requests.get(url, params=params, timeout=25)
    if resp.status_code == 401:
        raise Exception("GraphHopper requiert une clé API.")
    if resp.status_code != 200:
        raise Exception(f"GraphHopper {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    polygons = data.get("polygons", [])
    if polygons:
        return clean_geometry(shape(polygons[0]["geometry"]))
    raise Exception("GraphHopper : aucun polygone retourné")


def route_ors(start_lon, start_lat, end_lon, end_lat, profile, key):
    if not key or not key.strip():
        raise Exception("Clé API ORS manquante.")
    url = f"https://api.openrouteservice.org/v2/directions/{profile}/geojson"
    body = {"coordinates": [[start_lon, start_lat], [end_lon, end_lat]]}
    resp = requests.post(url, json=body, headers={"Authorization": key, "Content-Type": "application/json"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    feat = data["features"][0]
    geom = shape(feat["geometry"])
    summary = feat["properties"].get("summary", {})
    return _route_summary(geom, summary.get("distance"), summary.get("duration"), "ORS")


def route_osrm(start_lon, start_lat, end_lon, end_lat, profile):
    url = f"https://router.project-osrm.org/route/v1/{profile}/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson&steps=true"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    routes = data.get("routes", [])
    if not routes:
        raise Exception("OSRM : aucun itinéraire")
    route = routes[0]
    geom = shape(route["geometry"])
    return _route_summary(geom, route.get("distance"), route.get("duration"), "OSRM")


def route_graphhopper(start_lon, start_lat, end_lon, end_lat, profile, key):
    if not key or not key.strip():
        raise Exception("Clé GraphHopper manquante.")
    url = "https://graphhopper.com/api/1/route"
    params = {
        "point": [f"{start_lat},{start_lon}", f"{end_lat},{end_lon}"],
        "profile": profile,
        "points_encoded": False,
        "instructions": True,
        "key": key.strip(),
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    paths = data.get("paths", [])
    if not paths:
        raise Exception("GraphHopper : aucun itinéraire")
    path = paths[0]
    geom = shape(path["points"])
    return _route_summary(geom, path.get("distance"), path.get("time", 0) / 1000, "GraphHopper")


def route_osmnx(start_lon, start_lat, end_lon, end_lat, mode="drive", gdf_roads=None):
    dist = max(3000, int(np.sqrt((end_lon - start_lon) ** 2 + (end_lat - start_lat) ** 2) * 111000 * 2.2))
    dist = min(dist, 120000)
    G = ox.graph_from_point(((start_lat + end_lat) / 2, (start_lon + end_lon) / 2), dist=dist, network_type=mode, simplify=True)

    default_speeds = {"walk": 4.5, "bike": 15.0, "drive": 40.0}
    base_speed = default_speeds.get(mode, 40.0)

    for _, _, d in G.edges(data=True):
        ms = d.get("maxspeed")
        edge_speed = base_speed
        if isinstance(ms, list):
            ms = ms[0]
        if ms:
            try:
                edge_speed = float(str(ms).split()[0])
            except Exception:
                pass
        length = d.get("length", 1)
        d["travel_time"] = length / (max(edge_speed, 1) * 1000 / 3600)

    start_node = ox.nearest_nodes(G, start_lon, start_lat)
    end_node = ox.nearest_nodes(G, end_lon, end_lat)
    route = nx.shortest_path(G, start_node, end_node, weight="travel_time")
    coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in route]
    line = LineString(coords)

    distance_m = 0.0
    duration_s = 0.0
    for u, v in zip(route[:-1], route[1:]):
        edge_data = G.get_edge_data(u, v)
        if isinstance(edge_data, dict):
            first_key = next(iter(edge_data)) if 0 in edge_data or len(edge_data) else None
            if first_key is not None and isinstance(edge_data.get(first_key), dict):
                edge = edge_data[first_key]
            else:
                edge = edge_data
            distance_m += edge.get("length", 0)
            duration_s += edge.get("travel_time", 0)

    return _route_summary(line, distance_m, duration_s, "OSMnx")


def compute_isochrone(engine: str, lon: float, lat: float, minutes: int,
                      mode_api: str, mode_osm: str,
                      iso_method: str, alpha: float,
                      ors_key: str = "", gh_key: str = "",
                      gdf_roads=None):
    if gdf_roads is not None and not hasattr(gdf_roads, "geometry"):
        gdf_roads = read_geodata(gdf_roads)
    if gdf_roads is not None and hasattr(gdf_roads, "crs") and gdf_roads.crs:
        if gdf_roads.crs.to_epsg() != 4326:
            gdf_roads = gdf_roads.to_crs(4326)

    if "OSM Pur" in engine:
        return compute_osm_pur_isochrone(lon, lat, minutes, gdf_roads, iso_method, alpha)
    elif "OSM local" in engine:
        return compute_osmnx_isochrone(lon, lat, minutes, mode_osm, iso_method, alpha, gdf_roads)
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


def compute_route(engine: str, start_lon: float, start_lat: float,
                  end_lon: float, end_lat: float,
                  mode_api: str, mode_osm: str,
                  ors_key: str = "", gh_key: str = "", gdf_roads=None):
    if "ORS" in engine:
        return route_ors(start_lon, start_lat, end_lon, end_lat, mode_api, ors_key)
    elif "OSRM" in engine:
        return route_osrm(start_lon, start_lat, end_lon, end_lat, mode_api)
    elif "GraphHopper" in engine:
        return route_graphhopper(start_lon, start_lat, end_lon, end_lat, mode_api, gh_key)
    elif "OSM local" in engine or "OSM Pur" in engine:
        return route_osmnx(start_lon, start_lat, end_lon, end_lat, mode_osm, gdf_roads)
    elif "Valhalla" in engine:
        return route_osrm(start_lon, start_lat, end_lon, end_lat, "car" if mode_api in ["auto", "motorcycle"] else ("bike" if mode_api == "bicycle" else "foot"))
    else:
        raise ValueError(f"Moteur inconnu: {engine}")


def search_places_osm(query: str, limit: int = 10):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "jsonv2", "limit": limit, "addressdetails": 1}
    headers = {"User-Agent": "sante-isochrones-app/2.0"}
    resp = requests.get(url, params=params, headers=headers, timeout=25)
    resp.raise_for_status()
    data = resp.json()
    out = []
    for i, row in enumerate(data):
        out.append({
            "id": i,
            "name": row.get("display_name", query),
            "lon": float(row["lon"]),
            "lat": float(row["lat"]),
            "class": row.get("class", ""),
            "type": row.get("type", ""),
        })
    return out


def engines_metadata() -> dict:
    return {
        "OSM Pur (Tps_min local) — Recommandé": {
            "badge_class": "badge-local", "badge_label": "LOCAL ★",
            "description": "Construit le graphe 100% depuis votre couche OSM (Tps_min, Vit_kmh). Aucun téléchargement. Plus fidèle à la réalité terrain.",
            "needs_key": False, "needs_roads": True,
            "modes": {
                "🚗 Véhicule": ("drive", "drive"),
                "🚶 Marche": ("walk", "walk"),
                "🚲 Vélo": ("bike", "bike"),
            },
        },
        "OSM local (OSMnx + NetworkX) — Gratuit": {
            "badge_class": "badge-local", "badge_label": "LOCAL",
            "description": "Télécharge le réseau OSM + enrichit avec votre couche locale.",
            "needs_key": False, "needs_roads": False,
            "modes": {
                "🚶 Marche": ("walk", "walk"),
                "🚗 Véhicule": ("drive", "drive"),
                "🚲 Vélo": ("bike", "bike"),
            },
        },
        "OpenRouteService (ORS) — Clé API": {
            "badge_class": "badge-api", "badge_label": "API",
            "description": "Isochrones et itinéraires précis avec clé API.",
            "needs_key": True, "key_name": "ORS_API_KEY",
            "key_url": "https://openrouteservice.org/dev/#/signup",
            "needs_roads": False,
            "modes": {
                "🚶 Marche": ("foot-walking", "walk"),
                "🚗 Voiture": ("driving-car", "drive"),
                "🚲 Vélo": ("cycling-regular", "bike"),
                "🛵 HGV": ("driving-hgv", "drive"),
            },
        },
        "OSRM Public — Gratuit": {
            "badge_class": "badge-free", "badge_label": "GRATUIT",
            "description": "Sans clé. Itinéraires et isochrones rapides.",
            "needs_key": False, "needs_roads": False,
            "modes": {
                "🚗 Voiture": ("car", "drive"),
                "🚲 Vélo": ("bike", "bike"),
                "🚶 Marche": ("foot", "walk"),
            },
        },
        "Valhalla Public — Gratuit": {
            "badge_class": "badge-free", "badge_label": "GRATUIT",
            "description": "Polygones précis, multi-modes, sans clé.",
            "needs_key": False, "needs_roads": False,
            "modes": {
                "🚶 Marche": ("pedestrian", "walk"),
                "🚗 Voiture": ("auto", "drive"),
                "🚲 Vélo": ("bicycle", "bike"),
                "🛵 Moto": ("motorcycle", "drive"),
            },
        },
        "GraphHopper — Clé API": {
            "badge_class": "badge-api", "badge_label": "API",
            "description": "Itinéraires et polygones précis, clé requise.",
            "needs_key": True, "key_name": "GH_API_KEY",
            "key_url": "https://www.graphhopper.com/dashboard/",
            "needs_roads": False,
            "modes": {
                "🚗 Voiture": ("car", "drive"),
                "🚶 Marche": ("foot", "walk"),
                "🚲 Vélo": ("bike", "bike"),
                "🛵 Moto": ("motorcycle", "drive"),
                "🥾 Randonnée": ("hike", "walk"),
            },
        },
    }
