"""
utils.py — Fonctions utilitaires pour isochrones ET itinéraires
Moteurs : OSMnx (local), OSM Pur Tps_min, ORS (API), OSRM (public), Valhalla (public), GraphHopper (API)
Sources de points : fichier spatial, saisie manuelle, GPS, Nominatim, clic carte
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


# ─────────────────────────────────────────────────────────────────
# UTILITAIRE : lecture robuste de fichiers spatiaux
# ─────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────
# NOMINATIM — géocodage libre (sans clé, monde entier)
# ─────────────────────────────────────────────────────────────────
def geocode_nominatim(query: str) -> list:
    """Retourne une liste de dict {name, lon, lat, display_name}."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 6, "addressdetails": 1}
    try:
        resp = requests.get(url, params=params,
                            headers={"User-Agent": "GeoAnalyse-App/2.0"}, timeout=10)
        resp.raise_for_status()
        results = resp.json()
        out = []
        for r in results:
            out.append({
                "name":         r.get("display_name", query)[:80],
                "lon":          float(r["lon"]),
                "lat":          float(r["lat"]),
                "display_name": r.get("display_name", ""),
            })
        return out
    except Exception as e:
        raise Exception(f"Nominatim : {e}")


def reverse_geocode(lon: float, lat: float) -> str:
    """Nom du lieu à partir des coordonnées."""
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lon": lon, "lat": lat, "format": "json"}
    try:
        resp = requests.get(url, params=params,
                            headers={"User-Agent": "GeoAnalyse-App/2.0"}, timeout=8)
        data = resp.json()
        return data.get("display_name", f"{lat:.4f}, {lon:.4f}")[:80]
    except Exception:
        return f"{lat:.4f}, {lon:.4f}"


# ─────────────────────────────────────────────────────────────────
# UTILITAIRE : nettoyage et validation géométrie
# ─────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────
# ISOCHRONES — Moteur 1 : OSMnx local
# ─────────────────────────────────────────────────────────────────
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

    for u, v_node, d in G.edges(data=True):
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


# ─────────────────────────────────────────────────────────────────
# ISOCHRONES — Moteur 2 : OSM Pur (depuis couche locale)
# ─────────────────────────────────────────────────────────────────
def compute_osm_pur_isochrone(lon: float, lat: float, minutes: int,
                              gdf_roads: gpd.GeoDataFrame,
                              method: str, alpha: float):
    if gdf_roads is None or len(gdf_roads) == 0:
        raise ValueError("La couche de routes OSM est requise pour le moteur OSM Pur.")

    if gdf_roads.crs and gdf_roads.crs.to_epsg() != 4326:
        gdf_roads = gdf_roads.to_crs(4326)

    has_tps   = "Tps_min" in gdf_roads.columns
    has_vit   = "Vit_kmh" in gdf_roads.columns
    has_max   = "maxspeed" in gdf_roads.columns

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
            segments = [(geom.coords[i], geom.coords[i+1]) for i in range(len(geom.coords)-1)]
        elif geom.geom_type == "MultiLineString":
            segments = []
            for part in geom.geoms:
                for i in range(len(part.coords)-1):
                    segments.append((part.coords[i], part.coords[i+1]))
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

        n_segs = len(segments)
        time_per_seg = travel_time_sec / max(n_segs, 1)

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
    dists = np.sqrt((nodes_xy[:, 0] - lon)**2 + (nodes_xy[:, 1] - lat)**2)
    source = node_ids[np.argmin(dists)]

    target_sec = minutes * 60
    lengths = nx.single_source_dijkstra_path_length(G, source, cutoff=target_sec, weight="travel_time")
    accessible_nodes = list(lengths.keys())

    if len(accessible_nodes) < 4:
        return Point(lon, lat).buffer(0.005)

    pts = [Point(G.nodes[n]["x"], G.nodes[n]["y"]) for n in accessible_nodes]
    return build_polygon_from_points(pts, method, alpha)


# ─────────────────────────────────────────────────────────────────
# ISOCHRONES — Moteur 3 : ORS
# ─────────────────────────────────────────────────────────────────
def isochrone_ors(lon: float, lat: float, minutes: int, profile: str, key: str):
    if not key or not key.strip():
        raise Exception("Clé API ORS manquante. openrouteservice.org")
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


# ─────────────────────────────────────────────────────────────────
# ISOCHRONES — Moteur 4 : OSRM (72 points)
# ─────────────────────────────────────────────────────────────────
def isochrone_osrm(lon: float, lat: float, minutes: int, profile: str):
    speeds = {"foot": 4.5, "bike": 15.0, "car": 50.0}
    spd = speeds.get(profile, 50.0)
    R_base = (minutes / 60) * spd * 1000
    Re = 6371000.0
    radii = [R_base * 0.8, R_base * 1.2]
    N = 36
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    dests, dlats_all, dlons_all = [], [], []
    for R in radii:
        dlats = [np.degrees(R / Re * np.cos(a)) for a in angles]
        dlons = [np.degrees(R / Re * np.sin(a) / np.cos(np.radians(lat))) for a in angles]
        for i in range(N):
            dests.append((lon + dlons[i], lat + dlats[i]))
            dlats_all.append(dlats[i])
            dlons_all.append(dlons[i])

    coords_str = f"{lon},{lat};" + ";".join(f"{d[0]:.6f},{d[1]:.6f}" for d in dests)
    dest_indices = ";".join(str(i + 1) for i in range(len(dests)))
    url = (f"http://router.project-osrm.org/table/v1/{profile}/{coords_str}"
           f"?sources=0&destinations={dest_indices}&annotations=duration")

    resp = requests.get(url, timeout=35)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok":
        raise Exception(f"OSRM erreur : {data.get('message', 'inconnu')}")

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


# ─────────────────────────────────────────────────────────────────
# ISOCHRONES — Moteur 5 : Valhalla
# ─────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────
# ISOCHRONES — Moteur 6 : GraphHopper
# ─────────────────────────────────────────────────────────────────
def isochrone_graphhopper(lon: float, lat: float, minutes: int, profile: str, key: str = ""):
    url = "https://graphhopper.com/api/1/isochrone"
    params = {"point": f"{lat},{lon}", "time_limit": minutes * 60, "vehicle": profile, "buckets": 1, "type": "json"}
    if key and key.strip():
        params["key"] = key.strip()
    resp = requests.get(url, params=params, timeout=25)
    if resp.status_code == 401:
        raise Exception("GraphHopper requiert une clé API. graphhopper.com/dashboard")
    if resp.status_code != 200:
        raise Exception(f"GraphHopper {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    polygons = data.get("polygons", [])
    if polygons:
        return clean_geometry(shape(polygons[0]["geometry"]))
    raise Exception("GraphHopper : aucun polygone retourné")


# ─────────────────────────────────────────────────────────────────
# ISOCHRONES — Dispatcher central
# ─────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────
# ITINÉRAIRES — ORS
# ─────────────────────────────────────────────────────────────────
def route_ors(lon_o: float, lat_o: float, lon_d: float, lat_d: float,
              profile: str, key: str) -> dict:
    if not key or not key.strip():
        raise Exception("Clé API ORS manquante.")
    url = f"https://api.openrouteservice.org/v2/directions/{profile}/geojson"
    resp = requests.post(url,
        json={"coordinates": [[lon_o, lat_o], [lon_d, lat_d]], "instructions": True,
              "language": "fr", "units": "km"},
        headers={"Authorization": key, "Content-Type": "application/json"}, timeout=25)
    if resp.status_code != 200:
        raise Exception(f"ORS route {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    feature = data["features"][0]
    props = feature["properties"]
    summary = props["summary"]
    coords = [(c[0], c[1]) for c in feature["geometry"]["coordinates"]]
    steps = []
    for seg in props.get("segments", []):
        for s in seg.get("steps", []):
            steps.append({
                "instruction": s.get("instruction", ""),
                "distance_km": round(s.get("distance", 0), 2),
                "duration_min": round(s.get("duration", 0) / 60, 1),
            })
    return {
        "engine": "ORS",
        "distance_km": round(summary["distance"], 2),
        "duration_min": round(summary["duration"] / 60, 1),
        "geometry": LineString(coords) if len(coords) >= 2 else None,
        "steps": steps,
    }


# ─────────────────────────────────────────────────────────────────
# ITINÉRAIRES — OSRM
# ─────────────────────────────────────────────────────────────────
def route_osrm(lon_o: float, lat_o: float, lon_d: float, lat_d: float,
               profile: str) -> dict:
    url = (f"http://router.project-osrm.org/route/v1/{profile}/"
           f"{lon_o},{lat_o};{lon_d},{lat_d}"
           f"?overview=full&geometries=geojson&steps=true")
    resp = requests.get(url, timeout=25)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok":
        raise Exception(f"OSRM route : {data.get('message', 'inconnu')}")
    route = data["routes"][0]
    coords = [(c[0], c[1]) for c in route["geometry"]["coordinates"]]
    steps = []
    for leg in route.get("legs", []):
        for s in leg.get("steps", []):
            maneuver = s.get("maneuver", {})
            steps.append({
                "instruction": f"{maneuver.get('type', '')} {maneuver.get('modifier', '')}".strip(),
                "distance_km": round(s.get("distance", 0) / 1000, 2),
                "duration_min": round(s.get("duration", 0) / 60, 1),
                "name": s.get("name", ""),
            })
    return {
        "engine": "OSRM",
        "distance_km": round(route["distance"] / 1000, 2),
        "duration_min": round(route["duration"] / 60, 1),
        "geometry": LineString(coords) if len(coords) >= 2 else None,
        "steps": steps,
    }


# ─────────────────────────────────────────────────────────────────
# ITINÉRAIRES — Valhalla
# ─────────────────────────────────────────────────────────────────
def route_valhalla(lon_o: float, lat_o: float, lon_d: float, lat_d: float,
                   costing: str) -> dict:
    url = "https://valhalla1.openstreetmap.de/route"
    body = {
        "locations": [{"lon": lon_o, "lat": lat_o}, {"lon": lon_d, "lat": lat_d}],
        "costing": costing,
        "directions_options": {"language": "fr-FR", "units": "kilometers"},
    }
    resp = requests.post(url, json=body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    trip = data.get("trip", {})
    summary = trip.get("summary", {})
    legs = trip.get("legs", [])
    steps = []
    coords = []
    for leg in legs:
        try:
            import polyline as pl
            shape_pts = pl.decode(leg.get("shape", ""), precision=6)
            coords += [(pt[1], pt[0]) for pt in shape_pts]
        except Exception:
            pass
        for m in leg.get("maneuvers", []):
            steps.append({
                "instruction": m.get("instruction", ""),
                "distance_km": round(m.get("length", 0), 2),
                "duration_min": round(m.get("time", 0) / 60, 1),
            })
    return {
        "engine": "Valhalla",
        "distance_km": round(summary.get("length", 0), 2),
        "duration_min": round(summary.get("time", 0) / 60, 1),
        "geometry": LineString(coords) if len(coords) >= 2 else None,
        "steps": steps,
    }


# ─────────────────────────────────────────────────────────────────
# ITINÉRAIRES — GraphHopper
# ─────────────────────────────────────────────────────────────────
def route_graphhopper(lon_o: float, lat_o: float, lon_d: float, lat_d: float,
                      profile: str, key: str = "") -> dict:
    url = "https://graphhopper.com/api/1/route"
    params = {
        "point": [f"{lat_o},{lon_o}", f"{lat_d},{lon_d}"],
        "vehicle": profile,
        "locale": "fr",
        "calc_points": True,
        "points_encoded": False,
        "instructions": True,
    }
    if key and key.strip():
        params["key"] = key.strip()
    resp = requests.get(url, params=params, timeout=25)
    if resp.status_code == 401:
        raise Exception("GraphHopper requiert une clé API.")
    if resp.status_code != 200:
        raise Exception(f"GraphHopper route {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    path = data["paths"][0]
    coords = [(c[0], c[1]) for c in path["points"]["coordinates"]]
    steps = []
    for inst in path.get("instructions", []):
        steps.append({
            "instruction": inst.get("text", ""),
            "distance_km": round(inst.get("distance", 0) / 1000, 2),
            "duration_min": round(inst.get("time", 0) / 60000, 1),
        })
    return {
        "engine": "GraphHopper",
        "distance_km": round(path["distance"] / 1000, 2),
        "duration_min": round(path["time"] / 60000, 1),
        "geometry": LineString(coords) if len(coords) >= 2 else None,
        "steps": steps,
    }


# ─────────────────────────────────────────────────────────────────
# ITINÉRAIRES — OSMnx local
# ─────────────────────────────────────────────────────────────────
def route_osmnx(lon_o: float, lat_o: float, lon_d: float, lat_d: float,
                mode: str) -> dict:
    mid_lon = (lon_o + lon_d) / 2
    mid_lat = (lat_o + lat_d) / 2
    dist_deg = math.sqrt((lon_d - lon_o)**2 + (lat_d - lat_o)**2)
    dist_m = max(dist_deg * 111000 * 2.5, 2000)

    G = ox.graph_from_point((mid_lat, mid_lon), dist=int(dist_m), network_type=mode, simplify=True)
    orig = ox.nearest_nodes(G, lon_o, lat_o)
    dest = ox.nearest_nodes(G, lon_d, lat_d)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    route_nodes = ox.shortest_path(G, orig, dest, weight="travel_time")
    if route_nodes is None:
        raise Exception("OSMnx : aucun chemin trouvé entre les deux points.")

    coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in route_nodes]
    total_dist = 0.0
    total_time = 0.0
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        edge_data = list(G.get_edge_data(u, v).values())[0]
        total_dist += edge_data.get("length", 0)
        total_time += edge_data.get("travel_time", 0)

    return {
        "engine": "OSMnx",
        "distance_km": round(total_dist / 1000, 2),
        "duration_min": round(total_time / 60, 1),
        "geometry": LineString(coords) if len(coords) >= 2 else None,
        "steps": [],
    }


# ─────────────────────────────────────────────────────────────────
# ITINÉRAIRES — Dispatcher central
# ─────────────────────────────────────────────────────────────────
def compute_route(engine: str, lon_o: float, lat_o: float,
                  lon_d: float, lat_d: float,
                  mode_api: str, mode_osm: str,
                  ors_key: str = "", gh_key: str = "") -> dict:
    if "OSM local" in engine or "OSMnx" in engine:
        return route_osmnx(lon_o, lat_o, lon_d, lat_d, mode_osm)
    elif "ORS" in engine:
        return route_ors(lon_o, lat_o, lon_d, lat_d, mode_api, ors_key)
    elif "OSRM" in engine:
        return route_osrm(lon_o, lat_o, lon_d, lat_d, mode_api)
    elif "Valhalla" in engine:
        return route_valhalla(lon_o, lat_o, lon_d, lat_d, mode_api)
    elif "GraphHopper" in engine:
        return route_graphhopper(lon_o, lat_o, lon_d, lat_d, mode_api, gh_key)
    else:
        raise ValueError(f"Moteur inconnu: {engine}")


# ─────────────────────────────────────────────────────────────────
# Métadonnées moteurs (pour l'UI)
# ─────────────────────────────────────────────────────────────────
def engines_metadata() -> dict:
    return {
        "OSM Pur (Tps_min local) — Recommandé": {
            "badge_class": "badge-local", "badge_label": "LOCAL ★",
            "description": "Graphe 100% depuis votre couche OSM (Tps_min, Vit_kmh). Aucun téléchargement.",
            "needs_key": False, "needs_roads": True, "supports_route": False,
            "modes": {
                "🚗 Véhicule": ("drive", "drive"),
                "🚶 Marche":   ("walk",  "walk"),
                "🚲 Vélo":     ("bike",  "bike"),
            },
        },
        "OSM local (OSMnx + NetworkX) — Gratuit": {
            "badge_class": "badge-local", "badge_label": "LOCAL",
            "description": "Télécharge le réseau OSM + enrichit avec votre couche locale.",
            "needs_key": False, "needs_roads": False, "supports_route": True,
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
            "needs_roads": False, "supports_route": True,
            "modes": {
                "🚶 Marche":  ("foot-walking",   "walk"),
                "🚗 Voiture": ("driving-car",    "drive"),
                "🚲 Vélo":    ("cycling-regular", "bike"),
                "🛵 HGV":     ("driving-hgv",    "drive"),
            },
        },
        "OSRM Public — Gratuit": {
            "badge_class": "badge-free", "badge_label": "GRATUIT",
            "description": "Sans clé. 72 points de sondage. Isochrones et itinéraires.",
            "needs_key": False, "needs_roads": False, "supports_route": True,
            "modes": {
                "🚗 Voiture": ("car",  "drive"),
                "🚲 Vélo":    ("bike", "bike"),
                "🚶 Marche":  ("foot", "walk"),
            },
        },
        "Valhalla Public — Gratuit": {
            "badge_class": "badge-free", "badge_label": "GRATUIT",
            "description": "valhalla.openstreetmap.de — polygones et itinéraires précis, 4 modes.",
            "needs_key": False, "needs_roads": False, "supports_route": True,
            "modes": {
                "🚶 Marche":  ("pedestrian", "walk"),
                "🚗 Voiture": ("auto",       "drive"),
                "🚲 Vélo":    ("bicycle",    "bike"),
                "🛵 Moto":    ("motorcycle", "drive"),
            },
        },
        "GraphHopper — Clé API (gratuit 500 req/j)": {
            "badge_class": "badge-api", "badge_label": "API",
            "description": "graphhopper.com — polygones et itinéraires précis. Clé requise.",
            "needs_key": True, "key_name": "GH_API_KEY",
            "key_url": "https://www.graphhopper.com/dashboard/",
            "needs_roads": False, "supports_route": True,
            "modes": {
                "🚗 Voiture":   ("car",        "drive"),
                "🚶 Marche":    ("foot",        "walk"),
                "🚲 Vélo":      ("bike",        "bike"),
                "🛵 Moto":      ("motorcycle",  "drive"),
                "🥾 Randonnée": ("hike",        "walk"),
            },
        },
    }
