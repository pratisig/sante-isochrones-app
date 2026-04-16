import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import requests
import json
import time
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, shape
from shapely.ops import unary_union
import alphashape
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Isochrones — Zones de Desserte",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://api.fontshare.com/v2/css?f[]=satoshi@400,500,700&display=swap');
html, body, [class*="css"] { font-family: 'Satoshi', sans-serif; }
.stApp { background: #f7f6f2; }
.main-header {
  background: #01696f; color: white;
  padding: 1.2rem 1.5rem; border-radius: 0.75rem; margin-bottom: 1.5rem;
}
.main-header h1 { margin: 0; font-size: 1.35rem; font-weight: 700; }
.main-header p  { margin: 0.2rem 0 0; font-size: 0.83rem; opacity: 0.85; }
.engine-card {
  background: white; border: 1px solid rgba(0,0,0,0.08);
  border-radius: 0.7rem; padding: 0.8rem 1rem; margin-bottom: 0.5rem;
}
.engine-badge {
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 0.7rem; font-weight: 600; margin-left: 6px;
}
.badge-api   { background: #cedcd8; color: #01696f; }
.badge-local { background: #d4dfcc; color: #437a22; }
.badge-free  { background: #c6d8e4; color: #006494; }
.info-box {
  background: #f3f0ec; border-left: 3px solid #01696f;
  border-radius: 0 0.5rem 0.5rem 0; padding: 0.7rem 1rem;
  font-size: 0.83rem; color: #28251d; margin-bottom: 1rem;
}
div[data-testid="stSidebar"] { background: #f9f8f5; border-right: 1px solid #dcd9d5; }
div[data-testid="stButton"] button {
  background: #01696f !important; color: white !important;
  border: none !important; border-radius: 0.5rem !important;
  font-weight: 600 !important; width: 100%;
}
div[data-testid="stButton"] button:hover { background: #0c4e54 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>🗺️ Générateur d'Isochrones — Zones de Desserte</h1>
  <p>Multi-moteurs : OSM local · OpenRouteService · OSRM · Valhalla — Structures sanitaires</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Moteur
    st.markdown("#### 🔌 Moteur de Routing")
    engine = st.selectbox("Choisir le moteur", [
        "OSM local (OSMnx + NetworkX) — Gratuit",
        "OpenRouteService (ORS) — Clé API",
        "OSRM Public — Gratuit",
        "Valhalla Public — Gratuit",
    ])

    engine_info = {
        "OSM local (OSMnx + NetworkX) — Gratuit": ("badge-local","LOCAL",
            "Calcul 100% local avec routes OSM. Colonnes utilisées : Tps_min, Vit_kmh, maxspeed."),
        "OpenRouteService (ORS) — Clé API": ("badge-api","API",
            "Très précis. Clé gratuite sur openrouteservice.org → ajoutez ORS_API_KEY dans les secrets Streamlit."),
        "OSRM Public — Gratuit": ("badge-free","GRATUIT",
            "router.project-osrm.org — sans clé. Isochrones approximatifs par matrice de durées."),
        "Valhalla Public — Gratuit": ("badge-free","GRATUIT",
            "valhalla.openstreetmap.de — Isochrones polygonaux natifs, 4 modes, sans clé."),
    }
    bc, bl, bi = engine_info[engine]
    st.markdown(f'<div class="info-box"><span class="engine-badge {bc}">{bl}</span> {bi}</div>',
                unsafe_allow_html=True)

    # Clé ORS
    ors_key = ""
    if "ORS" in engine:
        ors_key = st.secrets.get("ORS_API_KEY", "")
        if ors_key:
            st.success("✅ Clé ORS chargée depuis Streamlit Secrets")
        else:
            ors_key = st.text_input("Clé API ORS", type="password",
                                     help="Obtenez une clé gratuite sur openrouteservice.org")

    st.divider()

    # Mode de transport
    st.markdown("#### 🚶 Mode de Transport")
    MODES = {
        "OSM local (OSMnx + NetworkX) — Gratuit": {
            "🚶 Marche": ("walk","walk"), "🚗 Véhicule": ("drive","drive"), "🚲 Vélo": ("bike","bike")},
        "OpenRouteService (ORS) — Clé API": {
            "🚶 Marche": ("foot-walking","walk"), "🚗 Voiture": ("driving-car","drive"),
            "🚲 Vélo": ("cycling-regular","bike"), "🛵 HGV": ("driving-hgv","drive")},
        "OSRM Public — Gratuit": {
            "🚗 Voiture": ("car","drive"), "🚲 Vélo": ("bike","bike"), "🚶 Marche": ("foot","walk")},
        "Valhalla Public — Gratuit": {
            "🚶 Marche": ("pedestrian","walk"), "🚗 Voiture": ("auto","drive"),
            "🚲 Vélo": ("bicycle","bike"), "🛵 Moto": ("motorcycle","drive")},
    }
    mode_label = st.selectbox("Mode", list(MODES[engine].keys()))
    mode_api, mode_osm = MODES[engine][mode_label]

    st.divider()

    # Intervalles
    st.markdown("#### ⏱️ Intervalles (minutes)")
    c1, c2 = st.columns(2)
    with c1:
        t1 = st.number_input("Seuil 1", value=15, min_value=5, max_value=180, step=5)
        t3 = st.number_input("Seuil 3", value=45, min_value=5, max_value=180, step=5)
    with c2:
        t2 = st.number_input("Seuil 2", value=30, min_value=5, max_value=180, step=5)
        t4 = st.number_input("Seuil 4", value=60, min_value=5, max_value=180, step=5)
    time_intervals = sorted({t for t in [t1, t2, t3, t4] if t > 0})

    st.divider()

    # Structures
    st.markdown("#### 📍 Structures Sanitaires")
    input_mode = st.radio("Source", [
        "Fichier GeoJSON/Shapefile", "Saisie manuelle", "Exemple (Ouagadougou)"], index=2)

    facilities = []
    if input_mode == "Fichier GeoJSON/Shapefile":
        up = st.file_uploader("Charger", type=["geojson","json","shp","gpkg"])
        if up:
            try:
                gdf = gpd.read_file(up).to_crs(4326)
                for i, row in gdf.iterrows():
                    if row.geometry and row.geometry.geom_type == "Point":
                        nm = row.get("nom", row.get("name", f"Structure {i}"))
                        facilities.append({"id": i, "name": nm,
                                           "lon": row.geometry.x, "lat": row.geometry.y})
                st.success(f"✅ {len(facilities)} structures")
            except Exception as e:
                st.error(str(e))
    elif input_mode == "Saisie manuelle":
        raw = st.text_area("nom, lon, lat (une par ligne)",
                           "CS Bogodogo, -1.5312, 12.3345\nHôpital Yalgado, -1.5167, 12.3611",
                           height=100)
        for i, line in enumerate(raw.strip().split("\n")):
            p = line.split(",")
            if len(p) == 3:
                try:
                    facilities.append({"id": i, "name": p[0].strip(),
                                       "lon": float(p[1]), "lat": float(p[2])})
                except: pass
        if facilities: st.success(f"✅ {len(facilities)} structures")
    else:
        facilities = [
            {"id": 1, "name": "CS Bogodogo",     "lon": -1.5312, "lat": 12.3345},
            {"id": 2, "name": "CS Baskuy",       "lon": -1.5600, "lat": 12.3700},
            {"id": 3, "name": "Hôpital Yalgado", "lon": -1.5167, "lat": 12.3611},
        ]
        st.info("3 structures exemples (Ouagadougou)")

    st.divider()

    # Méthode géométrique (OSM local)
    iso_method, alpha_val = "Alpha Shape (recommandé)", 0.4
    if "OSM local" in engine:
        st.markdown("#### 🔧 Méthode Géométrique")
        iso_method = st.selectbox("Algorithme", [
            "Alpha Shape (recommandé)", "Convex Hull", "Buffer sur noeuds"])
        if iso_method == "Alpha Shape (recommandé)":
            alpha_val = st.slider("Paramètre alpha", 0.1, 1.0, 0.4, 0.05)

    st.divider()
    run_btn = st.button("🚀 Calculer les Isochrones", use_container_width=True)


# ── FONCTIONS ────────────────────────────────────────────────────
COLORS   = ["#27ae60", "#f39c12", "#e74c3c", "#8e44ad"]
OPACITIES = [0.22, 0.19, 0.16, 0.13]


def isochrone_osmnx(lon, lat, minutes, mode, method, alpha):
    from utils import compute_osmnx_isochrone
    return compute_osmnx_isochrone(lon, lat, minutes, mode, method, alpha)


def isochrone_ors(lon, lat, minutes, profile, key):
    url = f"https://api.openrouteservice.org/v2/isochrones/{profile}"
    resp = requests.post(url, json={
        "locations": [[lon, lat]], "range": [minutes*60],
        "smoothing": 5, "attributes": ["area"]
    }, headers={"Authorization": key, "Content-Type": "application/json"}, timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        if "features" in data and data["features"]:
            return shape(data["features"][0]["geometry"])
    raise Exception(f"ORS {resp.status_code}: {resp.text[:200]}")


def isochrone_osrm(lon, lat, minutes, profile):
    speeds = {"foot": 4.5, "bike": 15.0, "car": 50.0}
    spd = speeds.get(profile, 50.0)
    R = (minutes / 60) * spd * 1000
    angles = np.linspace(0, 2*np.pi, 36, endpoint=False)
    Re = 6371000
    dlats = [np.degrees(R/Re * np.cos(a)) for a in angles]
    dlons = [np.degrees(R/Re * np.sin(a) / np.cos(np.radians(lat))) for a in angles]
    dests = [(lon+dlons[i], lat+dlats[i]) for i in range(36)]
    coords_str = f"{lon},{lat};" + ";".join([f"{d[0]},{d[1]}" for d in dests])
    url = (f"http://router.project-osrm.org/table/v1/{profile}/{coords_str}"
           f"?sources=0&destinations={';'.join(str(i+1) for i in range(36))}&annotations=duration")
    resp = requests.get(url, timeout=20)
    if resp.status_code != 200:
        raise Exception(f"OSRM {resp.status_code}")
    durations = resp.json()["durations"][0]
    target = minutes * 60
    pts = []
    for i, dur in enumerate(durations):
        if dur is None: dur = target * 2
        ratio = min(1.0, target / dur) if dur > 0 else 1.0
        pts.append(Point(lon + dlons[i]*ratio, lat + dlats[i]*ratio))
    return unary_union(pts).convex_hull if pts else Point(lon, lat).buffer(0.01)


def isochrone_valhalla(lon, lat, minutes, costing):
    url = "https://valhalla1.openstreetmap.de/isochrone"
    body = {
        "locations": [{"lon": lon, "lat": lat}],
        "costing": costing,
        "contours": [{"time": minutes, "color": "ff0000"}],
        "polygons": True, "generalize": 50
    }
    resp = requests.get(url, params={"json": json.dumps(body)}, timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        if "features" in data and data["features"]:
            return shape(data["features"][0]["geometry"])
    raise Exception(f"Valhalla {resp.status_code}: {resp.text[:200]}")


def compute_all(facilities, time_intervals, engine, mode_api, mode_osm,
                iso_method, alpha_val, ors_key):
    results = []
    prog = st.progress(0)
    stat = st.empty()
    total = len(facilities) * len(time_intervals)
    done = 0
    for fac in facilities:
        for mins in time_intervals:
            stat.markdown(f"⏳ **{fac['name']}** — {mins} min…")
            try:
                if "OSM local" in engine:
                    poly = isochrone_osmnx(fac["lon"], fac["lat"], mins, mode_osm, iso_method, alpha_val)
                elif "ORS" in engine:
                    if not ors_key: st.error("Clé ORS manquante !"); return []
                    poly = isochrone_ors(fac["lon"], fac["lat"], mins, mode_api, ors_key)
                    time.sleep(1.5)
                elif "OSRM" in engine:
                    poly = isochrone_osrm(fac["lon"], fac["lat"], mins, mode_api)
                    time.sleep(0.5)
                elif "Valhalla" in engine:
                    poly = isochrone_valhalla(fac["lon"], fac["lat"], mins, mode_api)
                    time.sleep(0.5)
                else: poly = None
                if poly and not poly.is_empty:
                    results.append({**fac, "minutes": mins, "geometry": poly})
            except Exception as e:
                st.warning(f"⚠️ {fac['name']} {mins} min : {e}")
            done += 1
            prog.progress(done / total)
    prog.empty(); stat.empty()
    return results


def build_map(facilities, results, time_intervals):
    clat = np.mean([f["lat"] for f in facilities]) if facilities else 12.36
    clon = np.mean([f["lon"] for f in facilities]) if facilities else -1.53
    m = folium.Map(location=[clat, clon], zoom_start=11, tiles="CartoDB positron")

    for i, mins in enumerate(sorted(time_intervals, reverse=True)):
        color = COLORS[i % len(COLORS)]
        layer = folium.FeatureGroup(name=f"⏱ {mins} min", show=True)
        for r in [x for x in results if x["minutes"] == mins]:
            folium.GeoJson(r["geometry"].__geo_interface__,
                style_function=lambda x, c=color, o=OPACITIES[i % len(OPACITIES)]: {
                    "fillColor": c, "fillOpacity": o, "color": c, "weight": 1.5, "opacity": 0.75},
                tooltip=f"{r['name']} — {mins} min"
            ).add_to(layer)
        layer.add_to(m)

    markers = folium.FeatureGroup(name="🏥 Structures", show=True)
    for fac in facilities:
        folium.Marker([fac["lat"], fac["lon"]],
            popup=folium.Popup(f"<b>{fac['name']}</b>", max_width=200),
            icon=folium.Icon(color="red", icon="plus", prefix="fa"),
            tooltip=fac["name"]).add_to(markers)
    markers.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    legend = '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.15);font-size:13px;"><b>Zones de desserte</b><br>'
    for i, mins in enumerate(sorted(time_intervals)):
        c = COLORS[i % len(COLORS)]
        legend += f'<span style="display:inline-block;width:13px;height:13px;background:{c};border-radius:3px;margin-right:6px;vertical-align:middle;"></span>{mins} min<br>'
    legend += "</div>"
    m.get_root().html.add_child(folium.Element(legend))
    return m


# ── LAYOUT PRINCIPAL ─────────────────────────────────────────────
col_map, col_info = st.columns([3, 1])

with col_info:
    st.markdown("### ℹ️ Moteurs")
    st.markdown("""
<div class="engine-card">
  <b>OSM local</b> <span class="engine-badge badge-local">LOCAL</span><br>
  <small>OSMnx + NetworkX. 100% local. Colonnes : <code>Tps_min</code>, <code>Vit_kmh</code>, <code>maxspeed</code>.</small>
</div>
<div class="engine-card">
  <b>OpenRouteService</b> <span class="engine-badge badge-api">API</span><br>
  <small>Clé gratuite. 500 req/jour. Isochrones très précis, 3 modes.</small>
</div>
<div class="engine-card">
  <b>OSRM Public</b> <span class="engine-badge badge-free">GRATUIT</span><br>
  <small>Sans clé. Approximatif (matrice de durées). Données OSM mondiales.</small>
</div>
<div class="engine-card">
  <b>Valhalla</b> <span class="engine-badge badge-free">GRATUIT</span><br>
  <small>Sans clé. Polygones natifs précis. 4 modes de transport.</small>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 📋 Clé ORS")
    st.markdown("""
<div class="info-box">
  Ajoutez votre clé dans <b>Streamlit Secrets</b> :<br><br>
  <code>ORS_API_KEY = "5b3ce35..."</code><br><br>
  → Settings → Secrets dans Streamlit Cloud.
</div>
""", unsafe_allow_html=True)

with col_map:
    map_ph = st.empty()
    if not run_btn:
        init_m = build_map(facilities, [], time_intervals)
        with map_ph: st_folium(init_m, width=None, height=560, returned_objects=[])
    else:
        if not facilities:
            st.error("⚠️ Aucune structure définie.")
        else:
            with st.spinner("Calcul en cours…"):
                results = compute_all(facilities, time_intervals, engine,
                                      mode_api, mode_osm, iso_method, alpha_val, ors_key)
            if results:
                m = build_map(facilities, results, time_intervals)
                with map_ph: st_folium(m, width=None, height=560, returned_objects=[])

                st.markdown("### 📊 Résultats")
                rows = []
                for r in results:
                    try:
                        aire = round(gpd.GeoSeries([r["geometry"]], crs=4326).to_crs(32630).area.values[0]/1e6, 2)
                    except: aire = None
                    rows.append({"Structure": r["name"], "Temps (min)": r["minutes"],
                                 "Aire (km²)": aire, "Moteur": engine.split("—")[0].strip(),
                                 "Mode": mode_label})
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                gdf_out = gpd.GeoDataFrame(df.copy(),
                    geometry=[r["geometry"] for r in results], crs=4326)
                st.download_button("⬇️ Télécharger GeoJSON", data=gdf_out.to_json(),
                    file_name="isochrones.geojson", mime="application/json")
            else:
                st.warning("Aucun résultat. Vérifiez vos paramètres.")

st.divider()
st.markdown('<div style="text-align:center;font-size:0.78rem;color:#7a7974;">Isochrones · OSMnx · ORS · OSRM · Valhalla · © OpenStreetMap contributors</div>',
            unsafe_allow_html=True)
