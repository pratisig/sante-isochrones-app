import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import io
import os
import shutil
import zipfile
import tempfile
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
from utils import (
    compute_isochrone,
    compute_route,
    engines_metadata,
    geocode_nominatim,
    reverse_geocode,
)
from projection import auto_utm_epsg
import warnings
warnings.filterwarnings("ignore")

gpd.options.io_engine = "pyogrio"

st.set_page_config(
    page_title="Générateur d'Isochrones & Itinéraires",
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
  background: linear-gradient(135deg, #01696f 0%, #0c4e54 100%);
  color: white; padding: 1.2rem 1.5rem; border-radius: 0.75rem; margin-bottom: 1.5rem;
}
.main-header h1 { margin: 0; font-size: 1.35rem; font-weight: 700; }
.main-header p  { margin: 0.2rem 0 0; font-size: 0.83rem; opacity: 0.88; }
.engine-card {
  background: white; border: 1px solid rgba(0,0,0,0.08);
  border-radius: 0.7rem; padding: 0.75rem 1rem; margin-bottom: 0.45rem;
  font-size: 0.82rem;
}
.engine-card.recommended { border: 1.5px solid #01696f; background: #f0f7f7; }
.engine-badge {
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 0.68rem; font-weight: 700; margin-left: 5px;
  vertical-align: middle;
}
.badge-api   { background: #cedcd8; color: #01696f; }
.badge-local { background: #d4dfcc; color: #437a22; }
.badge-free  { background: #c6d8e4; color: #006494; }
.info-box {
  background: #f3f0ec; border-left: 3px solid #01696f;
  border-radius: 0 0.5rem 0.5rem 0; padding: 0.65rem 1rem;
  font-size: 0.82rem; color: #28251d; margin-bottom: 0.9rem;
}
.warn-box {
  background: #fff8e1; border-left: 3px solid #d19900;
  border-radius: 0 0.5rem 0.5rem 0; padding: 0.55rem 0.9rem;
  font-size: 0.8rem; color: #28251d; margin-bottom: 0.6rem;
}
.roads-required {
  background: #fff3e0; border-left: 3px solid #da7101;
  border-radius: 0 0.5rem 0.5rem 0; padding: 0.55rem 0.9rem;
  font-size: 0.8rem; color: #28251d; margin-bottom: 0.6rem;
}
.stat-card {
  background: white; border-radius: 0.6rem; padding: 0.9rem 1.1rem;
  border: 1px solid rgba(0,0,0,0.07); text-align: center;
}
.stat-value { font-size: 1.5rem; font-weight: 700; color: #01696f; }
.stat-label { font-size: 0.78rem; color: #7a7974; margin-top: 2px; }
div[data-testid="stSidebar"] { background: #f9f8f5; border-right: 1px solid #dcd9d5; }
div[data-testid="stButton"] > button {
  background: #01696f !important; color: white !important;
  border: none !important; border-radius: 0.5rem !important;
  font-weight: 600 !important; width: 100%;
}
div[data-testid="stButton"] > button:hover { background: #0c4e54 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>🗺️ Générateur d'Isochrones & Itinéraires</h1>
  <p>Générique, multi-pays, multi-thèmes : établissements, marchés, écoles, dépôts, arrêts, GPS, chef-lieux, clic carte.</p>
</div>
""", unsafe_allow_html=True)

ENGINES   = engines_metadata()
COLORS    = ["#27ae60", "#f39c12", "#e74c3c", "#8e44ad", "#006494", "#01696f"]
OPACITIES = [0.22, 0.19, 0.16, 0.13, 0.10]


# ── helpers ─────────────────────────────────────────────────────────────────
def read_uploaded_file(uploaded_file) -> gpd.GeoDataFrame:
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".zip":
        tmp_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as zf:
                zf.extractall(tmp_dir)
            target = None
            for root, _, files in os.walk(tmp_dir):
                for f in files:
                    if f.lower().endswith(".shp"):
                        target = os.path.join(root, f); break
                if target: break
            if target is None:
                for root, _, files in os.walk(tmp_dir):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in (".geojson", ".json", ".gpkg"):
                            target = os.path.join(root, f); break
                    if target: break
            if target is None:
                raise ValueError("Aucun fichier spatial trouvé dans le ZIP.")
            return gpd.read_file(target, engine="pyogrio")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    driver_map = {".gpkg": "GPKG", ".geojson": "GeoJSON", ".json": "GeoJSON", ".kml": "KML"}
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        try:    return gpd.read_file(tmp_path, engine="pyogrio")
        except Exception: pass
        driver = driver_map.get(ext)
        if driver:
            try:    return gpd.read_file(tmp_path, engine="pyogrio", driver=driver)
            except Exception: pass
        return gpd.read_file(tmp_path)
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)


def area_km2(geometry, lon, lat):
    try:
        epsg = auto_utm_epsg(lon, lat)
        return round(gpd.GeoSeries([geometry], crs=4326).to_crs(epsg).area.iloc[0] / 1e6, 2)
    except Exception:
        return None


def parse_manual_points(raw_text, start_id=1):
    pts = []
    for i, line in enumerate(raw_text.strip().split("\n")):
        if not line.strip(): continue
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3:
            try:
                pts.append({"id": start_id+i, "name": p[0],
                             "lon": float(p[1]), "lat": float(p[2]), "source": "manuel"})
            except Exception: pass
    return pts


def points_from_gdf(gdf, label_field=None):
    rows = []
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        pt = geom.centroid if geom.geom_type != "Point" else geom
        nm = row.get(label_field) if label_field and label_field in gdf.columns else None
        if nm is None:
            for fld in ["nom","name","NOM","NAME","libelle","label"]:
                if fld in gdf.columns: nm = row.get(fld); break
        nm = str(nm) if nm is not None else f"Point {i+1}"
        rows.append({"id": i+1, "name": nm, "lon": pt.x, "lat": pt.y, "source": "fichier"})
    return rows


def build_iso_map(points, results, time_intervals, compare_mode, engines_list):
    clat = np.mean([p["lat"] for p in points]) if points else 12.36
    clon = np.mean([p["lon"] for p in points]) if points else -1.53
    m = folium.Map(location=[clat, clon], zoom_start=9, tiles="CartoDB positron")
    if compare_mode and len(engines_list) > 1:
        for i, eng in enumerate(engines_list):
            short = eng.split("—")[0].strip()
            color = COLORS[i % len(COLORS)]
            layer = folium.FeatureGroup(name=f"🔌 {short}", show=True)
            for mins in sorted(time_intervals, reverse=True):
                op = OPACITIES[min(time_intervals.index(mins), len(OPACITIES)-1)]
                for r in [x for x in results if x["minutes"]==mins and x["engine"]==short]:
                    folium.GeoJson(r["geometry"].__geo_interface__,
                        style_function=lambda x, c=color, o=op: {
                            "fillColor": c, "fillOpacity": o, "color": c, "weight": 2, "opacity": 0.85},
                        tooltip=f"{r['name']} · {short} · {mins} min").add_to(layer)
            layer.add_to(m)
    else:
        for i, mins in enumerate(sorted(time_intervals, reverse=True)):
            color = COLORS[i % len(COLORS)]
            layer = folium.FeatureGroup(name=f"⏱ {mins} min", show=True)
            for r in [x for x in results if x["minutes"]==mins]:
                folium.GeoJson(r["geometry"].__geo_interface__,
                    style_function=lambda x, c=color, o=OPACITIES[min(i, len(OPACITIES)-1)]: {
                        "fillColor": c, "fillOpacity": o, "color": c, "weight": 1.5, "opacity": 0.8},
                    tooltip=f"{r['name']} — {mins} min").add_to(layer)
            layer.add_to(m)
    pts_layer = folium.FeatureGroup(name="📍 Points", show=True)
    for p in points:
        folium.Marker([p["lat"], p["lon"]],
            popup=f"<b>{p['name']}</b><br>Lon: {p['lon']:.5f}<br>Lat: {p['lat']:.5f}",
            tooltip=p["name"],
            icon=folium.Icon(color="red", icon="map-marker", prefix="fa")).add_to(pts_layer)
    pts_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def build_route_map(origin, dest, route_result):
    clat = (origin["lat"] + dest["lat"]) / 2
    clon = (origin["lon"] + dest["lon"]) / 2
    m = folium.Map(location=[clat, clon], zoom_start=10, tiles="CartoDB positron")
    folium.Marker([origin["lat"], origin["lon"]], tooltip=f"Origine : {origin['name']}",
        icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
    folium.Marker([dest["lat"], dest["lon"]], tooltip=f"Destination : {dest['name']}",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")).add_to(m)
    if route_result.get("geometry") is not None:
        folium.GeoJson(route_result["geometry"].__geo_interface__,
            style_function=lambda x: {"color": "#01696f", "weight": 5, "opacity": 0.9}).add_to(m)
    return m


def export_shapefile(gdf_out):
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "export.shp")
        gdf_out.to_file(out_path, engine="pyogrio")
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(tmp_dir):
                zf.write(os.path.join(tmp_dir, fname), arcname=fname)
    buf.seek(0)
    return buf.read()


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # 1. MODE
    app_mode = st.radio("🗂️ Mode", ["🟢 Isochrones", "🔵 Itinéraire"], index=0)
    is_iso   = app_mode == "🟢 Isochrones"

    st.divider()

    # 2. MOTEUR
    st.markdown("#### 🔌 Moteur de routing")

    # Filtrer les moteurs selon le mode : OSM Pur ne supporte pas l'itinéraire
    available_engines = {
        k: v for k, v in ENGINES.items()
        if is_iso or v.get("supports_route", False)
    }

    compare_mode = False
    if is_iso:
        compare_mode = st.checkbox("Comparer plusieurs moteurs", value=False)

    if compare_mode:
        selected_engines = st.multiselect("Moteurs à comparer", list(available_engines.keys()),
                                          default=list(available_engines.keys())[:2])
        engine = selected_engines[0] if selected_engines else list(available_engines.keys())[0]
    else:
        engine = st.selectbox("Moteur", list(available_engines.keys()))
        selected_engines = [engine]

    meta = ENGINES[engine]
    st.markdown(
        f'<div class="info-box"><span class="engine-badge {meta["badge_class"]}">'
        f'{meta["badge_label"]}</span> {meta["description"]}</div>',
        unsafe_allow_html=True
    )

    needs_roads_engine = any(ENGINES.get(e, {}).get("needs_roads", False) for e in selected_engines)
    needs_ors = any("ORS" in e for e in selected_engines)
    needs_gh  = any("GraphHopper" in e for e in selected_engines)

    if needs_roads_engine:
        st.markdown('<div class="roads-required">⚠️ OSM Pur nécessite la couche Routes OSM.</div>',
                    unsafe_allow_html=True)

    ors_key = ""
    gh_key  = ""
    if needs_ors:
        ors_key = (st.secrets.get("ORS_API_KEY", "") if hasattr(st, "secrets") else "")
        if not ors_key:
            ors_key = st.text_input("Clé API ORS", type="password",
                                    help="openrouteservice.org — gratuit 500 req/j")
    if needs_gh:
        gh_key = (st.secrets.get("GH_API_KEY", "") if hasattr(st, "secrets") else "")
        if not gh_key:
            gh_key = st.text_input("Clé GraphHopper", type="password",
                                   help="graphhopper.com — gratuit 500 req/j")

    st.divider()

    # 3. MODE DE TRANSPORT
    st.markdown("#### 🚗 Mode de transport")
    mode_label = st.selectbox("Mode", list(meta["modes"].keys()))
    mode_api, mode_osm = meta["modes"][mode_label]

    st.divider()

    # 4. SOURCE TYPE
    source_type = st.text_input("🏷️ Type de source",
                                 value="Établissements",
                                 help="Marchés, écoles, dépôts, arrêts, chef-lieux…")

    st.divider()

    # ── SECTION POINTS (commune aux deux modes) ──────────────────────────────
    st.markdown("#### 📍 Points source")
    point_mode = st.radio("Entrée", [
        "Fichier spatial", "Saisie manuelle",
        "Recherche Nominatim", "Coordonnées GPS", "Exemple"
    ])

    points = []

    if point_mode == "Fichier spatial":
        up = st.file_uploader("Fichier points", type=["geojson","json","shp","gpkg","zip"])
        if up:
            try:
                gdf_pts = read_uploaded_file(up).to_crs(4326)
                sel_lbl = st.selectbox("Champ libellé", ["(auto)"] + list(gdf_pts.columns))
                points  = points_from_gdf(gdf_pts, None if sel_lbl == "(auto)" else sel_lbl)
                st.success(f"✅ {len(points)} points chargés")
            except Exception as e:
                st.error(f"Erreur : {e}")

    elif point_mode == "Saisie manuelle":
        raw = st.text_area("nom, lon, lat (une ligne par point)",
                           "Point A, -1.5312, 12.3345\nPoint B, -1.5167, 12.3611",
                           height=120)
        points = parse_manual_points(raw)
        if points:
            st.success(f"✅ {len(points)} points")

    elif point_mode == "Recherche Nominatim":
        q = st.text_input("Chercher lieu / établissement / chef-lieu…")
        if q:
            try:
                found  = geocode_nominatim(q)
                labels = [f"{x['name']} ({x['lat']:.5f}, {x['lon']:.5f})" for x in found]
                picked = st.multiselect("Résultats", labels, default=labels[:1])
                for i, item in enumerate(found):
                    lbl = f"{item['name']} ({item['lat']:.5f}, {item['lon']:.5f})"
                    if lbl in picked:
                        points.append({"id": i+1, "name": item["name"],
                                       "lon": item["lon"], "lat": item["lat"],
                                       "source": "nominatim"})
                if points:
                    st.success(f"✅ {len(points)} point(s) retenu(s)")
            except Exception as e:
                st.error(f"Nominatim : {e}")

    elif point_mode == "Coordonnées GPS":
        c1, c2 = st.columns(2)
        with c1:
            gps_lon = st.number_input("Longitude", value=-1.5167, format="%.6f")
        with c2:
            gps_lat = st.number_input("Latitude",  value=12.3611, format="%.6f")
        gps_nm = st.text_input("Nom", value="Point GPS")
        points = [{"id": 1, "name": gps_nm, "lon": gps_lon, "lat": gps_lat, "source": "gps"}]

    else:  # Exemple
        points = [
            {"id": 1, "name": "Marché central",    "lon": -1.5167, "lat": 12.3611, "source": "exemple"},
            {"id": 2, "name": "École secteur 10",   "lon": -1.5600, "lat": 12.3700, "source": "exemple"},
            {"id": 3, "name": "Dépôt logistique",  "lon": -1.5312, "lat": 12.3345, "source": "exemple"},
        ]
        st.info("3 points exemples chargés")

    st.divider()

    # ── SECTION ROUTES OSM ────────────────────────────────────────────────────
    st.markdown("#### 🛣️ Couche routes OSM")
    roads_file = st.file_uploader("Routes (optionnel sauf OSM Pur)",
                                   type=["geojson","json","shp","gpkg","zip"])
    gdf_roads = None
    if roads_file:
        try:
            gdf_roads = read_uploaded_file(roads_file).to_crs(4326)
            st.success(f"✅ {len(gdf_roads)} tronçons")
        except Exception as e:
            st.error(f"Erreur routes : {e}")

    st.divider()

    # ── PARAMÈTRES SPÉCIFIQUES AU MODE ────────────────────────────────────────
    if is_iso:
        st.markdown("#### ⏱️ Intervalles (minutes)")
        c1, c2 = st.columns(2)
        with c1:
            t1 = st.number_input("Seuil 1", value=15, min_value=5, max_value=240, step=5)
            t3 = st.number_input("Seuil 3", value=45, min_value=5, max_value=240, step=5)
        with c2:
            t2 = st.number_input("Seuil 2", value=30, min_value=5, max_value=240, step=5)
            t4 = st.number_input("Seuil 4", value=60, min_value=5, max_value=240, step=5)
        time_intervals = sorted({t for t in [t1, t2, t3, t4] if t > 0})

        st.markdown("#### 🔧 Méthode géométrique")
        iso_method = st.selectbox("Algorithme",
                                   ["Alpha Shape (recommandé)", "Convex Hull", "Buffer sur noeuds"])
        alpha_val = 0.4
        if iso_method == "Alpha Shape (recommandé)":
            alpha_val = st.slider("Alpha", 0.1, 1.0, 0.4, 0.05)

        origin = dest = None  # non utilisé en mode iso

    else:
        # ── MODE ITINÉRAIRE : O/D toujours accessibles ──
        time_intervals = []
        iso_method     = "Alpha Shape (recommandé)"
        alpha_val      = 0.4

        st.markdown("#### 🧭 Origine")
        orig_input = st.radio("Source origine", ["Depuis la liste", "Saisie directe"], key="orig_mode")
        if orig_input == "Depuis la liste" and points:
            pt_labels = [f"{p['name']} ({p['lat']:.5f}, {p['lon']:.5f})" for p in points]
            orig_lbl  = st.selectbox("Origine", pt_labels, key="orig_sel")
            origin    = next(p for p, l in zip(points, pt_labels) if l == orig_lbl)
        else:
            c1, c2 = st.columns(2)
            with c1: o_lon = st.number_input("Lon origine", value=-1.5167, format="%.6f", key="o_lon")
            with c2: o_lat = st.number_input("Lat origine", value=12.3611, format="%.6f", key="o_lat")
            o_nm = st.text_input("Nom origine", value="Origine", key="o_nm")
            origin = {"id": 0, "name": o_nm, "lon": o_lon, "lat": o_lat, "source": "direct"}

        st.markdown("#### 🏁 Destination")
        dest_input = st.radio("Source destination", ["Depuis la liste", "Saisie directe"], key="dest_mode")
        if dest_input == "Depuis la liste" and points:
            pt_labels = [f"{p['name']} ({p['lat']:.5f}, {p['lon']:.5f})" for p in points]
            default_d = min(1, len(pt_labels)-1)
            dest_lbl  = st.selectbox("Destination", pt_labels, index=default_d, key="dest_sel")
            dest      = next(p for p, l in zip(points, pt_labels) if l == dest_lbl)
        else:
            c1, c2 = st.columns(2)
            with c1: d_lon = st.number_input("Lon destination", value=-1.5312, format="%.6f", key="d_lon")
            with c2: d_lat = st.number_input("Lat destination", value=12.3345, format="%.6f", key="d_lat")
            d_nm = st.text_input("Nom destination", value="Destination", key="d_nm")
            dest = {"id": 1, "name": d_nm, "lon": d_lon, "lat": d_lat, "source": "direct"}

    st.divider()
    run_btn = st.button("🚀 Lancer", use_container_width=True)


# ── LAYOUT PRINCIPAL ─────────────────────────────────────────────────────────
col_map, col_info = st.columns([3, 1])

with col_info:
    st.markdown("### 📡 Moteurs")
    for eng, m in ENGINES.items():
        is_rec     = "OSM Pur" in eng
        card_class = "engine-card recommended" if is_rec else "engine-card"
        route_ok   = "✅" if m.get("supports_route") else "—"
        st.markdown(
            f'<div class="{card_class}"><b>{eng.split("—")[0].strip()}</b>'
            f'<span class="engine-badge {m["badge_class"]}">{m["badge_label"]}</span>'
            f'<br><small>{m["description"]}</small>'
            f'<br><small>Itinéraire : {route_ok}</small></div>',
            unsafe_allow_html=True
        )
    st.markdown(f"<div class='info-box'><b>Source :</b> {source_type}</div>", unsafe_allow_html=True)

with col_map:
    # Carte de prévisualisation
    if not run_btn:
        preview_pts = points if points else []
        if not is_iso and origin:
            preview_pts = [p for p in [origin, dest] if p]
        st_folium(build_iso_map(preview_pts, [], [15, 30], False, selected_engines),
                  width=None, height=580, returned_objects=[])

    else:
        # ── MODE ISOCHRONES ───────────────────────────────────────────────────
        if is_iso:
            if not points:
                st.error("⚠️ Aucun point disponible.")
            elif needs_roads_engine and gdf_roads is None:
                st.error("⚠️ Le moteur OSM Pur nécessite une couche de routes OSM.")
            else:
                results = []
                total   = len(points) * len(time_intervals) * len(selected_engines)
                prog    = st.progress(0)
                done    = 0
                for eng in selected_engines:
                    m_api, m_osm = ENGINES[eng]["modes"].get(
                        mode_label, list(ENGINES[eng]["modes"].values())[0])
                    for p in points:
                        for mins in time_intervals:
                            try:
                                poly = compute_isochrone(
                                    engine=eng, lon=p["lon"], lat=p["lat"],
                                    minutes=mins, mode_api=m_api, mode_osm=m_osm,
                                    iso_method=iso_method, alpha=alpha_val,
                                    ors_key=ors_key, gh_key=gh_key, gdf_roads=gdf_roads)
                                if poly is not None and not poly.is_empty:
                                    results.append({
                                        **p, "minutes": mins,
                                        "engine": eng.split("—")[0].strip(),
                                        "geometry": poly, "type_source": source_type,
                                        "aire_km2": area_km2(poly, p["lon"], p["lat"])
                                    })
                            except Exception as e:
                                st.warning(f"{eng.split('—')[0].strip()} | {p['name']} | {mins} min : {e}")
                            done += 1
                            prog.progress(done / total)
                prog.empty()

                if results:
                    st_folium(build_iso_map(points, results, time_intervals,
                                            compare_mode, selected_engines),
                              width=None, height=580, returned_objects=[])
                    st.markdown("### 📊 Résultats")
                    df = pd.DataFrame([{k: v for k, v in r.items() if k != "geometry"} for r in results])
                    st.dataframe(df[[c for c in df.columns if c != "id"]],
                                 use_container_width=True, hide_index=True)

                    gdf_out = gpd.GeoDataFrame(
                        [{k: v for k, v in r.items() if k != "geometry"} for r in results],
                        geometry=[r["geometry"] for r in results], crs=4326)
                    st.markdown("### ⬇️ Export")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button("⬇️ GeoJSON", gdf_out.to_json(),
                                           "isochrones.geojson", "application/json")
                    with c2:
                        try:
                            st.download_button("⬇️ Shapefile (.zip)", export_shapefile(gdf_out),
                                               "isochrones_shp.zip", "application/zip")
                        except Exception as e:
                            st.caption(f"SHP indisponible : {e}")
                else:
                    st.warning("Aucun résultat calculé.")

        # ── MODE ITINÉRAIRE ───────────────────────────────────────────────────
        else:
            if origin is None or dest is None:
                st.error("⚠️ Veuillez définir une origine et une destination.")
            elif (origin["lon"] == dest["lon"] and origin["lat"] == dest["lat"]):
                st.error("⚠️ L'origine et la destination sont identiques.")
            else:
                try:
                    route_result = compute_route(
                        engine=engine,
                        lon_o=origin["lon"], lat_o=origin["lat"],
                        lon_d=dest["lon"],   lat_d=dest["lat"],
                        mode_api=mode_api,   mode_osm=mode_osm,
                        ors_key=ors_key,     gh_key=gh_key)

                    st_folium(build_route_map(origin, dest, route_result),
                              width=None, height=580, returned_objects=[])

                    st.markdown("### 📊 Résumé")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f'<div class="stat-card"><div class="stat-value">'
                                    f'{route_result.get("distance_km", 0):.2f}</div>'
                                    f'<div class="stat-label">Distance (km)</div></div>',
                                    unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="stat-card"><div class="stat-value">'
                                    f'{route_result.get("duration_min", 0):.1f}</div>'
                                    f'<div class="stat-label">Durée (min)</div></div>',
                                    unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div class="stat-card"><div class="stat-value">'
                                    f'{route_result.get("engine", engine)}</div>'
                                    f'<div class="stat-label">Moteur</div></div>',
                                    unsafe_allow_html=True)

                    steps = route_result.get("steps", [])
                    if steps:
                        st.markdown("#### 📋 Feuille de route")
                        st.dataframe(pd.DataFrame(steps), use_container_width=True, hide_index=True)

                    if route_result.get("geometry") is not None:
                        gdf_route = gpd.GeoDataFrame([{
                            "origine": origin["name"], "destination": dest["name"],
                            "distance_km": route_result.get("distance_km"),
                            "duration_min": route_result.get("duration_min"),
                            "engine": route_result.get("engine"),
                            "mode": mode_label, "type_source": source_type,
                        }], geometry=[route_result["geometry"]], crs=4326)

                        st.markdown("### ⬇️ Export")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.download_button("⬇️ GeoJSON", gdf_route.to_json(),
                                               "itineraire.geojson", "application/json")
                        with c2:
                            try:
                                st.download_button("⬇️ Shapefile (.zip)", export_shapefile(gdf_route),
                                                   "itineraire_shp.zip", "application/zip")
                            except Exception as e:
                                st.caption(f"SHP indisponible : {e}")
                except Exception as e:
                    st.error(f"Erreur itinéraire : {e}")

st.divider()
st.markdown(
    '<div style="text-align:center;font-size:0.78rem;color:#7a7974;">'
    'Générateur générique d\'isochrones &amp; itinéraires · multi-pays · multi-sources · © OpenStreetMap contributors'
    '</div>',
    unsafe_allow_html=True
)
