import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import json
import time
import io
import os
import shutil
import zipfile
import tempfile
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point
from utils import compute_isochrone, engines_metadata
from projection import auto_utm_epsg
import warnings
warnings.filterwarnings("ignore")

gpd.options.io_engine = "pyogrio"

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
  background: linear-gradient(135deg, #01696f 0%, #0c4e54 100%);
  color: white; padding: 1.2rem 1.5rem; border-radius: 0.75rem; margin-bottom: 1.5rem;
}
.main-header h1 { margin: 0; font-size: 1.35rem; font-weight: 700; }
.main-header p  { margin: 0.2rem 0 0; font-size: 0.83rem; opacity: 0.85; }
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
.badge-api      { background: #cedcd8; color: #01696f; }
.badge-local    { background: #d4dfcc; color: #437a22; }
.badge-free     { background: #c6d8e4; color: #006494; }
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
.compare-tag {
  background: #e9e0c6; color: #8a5b00; border-radius: 999px;
  padding: 2px 9px; font-size: 0.68rem; font-weight: 700;
}
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
  <h1>🗺️ Générateur d'Isochrones — Zones de Desserte</h1>
  <p>6 moteurs : OSM Pur Tps_min ★ · OSMnx · ORS · OSRM (72pts) · Valhalla · GraphHopper</p>
</div>
""", unsafe_allow_html=True)

ENGINES   = engines_metadata()
COLORS    = ["#27ae60", "#f39c12", "#e74c3c", "#8e44ad"]
OPACITIES = [0.22, 0.19, 0.16, 0.13]


# ─────────────────────────────────────────────────────────────────
# Lecture robuste des fichiers uploadés
# ─────────────────────────────────────────────────────────────────
def read_uploaded_file(uploaded_file) -> gpd.GeoDataFrame:
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".zip":
        tmp_dir = tempfile.mkdtemp()
        try:
            zip_bytes = io.BytesIO(uploaded_file.read())
            with zipfile.ZipFile(zip_bytes) as zf:
                zf.extractall(tmp_dir)
            shp_path = None
            for root, dirs, files in os.walk(tmp_dir):
                for f in files:
                    if f.lower().endswith(".shp"):
                        shp_path = os.path.join(root, f)
                        break
                if shp_path:
                    break
            if shp_path is None:
                for root, dirs, files in os.walk(tmp_dir):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in (".geojson", ".json", ".gpkg"):
                            shp_path = os.path.join(root, f)
                            break
                    if shp_path:
                        break
            if shp_path is None:
                raise ValueError("Aucun fichier spatial trouvé dans le ZIP.")
            return gpd.read_file(shp_path, engine="pyogrio")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if ext == ".shp":
        tmp_dir = tempfile.mkdtemp()
        try:
            shp_path = os.path.join(tmp_dir, filename)
            with open(shp_path, "wb") as f:
                f.write(uploaded_file.read())
            os.environ["SHAPE_RESTORE_SHX"] = "YES"
            try:
                return gpd.read_file(shp_path, engine="pyogrio")
            finally:
                os.environ.pop("SHAPE_RESTORE_SHX", None)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    driver_map = {".gpkg": "GPKG", ".geojson": "GeoJSON", ".json": "GeoJSON", ".kml": "KML"}
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(uploaded_file.read())
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


# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # --- Moteur(s)
    st.markdown("#### 🔌 Moteur de Routing")
    compare_mode = st.checkbox("Comparer plusieurs moteurs", value=False)

    if compare_mode:
        selected_engines = st.multiselect(
            "Moteurs à comparer",
            list(ENGINES.keys()),
            default=list(ENGINES.keys())[:2]
        )
        engine = selected_engines[0] if selected_engines else list(ENGINES.keys())[0]
    else:
        engine = st.selectbox("Choisir le moteur", list(ENGINES.keys()))
        selected_engines = [engine]

    meta = ENGINES[engine]
    st.markdown(
        f'<div class="info-box"><span class="engine-badge {meta["badge_class"]}">{meta["badge_label"]}</span> {meta["description"]}</div>',
        unsafe_allow_html=True
    )

    # Avertissements
    needs_roads_engine = any(ENGINES.get(e, {}).get("needs_roads", False) for e in selected_engines)
    needs_gh = any("GraphHopper" in e for e in selected_engines)
    needs_ors = any("ORS" in e for e in selected_engines)

    if needs_roads_engine:
        st.markdown(
            '<div class="roads-required">⚠️ <b>OSM Pur</b> nécessite obligatoirement la couche Routes OSM ci-dessous.</div>',
            unsafe_allow_html=True
        )

    if needs_gh:
        st.markdown(
            '<div class="warn-box">⚠️ <b>GraphHopper</b> requiert une clé API. '
            'Clé gratuite sur <a href="https://www.graphhopper.com/dashboard/" target="_blank">graphhopper.com</a> (500 req/j).</div>',
            unsafe_allow_html=True
        )

    # --- Clés API
    ors_key = ""
    gh_key  = ""
    if needs_ors:
        ors_key = st.secrets.get("ORS_API_KEY", "") if hasattr(st, 'secrets') else ""
        if ors_key:
            st.success("✅ Clé ORS chargée (secrets)")
        else:
            ors_key = st.text_input("Clé API ORS", type="password",
                help="openrouteservice.org/dev/#/signup — gratuit 500 req/j")

    if needs_gh:
        gh_key = st.secrets.get("GH_API_KEY", "") if hasattr(st, 'secrets') else ""
        if gh_key:
            st.success("✅ Clé GraphHopper chargée (secrets)")
        else:
            gh_key = st.text_input("Clé GraphHopper", type="password",
                help="graphhopper.com — gratuit 500 req/j")

    st.divider()

    # --- Mode de transport
    st.markdown("#### 🚶 Mode de Transport")
    modes_available = meta["modes"]
    mode_label = st.selectbox("Mode", list(modes_available.keys()))
    mode_api, mode_osm = modes_available[mode_label]

    st.divider()

    # --- Intervalles
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

    # --- Couche routes OSM
    st.markdown("#### 🛣️ Couche Routes OSM")
    st.caption("💡 Shapefile : uploader un **ZIP** (.shp + .shx + .dbf + .prj)")
    roads_file = st.file_uploader(
        "Routes OSM (Tps_min, Vit_kmh, maxspeed, osm_id)",
        type=["geojson", "json", "shp", "gpkg", "zip"],
        help="Requis pour OSM Pur. Optionnel pour OSMnx."
    )
    gdf_roads = None
    if roads_file:
        try:
            gdf_roads = read_uploaded_file(roads_file).to_crs(4326)
            st.success(f"✅ {len(gdf_roads)} tronçons chargés")
            cols_ok = [c for c in ["Tps_min", "Vit_kmh", "maxspeed", "osm_id"] if c in gdf_roads.columns]
            if cols_ok:
                st.caption(f"Colonnes : {', '.join(cols_ok)}")
            if "Tps_min" in gdf_roads.columns:
                tps_vals = gdf_roads["Tps_min"].dropna()
                st.caption(f"Tps_min — moy: {tps_vals.mean():.1f} min · max: {tps_vals.max():.1f} min")
            if "Vit_kmh" in gdf_roads.columns:
                vit_vals = gdf_roads["Vit_kmh"].dropna()
                st.caption(f"Vit_kmh — moy: {vit_vals.mean():.1f} · max: {vit_vals.max():.1f}")
        except Exception as e:
            st.error(f"Erreur chargement routes : {e}")

    st.divider()

    # --- Structures sanitaires
    st.markdown("#### 📍 Structures Sanitaires")
    st.caption("💡 Shapefile : uploader un **ZIP** (.shp + .shx + .dbf + .prj)")
    input_mode = st.radio("Source", [
        "Fichier GeoJSON/Shapefile", "Saisie manuelle", "Exemple (Ouagadougou)"
    ], index=2)

    facilities = []
    if input_mode == "Fichier GeoJSON/Shapefile":
        up = st.file_uploader("Charger structures", type=["geojson", "json", "shp", "gpkg", "zip"])
        if up:
            try:
                gdf = read_uploaded_file(up).to_crs(4326)
                for i, row in gdf.iterrows():
                    geom = row.geometry
                    if geom is None:
                        continue
                    pt = geom.centroid if geom.geom_type != "Point" else geom
                    nm = row.get("nom", row.get("name", row.get("NOM",
                         row.get("NAME", f"Structure {i}"))))
                    facilities.append({"id": i, "name": str(nm),
                                       "lon": pt.x, "lat": pt.y})
                st.success(f"✅ {len(facilities)} structures chargées")
            except Exception as e:
                st.error(f"Erreur chargement structures : {e}")
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
                except Exception:
                    pass
        if facilities:
            st.success(f"✅ {len(facilities)} structures")
    else:
        facilities = [
            {"id": 1, "name": "CS Bogodogo",     "lon": -1.5312, "lat": 12.3345},
            {"id": 2, "name": "CS Baskuy",       "lon": -1.5600, "lat": 12.3700},
            {"id": 3, "name": "Hôpital Yalgado", "lon": -1.5167, "lat": 12.3611},
        ]
        st.info("3 structures exemples (Ouagadougou)")

    st.divider()

    # --- Méthode géométrique
    iso_method, alpha_val = "Alpha Shape (recommandé)", 0.4
    local_engines = [e for e in selected_engines if "OSM" in e]
    if local_engines:
        st.markdown("#### 🔧 Méthode Géométrique")
        iso_method = st.selectbox("Algorithme", [
            "Alpha Shape (recommandé)", "Convex Hull", "Buffer sur noeuds"
        ])
        if iso_method == "Alpha Shape (recommandé)":
            alpha_val = st.slider("Paramètre alpha", 0.1, 1.0, 0.4, 0.05,
                help="Plus alpha est grand, plus la forme est concave et précise. Valeur recommandée : 0.3-0.5")

    st.divider()
    run_btn = st.button("🚀 Calculer les Isochrones", use_container_width=True)


# ── FONCTIONS UTILITAIRES ───────────────────────────────────────────────────
def get_mode_for_engine(eng, lbl):
    modes = ENGINES[eng]["modes"]
    if lbl in modes:
        return modes[lbl]
    return list(modes.values())[0]


def compute_area_safe(geometry, lon: float, lat: float) -> float:
    try:
        epsg = auto_utm_epsg(lon, lat)
        gs = gpd.GeoSeries([geometry], crs=4326)
        return round(gs.to_crs(epsg).area.values[0] / 1e6, 2)
    except Exception:
        try:
            bounds = geometry.bounds
            dlat = bounds[3] - bounds[1]
            dlon = bounds[2] - bounds[0]
            lat_r = np.radians(lat)
            return round(dlat * 111.32 * dlon * 111.32 * np.cos(lat_r), 2)
        except Exception:
            return None


def compute_all(facilities, time_intervals, engines_list, mode_label,
                iso_method, alpha_val, ors_key, gh_key, gdf_roads):
    results = []
    total = len(facilities) * len(time_intervals) * len(engines_list)
    prog  = st.progress(0)
    stat  = st.empty()
    done  = 0
    delays = {"ORS": 1.5, "OSRM": 0.8, "Valhalla": 0.5, "GraphHopper": 0.5, "OSM": 0.0}

    for eng in engines_list:
        m_api, m_osm = get_mode_for_engine(eng, mode_label)
        for fac in facilities:
            for mins in time_intervals:
                stat.markdown(f"⏳ **[{eng.split('—')[0].strip()}]** {fac['name']} — {mins} min…")
                try:
                    poly = compute_isochrone(
                        engine=eng, lon=fac["lon"], lat=fac["lat"],
                        minutes=mins, mode_api=m_api, mode_osm=m_osm,
                        iso_method=iso_method, alpha=alpha_val,
                        ors_key=ors_key, gh_key=gh_key, gdf_roads=gdf_roads
                    )
                    if poly and not poly.is_empty:
                        results.append({
                            **fac,
                            "minutes":  mins,
                            "geometry": poly,
                            "engine":   eng.split("—")[0].strip()
                        })
                except Exception as e:
                    st.warning(f"⚠️ {eng.split('—')[0].strip()} | {fac['name']} {mins} min : {e}")

                for k, d in delays.items():
                    if k in eng:
                        time.sleep(d)
                        break
                done += 1
                prog.progress(done / total)
    prog.empty()
    stat.empty()
    return results


def build_map(facilities, results, time_intervals, compare_mode, engines_list):
    clat = np.mean([f["lat"] for f in facilities]) if facilities else 12.36
    clon = np.mean([f["lon"] for f in facilities]) if facilities else -1.53
    m = folium.Map(location=[clat, clon], zoom_start=11, tiles="CartoDB positron")

    if compare_mode and len(engines_list) > 1:
        engine_colors = {e: COLORS[i % len(COLORS)] for i, e in enumerate(engines_list)}
        for eng in engines_list:
            short = eng.split("—")[0].strip()
            color = engine_colors[eng]
            layer = folium.FeatureGroup(name=f"🔌 {short}", show=True)
            for mins in sorted(time_intervals, reverse=True):
                op = OPACITIES[time_intervals.index(mins) % len(OPACITIES)]
                for r in [x for x in results if x["minutes"] == mins and x["engine"] == short]:
                    folium.GeoJson(
                        r["geometry"].__geo_interface__,
                        style_function=lambda x, c=color, o=op: {
                            "fillColor": c, "fillOpacity": o,
                            "color": c, "weight": 2, "opacity": 0.85
                        },
                        tooltip=f"{r['name']} · {short} · {mins} min"
                    ).add_to(layer)
            layer.add_to(m)
    else:
        for i, mins in enumerate(sorted(time_intervals, reverse=True)):
            color = COLORS[i % len(COLORS)]
            layer = folium.FeatureGroup(name=f"⏱ {mins} min", show=True)
            for r in [x for x in results if x["minutes"] == mins]:
                area_km2 = compute_area_safe(r["geometry"], r["lon"], r["lat"])
                area_str = f" · {area_km2} km²" if area_km2 else ""
                folium.GeoJson(
                    r["geometry"].__geo_interface__,
                    style_function=lambda x, c=color, o=OPACITIES[i % len(OPACITIES)]: {
                        "fillColor": c, "fillOpacity": o,
                        "color": c, "weight": 1.5, "opacity": 0.75
                    },
                    tooltip=f"{r['name']} — {mins} min{area_str}"
                ).add_to(layer)
            layer.add_to(m)

    markers = folium.FeatureGroup(name="🏥 Structures", show=True)
    for fac in facilities:
        folium.Marker(
            [fac["lat"], fac["lon"]],
            popup=folium.Popup(f"<b>{fac['name']}</b><br>Lon: {fac['lon']:.4f}<br>Lat: {fac['lat']:.4f}", max_width=220),
            icon=folium.Icon(color="red", icon="plus", prefix="fa"),
            tooltip=fac["name"]
        ).add_to(markers)
    markers.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # Légende
    if compare_mode and len(engines_list) > 1:
        legend = '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.15);font-size:13px;"><b>Moteurs</b><br>'
        for i, eng in enumerate(engines_list):
            c = COLORS[i % len(COLORS)]
            legend += f'<span style="display:inline-block;width:13px;height:13px;background:{c};border-radius:3px;margin-right:6px;vertical-align:middle;"></span>{eng.split("—")[0].strip()}<br>'
    else:
        legend = '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.15);font-size:13px;"><b>Zones de desserte</b><br>'
        for i, mins in enumerate(sorted(time_intervals)):
            c = COLORS[i % len(COLORS)]
            legend += f'<span style="display:inline-block;width:13px;height:13px;background:{c};border-radius:3px;margin-right:6px;vertical-align:middle;"></span>{mins} min<br>'
    legend += "</div>"
    m.get_root().html.add_child(folium.Element(legend))
    return m


def export_shapefile(gdf_out):
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "isochrones.shp")
        gdf_out.to_file(out_path, engine="pyogrio")
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(tmp_dir):
                zf.write(os.path.join(tmp_dir, fname), arcname=fname)
    buf.seek(0)
    return buf.read()


def build_stats_section(results, facilities, time_intervals):
    """Affiche les statistiques détaillées après calcul."""
    if not results:
        return

    # Enrichir avec les aires
    for r in results:
        if "aire_km2" not in r:
            r["aire_km2"] = compute_area_safe(r["geometry"], r["lon"], r["lat"])

    aires = [r["aire_km2"] for r in results if r["aire_km2"] is not None]
    n_structures = len(set(r["name"] for r in results))
    n_zones = len(results)
    total_aire = sum(aires) if aires else 0
    moy_aire = np.mean(aires) if aires else 0

    st.markdown("### 📊 Statistiques")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{n_structures}</div><div class="stat-label">Structures traitées</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{n_zones}</div><div class="stat-label">Zones calculées</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{total_aire:.1f}</div><div class="stat-label">Aire totale (km²)</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{moy_aire:.1f}</div><div class="stat-label">Aire moyenne (km²)</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # Tableau détaillé
    rows = []
    for r in results:
        rows.append({
            "Structure":    r["name"],
            "Lon":          round(r["lon"], 4),
            "Lat":          round(r["lat"], 4),
            "Temps (min)":  r["minutes"],
            "Aire (km²)":   r["aire_km2"] if r["aire_km2"] is not None else "N/D",
            "Moteur":       r["engine"],
            "Mode":         mode_label,
            "Algorithme":   iso_method,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Tableau pivot : structure × intervalle de temps
    if len(time_intervals) > 1 and len(facilities) > 1:
        st.markdown("#### Aire (km²) par structure et intervalle")
        pivot_data = []
        for fac in facilities:
            row_p = {"Structure": fac["name"]}
            for mins in time_intervals:
                matching = [r["aire_km2"] for r in results
                            if r["name"] == fac["name"] and r["minutes"] == mins
                            and r["aire_km2"] is not None]
                row_p[f"{mins} min"] = f"{np.mean(matching):.1f}" if matching else "—"
            pivot_data.append(row_p)
        df_pivot = pd.DataFrame(pivot_data)
        st.dataframe(df_pivot, use_container_width=True, hide_index=True)

    return df


# ── LAYOUT PRINCIPAL ────────────────────────────────────────────────────────
col_map, col_info = st.columns([3, 1])

with col_info:
    st.markdown("### 📡 Moteurs")
    for eng, m in ENGINES.items():
        is_rec = "OSM Pur" in eng
        card_class = "engine-card recommended" if is_rec else "engine-card"
        st.markdown(
            f'<div class="{card_class}"><b>{eng.split("—")[0].strip()}</b>'
            f'<span class="engine-badge {m["badge_class"]}">{m["badge_label"]}</span><br>'
            f'<small>{m["description"]}</small></div>',
            unsafe_allow_html=True
        )
    if compare_mode:
        st.markdown(
            '<div style="margin-top:0.5rem;"><span class="compare-tag">MODE COMPARAISON</span>'
            ' Moteurs affichés en couleurs distinctes.</div>',
            unsafe_allow_html=True
        )

with col_map:
    map_ph = st.empty()
    if not run_btn:
        init_m = build_map(facilities, [], time_intervals, False, selected_engines)
        with map_ph:
            st_folium(init_m, width=None, height=580, returned_objects=[])
    else:
        # Validation
        needs_roads_any = any(ENGINES.get(e, {}).get("needs_roads", False) for e in selected_engines)
        if not facilities:
            st.error("⚠️ Aucune structure définie.")
        elif needs_roads_any and gdf_roads is None:
            st.error("⚠️ Le moteur **OSM Pur** nécessite la couche Routes OSM. Chargez-la dans la sidebar.")
        elif compare_mode and not selected_engines:
            st.error("⚠️ Sélectionnez au moins un moteur.")
        else:
            with st.spinner("Calcul en cours…"):
                results = compute_all(
                    facilities, time_intervals, selected_engines, mode_label,
                    iso_method, alpha_val, ors_key, gh_key, gdf_roads
                )
            if results:
                m = build_map(facilities, results, time_intervals, compare_mode, selected_engines)
                with map_ph:
                    st_folium(m, width=None, height=580, returned_objects=[])

                df = build_stats_section(results, facilities, time_intervals)

                # Export
                st.markdown("### ⬇️ Export")
                gdf_out = gpd.GeoDataFrame(
                    [{
                        "structure":  r["name"],
                        "lon":        round(r["lon"], 6),
                        "lat":        round(r["lat"], 6),
                        "temps_min":  r["minutes"],
                        "aire_km2":   r.get("aire_km2"),
                        "moteur":     r["engine"],
                        "mode":       mode_label,
                        "algorithme": iso_method,
                    } for r in results],
                    geometry=[r["geometry"] for r in results],
                    crs=4326
                )
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        "⬇️ GeoJSON",
                        data=gdf_out.to_json(),
                        file_name="isochrones.geojson",
                        mime="application/json"
                    )
                with col_dl2:
                    try:
                        shp_bytes = export_shapefile(gdf_out)
                        st.download_button(
                            "⬇️ Shapefile (.zip)",
                            data=shp_bytes,
                            file_name="isochrones_shp.zip",
                            mime="application/zip"
                        )
                    except Exception as e:
                        st.caption(f"Shapefile indisponible : {e}")
            else:
                st.warning("Aucun résultat. Vérifiez vos paramètres ou la connectivité.")

st.divider()
st.markdown(
    '<div style="text-align:center;font-size:0.78rem;color:#7a7974;">'
    'Isochrones · OSM Pur (Tps_min) · OSMnx · ORS · OSRM · Valhalla · GraphHopper · © OpenStreetMap contributors'
    '</div>',
    unsafe_allow_html=True
)
