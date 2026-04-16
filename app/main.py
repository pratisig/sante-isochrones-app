import warnings
warnings.filterwarnings("ignore")

import io
import json
import zipfile
import tempfile
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import folium
import streamlit as st
import osmnx as ox
import pydeck as pdk
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from shapely.geometry import Point, MultiPolygon
from streamlit_folium import st_folium

from utils import (
    PALETTE, MODE_COLORS, MODE_INTERVALS, MODE_SPEEDS,
    plug_shape_holes, get_gdf_corners,
    build_graph_from_roads, compute_accessible_subgraph,
    find_nearest_node, graph_to_gdfs_custom,
    isochrone_convex, isochrone_offset, isochrone_concave,
    prepare_interpolation_points, grid_interpolate,
    colorize_image, convert_image_to_bytes_url,
    prepare_facility_layer, prepare_roads_layer, prepare_isochrone_layer,
    gdf_to_geojson_bytes, compute_zone_stats, add_color_to_df
)

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Zones de Desserte — Santé",
    page_icon="🏥",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 340px; max-width: 400px; }
.block-container { padding-top: 1rem; }
.metric-box {
    background: #f3f0ec; border-radius: 8px;
    padding: 10px 16px; margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

colormap = plt.cm.YlOrRd

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    'roads_gdf': None,
    'facilities_gdf': None,
    'results': [],      # liste de dicts {name, mode, time_min, geometry}
    'stats': [],
    'graph': None,
    'graph_utm_crs': None,
    'use_local_roads': True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers chargement ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_gdf_from_upload(file_bytes, filename):
    suffix = filename.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(suffix=f'.{suffix}', delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        if suffix in ('geojson', 'json'):
            gdf = gpd.read_file(tmp_path)
        elif suffix in ('shp',):
            gdf = gpd.read_file(tmp_path)
        elif suffix in ('gpkg',):
            gdf = gpd.read_file(tmp_path)
        elif suffix in ('zip',):
            with zipfile.ZipFile(tmp_path) as z:
                z.extractall(tempfile.gettempdir())
                shp_files = [f for f in z.namelist() if f.endswith('.shp')]
                if shp_files:
                    gdf = gpd.read_file(os.path.join(tempfile.gettempdir(), shp_files[0]))
                else:
                    raise ValueError("Aucun .shp trouvé dans le ZIP")
        else:
            raise ValueError(f"Format non supporté: {suffix}")
        if gdf.crs is None:
            gdf = gdf.set_crs('epsg:4326')
        else:
            gdf = gdf.to_crs('epsg:4326')
        return gdf
    finally:
        os.unlink(tmp_path)


@st.cache_data(show_spinner=False)
def load_facilities_from_csv(file_bytes, lon_col, lat_col, name_col):
    df = pd.read_csv(io.BytesIO(file_bytes))
    geometry = [Point(row[lon_col], row[lat_col]) for _, row in df.iterrows()]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='epsg:4326')
    return gdf


@st.cache_data(show_spinner=False)
def download_osm_network(lat, lon, dist, network_type):
    return ox.graph_from_point((lat, lon), dist=dist, network_type=network_type)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 8px 0 4px 0;'>
        <svg width='36' height='36' viewBox='0 0 36 36' fill='none' xmlns='http://www.w3.org/2000/svg'>
          <circle cx='18' cy='18' r='18' fill='#01696f'/>
          <path d='M18 8v20M8 18h20' stroke='white' stroke-width='3.5' stroke-linecap='round'/>
          <circle cx='18' cy='18' r='6' stroke='white' stroke-width='2'/>
        </svg>
        <div style='font-size:17px; font-weight:700; color:#01696f; margin-top:4px;'>Zones de Desserte</div>
        <div style='font-size:12px; color:#7a7974;'>Établissements de Santé</div>
    </div>
    <hr style='border-color:#dcd9d5; margin: 8px 0;'/>
    """, unsafe_allow_html=True)

    # ── 1. DONNÉES ROUTES ───────────────────────────────────────────────────
    st.markdown("### 📁 1. Données Routes")
    data_source = st.radio(
        "Source des routes",
        ["Charger fichier local (OSM)", "Télécharger depuis OSM (en ligne)"],
        help="Utilisez vos routes locales avec Tps_min, Vit_kmh, maxspeed, osm_id"
    )

    if data_source == "Charger fichier local (OSM)":
        road_file = st.file_uploader(
            "Routes OSM (GeoJSON, Shapefile ZIP, GPKG)",
            type=['geojson', 'json', 'zip', 'gpkg'],
            help="Couche avec colonnes: osm_id, Tps_min, Vit_kmh, maxspeed"
        )
        if road_file:
            with st.spinner("Chargement des routes..."):
                try:
                    gdf = load_gdf_from_upload(road_file.read(), road_file.name)
                    st.session_state.roads_gdf = gdf
                    st.session_state.use_local_roads = True
                    # Afficher colonnes détectées
                    cols = list(gdf.columns)
                    detected = [c for c in ['osm_id','Tps_min','Vit_kmh','maxspeed'] if c in cols]
                    st.success(f"✅ {len(gdf)} tronçons chargés")
                    if detected:
                        st.info(f"Colonnes détectées : {', '.join(detected)}")
                    else:
                        st.warning("⚠️ Colonnes Tps_min/Vit_kmh non trouvées. Vitesse par défaut utilisée.")
                except Exception as e:
                    st.error(f"Erreur chargement : {e}")
    else:
        st.session_state.use_local_roads = False
        st.info("Les routes seront téléchargées automatiquement via OSMnx lors du calcul.")

    st.divider()

    # ── 2. ÉTABLISSEMENTS DE SANTÉ ──────────────────────────────────────────
    st.markdown("### 🏥 2. Établissements de Santé")
    fac_source = st.radio(
        "Source des établissements",
        ["Charger fichier", "Saisir manuellement"],
    )

    if fac_source == "Charger fichier":
        fac_file = st.file_uploader(
            "Établissements (GeoJSON, Shapefile ZIP, GPKG, CSV)",
            type=['geojson', 'json', 'zip', 'gpkg', 'csv'],
            key="fac_upload"
        )
        if fac_file:
            with st.spinner("Chargement des établissements..."):
                try:
                    if fac_file.name.endswith('.csv'):
                        df_tmp = pd.read_csv(fac_file)
                        st.write("Colonnes CSV :", list(df_tmp.columns))
                        lon_col = st.selectbox("Colonne Longitude", df_tmp.columns)
                        lat_col = st.selectbox("Colonne Latitude", df_tmp.columns)
                        name_col = st.selectbox("Colonne Nom", df_tmp.columns)
                        fac_file.seek(0)
                        gdf_fac = load_facilities_from_csv(fac_file.read(), lon_col, lat_col, name_col)
                    else:
                        gdf_fac = load_gdf_from_upload(fac_file.read(), fac_file.name)
                    st.session_state.facilities_gdf = gdf_fac
                    st.success(f"✅ {len(gdf_fac)} établissements chargés")
                except Exception as e:
                    st.error(f"Erreur : {e}")
    else:
        with st.form("manual_facility"):
            fac_name = st.text_input("Nom de l'établissement", value="Centre de Santé")
            fac_lat  = st.number_input("Latitude",  value=12.3700, format="%.6f")
            fac_lon  = st.number_input("Longitude", value=-1.5200, format="%.6f")
            fac_type = st.selectbox("Type", ["Hôpital", "Centre de Santé", "CSPS", "Dispensaire", "Maternité", "Autre"])
            add_btn = st.form_submit_button("➕ Ajouter")
            if add_btn:
                new_row = gpd.GeoDataFrame(
                    [{'nom': fac_name, 'type': fac_type, 'geometry': Point(fac_lon, fac_lat)}],
                    geometry='geometry', crs='epsg:4326'
                )
                if st.session_state.facilities_gdf is None:
                    st.session_state.facilities_gdf = new_row
                else:
                    st.session_state.facilities_gdf = pd.concat(
                        [st.session_state.facilities_gdf, new_row], ignore_index=True
                    )
                st.success(f"✅ {fac_name} ajouté")

    if st.session_state.facilities_gdf is not None:
        st.markdown(f"**{len(st.session_state.facilities_gdf)} établissement(s) chargé(s)**")
        if st.button("🗑️ Effacer les établissements"):
            st.session_state.facilities_gdf = None
            st.rerun()

    st.divider()

    # ── 3. PARAMÈTRES DE CALCUL ─────────────────────────────────────────────
    st.markdown("### ⚙️ 3. Paramètres")

    modes_selected = st.multiselect(
        "Modes de transport",
        options=["Marche", "Vehicule", "Velo"],
        default=["Marche"],
        help="Marche: 5km/h | Véhicule: 40km/h | Vélo: 15km/h (si pas de Vit_kmh dans les données)"
    )

    st.markdown("**Intervalles de temps (minutes)**")
    time_col1, time_col2 = st.columns(2)
    custom_times = {}
    for mode in modes_selected:
        default_times = MODE_INTERVALS.get(mode, [15, 30, 60])
        selected_times = st.multiselect(
            f"{mode}",
            options=[5, 10, 15, 20, 30, 45, 60, 90, 120],
            default=[t for t in default_times if t in [5,10,15,20,30,45,60,90,120]],
            key=f"times_{mode}"
        )
        custom_times[mode] = sorted(selected_times) if selected_times else default_times

    st.markdown("**Méthode isochrone**")
    method = st.selectbox(
        "Méthode de calcul",
        ["Offset (Tampon routes)", "Convexe", "Concave", "Interpolation IDW"],
        help="Offset = tampon autour des routes accessibles (recommandé pour données locales)"
    )

    with st.expander("Paramètres avancés"):
        offset_m = st.slider("Tampon offset (m)", 50, 500, 150, 50)
        padding_m = st.slider("Padding convexe/concave (m)", 10, 200, 50, 10)
        alpha_pct = st.slider("Alpha concave (%)", 10, 100, 80, 5)
        plug_holes = st.checkbox("Combler les trous", value=True)
        osm_dist   = st.slider("Rayon download OSM (m)", 1000, 20000, 5000, 500,
                               help="Distance de téléchargement si source OSM en ligne")

    st.divider()

    # ── 4. LANCER ───────────────────────────────────────────────────────────
    run_btn = st.button(
        "🚀 Calculer les zones de desserte",
        type="primary",
        disabled=(st.session_state.facilities_gdf is None),
        use_container_width=True
    )
    if st.session_state.facilities_gdf is None:
        st.caption("⚠️ Chargez d'abord les établissements de santé.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — TABS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🏥 Zones de Desserte — Établissements de Santé")
st.caption("Isochrones calculées à partir de vos données OSM locales (Tps_min, Vit_kmh, maxspeed, osm_id)")

tab_map, tab_table, tab_export, tab_help = st.tabs([
    "🗺️ Carte", "📊 Résultats", "💾 Export", "ℹ️ Guide"
])

# ── CALCUL ────────────────────────────────────────────────────────────────────
if run_btn and st.session_state.facilities_gdf is not None:
    st.session_state.results = []
    st.session_state.stats = []

    facilities = st.session_state.facilities_gdf
    roads_gdf  = st.session_state.roads_gdf
    use_local  = st.session_state.use_local_roads and roads_gdf is not None

    # Nom column
    nom_col = 'nom' if 'nom' in facilities.columns else (
              'name' if 'name' in facilities.columns else facilities.columns[0])

    progress_bar = st.progress(0, text="Démarrage du calcul...")
    total_steps = len(facilities) * len(modes_selected)
    step = 0

    for fac_idx, fac_row in facilities.iterrows():
        fac_name = str(fac_row.get(nom_col, f"Etab_{fac_idx}"))
        fac_pt   = fac_row.geometry
        fac_lon, fac_lat = fac_pt.x, fac_pt.y

        for mode in modes_selected:
            step += 1
            pct = int(step / total_steps * 100)
            progress_bar.progress(pct, text=f"Calcul {fac_name} — {mode} ({step}/{total_steps})")

            try:
                # Construire le graphe
                if use_local:
                    G, utm_crs = build_graph_from_roads(roads_gdf, mode)
                else:
                    net_type = {'Marche': 'walk', 'Vehicule': 'drive', 'Velo': 'bike'}[mode]
                    ox_graph = download_osm_network(fac_lat, fac_lon, osm_dist, net_type)
                    ox_graph = ox.add_edge_travel_times(ox.add_edge_speeds(ox_graph))
                    G = ox_graph
                    utm_crs = None

                # Noeud de départ
                start_node = find_nearest_node(G, fac_lon, fac_lat)
                if start_node is None:
                    st.warning(f"Noeud introuvable pour {fac_name}")
                    continue

                # Noeuds & arêtes
                nodes_gdf, edges_gdf = graph_to_gdfs_custom(G)

                times_min = custom_times.get(mode, MODE_INTERVALS[mode])
                max_time_sec = max(times_min) * 60

                # Sous-graphe accessible
                subgraph, node_lengths = compute_accessible_subgraph(
                    G, start_node, max_time_sec, weight='travel_time'
                )

                if len(subgraph.nodes) == 0:
                    st.warning(f"Aucun noeud accessible pour {fac_name} — {mode}")
                    continue

                acc_nodes_gdf, acc_edges_gdf = graph_to_gdfs_custom(subgraph)

                # Isochrone par intervalle de temps
                for t_min in times_min:
                    t_sec = t_min * 60
                    # Filtrer les noeuds accessibles dans ce temps
                    nodes_in_t = {n: d for n, d in node_lengths.items() if d <= t_sec}
                    if len(nodes_in_t) < 2:
                        continue

                    acc_nodes_t = nodes_gdf[nodes_gdf.index.isin(nodes_in_t.keys())].copy()
                    acc_edges_t = edges_gdf[
                        edges_gdf['u'].isin(nodes_in_t.keys()) &
                        edges_gdf['v'].isin(nodes_in_t.keys())
                    ].copy()

                    # Calculer l'isochrone selon la méthode choisie
                    try:
                        if method == "Offset (Tampon routes)":
                            iso_geom = isochrone_offset(acc_edges_t, offset_m, plug_holes=plug_holes)
                        elif method == "Convexe":
                            iso_geom = isochrone_convex(acc_nodes_t, padding_m)
                        elif method == "Concave":
                            iso_geom = isochrone_concave(acc_nodes_t, alpha_pct, padding_m)
                        else:  # IDW
                            iso_geom = isochrone_convex(acc_nodes_t, padding_m)

                        if iso_geom is None or iso_geom.is_empty:
                            continue

                        iso_gdf = gpd.GeoDataFrame(
                            [{'facility': fac_name, 'mode': mode,
                              'time_min': t_min, 'methode': method,
                              'geometry': iso_geom}],
                            geometry='geometry', crs='epsg:4326'
                        )

                        st.session_state.results.append({
                            'facility': fac_name, 'mode': mode,
                            'time_min': t_min, 'gdf': iso_gdf,
                            'nodes': acc_nodes_t, 'edges': acc_edges_t,
                        })

                        stats = compute_zone_stats(iso_gdf, fac_name, mode, t_min)
                        st.session_state.stats.append(stats)

                    except Exception as e:
                        st.warning(f"Erreur isochrone {fac_name}/{mode}/{t_min}min : {e}")

            except Exception as e:
                st.error(f"Erreur pour {fac_name} — {mode} : {e}")

    progress_bar.empty()
    if st.session_state.results:
        st.success(f"✅ {len(st.session_state.results)} zones calculées avec succès !")
    else:
        st.error("Aucune zone calculée. Vérifiez vos données.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CARTE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_map:
    if not st.session_state.results and st.session_state.facilities_gdf is None:
        st.info("👈 Chargez vos données dans la barre latérale et lancez le calcul.")
    else:
        # Carte de base centrée
        if st.session_state.facilities_gdf is not None:
            center_pt = st.session_state.facilities_gdf.geometry.unary_union.centroid
            map_center = [center_pt.y, center_pt.x]
        else:
            map_center = [12.37, -1.52]

        col_map, col_legend = st.columns([5, 1])

        with col_map:
            if st.session_state.results:
                # Préparer les couches PyDeck
                layers = []

                # Routes locales
                if st.session_state.roads_gdf is not None:
                    layers.append(prepare_roads_layer(st.session_state.roads_gdf))

                # Isochrones — par temps décroissant (les plus grandes en dessous)
                results_sorted = sorted(st.session_state.results,
                                        key=lambda r: r['time_min'], reverse=True)

                # Filtre interactif
                all_modes = list(set(r['mode'] for r in st.session_state.results))
                all_times = sorted(set(r['time_min'] for r in st.session_state.results))

                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    filter_modes = st.multiselect("Filtrer par mode", all_modes, default=all_modes, key="filt_mode")
                with filter_col2:
                    filter_times = st.multiselect("Filtrer par temps (min)", all_times, default=all_times, key="filt_time")

                for r in results_sorted:
                    if r['mode'] not in filter_modes:
                        continue
                    if r['time_min'] not in filter_times:
                        continue

                    base_color = MODE_COLORS.get(r['mode'], [100, 100, 200, 120])
                    # Opacité croissante avec le temps
                    max_t = max(all_times) if all_times else 60
                    opacity = int(40 + (r['time_min'] / max_t) * 120)
                    color = base_color[:3] + [opacity]

                    layers.append(prepare_isochrone_layer(r['gdf'], color))

                # Établissements
                if st.session_state.facilities_gdf is not None:
                    layers.append(prepare_facility_layer(st.session_state.facilities_gdf))

                viewport = pdk.ViewState(
                    latitude=map_center[0], longitude=map_center[1],
                    zoom=10, pitch=0
                )

                st.pydeck_chart(pdk.Deck(
                    initial_view_state=viewport,
                    layers=layers,
                    tooltip={
                        "html": "<b>{facility}</b><br/>Mode: {mode}<br/>Temps: {time_min} min",
                        "style": {"backgroundColor": "#01696f", "color": "white", "fontSize": "13px"}
                    },
                    map_style="mapbox://styles/mapbox/light-v11",
                ))

            else:
                # Carte simple folium avant calcul
                m = folium.Map(location=map_center, zoom_start=10, tiles="CartoDB positron")
                if st.session_state.facilities_gdf is not None:
                    for _, row in st.session_state.facilities_gdf.iterrows():
                        folium.Marker(
                            location=[row.geometry.y, row.geometry.x],
                            tooltip=str(row.get('nom', row.get('name', 'Établissement'))),
                            icon=folium.Icon(color='red', icon='plus-sign')
                        ).add_to(m)
                if st.session_state.roads_gdf is not None:
                    folium.GeoJson(
                        st.session_state.roads_gdf.__geo_interface__,
                        style_function=lambda x: {'color': '#aaa', 'weight': 1, 'opacity': 0.5}
                    ).add_to(m)
                st_folium(m, use_container_width=True, height=520)

        with col_legend:
            st.markdown("**Légende**")
            for mode, color in MODE_COLORS.items():
                if mode in (filter_modes if st.session_state.results else []):
                    c = f"rgba({color[0]},{color[1]},{color[2]},0.8)"
                    icon = {'Marche': '🚶', 'Vehicule': '🚗', 'Velo': '🚲'}[mode]
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">' +
                        f'<div style="width:18px;height:18px;background:{c};border-radius:3px;"></div>' +
                        f'<span>{icon} {mode}</span></div>',
                        unsafe_allow_html=True
                    )
            st.markdown("---")
            st.markdown("🟡 Établissements")
            if st.session_state.results:
                all_times_sorted = sorted(set(r['time_min'] for r in st.session_state.results))
                st.markdown("**Intervalles :**")
                for t in all_times_sorted:
                    st.markdown(f"• {t} min")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RÉSULTATS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_table:
    if st.session_state.stats:
        df_stats = pd.DataFrame(st.session_state.stats)
        st.markdown(f"### {len(df_stats)} zones calculées")

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Zones totales", len(df_stats))
        with k2:
            st.metric("Établissements", df_stats['Établissement'].nunique())
        with k3:
            st.metric("Modes", df_stats['Mode'].nunique())
        with k4:
            avg_area = df_stats['Superficie (km²)'].mean()
            st.metric("Superficie moy.", f"{avg_area:.2f} km²" if avg_area else "—")

        st.dataframe(
            df_stats.sort_values(['Établissement', 'Mode', 'Temps (min)']),
            use_container_width=True, hide_index=True
        )

        # Graphique surfaces
        if df_stats['Superficie (km²)'].notna().any():
            st.markdown("### Superficie des zones par établissement et mode")
            fig, ax = plt.subplots(figsize=(10, 4))
            pivot = df_stats.pivot_table(
                index='Temps (min)', columns=['Établissement', 'Mode'],
                values='Superficie (km²)', aggfunc='mean'
            )
            pivot.plot(ax=ax, marker='o')
            ax.set_xlabel("Temps (min)")
            ax.set_ylabel("Superficie (km²)")
            ax.set_title("Évolution de la superficie selon le temps de desserte")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=8)
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Lancez le calcul pour voir les résultats ici.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_export:
    if st.session_state.results:
        st.markdown("### 💾 Exporter les zones de desserte")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            # Export GeoJSON global
            all_gdfs = [r['gdf'] for r in st.session_state.results]
            merged = pd.concat(all_gdfs, ignore_index=True)
            merged_gdf = gpd.GeoDataFrame(merged, geometry='geometry', crs='epsg:4326')

            geojson_bytes = gdf_to_geojson_bytes(merged_gdf)
            st.download_button(
                label="⬇️ Télécharger GeoJSON (toutes les zones)",
                data=geojson_bytes,
                file_name="zones_desserte_sante.geojson",
                mime="application/json",
                use_container_width=True
            )

        with export_col2:
            # Export CSV stats
            if st.session_state.stats:
                df_export = pd.DataFrame(st.session_state.stats)
                csv_bytes = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Télécharger CSV (statistiques)",
                    data=csv_bytes,
                    file_name="stats_zones_desserte.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        st.markdown("---")
        # Export par mode
        st.markdown("#### Export par mode de transport")
        for mode in set(r['mode'] for r in st.session_state.results):
            mode_gdfs = [r['gdf'] for r in st.session_state.results if r['mode'] == mode]
            mode_merged = gpd.GeoDataFrame(
                pd.concat(mode_gdfs, ignore_index=True),
                geometry='geometry', crs='epsg:4326'
            )
            geojson_mode = gdf_to_geojson_bytes(mode_merged)
            icon = {'Marche': '🚶', 'Vehicule': '🚗', 'Velo': '🚲'}.get(mode, '📍')
            st.download_button(
                label=f"{icon} GeoJSON — {mode}",
                data=geojson_mode,
                file_name=f"zones_desserte_{mode.lower()}.geojson",
                mime="application/json",
            )
    else:
        st.info("Lancez le calcul pour pouvoir exporter les résultats.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — GUIDE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_help:
    st.markdown("""
    ## 📖 Guide d'utilisation

    Cette application génère des **zones de desserte (isochrones)** autour des établissements
    de santé en utilisant votre réseau routier OSM local.

    ---

    ### 🗂️ Étape 1 — Charger les routes OSM

    Chargez votre couche de routes avec les colonnes suivantes :

    | Colonne | Type | Description |
    |---------|------|-------------|
    | `osm_id` | Entier | Identifiant OSM de la route |
    | `Tps_min` | Décimal | Temps de parcours en **minutes** |
    | `Vit_kmh` | Décimal | Vitesse de circulation en **km/h** |
    | `maxspeed` | Texte | Vitesse maximale autorisée |
    | `geometry` | LineString | Géométrie de la route |

    > Si `Tps_min` et `Vit_kmh` sont absents, la vitesse par défaut du mode est utilisée.

    **Formats acceptés :** GeoJSON, Shapefile (ZIP), GeoPackage

    ---

    ### 🏥 Étape 2 — Charger les établissements

    Chargez votre couche de points de santé (Hôpitaux, CSPS, Centres de Santé...).
    Vous pouvez aussi **saisir manuellement** les coordonnées d'un établissement.

    **Formats acceptés :** GeoJSON, Shapefile (ZIP), GeoPackage, CSV (avec colonnes lat/lon)

    ---

    ### ⚙️ Étape 3 — Configurer le calcul

    - **Mode de transport** : Marche, Véhicule ou Vélo
    - **Intervalles de temps** : ex. 15, 30, 45, 60 minutes
    - **Méthode** :
        - *Offset* : tampon autour des routes accessibles (recommandé)
        - *Convexe* : enveloppe convexe des noeuds accessibles
        - *Concave* : forme plus précise (alpha-shape)
        - *IDW* : interpolation spatiale (raster)

    ---

    ### 🚀 Étape 4 — Lancer & Explorer

    Cliquez **"Calculer les zones de desserte"** puis explorez :
    - 🗺️ **Carte** interactive avec filtres par mode et temps
    - 📊 **Résultats** avec statistiques et graphiques
    - 💾 **Export** GeoJSON ou CSV

    ---

    ### 🔬 Méthodes de calcul

    L'application utilise **Dijkstra** sur le graphe routier pour trouver tous les
    noeuds accessibles dans un temps donné, puis construit la zone de desserte
    par l'une des 4 méthodes disponibles.

    Inspiré de [isochrone-app](https://github.com/adolmajian/isochrone-app)
    par [Arthur Dolmajian](https://medium.com/@arthur.dolmajian/creating-isochrones-what-is-the-optimal-way-dfc77a2ca13a).

    ---
    *Application développée pour l'analyse de l'accessibilité aux soins de santé au Burkina Faso.*
    """
    )
