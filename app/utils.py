import base64
import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import pydeck as pdk
import matplotlib.colors as mcolors

from io import BytesIO
from shapely.geometry import shape, MultiPoint, MultiPolygon, Polygon, LineString, MultiLineString, Point
from shapely.ops import unary_union
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from rasterio.io import MemoryFile
from rasterio.mask import mask
from PIL import Image

try:
    from shapelysmooth import taubin_smooth
    HAS_SMOOTH = True
except ImportError:
    HAS_SMOOTH = False
    def taubin_smooth(geom): return geom

# ── Palette couleurs ──────────────────────────────────────────────────────────
PALETTE = {
    'red':        [255, 75,  75],
    'white':      [255, 255, 255],
    'yellow':     [255, 249, 127],
    'teal':       [1,   105, 111],
    'aquamarine': [165, 255, 214],
    'melon':      [255, 166, 158],
    'beige':      [255, 235, 198],
    'orange':     [255, 177, 0],
    'purple':     [136, 67,  204],
    'onyx':       [46,  53,  50],
    'blue':       [0,   56,  68],
    'cyan':       [0,   148, 198],
    'green':      [67,  122, 34],
}

# Couleurs par mode de transport
MODE_COLORS = {
    'Marche':   [255, 177, 0,   160],  # orange
    'Vehicule': [255, 75,  75,  160],  # rouge
    'Velo':     [0,   148, 198, 160],  # cyan
}

# Intervalles de temps par mode (minutes)
MODE_INTERVALS = {
    'Marche':   [15, 30, 45, 60, 90],
    'Vehicule': [10, 20, 30, 60],
    'Velo':     [15, 30, 45, 60, 90],
}

# Vitesses moyennes km/h par mode
MODE_SPEEDS = {
    'Marche':   5.0,
    'Vehicule': 40.0,
    'Velo':     15.0,
}

# ── Helpers géométriques ──────────────────────────────────────────────────────
def plug_shape_holes(geom):
    if geom is None: return geom
    if geom.geom_type == 'MultiPolygon':
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms]).buffer(0)
    if geom.geom_type == 'Polygon':
        return Polygon(geom.exterior)
    return geom


def extract_exteriors(g):
    if g.geom_type == 'MultiPolygon':
        return MultiLineString([x.exterior for x in g.geoms])
    return LineString(g.exterior)


def get_gdf_corners(gdf):
    xmin, ymin, xmax, ymax = gdf.envelope.total_bounds
    return [[xmin, ymin],[xmin, ymax],[xmax, ymax],[xmax, ymin]]


# ── Couleurs ──────────────────────────────────────────────────────────────────
def add_color_to_df(df, col, colormap):
    df = df.copy()
    mn, mx = df[col].min(), df[col].max()
    if mx == mn:
        norm = df[col].apply(lambda x: 0.5)
    else:
        norm = (df[col] - mn) / (mx - mn)
    df['color_hex'] = norm.apply(lambda x: mcolors.to_hex(colormap(x)))
    df['color_rgb'] = norm.apply(
        lambda x: list((np.array(mcolors.to_rgb(colormap(x))) * 255).astype(np.uint8)))
    return df


# ── Construction du graphe depuis données locales ─────────────────────────────
def build_graph_from_roads(roads_gdf, mode='Marche'):
    """
    Construit un graphe NetworkX depuis une couche de routes OSM locale.
    Utilise Tps_min et Vit_kmh si disponibles, sinon calcule depuis la géométrie.
    """
    import osmnx as ox

    speed = MODE_SPEEDS[mode]
    G = nx.MultiDiGraph()
    G.graph['crs'] = 'epsg:4326'

    # Reprojeter en UTM pour mesurer les longueurs
    utm_crs = roads_gdf.estimate_utm_crs()
    roads_utm = roads_gdf.to_crs(utm_crs)

    node_id = 0
    node_coords = {}  # (lon, lat) -> node_id

    for idx, row in roads_utm.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Points de début et fin
        coords = list(geom.coords)
        start_coord = coords[0]
        end_coord   = coords[-1]

        # Convertir en WGS84 pour stocker
        from pyproj import Transformer
        transformer = Transformer.from_crs(utm_crs, 'epsg:4326', always_xy=True)
        s_lon, s_lat = transformer.transform(start_coord[0], start_coord[1])
        e_lon, e_lat = transformer.transform(end_coord[0], end_coord[1])

        # Créer les noeuds
        if (s_lon, s_lat) not in node_coords:
            node_coords[(s_lon, s_lat)] = node_id
            G.add_node(node_id, x=s_lon, y=s_lat, geometry=Point(s_lon, s_lat))
            node_id += 1
        if (e_lon, e_lat) not in node_coords:
            node_coords[(e_lon, e_lat)] = node_id
            G.add_node(node_id, x=e_lon, y=e_lat, geometry=Point(e_lon, e_lat))
            node_id += 1

        u = node_coords[(s_lon, s_lat)]
        v = node_coords[(e_lon, e_lat)]

        # Longueur en mètres
        length = geom.length

        # Temps de parcours
        if 'Tps_min' in roads_gdf.columns and pd.notna(row.get('Tps_min', None)):
            travel_time = float(row['Tps_min']) * 60  # secondes
        elif 'Vit_kmh' in roads_gdf.columns and pd.notna(row.get('Vit_kmh', None)):
            v_kmh = float(row['Vit_kmh'])
            travel_time = (length / 1000) / v_kmh * 3600
        else:
            travel_time = (length / 1000) / speed * 3600

        osm_id = row.get('osm_id', idx)
        G.add_edge(u, v, length=length, travel_time=travel_time, osmid=osm_id)
        G.add_edge(v, u, length=length, travel_time=travel_time, osmid=osm_id)  # bidirectionnel

    return G, utm_crs


# ── Calcul du sous-graphe accessible ─────────────────────────────────────────
def compute_accessible_subgraph(G, start_node, max_time_sec, weight='travel_time'):
    """
    Calcule le sous-graphe accessible depuis start_node dans max_time_sec secondes.
    """
    subgraph_nodes = set()
    lengths = nx.single_source_dijkstra_path_length(G, start_node, cutoff=max_time_sec, weight=weight)
    subgraph_nodes.update(lengths.keys())
    subgraph = G.subgraph(subgraph_nodes).copy()
    return subgraph, lengths


# ── Noeud le plus proche ──────────────────────────────────────────────────────
def find_nearest_node(G, lon, lat):
    """
    Trouve le noeud du graphe le plus proche d'un point (lon, lat).
    """
    import osmnx as ox
    try:
        return ox.nearest_nodes(G, lon, lat)
    except Exception:
        min_dist = float('inf')
        nearest = None
        pt = Point(lon, lat)
        for node, data in G.nodes(data=True):
            d = pt.distance(Point(data['x'], data['y']))
            if d < min_dist:
                min_dist = d
                nearest = node
        return nearest


# ── Conversion graphe → GeoDataFrames ────────────────────────────────────────
def graph_to_gdfs_custom(G):
    """
    Convertit le graphe custom en GeoDataFrames nodes + edges.
    """
    node_records = []
    for n, d in G.nodes(data=True):
        node_records.append({'osmid': n, 'x': d.get('x', 0), 'y': d.get('y', 0),
                             'geometry': d.get('geometry', Point(d.get('x',0), d.get('y',0)))})
    nodes_gdf = gpd.GeoDataFrame(node_records, geometry='geometry', crs='epsg:4326').set_index('osmid')

    edge_records = []
    for u, v, k, d in G.edges(data=True, keys=True):
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        line = LineString([(u_data.get('x',0), u_data.get('y',0)),
                           (v_data.get('x',0), v_data.get('y',0))])
        edge_records.append({'u': u, 'v': v, 'key': k,
                             'length': d.get('length', 0),
                             'travel_time': d.get('travel_time', 0),
                             'geometry': line})
    edges_gdf = gpd.GeoDataFrame(edge_records, geometry='geometry', crs='epsg:4326')
    return nodes_gdf, edges_gdf


# ── Méthodes isochrones ───────────────────────────────────────────────────────
def isochrone_convex(acc_nodes_gdf, padding_m=50):
    """Enveloppe convexe des noeuds accessibles."""
    if len(acc_nodes_gdf) < 3:
        return acc_nodes_gdf.geometry.buffer(0.001).unary_union
    utm_crs = acc_nodes_gdf.estimate_utm_crs()
    pts = acc_nodes_gdf.to_crs(utm_crs)
    hull = pts.geometry.unary_union.convex_hull.buffer(padding_m)
    return gpd.GeoDataFrame(geometry=[hull], crs=utm_crs).to_crs('epsg:4326').geometry.iloc[0]


def isochrone_offset(acc_edges_gdf, offset_m=100, cap_style='round', plug_holes=False):
    """Tampon autour des routes accessibles."""
    if len(acc_edges_gdf) == 0:
        return None
    utm_crs = acc_edges_gdf.estimate_utm_crs()
    edges_utm = acc_edges_gdf.to_crs(utm_crs)
    shape = edges_utm.buffer(offset_m, cap_style=cap_style).unary_union
    if plug_holes:
        shape = plug_shape_holes(shape)
    return gpd.GeoDataFrame(geometry=[shape], crs=utm_crs).to_crs('epsg:4326').geometry.iloc[0]


def isochrone_concave(acc_nodes_gdf, alpha=0.5, padding_m=50):
    """Hull concave (alpha shape) des noeuds accessibles."""
    try:
        from geonetworkx.tools import get_alpha_shape_polygon
        utm_crs = acc_nodes_gdf.estimate_utm_crs()
        pts_utm = acc_nodes_gdf.to_crs(utm_crs)
        pts = list(pts_utm.geometry.apply(lambda p: (p.x, p.y)))
        if len(pts) < 4:
            return isochrone_convex(acc_nodes_gdf, padding_m)
        shape = get_alpha_shape_polygon(pts, alpha).buffer(padding_m)
        return gpd.GeoDataFrame(geometry=[shape], crs=utm_crs).to_crs('epsg:4326').geometry.iloc[0]
    except Exception:
        return isochrone_convex(acc_nodes_gdf, padding_m)


# ── Interpolation grille ──────────────────────────────────────────────────────
def prepare_interpolation_points(G, start_node, acc_nodes_gdf, node_lengths, colormap):
    node_dists = pd.DataFrame(
        [(k, v) for k, v in node_lengths.items()],
        columns=['osmid', 'dist']
    ).set_index('osmid')
    node_dists = gpd.GeoDataFrame(
        node_dists.merge(acc_nodes_gdf[['geometry']], left_index=True, right_index=True),
        geometry='geometry', crs=acc_nodes_gdf.crs
    )
    node_dists = add_color_to_df(node_dists, 'dist', colormap)
    return node_dists


def grid_interpolate(node_vals, ref_gdf, target_resolution, algo, p=3):
    ref_crs = ref_gdf.crs
    points = node_vals.to_crs(ref_crs).geometry.values
    points = [(pt.x, pt.y) for pt in points]
    vals = node_vals['dist'].values
    xmin, ymin, xmax, ymax = ref_gdf.total_bounds
    num_x = max(int((xmax - xmin) / target_resolution), 10)
    num_y = max(int((ymax - ymin) / target_resolution), 10)
    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, num_x),
        np.linspace(ymin, ymax, num_y)
    )
    if algo == 'idw':
        distances = cdist(points, np.c_[grid_x.ravel(), grid_y.ravel()])
        weights = 1 / (np.power(distances, p) + 1e-10)
        interp = np.sum(weights * vals[:, np.newaxis], axis=0) / np.sum(weights, axis=0)
        interp = interp.reshape(grid_x.shape)
    else:  # tin
        interpolator = LinearNDInterpolator(points, vals)
        interp = interpolator(grid_x, grid_y).reshape(grid_x.shape)

    height, width = interp.shape
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    profile = {'driver': 'GTiff', 'dtype': rasterio.float32, 'count': 1,
               'width': width, 'height': height, 'transform': transform, 'crs': ref_crs}

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(interp[::-1], 1)
        src = memfile.open(**profile)
        img = src.read(1)
        shapes = [ref_gdf.values[0]]
        try:
            img_masked, transform = mask(src, shapes, crop=True, filled=False)
        except Exception:
            img_masked = np.ma.array(img[np.newaxis], mask=np.zeros_like(img[np.newaxis], dtype=bool))

    return img_masked, src, transform


def colorize_image(img, colormap):
    img.mask[np.isnan(img.data)] = True
    img_arr = np.ma.filled(img, fill_value=0).squeeze()
    mn, mx = img_arr.min(), img_arr.max()
    arr_norm = (img_arr - mn) / (mx - mn + 1e-10)
    arr_rgba = colormap(arr_norm)
    arr_rgba = (arr_rgba[:, :, :4] * 255).astype(np.uint8)
    arr_rgba[:, :, 3:][img.mask.squeeze()] = 0
    return Image.fromarray(arr_rgba)


def convert_image_to_bytes_url(img):
    im_file = BytesIO()
    img.save(im_file, format="png")
    im_bytes = im_file.getvalue()
    im_b64 = str(base64.b64encode(im_bytes))[2:-1]
    return fr'"data:image/png;base64,{im_b64}"'


# ── Couches PyDeck ────────────────────────────────────────────────────────────
def make_geojson_layer(gdf, fill_color, line_color, radius=None,
                       line_width=2, filled=True, stroked=True,
                       pickable=True, tooltip=False):
    params = dict(
        type="GeoJsonLayer",
        data=gdf.__geo_interface__ if hasattr(gdf, '__geo_interface__') else gdf,
        get_fill_color=fill_color,
        get_line_color=line_color,
        line_width_min_pixels=line_width,
        filled=filled, stroked=stroked,
        pickable=pickable, auto_highlight=pickable,
    )
    if radius:
        params['get_radius'] = radius
    return pdk.Layer("GeoJsonLayer", **{k: v for k, v in params.items() if k != 'type'}, **{'type': 'GeoJsonLayer'})


def prepare_facility_layer(facilities_gdf):
    return pdk.Layer(
        type="GeoJsonLayer",
        data=facilities_gdf.__geo_interface__,
        get_radius=8,
        get_fill_color=PALETTE['yellow'] + [240],
        get_line_color=PALETTE['white'] + [200],
        line_width_max_pixels=3,
        stroked=True, filled=True,
        pickable=True, auto_highlight=True,
    )


def prepare_roads_layer(roads_gdf):
    return pdk.Layer(
        type="GeoJsonLayer",
        data=roads_gdf.__geo_interface__,
        get_line_color=[200, 200, 200, 120],
        line_width_min_pixels=1,
        stroked=True, filled=False,
        pickable=False,
    )


def prepare_isochrone_layer(iso_gdf, color_rgba):
    return pdk.Layer(
        type="GeoJsonLayer",
        data=iso_gdf.__geo_interface__,
        get_fill_color=color_rgba[:3] + [80],
        get_line_color=color_rgba[:3] + [220],
        line_width_min_pixels=2,
        stroked=True, filled=True,
        pickable=True, auto_highlight=True,
    )


# ── Export GeoJSON ────────────────────────────────────────────────────────────
def gdf_to_geojson_bytes(gdf):
    buf = BytesIO()
    gdf.to_file(buf, driver='GeoJSON')
    return buf.getvalue()


# ── Stats zone ────────────────────────────────────────────────────────────────
def compute_zone_stats(iso_gdf, name, mode, time_min):
    try:
        utm_crs = iso_gdf.estimate_utm_crs()
        area_km2 = round(iso_gdf.to_crs(utm_crs).geometry.area.sum() / 1e6, 2)
    except Exception:
        area_km2 = None
    return {
        'Établissement': name,
        'Mode': mode,
        'Temps (min)': time_min,
        'Superficie (km²)': area_km2,
    }
