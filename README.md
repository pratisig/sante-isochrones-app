# 🗺️ Isochrones — Zones de Desserte Sanitaire

Application Streamlit de génération d'isochrones autour des établissements de santé,
basée sur les données OSM locales et plusieurs moteurs de routing.

## Moteurs disponibles

| Moteur | Type | Clé API | Description |
|--------|------|---------|-------------|
| **OSM Pur (Tps_min)** ★ | Local | Non | Graphe 100% depuis votre couche OSM (Tps_min, Vit_kmh). Aucun téléchargement. Le plus fidèle au terrain. |
| OSMnx + NetworkX | Local | Non | Télécharge le réseau OSM + enrichit avec votre couche locale |
| OpenRouteService (ORS) | API | Oui (gratuit 500/j) | Très précis |
| OSRM Public | Public | Non | Matrice de durées réseau (72 points, 2 couronnes) |
| Valhalla Public | Public | Non | Polygones précis, 4 modes |
| GraphHopper | API | Oui (gratuit 500/j) | Polygones précis |

## Colonnes de la couche OSM Routes

Le moteur **OSM Pur** exploite directement :
- `Tps_min` — temps de parcours en minutes (priorité maximale)
- `Vit_kmh` — vitesse en km/h (utilisé si Tps_min absent)
- `maxspeed` — vitesse max OSM (fallback)
- `osm_id` — identifiant du tronçon

## Installation

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Fonctionnalités

- **6 moteurs** de routing (local et API)
- **Mode comparaison** : superposition de plusieurs moteurs
- **3 algorithmes** géométriques : Alpha Shape, Convex Hull, Buffer
- **Statistiques** : aire km², tableau pivot structure × intervalle
- **Export** : GeoJSON + Shapefile (.zip) avec attributs complets
- **Modes** : Marche, Véhicule, Vélo, Moto (selon moteur)
- **Upload** : GeoJSON, Shapefile (ZIP), GPKG
