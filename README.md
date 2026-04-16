# 🗺️ Isochrones — Zones de Desserte Sanitaire

Application Streamlit de génération d'isochrones autour des structures de santé.

## 5 Moteurs supportés

| Moteur | Type | Clé requise | Remarque |
|---|---|---|---|
| **OSM local** (OSMnx + NetworkX) | Local | Non | Lit `Tps_min`, `Vit_kmh`, `maxspeed`, `osm_id` |
| **OpenRouteService (ORS)** | API | Oui (gratuit) | 500 req/jour — [signup](https://openrouteservice.org/dev/#/signup) |
| **OSRM Public** | API publique | Non | Approximatif, sans clé |
| **Valhalla Public** | API publique | Non | Polygones précis, 4 modes |
| **GraphHopper** | API publique | Non (opt.) | 500 req/jour sans clé |

## Fonctionnalités

- Chargement fichier structures sanitaires (GeoJSON / Shapefile)
- Chargement couche routes OSM avec colonnes **Tps_min / Vit_kmh / maxspeed / osm_id**
- Mode **comparaison** : superpose les résultats de plusieurs moteurs en couleurs différentes
- Export **GeoJSON** et **Shapefile (.zip)**
- 3 algorithmes de forme : Alpha Shape, Convex Hull, Buffer sur nœuds

## Déploiement Streamlit Cloud

```toml
# .streamlit/secrets.toml
ORS_API_KEY = "votre_clé_ors"
GH_API_KEY  = "votre_clé_graphhopper"  # optionnel
```

## Lancement local

```bash
pip install -r requirements.txt
streamlit run app/main.py
```
