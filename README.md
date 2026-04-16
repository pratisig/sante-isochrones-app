# 🏥 Zones de Desserte — Établissements de Santé

Application Streamlit pour générer des **isochrones et zones de desserte** autour des établissements de santé à partir de données OSM locales (routes avec colonnes `Tps_min`, `Vit_kmh`, `maxspeed`, `osm_id`).

## Fonctionnalités

- 📂 **Chargement de données locales** : routes OSM (GeoJSON/Shapefile/GPKG) + établissements de santé (GeoJSON/Shapefile/CSV)
- 🌐 **Téléchargement OSM automatique** via osmnx si aucune donnée locale
- 🚶 **3 modes de transport** : Marche, Véhicule, Vélo
- ⏱️ **Isochrones temporelles** basées sur `Tps_min` et `Vit_kmh` de vos routes
- 4 **méthodes de calcul** : Convexe, Concave, Offset (tampon routes), Interpolation grille (IDW/TIN)
- 📊 **Tableau de résultats** avec statistiques par établissement
- 💾 **Export GeoJSON** des zones calculées
- 🗺️ Visualisation interactive **PyDeck + Folium**

## Lancement

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Structure des données attendues

### Couche Routes OSM
| Colonne | Description |
|---------|-------------|
| `osm_id` | Identifiant OSM |
| `Tps_min` | Temps de parcours en minutes |
| `Vit_kmh` | Vitesse en km/h |
| `maxspeed` | Vitesse maximale |
| `geometry` | LineString |

### Couche Établissements de Santé
| Colonne | Description |
|---------|-------------|
| `nom` / `name` | Nom de l'établissement |
| `type` | Type (Hôpital, CS, CSPS...) |
| `geometry` | Point |

## Inspiré de
- [isochrone-app](https://github.com/adolmajian/isochrone-app) par Arthur Dolmajian
- [Article Medium](https://medium.com/@arthur.dolmajian/creating-isochrones-what-is-the-optimal-way-dfc77a2ca13a)
