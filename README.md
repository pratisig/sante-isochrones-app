# 🗺️ Générateur d'Isochrones — Zones de Desserte

Application **Streamlit** multi-moteurs pour calculer des zones de desserte (isochrones)
autour de structures sanitaires, sans ArcPy.

## 🔌 Moteurs de routing supportés

| Moteur | Clé API ? | Précision | Remarques |
|--------|-----------|-----------|----------|
| **OSM local** (OSMnx + NetworkX) | ❌ Non | ⭐⭐⭐⭐ | 100% local, utilise vos routes OSM |
| **OpenRouteService (ORS)** | ✅ Gratuite | ⭐⭐⭐⭐⭐ | 500 req/jour, 3 modes |
| **OSRM Public** | ❌ Non | ⭐⭐⭐ | Approx. par matrice de durées |
| **Valhalla Public** | ❌ Non | ⭐⭐⭐⭐ | Polygones natifs, 4 modes |

## 🚀 Installation

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## 🔑 Clé ORS — Streamlit Secrets

Créez `.streamlit/secrets.toml` :

```toml
ORS_API_KEY = "votre_clé_ors_ici"
```

Ou sur **Streamlit Cloud** → Settings → Secrets.

## 📦 Colonnes OSM utilisées (mode OSM local)

| Colonne | Description |
|---------|-------------|
| `Tps_min` | Temps de parcours en minutes |
| `Vit_kmh` | Vitesse en km/h |
| `maxspeed` | Vitesse maximale OSM |
| `osm_id` | Identifiant OSM |

## 📥 Entrées supportées

- **GeoJSON / Shapefile** : couche de structures sanitaires (points)
- **Saisie manuelle** : `nom, longitude, latitude`
- **Exemple intégré** : 3 structures à Ouagadougou (Burkina Faso)

## 📤 Sorties

- 🗺️ Carte interactive Folium avec couches par intervalle de temps
- 📊 Tableau récapitulatif (superficie en km²)
- ⬇️ Export **GeoJSON** des isochrones calculés

## 🏗️ Structure du projet

```
sante-isochrones-app/
├── app/
│   ├── main.py          # Application Streamlit principale
│   └── utils.py         # Fonctions de calcul d'isochrones
├── .streamlit/
│   └── config.toml      # Configuration Streamlit
├── requirements.txt
└── README.md
```
