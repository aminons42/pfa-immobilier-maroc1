from pathlib import Path
import requests
import pandas as pd
import time
import sys
import random
import os

sys.stdout.reconfigure(encoding='utf-8')

# =====================================================================
# CONFIGURATION
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
NOM_FICHIER_SORTIE = DATA_DIR / "avito_data_reel.csv"
DATE_LIMITE_STOP = '2022-12-01'  # Descendre jusqu'à décembre 2022
ANNONCES_PAR_PAGE = 24
DELAI_MIN = 2.0  # secondes entre chaque page
DELAI_MAX = 4.0
CATEGORY_ID = 1000  # Immobilier

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0',
]

GRAPHQL_QUERY = '''
query getListingAds($query: ListingAdsSearchQuery!) {
  getListingAds(query: $query) {
    ads {
      details {
        ... on PublishedAd {
          adId
          type { key }
          category { name }
          title
          listTime
          price { withoutCurrency }
          location { city { name } area { name } }
          params {
            secondary {
              __typename
              ... on NumericAdParam { id name numericValue }
              ... on TextAdParam { id name textValue }
              ... on BooleanAdParam { id name booleanValue }
            }
          }
        }
      }
    }
  }
}
'''

# =====================================================================
# FONCTIONS
# =====================================================================
def get_headers():
    return {
        'accept': 'application/json',
        'content-type': 'application/json',
        'origin': 'https://www.avito.ma',
        'referer': 'https://www.avito.ma/',
        'user-agent': random.choice(user_agents),
        'accept-language': 'fr-FR,fr;q=0.9,en;q=0.8',
    }


def extraire_params(secondaires):
    """Extrait les paramètres depuis la liste des champs secondaires."""
    chambres = salles_de_bain = etage = surface = etat = None
    for p in secondaires:
        if not isinstance(p, dict):
            continue
        p_id = p.get('id')
        if 'numericValue' in p:
            valeur = p['numericValue']
        elif 'textValue' in p:
            valeur = p['textValue']
        elif 'booleanValue' in p:
            valeur = p['booleanValue']
        else:
            continue

        if p_id == 'rooms':
            chambres = valeur
        elif p_id == 'bathrooms':
            salles_de_bain = valeur
        elif p_id == 'floor':
            etage = valeur
        elif p_id in ('size', 'surface'):
            surface = valeur
        elif p_id == 'estate_condition':
            etat = valeur

    return chambres, salles_de_bain, etage, surface, etat


def charger_existant():
    """Recharge les données déjà collectées si le fichier existe (reprise)."""
    if NOM_FICHIER_SORTIE.exists():
        df = pd.read_csv(NOM_FICHIER_SORTIE)
        print(f'  [REPRISE] {len(df)} annonces déjà collectées chargées.')
        return df.to_dict('records')
    return []


def sauvegarder(liste):
    pd.DataFrame(liste).to_csv(NOM_FICHIER_SORTIE, index=False, encoding='utf-8-sig')


def construire_cle(titre, prix, surface):
    return f"{titre}|{prix}|{surface}"


# =====================================================================
# SCRAPING PRINCIPAL
# =====================================================================
print('=' * 65)
print('  SCRAPER IMMOBILIER AVITO MAROC — DONNÉES RÉELLES')
print(f"  Objectif : Descendre jusqu'au {DATE_LIMITE_STOP}")
print(f'  Fichier de sortie : {NOM_FICHIER_SORTIE}')
print('=' * 65)

liste_annonces = charger_existant()

# Index pour déduplication rapide (titre|prix|surface)
cles_existantes = set(
    construire_cle(r.get('Titre', ''), r.get('Prix_DH', ''), r.get('Surface_m2', ''))
    for r in liste_annonces
)

atteint_limite = False
erreurs_consecutives = 0
MAX_ERREURS = 5

for numero_page in range(8000, 5_000_000):
    if atteint_limite:
        break

    time.sleep(random.uniform(DELAI_MIN, DELAI_MAX))

    json_data = {
        'operationName': 'getListingAds',
        'variables': {
            'query': {
                'filters': {
                    'ad': {'categoryId': CATEGORY_ID, 'hasImage': True}
                },
                'page': {'number': numero_page, 'size': ANNONCES_PAR_PAGE},
                'sort': {'adProperty': 'LIST_TIME', 'sortOrder': 'DESC'},
            },
        },
        'query': GRAPHQL_QUERY,
    }

    try:
        response = requests.post(
            'https://gateway.avito.ma/graphql',
            headers=get_headers(),
            json=json_data,
            timeout=15,
        )

        if response.status_code != 200:
            erreurs_consecutives += 1
            print(f'  [WARN] HTTP {response.status_code} page {numero_page} — attente 10s...')
            time.sleep(10)
            if erreurs_consecutives >= MAX_ERREURS:
                print(f'  [STOP] {MAX_ERREURS} erreurs consécutives. Arrêt.')
                break
            continue

        erreurs_consecutives = 0
        data = response.json()
        annonces = (
            data.get('data', {})
            .get('getListingAds', {})
            .get('ads', {})
            .get('details', [])
        )

        if not annonces:
            print('  [INFO] Fin du flux Avito — plus d\'annonces disponibles.')
            break

        nouvelles = 0

        for ad in annonces:
            # Uniquement les annonces de VENTE
            if not isinstance(ad, dict):
                continue
            if ad.get('type', {}).get('key') != 'SELL':
                continue

            prix = ad.get('price', {}).get('withoutCurrency') if ad.get('price') else None
            titre = ad.get('title', '')
            secondaires = ad.get('params', {}).get('secondary', []) if ad.get('params') else []

            chambres, salles_de_bain, etage, surface, etat = extraire_params(secondaires)

            # On exige au minimum : prix ET surface
            if prix is None or surface is None:
                continue

            ville = ad.get('location', {}).get('city', {}).get('name', 'N/A')
            quartier = ad.get('location', {}).get('area', {}).get('name', 'N/A')
            date_pub = ad.get('listTime', '')
            type_bien = ad.get('category', {}).get('name', 'N/A')

            # Déduplication rapide par clé
            cle = construire_cle(titre, prix, surface)
            if cle in cles_existantes:
                continue

            cles_existantes.add(cle)
            liste_annonces.append({
                'Date_Annonce': date_pub,
                'Type_Bien': type_bien,
                'Titre': titre,
                'Prix_DH': prix,
                'Surface_m2': surface,
                'Nombre_Chambres': chambres,
                'Salles_de_bain': salles_de_bain,
                'Etage': etage,
                'Etat': etat,
                'Ville': ville,
                'Quartier': quartier,
            })
            nouvelles += 1

        if nouvelles > 0:
            # Déterminer la date la plus ancienne collectée jusqu'ici
            dates = [r['Date_Annonce'] for r in liste_annonces if r.get('Date_Annonce')]
            date_plus_ancienne = min(dates) if dates else 'N/A'

            # Sauvegarde toutes les 10 pages
            if numero_page % 10 == 0:
                sauvegarder(liste_annonces)

            print(
                f'  Page {numero_page:5d} | +{nouvelles:2d} nouvelles | '
                f'Total : {len(liste_annonces):7,d} | '
                f'Date min : {date_plus_ancienne[:10]}'
            )

            # Condition d'arrêt
            if date_plus_ancienne[:10] < DATE_LIMITE_STOP:
                print(f"\n  [OBJECTIF] Date {DATE_LIMITE_STOP} atteinte — arrêt du scraper.")
                atteint_limite = True

        else:
            print(f'  Page {numero_page:5d} | 0 nouvelles (doublons ou sans prix/surface)')

    except requests.exceptions.Timeout:
        print(f'  [TIMEOUT] Page {numero_page} — on passe.')
        erreurs_consecutives += 1
    except Exception as e:
        print(f'  [ERREUR] Page {numero_page} : {str(e)[:80]}')
        erreurs_consecutives += 1

# Sauvegarde finale
sauvegarder(liste_annonces)
print(f"\n{'=' * 65}")
print(f'  TERMINÉ — {len(liste_annonces):,} annonces sauvegardées dans {NOM_FICHIER_SORTIE}')
print(f"{'=' * 65}")