import requests
import pandas as pd
import time
import sys
import random

# Force Windows à accepter les accents sans planter
sys.stdout.reconfigure(encoding='utf-8')

# Rotation d'user agents pour éviter le blocage
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]

def get_headers():
    return {
        'accept': 'application/json',
        'content-type': 'application/json',
        'user-agent': random.choice(user_agents),
    }

liste_finales = []
nom_du_fichier = 'avito_data_2022_collection.csv'
target_lines = 1000000  # 1 million de lignes
target_date_2022 = '2022-12-31'  # S'arrêter quand on trouve 2022
atteint_2022 = False  # Flag pour arrêter toutes les stratégies

# Essayer de charger les données existantes
try:
    df_existant = pd.read_csv(nom_du_fichier)
    liste_finales = df_existant.to_dict('records')
    print(f"📥 {len(liste_finales)} lignes existantes chargées")
except:
    print("📄 Nouveau fichier - démarrage de zéro")

# Stratégies multiples pour maximiser les données
strategies = [
    # (description, city_ids, category_ids)
    ("Casablanca - Immobilier", [1], [1000]),
    ("Rabat - Immobilier", [2], [1000]),
    ("Marrakech - Immobilier", [3], [1000]),
    ("Fès - Immobilier", [4], [1000]),
    ("Tanger - Immobilier", [5], [1000]),
    ("Agadir - Immobilier", [6], [1000]),
    ("Meknes - Immobilier", [7], [1000]),
    ("Salé - Immobilier", [8], [1000]),
    ("Tétouan - Immobilier", [9], [1000]),
    ("Autres villes", [], [1000]),  # Toutes les villes
]

print(f"\n🚀 Stratégie multi-villes : {len(strategies)} approches différentes")
print(f"🎯 Objectif : {target_lines} lignes jusqu'à date {target_date_2022}")

for strategy_idx, (strategy_name, city_ids, category_ids) in enumerate(strategies, 1):
    
    if atteint_2022:
        print(f"\n⏹️  Date 2022 atteinte ! Arrêt global.")
        break
    print(f"\n{'='*70}")
    print(f"🔍 Stratégie {strategy_idx}/{len(strategies)} : {strategy_name}")
    print(f"{'='*70}")
    
    # Pages illimitées pour cette stratégie (on s'arrête seulement si 2022 trouvé)
    pages_par_strategie = 10000  # Très élevé
    nouvelles_donnees = 0
    
    for numero_page in range(1, pages_par_strategie + 1):
        
        # Arrêt si on a atteint 2022
        if atteint_2022:
            print(f"⏹️  Date 2022 trouvée ! Arrêt de cette stratégie.")
            break
        
        # Arrêt si on a assez de données ET on a du temps (continue pour chercher 2022)
        if len(liste_finales) >= target_lines:
            print(f"📊 {len(liste_finales)} lignes atteintes ! Continuant pour trouver 2022...")
        
        # Délai randomisé pour éviter le blocage (1-3 secondes)
        time.sleep(random.uniform(1, 3))
        
        json_data = {
            'operationName': 'getListingAds',
            'variables': {
                'query': {
                    'filters': {
                        'ad': {
                            'categoryId': category_ids[0] if category_ids else 1000,
                            'hasImage': True
                        },
                        'extension': {
                            'extendPublishedAdsSearchIfNeeded': True,
                        },
                    },
                    'page': {'number': numero_page, 'size': 15},
                    'sort': {'adProperty': 'LIST_TIME', 'sortOrder': 'DESC'},
                },
            },
            'query': '''
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
                          location { city { name id } area { name } }
                          params {
                            secondary {
                              ... on NumericAdParam { id name numericValue }
                            }
                          }
                        }
                      }
                    }
                  }
                }
            '''
        }

        try:
            response = requests.post('https://gateway.avito.ma/graphql', headers=get_headers(), json=json_data, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                try:
                    annonces = data['data']['getListingAds']['ads']['details']
                except (KeyError, TypeError):
                    continue

                if not isinstance(annonces, list):
                    continue

                annonces_sur_cette_page = 0

                for ad in annonces:
                    if not isinstance(ad, dict):
                        continue
                    
                    # Vérification que c'est une VENTE
                    if ad.get('type', {}).get('key') != 'SELL': 
                        continue
                    
                    # --- EXTRACTION SÉCURISÉE DE LA SURFACE ---
                    surface = None
                    params_dict = ad.get('params', {})
                    if isinstance(params_dict, dict):
                        secondaires = params_dict.get('secondary', [])
                        if isinstance(secondaires, list):
                            for p in secondaires:
                                if isinstance(p, dict):
                                    if p.get('id') == 'size' or 'surface' in str(p.get('name')).lower():
                                        surface = p.get('numericValue')
                                        break
                    
                    # --- EXTRACTION SÉCURISÉE DU PRIX ---
                    prix_dict = ad.get('price', {})
                    prix = prix_dict.get('withoutCurrency') if isinstance(prix_dict, dict) else None
                    
                    # On ne garde que les annonces avec prix ET surface
                    if prix is not None and surface is not None:
                        
                        # Localisation sécurisée
                        loc = ad.get('location', {})
                        if isinstance(loc, dict):
                            ville = loc.get('city', {}).get('name') if isinstance(loc.get('city'), dict) else 'N/A'
                            quartier = loc.get('area', {}).get('name') if isinstance(loc.get('area'), dict) else 'N/A'
                        else:
                            ville, quartier = 'N/A', 'N/A'

                        # Créer un identifiant unique pour éviter les doublons
                        unique_id = f"{ad.get('title')}_{prix}_{surface}_{ville}"
                        
                        # Vérifier si cette annonce existe déjà
                        if not any(row.get('Titre') == ad.get('title') and 
                                  row.get('Prix_DH') == prix and 
                                  row.get('Surface_m2') == surface 
                                  for row in liste_finales):
                            
                            liste_finales.append({
                                'Date_Annonce': ad.get('listTime'),
                                'Type_Bien': ad.get('category', {}).get('name') if isinstance(ad.get('category'), dict) else 'N/A',
                                'Titre': ad.get('title'),
                                'Prix_DH': prix,
                                'Surface_m2': surface,
                                'Ville': ville,
                                'Quartier': quartier
                            })
                            annonces_sur_cette_page += 1
                            nouvelles_donnees += 1
                
                if annonces_sur_cette_page > 0:
                    oldest_date = min([row['Date_Annonce'] for row in liste_finales if row['Date_Annonce']], default='N/A')
                    print(f"  Page {numero_page:3d} : +{annonces_sur_cette_page:2d} | Total : {len(liste_finales):7d} | Date min : {oldest_date}")
                    
                    # ✅ VÉRIFIER SI ON A ATTEINT 2022
                    if oldest_date != 'N/A' and oldest_date[:4] <= '2022':
                        print(f"\n🎯 BINGO ! Date 2022 ou plus ancienne trouvée : {oldest_date}")
                        atteint_2022 = True
                    
                    # Sauvegarder chaque page pour éviter la perte de données
                    pd.DataFrame(liste_finales).to_csv(nom_du_fichier, index=False, encoding='utf-8')

            else:
                if response.status_code != 200:
                    print(f"  ❌ HTTP {response.status_code} - page {numero_page}")
                time.sleep(2)

        except Exception as e:
            print(f"  ⚠️ Erreur page {numero_page}: {str(e)[:50]}")
            time.sleep(2)
            continue
    
    print(f"  📊 Stratégie terminée : +{nouvelles_donnees} nouvelles lignes")

print(f"\n{'='*70}")
if atteint_2022:
    print(f"🎉 COLLECTE TERMINEE - DATE 2022 TROUVEE !")
else:
    print(f"🎉 COLLECTE TERMINEE - CIBLE 1M LIGNE ATTEINTE !")
print(f"{'='*70}")

# Affichage final avec statistiques de dates
if liste_finales:
    df_final = pd.DataFrame(liste_finales)
    
    # Supprimer les doublons stricts
    df_final = df_final.drop_duplicates(subset=['Titre', 'Prix_DH', 'Surface_m2', 'Ville'])
    
    # Sauvegarder
    df_final.to_csv(nom_du_fichier, index=False, encoding='utf-8')
    
    dates = [row['Date_Annonce'] for row in df_final.to_dict('records') if row['Date_Annonce']]
    if dates:
        oldest = min(dates)
        newest = max(dates)
        print(f"📊 Total final : {len(df_final)} lignes (après suppression des doublons)")
        print(f"📅 Période couverte : {oldest} → {newest}")
        print(f"✅ Fichier sauvegardé : {nom_du_fichier}")
    else:
        print("❌ Aucune donnée valide trouvée")
else:
    print("❌ Aucune donnée n'a pu être collectée")