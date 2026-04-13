# Validation du flux API (démo)

Date de validation: 2026-02-24

## Endpoints validés

- `POST /predict`
- `POST /explain`
- `POST /explain/by-cin`

## Résultats des tests

### Cas 1 - Profil faible (manuel)
- Endpoint: `POST /predict`
- Résultat obtenu:
  - `prediction = 0`
  - `default_proba = 0.0001`
  - `risk_level = FAIBLE`

### Cas 2 - Profil élevé (manuel)
- Endpoint: `POST /predict`
- Résultat obtenu:
  - `prediction = 1`
  - `default_proba = 1.0000`
  - `risk_level = ELEVE`

### Cas 3 - Explication LLM locale (manuel)
- Endpoint: `POST /explain`
- Résultat obtenu:
  - score retourné correctement (`FAIBLE`)
  - message textuel généré par Ollama avec justification du niveau de risque

### Cas 4 - CIN valide
- Endpoint: `POST /explain/by-cin`
- Payload exemple:
```json
{
  "cin": "95822412"
}
```
- Résultat obtenu:
  - CIN retrouvé
  - `credit_id` sélectionné automatiquement
  - score + message LLM retournés correctement

### Cas 5 - CIN invalide (gestion d'erreur)
- Endpoint: `POST /explain/by-cin`
- Payload exemple:
```json
{
  "cin": "00000000"
}
```
- Résultat obtenu:
  - HTTP `404`
  - message: `CIN not found: 00000000`

## Note sur le niveau MODERE

Lors des tests de plusieurs payloads manuels et aléatoires, le modèle actuel retourne
majoritairement des probabilités très basses ou très hautes, avec peu/pas de scores
dans l'intervalle `0.30-0.60`.

Ce comportement est cohérent avec un dataset synthétique fortement séparé et explique
la rareté de `risk_level = MODERE`.

## Recommandations

1. Recalibrer le modèle (Platt/Isotonic) pour obtenir des probabilités mieux distribuées.
2. Revoir les features potentiellement trop discriminantes.
3. Ajouter un jeu de validation plus réaliste pour observer davantage de cas intermédiaires.
