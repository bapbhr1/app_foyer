# Assistant Contrats — démonstrateur RAG assurance

Application [Streamlit](https://streamlit.io/) d'aide à la décision pour un gestionnaire de sinistres. À partir d'une question en langage naturel, elle retrouve les clauses contractuelles pertinentes, génère une réponse structurée via LLM Groq (ici `gpt-oss-120b`), **vérifie chaque citation** dans les clauses sources, et signale les dossiers qui nécessitent une validation humaine.

> Corpus 100 % **synthétique** (7 contrats, 46 clauses). Aucune donnée client réelle.

## Le problème

Sur un assistant contractuel, une réponse fausse mais crédible coûte plus cher qu'une absence de réponse. Deux risques dominent :

1. **la recherche remonte la mauvaise clause** → le modèle répond sur un texte hors sujet ;
2. **le modèle invente ou reformule** une clause qui n'existe pas.

Le projet traite les deux : une recherche hybride *mesurée*, et un contrôle déterministe qui rejette toute affirmation non retrouvée **littéralement** dans les clauses transmises. Aucun framework RAG, chaque étape est écrite explicitement pour rester maîtrisée et testable.

## Résultats d'évaluation

Mesurés par `evaluate.py` sur 20 questions annotées (`data/eval_questions.json`) couvrant les 7 branches — cas couverts, exclusions, et cas piégeux (suicide la première année, défaut d'entretien, chirurgie esthétique). Recherche et génération sont évaluées **séparément**, car un échec de recherche est la première cause d'hallucination.

**Recherche** (sans appel modèle — pertinence jugée sur le couple `contract_id`/`article_id`)

| Métrique  | Valeur          | Lecture                                      |
| ---------- | --------------- | -------------------------------------------- |
| Hit-rate@5 | **100 %** | au moins une clause attendue est remontée   |
| Recall@5   | **100 %** | proportion des clauses attendues remontées  |
| MRR        | **1.00**  | rang moyen de la première clause pertinente |

**Génération** (avec les contrôles déterministes de l'application)

| Métrique            | Valeur          | Lecture                                                  |
| -------------------- | --------------- | -------------------------------------------------------- |
| Exactitude du statut | **90 %**  | décision conforme à l'attendu                          |
| Taux de citations    | **100 %** | réponses appuyées par ≥ 1 extrait vérifié           |
| Fidélité           | **100 %** | affirmations retrouvées littéralement dans les clauses |
| Taux d'escalade      | **10 %**  | dossiers renvoyés vers un gestionnaire                  |

Ces chiffres sont élevés parce que le corpus est petit et le vocabulaire distinctif : c'est la limite assumée d'une matrice d'embeddings en mémoire. À plus grande échelle ils baisseraient — et c'est justement l'intérêt d'un harnais d'évaluation : rendre cette dégradation visible et chiffrée avant qu'elle n'atteigne l'utilisateur.

```bash
python3 evaluate.py                 # recherche + génération (si GROQ_API_KEY est défini)
python3 evaluate.py --no-llm        # recherche seule, sans appel modèle
python3 evaluate.py --report eval_report.md
```

## Fonctionnement

```text
Contrats JSON
  → découpage par article
  → recherche hybride (embeddings multilingues + lexical) et filtrage par branche
  → sélection des clauses les plus pertinentes
  → génération JSON avec Groq
  → validation Pydantic
  → contrôle exact des citations dans les clauses sources
  → calcul de fiabilité et garde-fou métier
```

La **fiabilité** affichée additionne trois contrôles automatiques : informations vérifiées (50 %), réponse adaptée aux preuves (30 %), utilisation des sources (20 %). Une abstention justifiée peut donc obtenir une fiabilité élevée. Cet indicateur mesure les contrôles de l'application ; il ne constitue pas une probabilité juridique de prise en charge.

## Fonctionnalités

- assistant conversationnel avec plusieurs discussions et questions de suivi ;
- recherche hybride multilingue avec filtrage par branche d'assurance ;
- réponse structurée : décision, conditions, informations manquantes ;
- contrôle exact des citations, tolérant à la typographie mais rejetant les reformulations ;
- bascule vers une validation humaine quand les preuves sont insuffisantes ;
- consultation et recherche dans le portefeuille documentaire synthétique.

## Installation locale

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Créer `.streamlit/secrets.toml` (ignoré par Git) :

```toml
GROQ_API_KEY = "gsk_..."
# HTTPS_PROXY = "http://..."   # facultatif, réseau d'entreprise
```

Lancer l'application depuis la racine :

```bash
streamlit run app.py
```

Prérequis : Python 3.10+ (3.12 recommandé), clé API Groq, une connexion internet au premier lancement (téléchargement du modèle d'embeddings).

## Structure du projet

```text
app.py                          Interface Streamlit
evaluate.py                     Harnais d'évaluation hors ligne
data/synthetic_contracts.json   Corpus contractuel synthétique
data/eval_questions.json        Jeu de questions annotées
src/chunking.py                 Chargement et découpage des contrats
src/models.py                   Modèles Pydantic
src/retriever.py                Recherche hybride
src/rag.py                      Génération, contrôles et garde-fous
```

## Limites

- Pas de gestion des versions de contrats, dates d'effet, pièces de sinistre ni habilitations.
- La matrice d'embeddings en mémoire convient aux 46 clauses actuelles, pas à un corpus volumineux.
- Les conversations ne sont ni persistées ni partagées entre utilisateurs.
- Toute décision destinée à un client doit rester validée par un gestionnaire.
