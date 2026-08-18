# Rapport d'évaluation RAG

- Questions annotées : **20**
- Profondeur de recherche : **top-5**

## Retrieval (sans appel modèle)

| Métrique | Valeur | Lecture |
| --- | --- | --- |
| Hit-rate@5 | 100% | au moins une clause attendue est remontée |
| Recall@5 | 100% | proportion des clauses attendues remontées |
| MRR | 1.00 | rang moyen de la première clause pertinente |

## Génération (avec contrôles déterministes)

| Métrique | Valeur | Lecture |
| --- | --- | --- |
| Exactitude du statut | 90% | décision conforme à l'attendu |
| Taux de citations | 100% | réponses appuyées par ≥ 1 extrait vérifié |
| Fidélité | 100% | part des affirmations retrouvées littéralement |
| Taux d'escalade | 10% | réponses renvoyées vers un gestionnaire |
