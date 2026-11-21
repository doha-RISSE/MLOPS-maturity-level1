# Pipeline MLOps — Telco Churn

Ce dépôt industrialise la prédiction du désabonnement client : validation des lots, détection de dérive, entraînement scikit-learn, registry MLflow, API FastAPI et monitoring Prometheus/Grafana.

## Démarrage

1. Copier `.env.example` vers `.env` et modifier les valeurs de démonstration.
2. Placer le fichier fourni `WA_Fn-UseC_-Telco-Customer-Churn.csv` dans `data/raw/`.
3. Générer les lots : `python scripts/make_batches.py --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --output data/batches --seed 42`.
4. Démarrer la plateforme : `docker compose up -d --build`.
5. Lancer un entraînement : `docker compose --profile pipeline run --rm pipeline --input /data/batches/batch_initial.csv`.

MLflow est exposé sur `http://localhost:5000`, l’API sur `http://localhost:8000`, Prometheus sur `http://localhost:9090` et Grafana sur `http://localhost:3000`.

Exemple de prédiction :

```powershell
Invoke-RestMethod http://localhost:8000/predict -Method Post -ContentType 'application/json' -Body '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":50.0,"TotalCharges":600.0}'
```

Les tests s’exécutent avec `pytest -q`. Les dépendances de développement sont dans `requirements-dev.txt`.

## Décisions

- MLflow 2.14.3 et les bibliothèques associées sont épinglés pour une API registry stable ; l’image MLflow est construite localement.
- Le champion est comparé sur le jeu de test du nouveau lot ; l’égalité ne le remplace jamais.
- Les artefacts intermédiaires sont créés dans un répertoire temporaire, puis envoyés dans MLflow, afin de ne pas persister de données non nécessaires dans le dépôt.
- L’API continue de démarrer sans champion ; `/health` reste donc un endpoint de liveness, tandis que `/predict` répond 503 jusqu’au chargement.
- L’accès administrateur est volontairement limité au token statique demandé et les endpoints de panne sont désactivés par défaut.
