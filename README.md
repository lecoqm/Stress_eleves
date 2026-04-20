# Stress des élèves dans le supérieur

## Description du projet

A partir d'un jeu de données anonymisées sur le stress des élèves dans le supérieur en France et les facteurs physiques, psychologiques et de contexte scolaire et économique associés, nous mettons en oeuvre plusieurs modèles de classification et prédiction du niveau de stress (ou éventuellement d'une autre variable d'intérêt choisie par l'utilisateur au sein du jeu de données). Nous comparons ensuite les performances de chacun de ces modèles. Ces derniers sont : 
- Régression logistique simple multiclasse ;
- Régression logistique simple OVR ;
- Régression Lasso avec validation croisée ;
- Random Forest avec *fine tuning* ;
- Boosting avec *fine tuning*.

Les résultats sont accessibles à [ce lien](https://lecoqm.github.io/Stress_eleves/)

## Utilisation du dépôt

### Prérequis

Version de Python utilisée pour ce projet : `Python 3.13`.

### Installation
1. **Cloner le dépôt**

```bash
git clone https://github.com/lecoqm/Stress_eleves.git
cd Stress_eleves
```
2. **Installation des dépendances**

```bash 
pip install -r requirements.txt
```

ou :

```bash
uv sync
```

3. **Exécution du code**

```bash
python main.py
```

ou 

```bash
uv run main.py
```

Si l'utilisateur souhaite prendre une variable d'intérêt différente de `niveau_stress`:

```bash
python main.py --target_col mavariable
```

ou 

```bash
uv run main.py --target_col mavariable
```
Si l'utilisateur souhaite appliquer le code à un nouveau jeu de données, par exemple une nouvelle version de l'enquête, chargeable à partir de l'url `url_source` : 

```bash
python main.py --url url_source
```

ou

```bash
uv run main.py --url url_source
```

### Structure du dépôt
- **.github/*** : fichiers de configuration du dépôt.
- **notebooks/** : contient le notebook initial duquel est issu le code du projet.
- **src/** : code source du projet comprenant les scripts de configuration des hyperparamètres, de pré-traitement des données, de définition des modèles, de traitement des résultats.
- **main.py** : fichier principal d'exécution des modèles.
- **scripts/ingest.py** : gère l'ingestion de nouvelles données sur lesquelles exécuter les modèles.
- **index.qmd**, **_quarto.yml** et **styles.css**, **_freeze/** : fichiers de configuration du site hébergeant le rapport.
- **models** : fichiers quarto décrivant les résultats pour chacun des modèles mis en oeuvre.
- **reports/** : fichier stockant les figures et les tableaux générés par l'exécution du code.
- **slides.qmd** : fichier quarto des slides du site internet. 
- **requirements.txt** et **pyproject.toml**: dépendances Python à installer.
- **uv.lock** : garantissent que les versions des packages installés correspondent à celles utilisées par l'auteur du dépôt.
- **Dockerfile** : fichier servant à la construction de l'image Docker du dépôt.
- **README.md** : description du dépôt et guide d'utilisation.

### Dépôt d'origine
https://github.com/averniere/Stress_eleves 