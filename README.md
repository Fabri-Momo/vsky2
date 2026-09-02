# vSky

## Installation

vSky fonctionne sur Windows, Linux et macOS avec accélération GPU cross-platform via [Taichi](https://taichi.graphics/) (NVIDIA CUDA, AMD/Intel Vulkan, Apple Silicon Metal, ou CPU).

### Windows / Linux / macOS

```bash
conda env create -f environment.yml
conda activate vsky
```

L'environnement `environment.yml` inclut Taichi. Pour macOS, vous pouvez aussi utiliser `environment_mac.yml` qui est identique.

Si Taichi n'est pas disponible, vSky utilise automatiquement NumPy en mode CPU.

Puis lancez le programme :

```bash
python vSky2.py
```

## Build d'une application autonome

### Windows

```bash
conda activate vsky
python setup.py
```

L'exécutable se trouve dans `dist/vSky/vSky.exe`.

### macOS

```bash
conda activate vsky
python -m PyInstaller --noconfirm --clean vSky_mac.spec
```

Le bundle `.app` se trouve dans `dist/vSky/vSky.app`.

## Documentation

La documentation est contenue dans le dossier resources/doc.
La mise à disposition des fichiers la composant est faite par le moyen du fichier de ressource resources.qrc.
Pour chaque fichier, il est nécessaire d’ajouter une ligne dans ce fichier de ressources.

Après chaque modification de ce fichier de ressources, il est nécessaire d’exécuter la commande `pyrcc5 -o qrc_resources.py resources.qrc` de manière à mettre à jour le fichier qrc_resources.py qui est chargé par le programme.

La documentation est construite sous la forme de fichiers HTML.
La page d’accueil est le fichier index.html, qui doit donc intégrer des liens vers les autres pages.
Les cibles des liens sont les alias déclarés dans le fichier de ressources.

## Traduction

La traduction de l’application est effectuée via les mécanismes de PyQt5 : création d’un fichier qm ensuite déclaré dans le fichier de ressources et donc chargé par le programme au démarrage.

Pour qu’une chaîne de caractères puisse être traduite, il faut qu’elle soit passée en argument de la méthode `self.tr`.La création du fichier qm se fait en compilant un fichier intermédiaire éditable : un fichier ts.

La création du fichier ts est effectué grâce à la commande `pylupdate5 vSky.pro` (le fichier vSky.pro déclarant le fichier de traduction ts).
Le fichier ts peut être édité dans n’importe quel éditeur de texte.
Une fois le fichier ts complété, sa compilation est effectuée via la commande `lrelease-qt5 vSky_fr_FR.ts`.
Il est alors nécessaire de lancer à nouveau `pyrcc5 -o qrc_resources.py resources.qrc` pour intégrer les modifications du fichier qm dans le fichier de ressources.

## Licence

Le choix d’une licence est contraint par l’utilisation de divers briques.

- PyQt5 : https://github.com/PyQt5/PyQt/blob/master/LICENSE (GPL v3.0)
- pyside2 (éventuellement en remplacement de PyQt5) : https://doc.qt.io/qtforpython/licenses.html (LGPL ?)
- cupy, numpy, scipy
- python 3
(https://www.gnu.org/licenses/gpl-faq.html#AllCompatibility)
