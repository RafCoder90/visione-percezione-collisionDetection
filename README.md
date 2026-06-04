# Sistema di Rilevazione di Collisioni e Posa (Objectron 3D)

Questo repository contiene il codice sorgente sviluppato per l'esame di **Visione e Percezione**. 
L'obiettivo è stimare la posa 3D e le collisioni (cuboidi e Bounding Box Orientati - OBB) di oggetti in uno spazio metrico a partire da un singolo flusso monoculare RGB.

#####    Architettura e Modelli di Deep Learning   #####

Il sistema è basato principalmente su:
1. **Segmentazione Semantica (2D):** Gestita tramite **YOLOv8-seg** (`yolov8n-seg.pt`), una rete neurale convoluzionale profonda (CNN) ottimizzata per l'inferenza in tempo reale, utilizzata per isolare le maschere binarie dei pixel dei target.
2. **Stima della Profondità Densa:** Affidata a **Depth Anything V2** (`depth-anything-v2-small`), un modello basato su un'architettura **Vision Transformer (ViT)** che sfrutta meccanismi di *Self-Attention* per generare mappe di profondità relative ad alta fedeltà spaziale in modalità *Zero-Shot*.

### Modulo Geometrico e Decisionale (Oltre il Deep Learning)
* **Ancoraggio Semantico e Retroproiezione:** I valori relativi della mappa di profondità vengono convertiti in metri assoluti. Sfruttando la matrice dei parametri intrinseci della fotocamera $K$, il centroide 2D dell'oggetto viene retroproiettato nello spazio metrico 3D ($X_c, Y_c, Z_c$).
* **Geometric Prior Adattivo:** Il sistema integra un dizionario di altezze e spessori medi delle classi. Se l'oggetto reale è fuori scala (es. sfere macroscopiche), un algoritmo adattivo confronta i limiti dei pixel proiettati estendendo dinamicamente il cuboide.
* **Tracking Temporale (Filtro di Kalman):** La coerenza tra i frame è garantita dall'**Algoritmo Ungherese** (Data Association 3D), mentre lo stato cinematico a 7 variabili è stabilizzato da un Filtro di Kalman lineare indipendente per assorbire lo sfarfallio della profondità.
* **Collision Detection (OBB-SAT):** I cuboidi vengono proiettati sul piano orizzontale $XZ$. Su questa mappa bidimensionale isometrica viene eseguito il **Teorema degli Assi Separatori (SAT)**; in caso di sovrapposizione geometrica, lo stato dell'oggetto muta istantaneamente in `COLLISION` colorandosi di rosso.



#####    Installazione   #####

Il progetto è stato sviluppato isolando le dipendenze tramite un ambiente virtuale di Python (`venv`).

### 1. Clonare il repository

git clone [https://github.com/RafCoder90/visione-percezione-collisionDetection.git](https://github.com/RafCoder90/visione-percezione-collisionDetection.git)
cd visione-percezione-collisionDetection



### 2. Creare e attivare l'ambiente virtuale venv

## Su Windows:
python -m venv venv
.\venv\Scripts\activate

#] Su Linux/macOS:
python3 -m venv venv
source venv/bin/activate



### 3. Installare le dipendenze richieste

pip install --upgrade pip
pip install -r requirements.txt


### 4. Avviare lo script
python src\main.py
