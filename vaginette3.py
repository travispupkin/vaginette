import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import pygame
import numpy as np
from filterpy.kalman import KalmanFilter
import threading
import time
import sys
import time

# Variables globales et verrous pour la synchronisation
lock = threading.Lock()
en_mouvement = False
arret_demande = False

# Initialisation de pygame pour la gestion de la musique
def initialiser_musique():
    pygame.mixer.init()
    musique_chemin = 'STEREOLIGHT.wav'  # Remplacez par le chemin de votre fichier musical
    pygame.mixer.music.load(musique_chemin)
    pygame.mixer.music.play(-1)  # Jouer la musique en boucle
    pygame.mixer.music.pause()  # Commencer par mettre la musique en pause

# Fonction pour gérer la musique
def gerer_musique():
    global en_mouvement, arret_demande
    while not arret_demande:
        with lock:
            if en_mouvement:
                pygame.mixer.music.unpause()
            else:
                pygame.mixer.music.pause()
        time.sleep(0.1)

# Charger le modèle YOLOv8 pour le suivi d'objets
model = YOLO('yolov8n.pt')

# Paramètres de la caméra et URL RTSP
adresse_ip = '192.168.8.101'
identifiant = 'admin'
mot_de_passe = 'uPzn1J1@'
url = f'rtsp://{identifiant}:{mot_de_passe}@{adresse_ip}/Streaming/Channels/2'

# Initialiser le filtre de Kalman
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 0, 1, 0],   # Matrice d'état
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0],   # Matrice de mesure
                 [0, 1, 0, 0]])
kf.R *= 10                        # Bruit de mesure
kf.P *= 1000                      # Covariance de l'état initial
kf.Q[-1, -1] *= 0.01
kf.Q[2, 2] *= 0.01

# Fonction pour mettre à jour et prédire la position avec le filtre de Kalman
def update_kalman(x, y):
    kf.predict()
    kf.update(np.array([x, y]))
    return kf.x[:2]

# Fonction principale pour afficher le flux vidéo
def afficher_flux_video():
    global en_mouvement, arret_demande
    try:
        cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            print("Erreur: Impossible d'ouvrir le flux vidéo.")
            sys.exit(1)

        seuil_mouvement = 2
        position_precedente = None
        temps_precedent = time.time()

        while not arret_demande:
            temps_actuel = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Avertissement: Impossible de lire la trame du flux vidéo.")
                continue

            # Exécuter le suivi d'objets avec YOLOv8 sur la trame
            results = model.track(frame, verbose=False)

            # Détecter et suivre la personne
            personne_detectee = False
            position_actuelle = None
            for result in results:
                for box in result.boxes:
                    if box.cls == 0:  # ID de classe pour 'personne'
                        xywh = box.xywh.cpu().numpy()[0]
                        x_center, y_center, w, h = xywh[0], xywh[1], xywh[2], xywh[3]

                        # Convertir les coordonnées du centre en coordonnées du coin supérieur gauche
                        x = int(x_center - w / 2)
                        y = int(y_center - h / 2)

                        position_actuelle = (x_center, y_center)
                        position_actuelle = update_kalman(*position_actuelle)
                        personne_detectee = True
                        break
                if personne_detectee:
                    break

            # Mise à jour du statut de mouvement et calcul de la vitesse
            if position_actuelle is not None and position_precedente is not None:
                distance = np.linalg.norm(np.array(position_actuelle) - np.array(position_precedente))
                temps_ecoule = temps_actuel - temps_precedent
                if temps_ecoule > 0:
                    vitesse = distance / temps_ecoule
                    print(f"Vitesse: {vitesse:.2f} pixels/s")
                with lock:
                    en_mouvement = distance > seuil_mouvement
            position_precedente = position_actuelle
            temps_precedent = temps_actuel

            # Utilisation de l'Annotator pour l'annotation des images
            annotator = Annotator(frame, line_width=2, pil=True)
            if personne_detectee:
                annotator.box_label([x, y, x + int(w), y + int(h)], label="Personne", color=(255, 0, 0))

            annotated_frame = annotator.result()
            cv2.imshow('Suivi YOLOv8', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                with lock:
                    arret_demande = True
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Erreur inattendue: {e}")
        cap.release()
        cv2.destroyAllWindows()


# Appeler la fonction principale
if __name__ == "__main__":
    initialiser_musique()
    video_thread = threading.Thread(target=afficher_flux_video)
    musique_thread = threading.Thread(target=gerer_musique)

    video_thread.start()
    musique_thread.start()

    video_thread.join()
    musique_thread.join()