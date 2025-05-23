
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from glob import glob
from mmengine.registry import init_default_scope
from mmpose.registry import VISUALIZERS
from mmpose.apis import init_model, inference_topdown
from mmdet.apis import init_detector, inference_detector

def init_models():
    """Initialiser les modèles de détection et d'estimation de pose."""
    # Modèle de détection
    det_config = 'configs/rtmdet/rtmdet-tiny_8xb32-300e_coco.py'
    det_checkpoint = 'checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
    det_model = init_detector(det_config, det_checkpoint, device='cpu')
    
    # Modèle d'estimation de pose
    pose_config = 'configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py'
    pose_checkpoint = 'checkpoints/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth'
    pose_model = init_model(pose_config, pose_checkpoint, device='cpu')
    
    return det_model, pose_model

def detect_and_draw_keypoints(img_path, det_model, pose_model, output_dir='output'):
    """Détecter les chevaux et leurs keypoints."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Traitement de l'image: {img_path}")
    
    # Charger l'image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Impossible de lire l'image: {img_path}")
        return None
    
    # Détecter les objets (chevaux)
    det_results = inference_detector(det_model, img)
    
    # Filtrer les détections de chevaux (classe 15 dans COCO)
    horse_class = 15  # ID de classe pour le cheval dans COCO
    pred_instances = det_results.pred_instances
    
    valid_indices = []
    for i in range(len(pred_instances.labels)):
        if pred_instances.labels[i] == horse_class and pred_instances.scores[i] > 0.5:
            valid_indices.append(i)
    
    if not valid_indices:
        print("Aucun cheval détecté dans l'image.")
        return None
    
    # Préparer les boîtes pour l'estimation de pose
    bboxes = []
    for i in valid_indices:
        x1, y1, x2, y2 = pred_instances.bboxes[i].tolist()
        score = float(pred_instances.scores[i])
        bboxes.append([x1, y1, x2, y2, score])
    
    # Estimer la pose
    pose_results = inference_topdown(pose_model, img, bboxes)
    
    # Initialiser le scope par défaut pour le visualiseur
    init_default_scope('mmpose')
    
    # Créer un visualiseur
    visualizer = VISUALIZERS.build(dict(type='PoseLocalVisualizer'))
    visualizer.set_dataset_meta(pose_model.dataset_meta)
    
    # Dessiner sur une copie de l'image
    img_vis = img.copy()
    
    # Afficher les résultats
    visualizer.add_datasample(
        'result',
        img_vis,
        data_sample=pose_results[0],
        draw_bbox=True,
        draw_heatmap=False,
        show=False)
    
    img_vis = visualizer.get_image()
    
    # Sauvegarder l'image
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, img_vis)
    print(f"Image sauvegardée dans {output_path}")
    
    # Afficher l'image
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.title("Détection de keypoints de cheval")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"viz_{os.path.splitext(os.path.basename(img_path))[0]}.png"), dpi=300)
    plt.show()
    
    return img_vis

def main():
    """Fonction principale."""
    # Initialiser les modèles
    print("Initialisation des modèles...")
    det_model, pose_model = init_models()
    
    # Définir le dossier d'entrée et de sortie
    img_dir = 'data/custom_images'
    output_dir = 'output'
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Vérifier si le dossier d'images existe
    if not os.path.exists(img_dir):
        print(f"Erreur: le dossier {img_dir} n'existe pas.")
        print("Veuillez créer ce dossier et y placer vos images.")
        return
    
    # Rechercher toutes les images
    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        img_files.extend(glob(os.path.join(img_dir, ext)))
    
    if not img_files:
        print(f"Aucune image trouvée dans {img_dir}")
        # Télécharger une image de test
        print("Téléchargement d'une image de test...")
        import urllib.request
        os.makedirs(img_dir, exist_ok=True)
        test_image_url = "https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/animalpose/ca110.jpeg"
        test_image_path = os.path.join(img_dir, "test_horse.jpeg")
        urllib.request.urlretrieve(test_image_url, test_image_path)
        print(f"Image téléchargée: {test_image_path}")
        img_files = [test_image_path]
    
    print(f"Trouvé {len(img_files)} image(s)")
    
    # Traiter chaque image
    for img_path in img_files:
        detect_and_draw_keypoints(img_path, det_model, pose_model, output_dir)
        print("-" * 50)
    
    print("Traitement terminé!")

if __name__ == '__main__':
    main()
