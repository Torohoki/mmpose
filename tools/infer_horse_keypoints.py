
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from mmpose.apis import MMPoseInferencer

v
from glob import glob

def detect_horse_keypoints(img_path, output_dir='output'):
    """Détecte les keypoints d'un cheval dans une image."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Traitement de l'image: {img_path}")
    
    # Initialiser MMPoseInferencer avec les modèles spécifiques pour les animaux
    inferencer = MMPoseInferencer(
        # Modèle de pose
        pose2d='configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py',
        pose2d_weights='checkpoints/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth',
        
        # Modèle de détection
        det_model='mmdet::rtmdet_tiny_8xb32-300e_coco.py',
        det_weights='checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
        
        # Filtrer pour détecter seulement les chevaux (ID=15 dans COCO)
        det_cat_ids=[15],  
        
        # Utiliser CPU ou GPU selon disponibilité
        device='cuda:0' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    )
    
    # Inférence
    result_generator = inferencer(img_path, show=False)
    results = next(result_generator)
    
    # Sauvegarder et afficher les résultats
    if 'visualization' in results:
        vis_img = results['visualization']
        img_filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"pose_{img_filename}")
        
        # Sauvegarder l'image avec annotations
        cv2.imwrite(output_path, vis_img)
        print(f"Image annotée sauvegardée dans: {output_path}")
        
        # Afficher les informations des détections
        if 'predictions' in results:
            for i, pred in enumerate(results['predictions']):
                if 'instances' in pred:
                    for j, inst in enumerate(pred['instances']):
                        if 'keypoints' in inst and 'keypoint_scores' in inst:
                            scores = inst['keypoint_scores']
                            print(f"Détection {j+1}: {sum(scores > 0.3)}/{len(scores)} keypoints fiables")
        
        # Afficher l'image
        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Détection de keypoints de cheval - {img_filename}")
        plt.axis('off')
        plt.tight_layout()
        plt_output = os.path.join(output_dir, f"viz_{os.path.splitext(img_filename)[0]}.png")
        plt.savefig(plt_output)
        plt.show()
        
        return True
    else:
        print("Aucune visualisation générée pour cette image.")
        return False

def main():
    # Chemin vers le dossier d'images
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
        return
    
    print(f"Trouvé {len(img_files)} image(s)")
    
    # Traiter chaque image
    for img_path in img_files:
        detect_horse_keypoints(img_path, output_dir)
        print("-" * 50)
    
    print("Traitement terminé!")

if __name__ == '__main__':
    main()
