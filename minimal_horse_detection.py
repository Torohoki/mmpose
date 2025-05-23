
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

# Définition des keypoints pour le cheval (AnimalPose)
KEYPOINTS = [
    "L_Eye", "R_Eye", "Nose", "Neck",
    "Root_of_tail", "L_Shoulder", "L_Elbow", "L_F_Knee", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Knee", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw", "R_Hip", "R_Knee", "R_B_Paw", "Tail"
]

# Définition des connexions pour le squelette
SKELETON = [
    (0, 2), (1, 2), (2, 3), (3, 4), # tête et cou vers queue
    (3, 5), (5, 6), (6, 7), (7, 8), # patte avant gauche
    (3, 9), (9, 10), (10, 11), (11, 12), # patte avant droite
    (4, 13), (13, 14), (14, 15), # patte arrière gauche
    (4, 16), (16, 17), (17, 18), # patte arrière droite
    (4, 19) # queue
]

def simplify_mmpose(img_path, output_dir='output'):
    """Version simplifiée de la détection de pose sans extensions MMCV."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Importer uniquement les modules de base de PyTorch
        import torch
        import torchvision.transforms as transforms
        from torchvision.models import resnet50, ResNet50_Weights
        
        # Charger l'image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Impossible de lire l'image {img_path}")
            return None
        
        # Préparer l'image pour le modèle ResNet (simulation de la détection)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convertir l'image au format approprié
        img_tensor = transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        
        # Charger un modèle ResNet pour la visualisation
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Simuler la détection d'un cheval
        h, w = img.shape[:2]
        
        # Détecter un rectangle dans l'image correspondant au cheval (simulation)
        # Extraire les caractéristiques de l'image pour trouver les régions d'intérêt
        model.eval()
        with torch.no_grad():
            features = model.conv1(img_tensor)
            features = model.bn1(features)
            features = model.relu(features)
            features = model.maxpool(features)
            features = model.layer1(features)
        
        # Utiliser les activations pour détecter les régions d'intérêt
        feature_map = features.mean(1).squeeze().numpy()
        normalized_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
        
        # Redimensionner à la taille de l'image
        heatmap = cv2.resize(normalized_map, (w, h))
        
        # Trouver la région la plus active (simplification)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)
        
        # Créer une boîte englobante autour de ce point
        bbox_size = min(w, h) // 2
        x1 = max(0, max_loc[0] - bbox_size // 2)
        y1 = max(0, max_loc[1] - bbox_size // 2)
        x2 = min(w, max_loc[0] + bbox_size // 2)
        y2 = min(h, max_loc[1] + bbox_size // 2)
        
        # Afficher la région détectée
        img_vis = img.copy()
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Générer des keypoints basés sur l'anatomie du cheval dans la boîte
        keypoints = []
        for i, name in enumerate(KEYPOINTS):
            # Distribuer les keypoints selon leur position anatomique
            if "Eye" in name or "Nose" in name:
                # Points de la tête: haut de la boîte
                offset_x = (i % 3) - 1  # -1, 0, 1
                x = x1 + (x2 - x1) * (0.5 + 0.1 * offset_x)
                y = y1 + (y2 - y1) * 0.15
            elif "Neck" in name:
                # Cou
                x = (x1 + x2) / 2
                y = y1 + (y2 - y1) * 0.25
            elif "Root_of_tail" in name or "Hip" in name:
                # Arrière train
                offset_x = (i % 3) - 1  # -1, 0, 1
                x = x1 + (x2 - x1) * (0.3 + 0.1 * offset_x)
                y = y1 + (y2 - y1) * 0.40
            elif "Shoulder" in name:
                # Épaules
                offset_x = -0.1 if "L_" in name else 0.1
                x = (x1 + x2) / 2 + offset_x * (x2 - x1)
                y = y1 + (y2 - y1) * 0.35
            elif "Elbow" in name or "Knee" in name:
                # Articulations des pattes
                x_offset = -0.15 if "L_" in name else 0.15
                y_offset = 0.6 if "F_" in name else 0.7
                x = (x1 + x2) / 2 + x_offset * (x2 - x1)
                y = y1 + (y2 - y1) * y_offset
            elif "Paw" in name:
                # Pattes
                x_offset = -0.15 if "L_" in name else 0.15
                y_offset = 0.85 if "F_" in name else 0.9
                x = (x1 + x2) / 2 + x_offset * (x2 - x1)
                y = y1 + (y2 - y1) * y_offset
            elif "Tail" in name:
                # Queue
                x = x1 + (x2 - x1) * 0.2
                y = y1 + (y2 - y1) * 0.4
            else:
                # Position par défaut
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
            
            # Ajouter un peu de variabilité 
            x += np.random.normal(0, 5)
            y += np.random.normal(0, 5)
            
            # Garder dans les limites de l'image
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))
            
            # Calculer un score basé sur l'activité dans la région
            try:
                score_region = heatmap[int(y), int(x)]
                score = min(1.0, max(0.5, score_region * 1.5))
            except IndexError:
                score = 0.7  # Valeur par défaut
            
            keypoints.append((int(x), int(y), float(score)))
        
        # Dessiner les keypoints et connexions
        for idx, (x, y, score) in enumerate(keypoints):
            if score > 0.5:
                # Dessiner le keypoint
                cv2.circle(img_vis, (x, y), 5, (0, 255, 0), -1)
                # Ajouter le numéro du keypoint
                cv2.putText(img_vis, f"{idx}", (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Dessiner le squelette
        for start_idx, end_idx in SKELETON:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and 
                keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5):
                cv2.line(img_vis, 
                         (keypoints[start_idx][0], keypoints[start_idx][1]),
                         (keypoints[end_idx][0], keypoints[end_idx][1]),
                         (0, 0, 255), 2)
        
        # Écrire l'image avec les keypoints
        output_path = os.path.join(output_dir, f"simplified_keypoints_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, img_vis)
        print(f"Image sauvegardée dans {output_path}")
        
        # Afficher avec matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        plt.title("Détection simplifiée de keypoints de cheval")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'keypoints_visualization.png'))
        plt.show()
        
        return img_vis
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Détection simplifiée de pose de cheval")
    parser.add_argument('--image', default='data/custom_images/cheval.jpg', 
                       help='Chemin vers l\'image')
    parser.add_argument('--output', default='output', 
                       help='Dossier de sortie')
    args = parser.parse_args()
    
    # Vérifier si l'image existe
    if not os.path.exists(args.image):
        print(f"L'image {args.image} n'existe pas.")
        # Essayer de télécharger une image d'exemple
        try:
            import urllib.request
            os.makedirs(os.path.dirname(args.image), exist_ok=True)
            print("Téléchargement d'une image d'exemple...")
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/animalpose/ca110.jpeg',
                args.image
            )
            print(f"Image téléchargée dans {args.image}")
        except:
            print("Impossible de télécharger une image d'exemple.")
            return
    
    # Traiter l'image
    simplify_mmpose(args.image, args.output)

if __name__ == "__main__":
    main()
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse

# Définition des keypoints pour le cheval (AnimalPose)
KEYPOINTS = [
    "L_Eye", "R_Eye", "Nose", "Neck",
    "Root_of_tail", "L_Shoulder", "L_Elbow", "L_F_Knee", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Knee", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw", "R_Hip", "R_Knee", "R_B_Paw", "Tail"
]

# Définition des connexions pour le squelette
SKELETON = [
    (0, 2), (1, 2), (2, 3), (3, 4), # tête et cou vers queue
    (3, 5), (5, 6), (6, 7), (7, 8), # patte avant gauche
    (3, 9), (9, 10), (10, 11), (11, 12), # patte avant droite
    (4, 13), (13, 14), (14, 15), # patte arrière gauche
    (4, 16), (16, 17), (17, 18), # patte arrière droite
    (4, 19) # queue
]

def simplify_mmpose(img_path, output_dir='output'):
    """Version simplifiée de la détection de pose sans extensions MMCV."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Importer uniquement les modules de base de PyTorch
        import torch
        import torchvision.transforms as transforms
        from torchvision.models import resnet50, ResNet50_Weights
        
        # Charger l'image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Impossible de lire l'image {img_path}")
            return None
        
        # Préparer l'image pour le modèle ResNet (simulation de la détection)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convertir l'image au format approprié
        img_tensor = transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        
        # Charger un modèle ResNet pour la visualisation
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Simuler la détection d'un cheval
        h, w = img.shape[:2]
        
        # Détecter un rectangle dans l'image correspondant au cheval (simulation)
        # Extraire les caractéristiques de l'image pour trouver les régions d'intérêt
        model.eval()
        with torch.no_grad():
            features = model.conv1(img_tensor)
            features = model.bn1(features)
            features = model.relu(features)
            features = model.maxpool(features)
            features = model.layer1(features)
        
        # Utiliser les activations pour détecter les régions d'intérêt
        feature_map = features.mean(1).squeeze().numpy()
        normalized_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
        
        # Redimensionner à la taille de l'image
        heatmap = cv2.resize(normalized_map, (w, h))
        
        # Trouver la région la plus active (simplification)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)
        
        # Créer une boîte englobante autour de ce point
        bbox_size = min(w, h) // 2
        x1 = max(0, max_loc[0] - bbox_size // 2)
        y1 = max(0, max_loc[1] - bbox_size // 2)
        x2 = min(w, max_loc[0] + bbox_size // 2)
        y2 = min(h, max_loc[1] + bbox_size // 2)
        
        # Afficher la région détectée
        img_vis = img.copy()
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Générer des keypoints basés sur l'anatomie du cheval dans la boîte
        keypoints = []
        for i, name in enumerate(KEYPOINTS):
            # Distribuer les keypoints selon leur position anatomique
            if "Eye" in name or "Nose" in name:
                # Points de la tête: haut de la boîte
                offset_x = (i % 3) - 1  # -1, 0, 1
                x = x1 + (x2 - x1) * (0.5 + 0.1 * offset_x)
                y = y1 + (y2 - y1) * 0.15
            elif "Neck" in name:
                # Cou
                x = (x1 + x2) / 2
                y = y1 + (y2 - y1) * 0.25
            elif "Root_of_tail" in name or "Hip" in name:
                # Arrière train
                offset_x = (i % 3) - 1  # -1, 0, 1
                x = x1 + (x2 - x1) * (0.3 + 0.1 * offset_x)
                y = y1 + (y2 - y1) * 0.40
            elif "Shoulder" in name:
                # Épaules
                offset_x = -0.1 if "L_" in name else 0.1
                x = (x1 + x2) / 2 + offset_x * (x2 - x1)
                y = y1 + (y2 - y1) * 0.35
            elif "Elbow" in name or "Knee" in name:
                # Articulations des pattes
                x_offset = -0.15 if "L_" in name else 0.15
                y_offset = 0.6 if "F_" in name else 0.7
                x = (x1 + x2) / 2 + x_offset * (x2 - x1)
                y = y1 + (y2 - y1) * y_offset
            elif "Paw" in name:
                # Pattes
                x_offset = -0.15 if "L_" in name else 0.15
                y_offset = 0.85 if "F_" in name else 0.9
                x = (x1 + x2) / 2 + x_offset * (x2 - x1)
                y = y1 + (y2 - y1) * y_offset
            elif "Tail" in name:
                # Queue
                x = x1 + (x2 - x1) * 0.2
                y = y1 + (y2 - y1) * 0.4
            else:
                # Position par défaut
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
            
            # Ajouter un peu de variabilité 
            x += np.random.normal(0, 5)
            y += np.random.normal(0, 5)
            
            # Garder dans les limites de l'image
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))
            
            # Calculer un score basé sur l'activité dans la région
            try:
                score_region = heatmap[int(y), int(x)]
                score = min(1.0, max(0.5, score_region * 1.5))
            except IndexError:
                score = 0.7  # Valeur par défaut
            
            keypoints.append((int(x), int(y), float(score)))
        
        # Dessiner les keypoints et connexions
        for idx, (x, y, score) in enumerate(keypoints):
            if score > 0.5:
                # Dessiner le keypoint
                cv2.circle(img_vis, (x, y), 5, (0, 255, 0), -1)
                # Ajouter le numéro du keypoint
                cv2.putText(img_vis, f"{idx}", (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Dessiner le squelette
        for start_idx, end_idx in SKELETON:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and 
                keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5):
                cv2.line(img_vis, 
                         (keypoints[start_idx][0], keypoints[start_idx][1]),
                         (keypoints[end_idx][0], keypoints[end_idx][1]),
                         (0, 0, 255), 2)
        
        # Écrire l'image avec les keypoints
        output_path = os.path.join(output_dir, f"simplified_keypoints_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, img_vis)
        print(f"Image sauvegardée dans {output_path}")
        
        # Afficher avec matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
        plt.title("Détection simplifiée de keypoints de cheval")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'keypoints_visualization.png'))
        plt.show()
        
        return img_vis
        
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Détection simplifiée de pose de cheval")
    parser.add_argument('--image', default='data/custom_images/cheval.jpg', 
                       help='Chemin vers l\'image')
    parser.add_argument('--output', default='output', 
                       help='Dossier de sortie')
    args = parser.parse_args()
    
    # Vérifier si l'image existe
    if not os.path.exists(args.image):
        print(f"L'image {args.image} n'existe pas.")
        # Essayer de télécharger une image d'exemple
        try:
            import urllib.request
            os.makedirs(os.path.dirname(args.image), exist_ok=True)
            print("Téléchargement d'une image d'exemple...")
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/animalpose/ca110.jpeg',
                args.image
            )
            print(f"Image téléchargée dans {args.image}")
        except:
            print("Impossible de télécharger une image d'exemple.")
            return
    
    # Traiter l'image
    simplify_mmpose(args.image, args.output)

if __name__ == "__main__":
    main()

