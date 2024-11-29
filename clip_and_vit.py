import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import open_clip
import timm
import numpy as np
import os
from typing import List, Tuple

class AdvancedImageSimilaritySearch:
    def __init__(self, database_path: str, device: str = None):
        """
        Initialize image similarity search with ViT and CLIP models
        
        Args:
            database_path (str): Path to directory containing reference images
            device (str): Device to run models on (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.database_path = database_path
        
        # Load CLIP model
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', 
            pretrained='laion2b_s34b_b79k'
        )
        self.clip_model = self.clip_model.to(self.device)
        
        # Load Vision Transformer
        self.vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.vit_model.to(self.device)
        self.vit_model.eval()
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Feature cache
        self.feature_cache = {
            'clip': {},
            'vit': {}
        }
        
        self._load_database()
    
    def _load_image(self, img_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess image for different models
        
        Args:
            img_path (str): Path to image
        
        Returns:
            Processed image tensors
        """
        img = Image.open(img_path).convert('RGB')
        
        # CLIP preprocessing
        clip_img = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        
        # ViT preprocessing
        vit_img = self.transform(img).unsqueeze(0).to(self.device)
        
        return clip_img, vit_img
    
    def _extract_clip_features(self, img: torch.Tensor) -> np.ndarray:
        """
        Extract features using CLIP model
        
        Args:
            img (torch.Tensor): Preprocessed image tensor
        
        Returns:
            Feature vector
        """
        with torch.no_grad():
            features = self.clip_model.encode_image(img)
            features = F.normalize(features, p=2, dim=-1)
        return features.cpu().numpy().flatten()
    
    def _extract_vit_features(self, img: torch.Tensor) -> np.ndarray:
        """
        Extract features using Vision Transformer
        
        Args:
            img (torch.Tensor): Preprocessed image tensor
        
        Returns:
            Feature vector
        """
        with torch.no_grad():
            features = self.vit_model(img)
            features = F.normalize(features, p=2, dim=-1)
        return features.cpu().numpy().flatten()
    
    def _load_database(self):
        """
        Load and precompute features for database images
        """
        for filename in os.listdir(self.database_path):
            filepath = os.path.join(self.database_path, filename)
            
            # Skip non-image files
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
            
            # Load and preprocess images
            clip_img, vit_img = self._load_image(filepath)
            
            # Compute features
            self.feature_cache['clip'][filename] = self._extract_clip_features(clip_img)
            self.feature_cache['vit'][filename] = self._extract_vit_features(vit_img)
    
    def search(
        self, 
        query_path: str, 
        top_k: int = 1, 
        weights: dict = {'clip': 0.6, 'vit': 0.4}
    ) -> List[Tuple[str, float]]:
        """
        Perform multi-approach similarity search
        
        Args:
            query_path (str): Path to query image
            top_k (int): Number of top similar images to return
            weights (dict): Approach weights for hybrid search
        
        Returns:
            List of similar images with scores
        """
        # Load query image
        clip_query, vit_query = self._load_image(query_path)
        
        # Extract query features
        query_clip = self._extract_clip_features(clip_query)
        query_vit = self._extract_vit_features(vit_query)
        
        # Compute similarities
        results = {}
        for filename, clip_feat in self.feature_cache['clip'].items():
            # CLIP features similarity
            clip_sim = np.dot(query_clip, clip_feat)
            
            # ViT features similarity
            vit_feat = self.feature_cache['vit'][filename]
            vit_sim = np.dot(query_vit, vit_feat)
            
            # Weighted hybrid score
            hybrid_score = (
                weights['clip'] * clip_sim + 
                weights['vit'] * vit_sim
            )
            
            results[filename] = hybrid_score
        
        # Sort and return top results
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

def main():
    # Example usage
    similarity_search = AdvancedImageSimilaritySearch('./image_database')
    similar_images = similarity_search.search('./query_image4.jpg')
    
    for img, score in similar_images:
        print(f"Similar Image: {img}, Similarity Score: {score}")

if __name__ == "__main__":
    main()