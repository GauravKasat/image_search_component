import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
from PIL import Image
import imagehash
import numpy as np
from typing import List, Tuple

class ImageSimilaritySearch:
    def __init__(self, database_path: str, device: str = None):
        """
        Initialize image similarity search with PyTorch models
        
        Args:
            database_path (str): Path to directory containing reference images
            device (str): Device to run models on (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.database_path = database_path
        
        # Feature extraction models
        self.resnet = models.resnet50(pretrained=True).to(self.device)
        self.efficientnet = models.efficientnet_b0(pretrained=True).to(self.device)
        
        # Remove classification layers
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.efficientnet = torch.nn.Sequential(*list(self.efficientnet.children())[:-1])
        
        # Set models to evaluation mode
        self.resnet.eval()
        self.efficientnet.eval()
        
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
            'resnet': {},
            'efficientnet': {},
            'phash': {}
        }
        
        self._load_database()
    
    def _load_image(self, img_path: str) -> torch.Tensor:
        """
        Load and preprocess image
        
        Args:
            img_path (str): Path to image
        
        Returns:
            Preprocessed image tensor
        """
        img = Image.open(img_path).convert('RGB')
        return self.transform(img).unsqueeze(0).to(self.device)
    
    def _extract_resnet_features(self, img: torch.Tensor) -> np.ndarray:
        """
        Extract features using ResNet model
        
        Args:
            img (torch.Tensor): Preprocessed image tensor
        
        Returns:
            Feature vector
        """
        with torch.no_grad():
            features = self.resnet(img)
        return features.cpu().numpy().flatten()
    
    def _extract_efficientnet_features(self, img: torch.Tensor) -> np.ndarray:
        """
        Extract features using EfficientNet model
        
        Args:
            img (torch.Tensor): Preprocessed image tensor
        
        Returns:
            Feature vector
        """
        with torch.no_grad():
            features = self.efficientnet(img)
        return features.cpu().numpy().flatten()
    
    def _compute_phash(self, img_path: str) -> str:
        """
        Compute perceptual hash for image
        
        Args:
            img_path (str): Path to image
        
        Returns:
            Perceptual hash
        """
        img = Image.open(img_path)
        return str(imagehash.phash(img))
    
    def _load_database(self):
        """
        Load and precompute features for database images
        """
        for filename in os.listdir(self.database_path):
            filepath = os.path.join(self.database_path, filename)
            
            # Skip non-image files
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
            
            # Load and preprocess image
            img = self._load_image(filepath)
            
            # Compute features
            self.feature_cache['resnet'][filename] = self._extract_resnet_features(img)
            self.feature_cache['efficientnet'][filename] = self._extract_efficientnet_features(img)
            self.feature_cache['phash'][filename] = self._compute_phash(filepath)
    
    def search(
        self, 
        query_path: str, 
        top_k: int = 1, 
        weights: dict = {'resnet': 0.5, 'efficientnet': 0.4, 'phash': 0.1}
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
        query_img = self._load_image(query_path)
        query_phash = self._compute_phash(query_path)
        
        # Extract query features
        query_resnet = self._extract_resnet_features(query_img)
        query_efficientnet = self._extract_efficientnet_features(query_img)
        
        # Compute similarities
        results = {}
        for filename, resnet_feat in self.feature_cache['resnet'].items():
            # ResNet features similarity
            resnet_sim = 1 - np.linalg.norm(query_resnet - resnet_feat)
            
            # EfficientNet features similarity
            efficientnet_feat = self.feature_cache['efficientnet'][filename]
            efficientnet_sim = 1 - np.linalg.norm(query_efficientnet - efficientnet_feat)
            
            # Perceptual hash similarity
            phash_sim = 1.0 - (bin(int(self.feature_cache['phash'][filename], 16) ^ int(query_phash, 16)).count('1') / 64.0)
            
            # Weighted hybrid score
            hybrid_score = (
                weights['resnet'] * resnet_sim + 
                weights['efficientnet'] * efficientnet_sim + 
                weights['phash'] * phash_sim
            )
            
            results[filename] = hybrid_score
        
        # Sort and return top results
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

def main():
    # Example usage
    print("hello in here\n")
    similarity_search = ImageSimilaritySearch('./image_database',device='mps')
    similar_images = similarity_search.search('./query_image2.jpg')
    
    for img, score in similar_images:
        print(f"Similar Image: {img}, Similarity Score: {score}")

if __name__ == "__main__":
    main()