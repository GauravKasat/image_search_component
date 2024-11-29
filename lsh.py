import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from typing import List, Tuple

class DeepHashNet(nn.Module):
    """
    Deep neural network for learning hash codes
    """
    def __init__(self, hash_bits: int = 64):
        super(DeepHashNet, self).__init__()
        # Use ResNet as feature extractor
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Hash layer
        self.hash_layer = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, hash_bits)
        )
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        hash_code = torch.tanh(self.hash_layer(features))
        return hash_code

class LocalitySensitiveHashing:
    def __init__(self, dim: int, num_hashes: int = 10, hash_bits: int = 64):
        """
        Initialize LSH with random projection
        
        Args:
            dim (int): Dimension of feature vectors
            num_hashes (int): Number of hash tables
            hash_bits (int): Number of bits in hash
        """
        self.num_hashes = num_hashes
        self.hash_bits = hash_bits
        
        # Random projection matrices
        self.random_matrices = [
            np.random.randn(dim, hash_bits) for _ in range(num_hashes)
        ]
    
    def hash(self, features: np.ndarray) -> List[str]:
        """
        Generate hash buckets for given features
        
        Args:
            features (np.ndarray): Feature vector
        
        Returns:
            List of hash bucket identifiers
        """
        hashes = []
        for matrix in self.random_matrices:
            # Project features and binarize
            projection = np.dot(features.reshape(-1, 1).T, matrix)
            binary_hash = (projection > 0).astype(int)
            hashes.append(''.join(map(str, binary_hash[0])))
        return hashes

class ImageSimilaritySearch:
    def __init__(self, database_path: str, hash_bits: int = 64,device: str = None):
        """
        Initialize image similarity search
        
        Args:
            database_path (str): Path to image database
            hash_bits (int): Number of bits for hashing
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Deep hashing network
        self.deep_hash_net = DeepHashNet(hash_bits).to(self.device)
        self.deep_hash_net.eval()
        
        # LSH index
        self.lsh_index = {}
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load and index database
        self._load_database(database_path)
    
    def _extract_features(self, img_path: str) -> torch.Tensor:
        """
        Extract deep features from image
        
        Args:
            img_path (str): Path to image
        
        Returns:
            Deep feature tensor
        """
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.deep_hash_net(img_tensor)
        
        return features
    
    def _load_database(self, database_path: str):
        """
        Index images using deep hashing and LSH
        
        Args:
            database_path (str): Path to image database
        """
        # LSH with multiple hash tables
        self.lsh = LocalitySensitiveHashing(
            dim=2048,  # ResNet feature dimension
            num_hashes=10,
            hash_bits=64
        )
        
        # Store hash codes and feature vectors
        self.feature_vectors = {}
        
        for filename in os.listdir(database_path):
            filepath = os.path.join(database_path, filename)
            
            # Skip non-image files
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
            
            # Extract features and hash codes
            features = self._extract_features(filepath)
            feature_np = features.cpu().numpy().flatten()
            
            # Generate LSH hashes
            lsh_hashes = self.lsh.hash(feature_np)
            
            # Index images in hash buckets
            for lsh_hash in lsh_hashes:
                if lsh_hash not in self.lsh_index:
                    self.lsh_index[lsh_hash] = []
                self.lsh_index[lsh_hash].append(filename)
            
            # Store feature vector
            self.feature_vectors[filename] = feature_np
    
    def search(
        self, 
        query_path: str, 
        top_k: int = 1, 
        similarity_threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Search for similar images
        
        Args:
            query_path (str): Path to query image
            top_k (int): Number of top results
            similarity_threshold (float): Similarity cutoff
        
        Returns:
            List of similar images with scores
        """
        # Extract query features
        query_features = self._extract_features(query_path)
        query_np = query_features.cpu().numpy().flatten()
        
        # Generate LSH hashes for query
        query_lsh_hashes = self.lsh.hash(query_np)
        
        # Find candidate images from LSH buckets
        candidates = set()
        for lsh_hash in query_lsh_hashes:
            if lsh_hash in self.lsh_index:
                candidates.update(self.lsh_index[lsh_hash])
        
        # Compute similarities for candidates
        similarities = []
        for candidate in candidates:
            candidate_features = self.feature_vectors[candidate]
            
            # Cosine similarity
            similarity = np.dot(query_np, candidate_features) / (
                np.linalg.norm(query_np) * np.linalg.norm(candidate_features)
            )
            
            if similarity >= similarity_threshold:
                similarities.append((candidate, similarity))
        
        # Sort and return top results
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

def main():
    # Example usage
    similarity_search = ImageSimilaritySearch('./image_database')
    similar_images = similarity_search.search('./query_image2.jpg')
    
    for img, score in similar_images:
        print(f"Similar Image: {img}, Similarity Score: {score}")

if __name__ == "__main__":
    main()