import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import faiss
import cv2

class ImageSimilaritySearch(nn.Module):
    def __init__(self, embedding_dim=256,device: str = None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = embedding_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Latent space layers
        self.fc_encoder = nn.Linear(128 * 28 * 28, embedding_dim)
        
        # Decoder
        self.fc_decoder = nn.Linear(embedding_dim, 128 * 28 * 28)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 28, 28)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        # Pre-trained feature extractor
        self.feature_extractor = models.mobilenet_v2(pretrained=True).features
        
        # FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.image_paths = []
        
        self.to(self.device)

    def forward(self, x):
        x = self.encoder(x)
        latent = self.fc_encoder(x)
        reconstructed = self.fc_decoder(latent)
        decoded = self.decoder(reconstructed)
        return decoded, latent

    def preprocess_image(self, image_path):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return transform(img).unsqueeze(0).to(self.device)

    def extract_embeddings(self, image_path):
        with torch.no_grad():
            img = self.preprocess_image(image_path)
            
            # Autoencoder embedding
            _, autoencoder_embedding = self(img)
            
            # Feature extraction
            feature_embedding = self.feature_extractor(img)
            feature_embedding = torch.nn.functional.adaptive_avg_pool2d(feature_embedding, 1).squeeze()
            
            # Combined embedding
            print(autoencoder_embedding.shape)
            print(feature_embedding.shape)
            feature_embedding = feature_embedding.unsqueeze(0)

            print(autoencoder_embedding.shape)
            print(feature_embedding.shape)
            combined_embedding = torch.cat([autoencoder_embedding, feature_embedding]).cpu().numpy()
            
            return combined_embedding

    def add_image(self, image_path):
        embedding = self.extract_embeddings(image_path)
        self.index.add(embedding.reshape(1, -1))
        self.image_paths.append(image_path)

    def search_similar_images(self, query_image_path, top_k=1):
        query_embedding = self.extract_embeddings(query_image_path)
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), top_k
        )
        
        similar_images = [
            (self.image_paths[idx], dist) 
            for idx, dist in zip(indices[0], distances[0])
        ]
        
        return similar_images

def train_autoencoder(model, dataloader, epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

def main():
    # Example usage
    print("hello in here\n")
    search_engine = ImageSimilaritySearch(device='mps')
    
    # Add images to index
    image_directory = './image_database'
    for filename in os.listdir(image_directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            search_engine.add_image(os.path.join(image_directory, filename))
    
    # Search similar images
    query_image = './query_image2.jpg'
    similar_images = search_engine.search_similar_images(query_image)
    
    for image, distance in similar_images:
        print(f"Similar Image: {image}, Distance: {distance}")

if __name__ == "__main__":
    main()