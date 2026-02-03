import torch
import open_clip
from PIL import Image

class FeatureExtractor:
    """A wrapper class for the CLIP model to extract image features."""

    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
        """
        Initializes the feature extractor.
        Args:
            model_name (str): The name of the CLIP model architecture.
            pretrained (str): The name of the pretrained dataset.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing FeatureExtractor on device: {self.device}")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=self.device
        )
        self.model.eval()  # Set model to evaluation mode
        print("CLIP model loaded successfully.")

    def extract_features(self, image):
        """
        Extracts a feature vector from a PIL Image.
        Args:
            image (PIL.Image.Image): The input image.
        Returns:
            torch.Tensor: The normalized feature vector.
        """
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image_tensor)
            # Normalize the features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features

    @staticmethod
    def cosine_similarity(features1, features2):
        """
        Computes the cosine similarity between two feature vectors.
        Args:
            features1 (torch.Tensor): The first feature vector.
            features2 (torch.Tensor): The second feature vector.
        Returns:
            float: The cosine similarity score.
        """
        # Ensure features are on the same device
        features2 = features2.to(features1.device)
        # Calculate cosine similarity
        similarity = (features1 @ features2.T).item()
        return similarity
