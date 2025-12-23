"""
Vision-Language Models for Scene Understanding
Implements multimodal models from recent CV/NLP conferences:
- CLIP (ICML 2021)
- BLIP-2 (ICML 2023)
- LLaVA for visual question answering
"""
import torch
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class CLIPModel:
    """
    CLIP for zero-shot image classification and retrieval
    """
    
    def __init__(self, model_name: str = "ViT-L/14", device: str = "auto"):
        """
        Initialize CLIP
        
        Args:
            model_name: CLIP model variant
            device: Device to run on
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.load_model()
    
    def load_model(self):
        """Load CLIP model"""
        try:
            import clip
            
            logger.info(f"Loading CLIP {self.model_name}")
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            logger.info("CLIP loaded successfully")
            
        except ImportError:
            logger.warning("CLIP not installed. Using alternative...")
            try:
                from transformers import CLIPProcessor, CLIPModel
                
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
                self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
                self.model.to(self.device)
                self.model.eval()
                logger.info("CLIP loaded via transformers")
                
            except Exception as e:
                logger.error(f"Failed to load CLIP: {e}")
    
    def classify_image(self, frame: np.ndarray, categories: List[str]) -> Dict[str, float]:
        """
        Zero-shot image classification
        
        Args:
            frame: Input image (BGR)
            categories: List of category names
            
        Returns:
            Dictionary of category probabilities
        """
        if self.model is None:
            return {}
        
        try:
            # Convert to PIL Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            
            # Use transformers CLIP
            if hasattr(self, 'processor'):
                inputs = self.processor(
                    text=categories,
                    images=pil_image,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)[0]
                
                results = {cat: float(prob) for cat, prob in zip(categories, probs)}
                return results
            
            # Use OpenAI CLIP
            else:
                import clip
                
                image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                text = clip.tokenize(categories).to(self.device)
                
                with torch.no_grad():
                    image_features = self.model.encode_image(image)
                    text_features = self.model.encode_text(text)
                    
                    # Calculate similarity
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                results = {cat: float(sim) for cat, sim in zip(categories, similarity[0])}
                return results
                
        except Exception as e:
            logger.error(f"CLIP classification error: {e}")
            return {}
    
    def get_image_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract image embedding
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            Image embedding vector
        """
        if self.model is None:
            return None
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            
            if hasattr(self, 'processor'):
                inputs = self.processor(images=pil_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                    embedding = outputs[0].cpu().numpy()
                
                return embedding
            
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return None


class BLIP2:
    """
    BLIP-2 for image captioning and visual question answering
    """
    
    def __init__(self, model_type: str = "blip2-opt-2.7b", device: str = "auto"):
        """
        Initialize BLIP-2
        
        Args:
            model_type: Model variant
            device: Device to run on
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model_type = model_type
        self.model = None
        self.processor = None
        self.load_model()
    
    def load_model(self):
        """Load BLIP-2 model"""
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            logger.info(f"Loading BLIP-2 {self.model_type}")
            self.processor = Blip2Processor.from_pretrained(f"Salesforce/{self.model_type}")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                f"Salesforce/{self.model_type}",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("BLIP-2 loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load BLIP-2: {e}")
    
    def generate_caption(self, frame: np.ndarray) -> str:
        """
        Generate image caption
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            Caption text
        """
        if self.model is None:
            return "Caption generation not available"
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=50)
                caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return caption.strip()
            
        except Exception as e:
            logger.error(f"Caption generation error: {e}")
            return "Error generating caption"
    
    def visual_question_answering(self, frame: np.ndarray, question: str) -> str:
        """
        Answer questions about image
        
        Args:
            frame: Input image (BGR)
            question: Question text
            
        Returns:
            Answer text
        """
        if self.model is None:
            return "VQA not available"
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            
            inputs = self.processor(images=pil_image, text=question, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=20)
                answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"VQA error: {e}")
            return "Error processing question"


class SceneUnderstanding:
    """
    High-level scene understanding and analysis
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize scene understanding"""
        self.clip_model = CLIPModel(device=device)
        
        # Predefined scene categories
        self.scene_categories = [
            "indoor scene", "outdoor scene", "urban environment", "natural landscape",
            "office", "home", "street", "park", "beach", "forest", "mountain"
        ]
        
        self.activity_categories = [
            "people walking", "people sitting", "people standing", "people running",
            "people talking", "people working", "no visible activity"
        ]
        
        self.weather_categories = [
            "sunny", "cloudy", "rainy", "snowy", "foggy", "clear sky"
        ]
    
    def analyze_scene(self, frame: np.ndarray) -> Dict:
        """
        Comprehensive scene analysis
        
        Args:
            frame: Input image
            
        Returns:
            Dictionary with scene analysis
        """
        analysis = {}
        
        # Scene type
        scene_probs = self.clip_model.classify_image(frame, self.scene_categories)
        if scene_probs:
            analysis['scene_type'] = max(scene_probs.items(), key=lambda x: x[1])
        
        # Activity detection
        activity_probs = self.clip_model.classify_image(frame, self.activity_categories)
        if activity_probs:
            analysis['activity'] = max(activity_probs.items(), key=lambda x: x[1])
        
        # Weather conditions (for outdoor scenes)
        weather_probs = self.clip_model.classify_image(frame, self.weather_categories)
        if weather_probs:
            analysis['weather'] = max(weather_probs.items(), key=lambda x: x[1])
        
        return analysis
