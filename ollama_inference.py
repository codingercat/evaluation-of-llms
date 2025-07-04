import requests
import json
import base64
import os
from pathlib import Path
import time

class OllamaInference:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.prompt = "Describe this image for a blind student. Be detailed and focus on the educational content."
    
    def encode_image(self, image_path):
        """Encode image to base64 for Ollama API"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def generate_description(self, model_name, image_path, max_retries=3):
        """Generate description for a single image using Ollama"""
        encoded_image = self.encode_image(image_path)
        if not encoded_image:
            return None
        
        payload = {
            "model": model_name,
            "prompt": self.prompt,
            "images": [encoded_image],
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=300 
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '').strip()
                else:
                    print(f"API error (attempt {attempt + 1}): {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retry
                    
            except requests.exceptions.RequestException as e:
                print(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return None
    
    def process_model(self, model_name, images_dir="data/images"):
        """Process all images for a specific model"""
        print(f"\n{'='*50}")
        print(f"Processing model: {model_name}")
        print(f"{'='*50}")
        
        # Load ground truth to get image list
        ground_truth_path = "data/ground_truth/ground_truth.json"
        try:
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
        except FileNotFoundError:
            print(f"Ground truth file not found at {ground_truth_path}")
            return False
        
        # Remove characters like ":" that are invalid in filenames
        safe_model_name = model_name.replace(":", "_")
        model_file = f"data/model_outputs/{safe_model_name}_descriptions.json"

        try:
            with open(model_file, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
        except FileNotFoundError:
            print(f"Model template not found at {model_file}")
            print("Please create the template first using the data setup script")
            return False
        
        # Process each image
        total_images = len(ground_truth)
        processed = 0
        failed = 0
        
        for image_id, gt_data in ground_truth.items():
            print(f"\nProcessing {image_id} ({processed + 1}/{total_images})")
            
            # Skip if already processed
            if model_data[image_id]['model_description'].strip():
                print(f"  ✓ Already processed, skipping")
                processed += 1
                continue
            
            # Find image file
            filename = gt_data['filename']
            image_path = os.path.join(images_dir, filename)
            
            if not os.path.exists(image_path):
                print(f"  ✗ Image not found: {image_path}")
                failed += 1
                continue
            
            # Generate description
            print(f"  → Generating description...")
            description = self.generate_description(model_name, image_path)
            
            if description:
                model_data[image_id]['model_description'] = description
                print(f"  ✓ Generated ({len(description)} chars)")
                processed += 1
                
                # Save progress after each successful generation
                with open(model_file, 'w', encoding='utf-8') as f:
                    json.dump(model_data, f, indent=2, ensure_ascii=False)
                
            else:
                print(f"  ✗ Failed to generate description")
                failed += 1
            
            # Small delay to avoid overwhelming the API
            time.sleep(1)
        
        print(f"\n{'='*50}")
        print(f"Model {model_name} processing complete!")
        print(f"Successfully processed: {processed}/{total_images}")
        print(f"Failed: {failed}/{total_images}")
        print(f"{'='*50}")
        
        return failed == 0
    
    def check_model_availability(self, model_name):
        """Check if a model is available in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                return model_name in available_models
            return False
        except:
            return False
    
    def list_available_models(self):
        """List all available models in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except:
            return []

def run_ollama_models(model_list, images_dir="data/images"):
    """Run inference for multiple Ollama models"""
    ollama = OllamaInference()
    
    print("Checking Ollama connection...")
    available_models = ollama.list_available_models()
    
    if not available_models:
        print("❌ Cannot connect to Ollama or no models available")
        print("Make sure Ollama is running: ollama serve")
        return
    
    print(f"✅ Connected to Ollama. Available models: {available_models}")
    
    results = {}
    for model_name in model_list:
        if not ollama.check_model_availability(model_name):
            print(f"\n⚠️  Model '{model_name}' not available in Ollama")
            print(f"Available models: {available_models}")
            print(f"To install: ollama pull {model_name}")
            results[model_name] = False
            continue
        
        success = ollama.process_model(model_name, images_dir)
        results[model_name] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("OLLAMA PROCESSING SUMMARY")
    print(f"{'='*60}")
    for model, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model}: {status}")

# Example usage
if __name__ == "__main__":
    # List of Ollama models to process
    # Update this list with your specific models
    ollama_models = [
            "moondream:latest",
            "mistral-small3.2:latest",
            "llama3.2-vision:latest"
        ]
    
    print("Starting Ollama inference for image descriptions...")
    print(f"Models to process: {ollama_models}")
    
    # Make sure images directory exists
    images_dir = "data/images"
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        print("Please place your images in the data/images/ directory")
    else:
        run_ollama_models(ollama_models, images_dir)