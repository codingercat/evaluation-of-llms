import pandas as pd
import json
import os
from pathlib import Path

def setup_project_structure():
    """Create the project directory structure"""
    directories = [
        'data/images',
        'data/ground_truth',
        'data/model_outputs',
        'evaluation/scripts',
        'evaluation/results',
        'evaluation/analysis',
        'utils'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("Project structure created successfully!")

def convert_excel_to_ground_truth(excel_file_path, output_path='data/ground_truth/ground_truth.json'):
    """
    Convert Excel file with DIAGRAM descriptions to JSON format
    
    Args:
        excel_file_path: Path to your Excel file
        output_path: Where to save the JSON file
    """
    try:
        # Read the Excel file
        df = pd.read_excel('D:\Shambhavi NCAHT\evaluation_of_different_vision_models_research_project\data\DIAGRAM CENTER DATASET.xlsx')
        
        # Print column names to help you identify the right columns
        print("Available columns in Excel:")
        print(df.columns.tolist())
        
        # You'll need to adjust these column names based on your Excel structure
        # Common possibilities:
        required_columns = ['image_id', 'filename', 'description', 'category']
        
        print(f"\nPlease check if your Excel has columns similar to: {required_columns}")
        print("If not, you'll need to modify the column names in this script")
        
        # Assuming your Excel has these columns (adjust as needed):
        # - 'image_id' or 'Image ID' or similar
        # - 'filename' or 'Image Name' or similar  
        # - 'description' or 'Description' or 'Ground Truth' or similar
        # - 'category' or 'Category' or 'Type' or similar
        
        ground_truth_data = {}
        
        for index, row in df.iterrows():
            # Adjust these column names based on your Excel structure
            image_id = row.get('image_id', f'image_{index+1:03d}')
            filename = row.get('filename', row.get('Image Name', f'image_{index+1}.jpg'))
            description = row.get('description', row.get('Description', row.get('Ground Truth', '')))
            category = row.get('category', row.get('Category', row.get('Type', 'unknown')))
            
            ground_truth_data[image_id] = {
                'filename': filename,
                'category': category,
                'ground_truth': description
            }
        
        # Save to JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ground_truth_data, f, indent=2, ensure_ascii=False)
        
        print(f"Ground truth data saved to {output_path}")
        print(f"Processed {len(ground_truth_data)} images")
        
        return ground_truth_data
        
    except Exception as e:
        print(f"Error processing Excel file: {e}")
        return None

def create_model_template(model_name, ground_truth_path='data/ground_truth/ground_truth.json'):
    """
    Create a template JSON file for a new model's descriptions
    
    Args:
        model_name: Name of the model (e.g., 'gpt4v', 'llava', etc.)
        ground_truth_path: Path to ground truth JSON
    """
    try:
        # Load ground truth to get image IDs
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        # Create template with empty descriptions
        model_data = {}
        for image_id, gt_data in ground_truth.items():
            model_data[image_id] = {
                'filename': gt_data['filename'],
                'category': gt_data['category'],
                'ground_truth': gt_data['ground_truth'],
                'model_description': ''  # Empty field to fill
            }
        
        # Save template
        output_path = f'data/model_outputs/{model_name}_descriptions.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
        
        print(f"Template created for {model_name} at {output_path}")
        print("You can now fill in the 'model_description' fields manually or programmatically")
        
        return output_path
        
    except Exception as e:
        print(f"Error creating template: {e}")
        return None

def add_descriptions_to_model(model_name, descriptions_dict):
    """
    Add descriptions to an existing model JSON file
    
    Args:
        model_name: Name of the model
        descriptions_dict: Dictionary with image_id -> description mapping
    """
    file_path = f'data/model_outputs/{model_name}_descriptions.json'
    
    try:
        # Load existing data
        with open(file_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        # Update descriptions
        for image_id, description in descriptions_dict.items():
            if image_id in model_data:
                model_data[image_id]['model_description'] = description
            else:
                print(f"Warning: Image ID {image_id} not found in model data")
        
        # Save updated data
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
        
        print(f"Updated descriptions for {model_name}")
        
    except Exception as e:
        print(f"Error updating model descriptions: {e}")

# Example usage
if __name__ == "__main__":
    # Step 1: Set up project structure
    setup_project_structure()
    
    # Step 2: Convert your Excel file to ground truth JSON
    excel_file = r"D:\Shambhavi NCAHT\evaluation_of_different_vision_models_research_project\data\DIAGRAM CENTER DATASET.xlsx"
    
    print("=" * 50)
    print("STEP 1: Converting Excel to Ground Truth JSON")
    print("=" * 50)
    
    if os.path.exists(excel_file):
        ground_truth = convert_excel_to_ground_truth(excel_file)
    else:
        print(f"Excel file '{excel_file}' not found. Please update the file path.")
    
    print("\n" + "=" * 50)
    print("STEP 2: Creating Model Templates")
    print("=" * 50)
    
    # Step 3: Create templates for your models
    # Add your model names here
    web_ui_models = ['gpt4v', 'claude_vision', 'gemini_vision']  # Update with your models
    ollama_models = ['llava', 'bakllava', 'moondream']  # Update with your models
    
    all_models = web_ui_models + ollama_models
    
    for model in all_models:
        create_model_template(model)
    
    print("\n" + "=" * 50)
    print("SETUP COMPLETE!")
    print("=" * 50)
    print("Next steps:")
    print("1. Check that your Excel was converted correctly")
    print("2. For web UI models: manually fill the model_description fields")
    print("3. For Ollama models: use the ollama_inference.py script")
    print("4. Run evaluation metrics using the evaluation script")