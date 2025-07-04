import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import evaluation libraries
try:
    from bert_score import score as bert_score
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    
except ImportError as e:
    print(f"Missing required libraries. Please install:")
    print("pip install bert-score rouge-score nltk matplotlib seaborn")
    print(f"Error: {e}")
    exit(1)

class EvaluationMetrics:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
    
    def compute_bleu(self, reference, candidate):
        """Compute BLEU score"""
        try:
            # Tokenize sentences
            ref_tokens = reference.lower().split()
            cand_tokens = candidate.lower().split()
            
            # BLEU expects list of reference sentences
            return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self.smoothing)
        except:
            return 0.0
    
    def compute_rouge(self, reference, candidate):
        """Compute ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def compute_bert_score(self, references, candidates):
        """Compute BERTScore for all pairs"""
        try:
            P, R, F1 = bert_score(candidates, references, lang='en', verbose=False)
            return {
                'precision': P.tolist(),
                'recall': R.tolist(),
                'f1': F1.tolist()
            }
        except:
            return {
                'precision': [0.0] * len(candidates),
                'recall': [0.0] * len(candidates),
                'f1': [0.0] * len(candidates)
            }
    
    def compute_meteor(self, reference, candidate):
        """Compute METEOR score"""
        try:
            # Tokenize
            ref_tokens = reference.lower().split()
            cand_tokens = candidate.lower().split()
            return meteor_score([ref_tokens], cand_tokens)
        except:
            return 0.0

class ModelEvaluator:
    def __init__(self):
        self.metrics = EvaluationMetrics()
        self.results_dir = "evaluation/results"
        self.analysis_dir = "evaluation/analysis"
        
        # Create directories
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.analysis_dir).mkdir(parents=True, exist_ok=True)
    
    def load_model_data(self, model_name):
        """Load model descriptions from JSON"""
        file_path = f"data/model_outputs/{model_name}_descriptions.json"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Model file not found: {file_path}")
            return None
    
    def evaluate_single_model(self, model_name):
        """Evaluate a single model against ground truth"""
        print(f"\n{'='*50}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*50}")
        
        # Load model data
        model_data = self.load_model_data(model_name)
        if not model_data:
            return None
        
        # Extract references and candidates
        references = []
        candidates = []
        image_ids = []
        categories = []
        
        empty_descriptions = 0
        
        for image_id, data in model_data.items():
            ground_truth = data.get('ground_truth', '').strip()
            model_desc = data.get('model_description', '').strip()
            
            if not ground_truth:
                print(f"⚠️  No ground truth for {image_id}")
                continue
                
            if not model_desc:
                print(f"⚠️  No model description for {image_id}")
                empty_descriptions += 1
                model_desc = ""  # Will get 0 scores
            
            references.append(ground_truth)
            candidates.append(model_desc)
            image_ids.append(image_id)
            categories.append(data.get('category', 'unknown'))
        
        if empty_descriptions > 0:
            print(f"⚠️  Found {empty_descriptions} empty descriptions")
        
        if not references:
            print("❌ No valid data to evaluate")
            return None
        
        print(f"✅ Evaluating {len(references)} image descriptions")
        
        # Compute metrics
        results = {
            'model_name': model_name,
            'image_ids': image_ids,
            'categories': categories,
            'scores': {}
        }
        
        # BLEU scores
        print("Computing BLEU scores...")
        bleu_scores = [self.metrics.compute_bleu(ref, cand) 
                      for ref, cand in zip(references, candidates)]
        results['scores']['bleu'] = bleu_scores
        
        # ROUGE scores
        print("Computing ROUGE scores...")
        rouge_scores = [self.metrics.compute_rouge(ref, cand) 
                       for ref, cand in zip(references, candidates)]
        results['scores']['rouge1'] = [s['rouge1'] for s in rouge_scores]
        results['scores']['rouge2'] = [s['rouge2'] for s in rouge_scores]
        results['scores']['rougeL'] = [s['rougeL'] for s in rouge_scores]
        
        # METEOR scores
        print("Computing METEOR scores...")
        meteor_scores = [self.metrics.compute_meteor(ref, cand) 
                        for ref, cand in zip(references, candidates)]
        results['scores']['meteor'] = meteor_scores
        
        # BERTScore
        print("Computing BERTScore...")
        bert_scores = self.metrics.compute_bert_score(references, candidates)
        results['scores']['bert_precision'] = bert_scores['precision']
        results['scores']['bert_recall'] = bert_scores['recall']
        results['scores']['bert_f1'] = bert_scores['f1']
        
        # Calculate averages
        avg_scores = {}
        for metric, scores in results['scores'].items():
            avg_scores[f'avg_{metric}'] = np.mean(scores)
            avg_scores[f'std_{metric}'] = np.std(scores)
        
        results['average_scores'] = avg_scores
        
        # Save results
        output_file = f"{self.results_dir}/{model_name}_evaluation.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to {output_file}")
        
        # Print summary
        print(f"\n📊 SUMMARY FOR {model_name.upper()}:")
        print(f"BLEU:      {avg_scores['avg_bleu']:.4f} ± {avg_scores['std_bleu']:.4f}")
        print(f"ROUGE-1:   {avg_scores['avg_rouge1']:.4f} ± {avg_scores['std_rouge1']:.4f}")
        print(f"ROUGE-L:   {avg_scores['avg_rougeL']:.4f} ± {avg_scores['std_rougeL']:.4f}")
        print(f"METEOR:    {avg_scores['avg_meteor']:.4f} ± {avg_scores['std_meteor']:.4f}")
        print(f"BERT-F1:   {avg_scores['avg_bert_f1']:.4f} ± {avg_scores['std_bert_f1']:.4f}")
        
        return results
    
    def evaluate_all_models(self):
        """Evaluate all available models"""
        model_files = list(Path("data/model_outputs").glob("*_descriptions.json"))
        
        if not model_files:
            print("❌ No model files found in data/model_outputs/")
            return {}
        
        all_results = {}
        
        for file_path in model_files:
            model_name = file_path.stem.replace('_descriptions', '')
            results = self.evaluate_single_model(model_name)
            if results:
                all_results[model_name] = results
        
        return all_results
    
    def create_comparison_report(self, all_results=None):
        """Create comprehensive comparison report"""
        if all_results is None:
            all_results = self.evaluate_all_models()
        
        if not all_results:
            print("❌ No results to compare")
            return
        
        print(f"\n{'='*60}")
        print("CREATING COMPARISON REPORT")
        print(f"{'='*60}")
        
        # Create comparison DataFrame
        comparison_data = []
        
        for model_name, results in all_results.items():
            avg_scores = results['average_scores']
            row = {
                'Model': model_name,
                'BLEU': avg_scores['avg_bleu'],
                'ROUGE-1': avg_scores['avg_rouge1'],
                'ROUGE-2': avg_scores['avg_rouge2'],
                'ROUGE-L': avg_scores['avg_rougeL'],
                'METEOR': avg_scores['avg_meteor'],
                'BERT-Precision': avg_scores['avg_bert_precision'],
                'BERT-Recall': avg_scores['avg_bert_recall'],
                'BERT-F1': avg_scores['avg_bert_f1']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        csv_file = f"{self.analysis_dir}/model_comparison.csv"
        df.to_csv(csv_file, index=False)
        print(f"✅ Comparison table saved to {csv_file}")
        
        # Create visualizations
        self.create_visualizations(df, all_results)
        
        # Print ranking
        print(f"\n{'='*60}")
        print("MODEL RANKINGS")
        print(f"{'='*60}")
        
        metrics_to_rank = ['BLEU', 'ROUGE-L', 'METEOR', 'BERT-F1']
        
        for metric in metrics_to_rank:
            ranked = df.nlargest(len(df), metric)
            print(f"\n🏆 {metric} Rankings:")
            for i, (_, row) in enumerate(ranked.iterrows(), 1):
                print(f"  {i}. {row['Model']}: {row[metric]:.4f}")
        
        return df
    
    def create_visualizations(self, df, all_results):
        """Create comparison visualizations"""
        plt.style.use('default')
        
        # 1. Bar plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['BLEU', 'ROUGE-L', 'METEOR', 'BERT-F1']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            df_sorted = df.sort_values(metric, ascending=True)
            bars = ax.barh(df_sorted['Model'], df_sorted[metric])
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} Scores')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{self.analysis_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Heatmap
        plt.figure(figsize=(12, 8))
        metrics_df = df.set_index('Model')[['BLEU', 'ROUGE-1', 'ROUGE-L', 'METEOR', 'BERT-F1']]
        sns.heatmap(metrics_df, annot=True, cmap='YlOrRd', fmt='.3f', cbar_kws={'label': 'Score'})
        plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.analysis_dir}/performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Category-wise analysis (if categories available)
        self.create_category_analysis(all_results)
        
        print(f"✅ Visualizations saved to {self.analysis_dir}/")
    
    def create_category_analysis(self, all_results):
        """Analyze performance by image category"""
        category_data = []
        
        for model_name, results in all_results.items():
            categories = results['categories']
            scores = results['scores']
            
            for i, category in enumerate(categories):
                row = {
                    'Model': model_name,
                    'Category': category,
                    'BLEU': scores['bleu'][i],
                    'ROUGE-L': scores['rougeL'][i],
                    'METEOR': scores['meteor'][i],
                    'BERT-F1': scores['bert_f1'][i]
                }
                category_data.append(row)
        
        if not category_data:
            return
        
        cat_df = pd.DataFrame(category_data)
        
        # Save category analysis
        cat_df.to_csv(f"{self.analysis_dir}/category_analysis.csv", index=False)
        
        # Create category-wise comparison plot
        plt.figure(figsize=(14, 10))
        
        categories = cat_df['Category'].unique()
        metrics = ['BLEU', 'ROUGE-L', 'METEOR', 'BERT-F1']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance by Image Category', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # Group by category and calculate means
            cat_means = cat_df.groupby(['Category', 'Model'])[metric].mean().unstack()
            cat_means.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(f'{metric} by Category')
            ax.set_xlabel('Category')
            ax.set_ylabel(metric)
            ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{self.analysis_dir}/category_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Category summary
        print(f"\n📊 CATEGORY-WISE PERFORMANCE SUMMARY:")
        for category in categories:
            cat_subset = cat_df[cat_df['Category'] == category]
            print(f"\n{category.upper()}:")
            for metric in metrics:
                best_model = cat_subset.loc[cat_subset[metric].idxmax(), 'Model']
                best_score = cat_subset[metric].max()
                print(f"  Best {metric}: {best_model} ({best_score:.4f})")

def run_evaluation():
    """Main function to run the complete evaluation"""
    evaluator = ModelEvaluator()
    
    print("🚀 Starting Model Evaluation...")
    
    # Check if we have any model files
    model_files = list(Path("data/model_outputs").glob("*_descriptions.json"))
    
    if not model_files:
        print("❌ No model files found!")
        print("Please ensure you have model description files in data/model_outputs/")
        print("File format: {model_name}_descriptions.json")
        return
    
    print(f"📁 Found {len(model_files)} model files:")
    for file_path in model_files:
        model_name = file_path.stem.replace('_descriptions', '')
        print(f"  - {model_name}")
    
    # Run evaluation
    all_results = evaluator.evaluate_all_models()
    
    if all_results:
        # Create comparison report
        comparison_df = evaluator.create_comparison_report(all_results)
        
        print(f"\n🎉 EVALUATION COMPLETE!")
        print(f"📊 Results saved in: evaluation/results/")
        print(f"📈 Analysis saved in: evaluation/analysis/")
        print(f"📋 Summary table: evaluation/analysis/model_comparison.csv")
        
        return comparison_df
    else:
        print("❌ No successful evaluations completed")
        return None

def evaluate_specific_models(model_names):
    """Evaluate only specific models"""
    evaluator = ModelEvaluator()
    results = {}
    
    for model_name in model_names:
        result = evaluator.evaluate_single_model(model_name)
        if result:
            results[model_name] = result
    
    if results:
        comparison_df = evaluator.create_comparison_report(results)
        return comparison_df
    return None

# Example usage and utility functions
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate vision-language models')
    parser.add_argument('--models', nargs='+', help='Specific models to evaluate')
    parser.add_argument('--all', action='store_true', help='Evaluate all available models')
    
    args = parser.parse_args()
    
    if args.models:
        print(f"Evaluating specific models: {args.models}")
        evaluate_specific_models(args.models)
    else:
        print("Evaluating all available models...")
        run_evaluation()