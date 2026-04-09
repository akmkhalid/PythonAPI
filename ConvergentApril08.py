"""
Convergence Analysis for Bangladeshi English Texts (Lightweight Version)
Uses TF-IDF instead of sentence transformers for better compatibility
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# DATA PREPARATION
# ============================================================================

class TextPreprocessor:
    """Clean and prepare text for analysis"""

    @staticmethod
    def clean_text(text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = ' '.join(text.split())
        text = re.sub(r'[^\w\s\.,;:!?\'\"-]', ' ', text)
        return text

    @staticmethod
    def tokenize(text):
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())


# ============================================================================
# CONVERGENCE ANALYZER (Lightweight)
# ============================================================================

class ConvergenceAnalyzer:
    """Simplified convergence analysis using TF-IDF"""

    def __init__(self):
        self.ai_lexical_markers = [
            'delve', 'crucial', 'landscape', 'navigate', 'leverage',
            'moreover', 'furthermore', 'in conclusion', 'paramount',
            'pivotal', 'synergy', 'optimize', 'streamline', 'holistic'
        ]

    def compute_pairwise_tfidf_similarity(self, texts, max_features=5000):
        """Compute pairwise TF-IDF similarity between texts"""
        if len(texts) < 2:
            return np.array([])

        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf_matrix)

            # Extract upper triangle (excluding diagonal)
            n = len(sim_matrix)
            upper_indices = np.triu_indices(n, k=1)
            scores = sim_matrix[upper_indices]
            return scores
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return np.array([])

    def compute_lexical_diversity(self, texts):
        """Compute type-token ratio for each text"""
        ttr_scores = []
        for text in texts:
            tokens = TextPreprocessor.tokenize(text)
            if len(tokens) > 0:
                types = len(set(tokens))
                ttr = types / len(tokens)
            else:
                ttr = 0
            ttr_scores.append(ttr)
        return np.array(ttr_scores)

    def compute_ai_fingerprint(self, texts):
        """Compute frequency of AI lexical markers"""
        scores = []
        for text in texts:
            text_lower = text.lower()
            marker_count = sum(1 for marker in self.ai_lexical_markers
                               if marker in text_lower)
            words = len(TextPreprocessor.tokenize(text))
            if words > 0:
                score = (marker_count / words) * 1000
            else:
                score = 0
            scores.append(score)
        return np.array(scores)

    def compute_vocabulary_size(self, texts):
        """Compute unique vocabulary size for each text"""
        vocab_sizes = []
        for text in texts:
            tokens = set(TextPreprocessor.tokenize(text))
            vocab_sizes.append(len(tokens))
        return np.array(vocab_sizes)

    def analyze_year_group(self, texts, year):
        """Analyze all texts from a single year"""
        results = {
            'year': year,
            'n_texts': len(texts)
        }

        if len(texts) < 2:
            print(f"Warning: Year {year} has only {len(texts)} text(s)")
            return results

        # Clean texts
        cleaned_texts = [TextPreprocessor.clean_text(t) for t in texts]

        # Compute metrics
        similarity_scores = self.compute_pairwise_tfidf_similarity(cleaned_texts)

        if len(similarity_scores) > 0:
            results['similarity_mean'] = np.mean(similarity_scores)
            results['similarity_std'] = np.std(similarity_scores)
            results['similarity_scores'] = similarity_scores
        else:
            results['similarity_mean'] = 0
            results['similarity_std'] = 0
            results['similarity_scores'] = np.array([])

        # Lexical diversity
        ttr_scores = self.compute_lexical_diversity(cleaned_texts)
        results['lexical_diversity_mean'] = np.mean(ttr_scores)
        results['lexical_diversity_std'] = np.std(ttr_scores)
        results['lexical_diversity_scores'] = ttr_scores

        # AI fingerprint
        ai_scores = self.compute_ai_fingerprint(cleaned_texts)
        results['ai_fingerprint_mean'] = np.mean(ai_scores)
        results['ai_fingerprint_std'] = np.std(ai_scores)
        results['ai_fingerprint_scores'] = ai_scores

        # Vocabulary size
        vocab_sizes = self.compute_vocabulary_size(cleaned_texts)
        results['vocab_size_mean'] = np.mean(vocab_sizes)
        results['vocab_size_std'] = np.std(vocab_sizes)

        return results


# ============================================================================
# STATISTICAL COMPARISON
# ============================================================================

class StatisticalComparison:
    """Compare year groups"""

    @staticmethod
    def compare_two_years(results_year1, results_year2, year1_label=2003, year2_label=2026, metric='similarity_mean'):
        """Direct comparison between two years (flexible year labels)"""
        if metric == 'similarity_mean':
            scores_year1 = results_year1.get('similarity_scores', np.array([]))
            scores_year2 = results_year2.get('similarity_scores', np.array([]))
            metric_name = "Similarity Score"
        elif metric == 'lexical_diversity_mean':
            scores_year1 = results_year1.get('lexical_diversity_scores', np.array([]))
            scores_year2 = results_year2.get('lexical_diversity_scores', np.array([]))
            metric_name = "Lexical Diversity"
        elif metric == 'ai_fingerprint_mean':
            scores_year1 = results_year1.get('ai_fingerprint_scores', np.array([]))
            scores_year2 = results_year2.get('ai_fingerprint_scores', np.array([]))
            metric_name = "AI Markers"
        else:
            return None

        if len(scores_year1) == 0 or len(scores_year2) == 0:
            return None

        # Mann-Whitney U test
        stat, p_value = stats.mannwhitneyu(scores_year1, scores_year2, alternative='two-sided')

        # Calculate effect size (Cohen's d)
        mean1, mean2 = np.mean(scores_year1), np.mean(scores_year2)
        std1, std2 = np.std(scores_year1), np.std(scores_year2)
        pooled_std = np.sqrt((std1 ** 2 + std2 ** 2) / 2)
        effect_size = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0

        return {
            'metric': metric_name,
            f'mean_{year1_label}': mean1,
            f'mean_{year2_label}': mean2,
            f'std_{year1_label}': std1,
            f'std_{year2_label}': std2,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': effect_size,
            'direction': 'higher' if mean2 > mean1 else 'lower',
            'year1': year1_label,
            'year2': year2_label
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

class ConvergenceVisualizer:
    """Create visualizations"""

    @staticmethod
    def plot_comparison(results_year1, results_year2, year1_label=2003, year2_label=2026,
                        output_dir='convergence_results'):
        """Create comparison plots for two years"""
        os.makedirs(output_dir, exist_ok=True)

        metrics = [
            ('similarity_mean', 'Text Similarity (TF-IDF)', 'Higher = More Converged', '#2E86AB'),
            ('lexical_diversity_mean', 'Lexical Diversity (Type-Token Ratio)', 'Higher = More Diverse', '#A23B72'),
            ('ai_fingerprint_mean', 'AI Markers (per 1000 words)', 'Higher = More AI-like', '#73AB84'),
            ('vocab_size_mean', 'Vocabulary Size (Unique Words)', 'Higher = Richer Vocabulary', '#F39C12')
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, (metric, title, ylabel, color) in enumerate(metrics):
            mean1 = results_year1.get(metric, 0)
            mean2 = results_year2.get(metric, 0)
            std1 = results_year1.get(metric.replace('mean', 'std'), 0)
            std2 = results_year2.get(metric.replace('mean', 'std'), 0)

            axes[idx].bar([str(year1_label), str(year2_label)], [mean1, mean2],
                          yerr=[std1, std2], capsize=10,
                          color=[color, color], alpha=0.7, edgecolor='black')
            axes[idx].set_ylabel(ylabel, fontsize=10)
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

            # Add value labels on bars
            axes[idx].text(0, mean1 + std1 + 0.01, f'{mean1:.3f}',
                           ha='center', va='bottom', fontsize=9)
            axes[idx].text(1, mean2 + std2 + 0.01, f'{mean2:.3f}',
                           ha='center', va='bottom', fontsize=9)

        plt.suptitle(f'Convergence Analysis: {year1_label} vs {year2_label}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparison_plot.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    @staticmethod
    def plot_similarity_distributions(results_year1, results_year2, year1_label=2003, year2_label=2026,
                                      output_dir='convergence_results'):
        """Plot distribution of similarity scores"""
        scores1 = results_year1.get('similarity_scores', np.array([]))
        scores2 = results_year2.get('similarity_scores', np.array([]))

        if len(scores1) == 0 or len(scores2) == 0:
            print("Not enough data for distribution plot")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(scores1, bins=20, alpha=0.5, label=str(year1_label), color='#2E86AB', density=True)
        ax.hist(scores2, bins=20, alpha=0.5, label=str(year2_label), color='#E63946', density=True)

        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Distribution of Text Similarity Scores: {year1_label} vs {year2_label}', fontsize=14,
                     fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.savefig(f'{output_dir}/similarity_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data_from_folders(base_path='Data'):
    """Load text files from folder structure"""
    data_dict = {}

    if not os.path.exists(base_path):
        print(f"Error: Folder '{base_path}' not found!")
        return {}

    for year_folder in os.listdir(base_path):
        year_path = os.path.join(base_path, year_folder)

        if os.path.isdir(year_path) and year_folder.isdigit():
            year = int(year_folder)
            data_dict[year] = {}

            for genre_folder in os.listdir(year_path):
                genre_path = os.path.join(year_path, genre_folder)

                if os.path.isdir(genre_path):
                    texts = []
                    for filename in os.listdir(genre_path):
                        if filename.lower().endswith('.txt'):
                            filepath = os.path.join(genre_path, filename)
                            try:
                                with open(filepath, 'r', encoding='utf-8') as f:
                                    text = f.read()
                                    if text.strip():
                                        texts.append(text)
                                        print(f"  Loaded: {year}/{genre_folder}/{filename}")
                            except Exception as e:
                                print(f"  Error reading {filepath}: {e}")

                    if texts:
                        data_dict[year][genre_folder] = texts
                        print(f"\n✓ Loaded {len(texts)} texts from {year}/{genre_folder}")

    return data_dict


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("=" * 60)
    print("CONVERGENCE ANALYSIS FOR BANGLADESHI ENGLISH TEXTS")
    print("Testing: Are 2026 texts more similar/converged than 2003 texts?")
    print("=" * 60)

    # Load data
    print("\nLOADING DATA...")
    data_dict = load_data_from_folders('Data')

    if not data_dict:
        print("\nNo data loaded! Please check folder structure.")
        print("Expected structure:")
        print("  Data/")
        print("    ├── 2003/")
        print("    │   ├── Aunties/")
        print("    │   │   ├── file1.txt")
        print("    │   │   └── ...")
        print("    │   └── Fashion/")
        print("    │       ├── file1.txt")
        print("    │       └── ...")
        print("    └── 2026/")
        print("        ├── Aunties/")
        print("        │   ├── file1.txt")
        print("        │   └── ...")
        print("        └── Fashion/")
        print("            ├── file1.txt")
        print("            └── ...")
        return

    # Display summary
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    for year in sorted(data_dict.keys()):
        for genre, texts in data_dict[year].items():
            print(f"  {year} - {genre}: {len(texts)} texts")

    # Get available years (should have at least 2003 and 2026)
    available_years = sorted([y for y in data_dict.keys() if y in [2003, 2026]])

    if 2003 not in available_years or 2026 not in available_years:
        print("\nError: Both 2003 and 2026 data required for comparison!")
        print(f"Found years: {available_years}")
        return

    # Get all genres that exist in BOTH years
    genres_2003 = set(data_dict[2003].keys())
    genres_2026 = set(data_dict[2026].keys())
    common_genres = genres_2003.intersection(genres_2026)

    if not common_genres:
        print("\nError: No common genres found between 2003 and 2026!")
        print(f"2003 genres: {genres_2003}")
        print(f"2026 genres: {genres_2026}")
        return

    print(f"\nAnalyzing genres: {list(common_genres)}")

    # Initialize analyzer
    analyzer = ConvergenceAnalyzer()

    # Store all results for summary
    all_results = {}

    # Analyze each genre found in BOTH years
    for genre in sorted(common_genres):
        print(f"\n" + "=" * 60)
        print(f"ANALYZING: {genre}")
        print("=" * 60)

        # Get texts for both years
        texts_2003 = data_dict[2003][genre]
        texts_2026 = data_dict[2026][genre]

        print(f"  Processing {len(texts_2003)} texts from 2003...")
        results_2003 = analyzer.analyze_year_group(texts_2003, 2003)

        print(f"  Processing {len(texts_2026)} texts from 2026...")
        results_2026 = analyzer.analyze_year_group(texts_2026, 2026)

        # Create output directory for this genre
        output_dir = f'convergence_results/{genre}'
        os.makedirs(output_dir, exist_ok=True)

        # Statistical comparisons
        comparator = StatisticalComparison()

        print("\n" + "-" * 40)
        print("STATISTICAL RESULTS")
        print("-" * 40)

        metrics_to_test = ['similarity_mean', 'lexical_diversity_mean', 'ai_fingerprint_mean']
        results_list = []

        for metric in metrics_to_test:
            comp = comparator.compare_two_years(results_2003, results_2026, 2003, 2026, metric)
            if comp:
                results_list.append(comp)
                print(f"\n{comp['metric']}:")
                print(f"  2003: {comp['mean_2003']:.4f} (±{comp['std_2003']:.4f})")
                print(f"  2026: {comp['mean_2026']:.4f} (±{comp['std_2026']:.4f})")
                if comp['mean_2003'] != 0:
                    print(f"  Change: {((comp['mean_2026'] - comp['mean_2003']) / comp['mean_2003'] * 100):.1f}%")
                else:
                    print(f"  Change: N/A (baseline was 0)")
                print(f"  P-value: {comp['p_value']:.4f}")
                print(f"  Significant: {'✓ YES' if comp['significant'] else '✗ NO'}")
                print(f"  Direction: {comp['direction']} in 2026")
                print(f"  Effect size: {comp['effect_size']:.3f}")

        # Generate visualizations
        visualizer = ConvergenceVisualizer()

        print("\n" + "-" * 40)
        print("GENERATING PLOTS")
        print("-" * 40)

        visualizer.plot_comparison(results_2003, results_2026, 2003, 2026, output_dir)
        visualizer.plot_similarity_distributions(results_2003, results_2026, 2003, 2026, output_dir)

        # Save results to CSV
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f'{output_dir}/statistical_results.csv', index=False)

        print(f"\n✓ Results saved to {output_dir}/")

        # Store for summary
        all_results[genre] = results_list

        # Final interpretation for this genre
        print("\n" + "=" * 60)
        print(f"INTERPRETATION - {genre}")
        print("=" * 60)

        # Check similarity result
        sim_comp = next((r for r in results_list if r['metric'] == 'Similarity Score'), None)
        if sim_comp:
            if sim_comp['significant'] and sim_comp['direction'] == 'higher':
                print("\n✓✓✓ HYPOTHESIS SUPPORTED!")
                print("   Texts in 2026 are SIGNIFICANTLY MORE SIMILAR than in 2003.")
                print("   This indicates convergence/homogenization over time.")
                print(f"   Effect size: {sim_comp['effect_size']:.3f} (moderate to large)")
            elif sim_comp['significant'] and sim_comp['direction'] == 'lower':
                print("\n✗ OPPOSITE TREND DETECTED!")
                print("   Texts in 2026 are LESS similar than in 2003.")
                print("   This suggests divergence rather than convergence.")
            else:
                print("\n○ NO SIGNIFICANT CHANGE DETECTED")
                print("   Similarity scores are not statistically different between years.")

    # Print cross-genre summary
    print("\n" + "=" * 60)
    print("CROSS-GENRE SUMMARY")
    print("=" * 60)

    summary_data = []
    for genre, results_list in all_results.items():
        sim_comp = next((r for r in results_list if r['metric'] == 'Similarity Score'), None)
        if sim_comp:
            summary_data.append({
                'Genre': genre,
                '2003 Similarity': f"{sim_comp['mean_2003']:.4f}",
                '2026 Similarity': f"{sim_comp['mean_2026']:.4f}",
                'Change': f"{((sim_comp['mean_2026'] - sim_comp['mean_2003']) / sim_comp['mean_2003'] * 100):.1f}%",
                'Significant': '✓' if sim_comp['significant'] else '✗',
                'Effect Size': f"{sim_comp['effect_size']:.3f}"
            })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
        summary_df.to_csv('convergence_results/cross_genre_summary.csv', index=False)
        print("\n✓ Cross-genre summary saved to convergence_results/cross_genre_summary.csv")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()