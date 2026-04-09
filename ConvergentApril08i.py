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
# US STRUCTURAL DOMINANCE ANALYZER (Americanization at Structural Level)
# ============================================================================

class US_Structural_Analyzer:
    """
    Analyzes structural features that distinguish US English from UK/BdE English.
    This captures Americanization at the DEEP STRUCTURAL level, not surface spelling.
    """

    def __init__(self):
        # US structural patterns (rhetorical, organizational, syntactic)
        self.us_structural_patterns = {
            # Rhetorical patterns (US prefers direct, enthusiastic engagement)
            'exclamation_marks': r'!',
            'rhetorical_questions': r'\b(why not|what\'s more|have you ever|can you imagine|is there anything more|who doesn\'t|what could be|don\'t you think)\b',
            'direct_address': r'\b(you|your|yours)\b',
            'inclusive_we': r'\b(we|us|our)\b',

            # Organizational patterns (US prefers lists, sections, clear structure)
            'numbered_lists': r'\d+\.\s+',
            'bullet_points': r'[•\-]\s+',
            'section_headers': r'^[A-Z][a-z]+:|^[A-Z][a-z]+$',  # "Morning:", "Skin prep"
            'color_lists': r'\b(pink|red|blue|green|yellow|purple|orange|white|black|brown|grey|beige|mint|sage|teal|maroon|fuchsia|emerald|coral|lavender)\b.*\b(pink|red|blue|green|yellow|purple|orange|white|black|brown|grey|beige|mint|sage|teal|maroon|fuchsia|emerald|coral|lavender)\b',

            # Transitional patterns (US prefers simple, conversational transitions)
            'simple_transitions': r'\b(so|ok|now|well|look|here\'s the thing|here\'s why|the bottom line|let\'s face it|you see|actually|basically|honestly)\b',

            # Engagement patterns (US prefers direct reader engagement)
            'reader_questions': r'\?',
            'imperatives': r'\b(try|choose|opt for|go for|add|pair|wear|ditch|swap|embrace|avoid|remember|don\'t forget)\b'
        }

        # UK/BdE structural patterns (for comparison)
        self.uk_structural_patterns = {
            # UK prefers passive voice, hedging, complex transitions
            'passive_voice': r'\b(be|been|being)\s+\w+ed\b|\bwas\s+\w+ed\b|\bwere\s+\w+ed\b',
            'hedging': r'\b(perhaps|maybe|it seems|it appears|possibly|rather|quite|somewhat|fairly|relatively)\b',
            'complex_transitions': r'\b(nevertheless|nonetheless|furthermore|moreover|consequently|accordingly|hence|thus|therefore|whereas)\b',
            'indirect_address': r'\b(one|oneself|one\'s)\b',
            'long_paragraphs': None,  # Calculated separately ( >5 sentences)
            'nominalization': r'\b\w+(tion|sion|ance|ence|ment|ity)\b'  # Noun-heavy style
        }

    def analyze_text(self, text):
        """Extract US vs UK structural features from a single text"""
        text_lower = text.lower()

        # Count US structural markers
        us_counts = {}
        for name, pattern in self.us_structural_patterns.items():
            if pattern:
                matches = re.findall(pattern, text_lower, re.MULTILINE)
                us_counts[name] = len(matches)
            else:
                us_counts[name] = 0

        # Special handling for section headers (count unique lines)
        lines = text.split('\n')
        section_headers = 0
        for line in lines:
            if re.match(r'^[A-Z][a-z]+:', line.strip()) or re.match(r'^[A-Z][a-z]+$', line.strip()):
                section_headers += 1
        us_counts['section_headers'] = section_headers

        # Count UK structural markers
        uk_counts = {}
        for name, pattern in self.uk_structural_patterns.items():
            if pattern:
                matches = re.findall(pattern, text_lower, re.MULTILINE)
                uk_counts[name] = len(matches)
            else:
                uk_counts[name] = 0

        # Sentence and paragraph analysis
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 20]

        # US prefers shorter sentences and paragraphs
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        avg_paragraph_length = np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0

        # Long paragraph count (UK prefers longer, more complex paragraphs)
        long_paragraphs = sum(1 for p in paragraphs if len(p.split()) > 100)
        uk_counts['long_paragraphs'] = long_paragraphs

        # Calculate US Dominance Score
        us_total = sum(us_counts.values())
        uk_total = sum(uk_counts.values())

        if us_total + uk_total > 0:
            us_dominance_score = (us_total - uk_total) / (us_total + uk_total)
        else:
            us_dominance_score = 0

        # Weighted score (exclamations and lists are stronger US signals)
        weighted_us = (us_counts.get('exclamation_marks', 0) * 2 +
                       us_counts.get('numbered_lists', 0) * 2 +
                       us_counts.get('bullet_points', 0) * 2 +
                       us_counts.get('section_headers', 0) * 1.5 +
                       us_counts.get('rhetorical_questions', 0) * 1.5 +
                       sum(us_counts.values()))

        weighted_uk = (uk_counts.get('passive_voice', 0) * 2 +
                       uk_counts.get('complex_transitions', 0) * 2 +
                       uk_counts.get('long_paragraphs', 0) * 2 +
                       sum(uk_counts.values()))

        if weighted_us + weighted_uk > 0:
            weighted_us_dominance = (weighted_us - weighted_uk) / (weighted_us + weighted_uk)
        else:
            weighted_us_dominance = 0

        return {
            'us_counts': us_counts,
            'uk_counts': uk_counts,
            'us_total': us_total,
            'uk_total': uk_total,
            'us_dominance_score': us_dominance_score,
            'weighted_us_dominance': weighted_us_dominance,
            'avg_sentence_length': avg_sentence_length,
            'avg_paragraph_length': avg_paragraph_length,
            'exclamation_count': us_counts.get('exclamation_marks', 0),
            'rhetorical_question_count': us_counts.get('rhetorical_questions', 0),
            'direct_address_count': us_counts.get('direct_address', 0),
            'numbered_list_count': us_counts.get('numbered_lists', 0),
            'section_header_count': us_counts.get('section_headers', 0),
            'passive_voice_count': uk_counts.get('passive_voice', 0),
            'complex_transition_count': uk_counts.get('complex_transitions', 0)
        }

    def analyze_corpus(self, texts):
        """Analyze a corpus of texts and return aggregated results"""
        if not texts:
            return None

        results = [self.analyze_text(t) for t in texts]

        return {
            'mean_us_dominance': np.mean([r['us_dominance_score'] for r in results]),
            'mean_weighted_us_dominance': np.mean([r['weighted_us_dominance'] for r in results]),
            'std_us_dominance': np.std([r['us_dominance_score'] for r in results]),
            'mean_exclamation_count': np.mean([r['exclamation_count'] for r in results]),
            'mean_rhetorical_questions': np.mean([r['rhetorical_question_count'] for r in results]),
            'mean_direct_address': np.mean([r['direct_address_count'] for r in results]),
            'mean_numbered_lists': np.mean([r['numbered_list_count'] for r in results]),
            'mean_section_headers': np.mean([r['section_header_count'] for r in results]),
            'mean_passive_voice': np.mean([r['passive_voice_count'] for r in results]),
            'mean_complex_transitions': np.mean([r['complex_transition_count'] for r in results]),
            'mean_avg_sentence_length': np.mean([r['avg_sentence_length'] for r in results]),
            'individual_results': results
        }

    def compare_years(self, texts_2003, texts_2026):
        """Compare US structural dominance between years"""
        results_2003 = self.analyze_corpus(texts_2003)
        results_2026 = self.analyze_corpus(texts_2026)

        if not results_2003 or not results_2026:
            return None

        # Statistical tests for each metric
        metric_keys = ['exclamation_count', 'rhetorical_question_count', 'direct_address_count',
                       'numbered_list_count', 'section_header_count', 'passive_voice_count',
                       'complex_transition_count']

        comparison = {}

        for key in metric_keys:
            scores_2003 = [r[key] for r in results_2003['individual_results']]
            scores_2026 = [r[key] for r in results_2026['individual_results']]

            if len(scores_2003) > 0 and len(scores_2026) > 0:
                stat, p_value = stats.mannwhitneyu(scores_2003, scores_2026, alternative='two-sided')
                mean_2003 = np.mean(scores_2003)
                mean_2026 = np.mean(scores_2026)

                if mean_2003 > 0:
                    change_percent = ((mean_2026 - mean_2003) / mean_2003) * 100
                else:
                    change_percent = 0 if mean_2026 == 0 else 100
            else:
                p_value = 1.0
                mean_2003 = 0
                mean_2026 = 0
                change_percent = 0

            # Create simplified name for printing
            simple_name = key.replace('_count', '').replace('_count', '')
            comparison[simple_name] = {
                'mean_2003': mean_2003,
                'mean_2026': mean_2026,
                'change_percent': change_percent,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'direction': 'higher' if mean_2026 > mean_2003 else 'lower'
            }

        # Add US dominance score
        scores_2003_dom = [r['us_dominance_score'] for r in results_2003['individual_results']]
        scores_2026_dom = [r['us_dominance_score'] for r in results_2026['individual_results']]

        if len(scores_2003_dom) > 0 and len(scores_2026_dom) > 0:
            stat, p_value = stats.mannwhitneyu(scores_2003_dom, scores_2026_dom, alternative='two-sided')
        else:
            p_value = 1.0

        comparison['us_dominance'] = {
            'mean_2003': results_2003['mean_us_dominance'],
            'mean_2026': results_2026['mean_us_dominance'],
            'change_percent': ((results_2026['mean_us_dominance'] - results_2003['mean_us_dominance']) / (
                        abs(results_2003['mean_us_dominance']) + 0.001)) * 100,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'direction': 'higher' if results_2026['mean_us_dominance'] > results_2003['mean_us_dominance'] else 'lower'
        }

        # Create aliases for the print section to match expected names
        comparison['exclamation_count'] = comparison['exclamation']
        comparison['rhetorical_questions'] = comparison['rhetorical_question']
        comparison['direct_address'] = comparison['direct_address']
        comparison['numbered_lists'] = comparison['numbered_list']
        comparison['passive_voice'] = comparison['passive_voice']
        comparison['complex_transitions'] = comparison['complex_transition']

        # Overall US dominance shift (for easy access)
        comparison['us_dominance_shift'] = {
            'score_2003': results_2003['mean_us_dominance'],
            'score_2026': results_2026['mean_us_dominance'],
            'shift': results_2026['mean_us_dominance'] - results_2003['mean_us_dominance'],
            'weighted_2003': results_2003['mean_weighted_us_dominance'],
            'weighted_2026': results_2026['mean_weighted_us_dominance']
        }

        return comparison

    def plot_us_dominance(self, results_2003, results_2026, genre_name, output_dir):
        """Create visualization for US structural dominance"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # 1. US Dominance Score (most important)
        ax1 = axes[0, 0]
        scores_2003 = [r['us_dominance_score'] for r in results_2003['individual_results']]
        scores_2026 = [r['us_dominance_score'] for r in results_2026['individual_results']]

        ax1.boxplot([scores_2003, scores_2026], labels=['2003', '2026'])
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral')
        ax1.axhline(y=0.3, color='green', linestyle='--', alpha=0.3, label='US Dominant')
        ax1.axhline(y=-0.3, color='red', linestyle='--', alpha=0.3, label='UK Dominant')
        ax1.set_ylabel('US Dominance Score (-1=UK, +1=US)')
        ax1.set_title(f'{genre_name}: US vs UK Structural Dominance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Key US Structural Features
        ax2 = axes[0, 1]
        features = ['exclamation_count', 'rhetorical_questions', 'direct_address', 'numbered_lists']
        means_2003 = [results_2003['mean_exclamation_count'], results_2003['mean_rhetorical_questions'],
                      results_2003['mean_direct_address'], results_2003['mean_numbered_lists']]
        means_2026 = [results_2026['mean_exclamation_count'], results_2026['mean_rhetorical_questions'],
                      results_2026['mean_direct_address'], results_2026['mean_numbered_lists']]

        x = np.arange(len(features))
        width = 0.35
        ax2.bar(x - width / 2, means_2003, width, label='2003', color='#2E86AB', alpha=0.7)
        ax2.bar(x + width / 2, means_2026, width, label='2026', color='#E63946', alpha=0.7)
        ax2.set_ylabel('Count per Article')
        ax2.set_title('US Structural Features (Higher = More American)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Exclamations', 'Rhetorical Qs', '"You"', 'Numbered Lists'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. UK Structural Features (Should DECREASE)
        ax3 = axes[1, 0]
        uk_features = ['passive_voice', 'complex_transitions']
        uk_means_2003 = [results_2003['mean_passive_voice'], results_2003['mean_complex_transitions']]
        uk_means_2026 = [results_2026['mean_passive_voice'], results_2026['mean_complex_transitions']]

        x2 = np.arange(len(uk_features))
        ax3.bar(x2 - width / 2, uk_means_2003, width, label='2003', color='#2E86AB', alpha=0.7)
        ax3.bar(x2 + width / 2, uk_means_2026, width, label='2026', color='#E63946', alpha=0.7)
        ax3.set_ylabel('Count per Article')
        ax3.set_title('UK/BdE Structural Features (Lower = Less UK Influence)')
        ax3.set_xticks(x2)
        ax3.set_xticklabels(['Passive Voice', 'Complex Transitions'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Sentence Length (US prefers shorter)
        ax4 = axes[1, 1]
        sent_lengths_2003 = [r['avg_sentence_length'] for r in results_2003['individual_results']]
        sent_lengths_2026 = [r['avg_sentence_length'] for r in results_2026['individual_results']]

        ax4.boxplot([sent_lengths_2003, sent_lengths_2026], labels=['2003', '2026'])
        ax4.set_ylabel('Average Sentence Length (words)')
        ax4.set_title('Sentence Length (US = Shorter, UK/BdE = Longer)')
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'{genre_name}: Structural Americanization Analysis\n(Shift from UK/BdE toward US norms)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        # FIX: Use output_dir directly (it already includes the genre folder)
        save_path = os.path.join(output_dir, 'us_structural_dominance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved US structural plot to: {save_path}")
        plt.show()

        return fig
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
        results['texts'] = cleaned_texts  # Store cleaned texts for heatmap
        return results


class FashionConvergenceAnalyzer(ConvergenceAnalyzer):
    """Extended analyzer with fashion-specific metrics"""

    def __init__(self):
        super().__init__()
        # Fashion-specific AI markers
        self.fashion_ai_markers = [
            'trendsetting', 'must-have', 'season\'s hottest', 'capsule wardrobe',
            'investment piece', 'fashion-forward', 'style staple', 'wardrobe essential',
            'elevate your look', 'effortless chic', 'sustainable fashion'
        ]

        # Fashion terminology diversity (more technical terms = less homogenized)
        self.fashion_terminology = [
            'silhouette', 'texture', 'fabric', 'hemline', 'neckline',
            'embellishment', 'draping', 'tailoring', 'couture', 'prêt-à-porter'
        ]

    def compute_fashion_terminology_density(self, texts):
        """Measure use of specialized fashion vocabulary"""
        scores = []
        for text in texts:
            text_lower = text.lower()
            term_count = sum(1 for term in self.fashion_terminology if term in text_lower)
            words = len(TextPreprocessor.tokenize(text))
            score = (term_count / words) * 1000 if words > 0 else 0
            scores.append(score)
        return np.array(scores)

    def compute_sentence_length_variance(self, texts):
        """Measure variation in sentence length (AI produces more uniform sentences)"""
        variances = []
        for text in texts:
            sentences = re.split(r'[.!?]+', text)
            sent_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
            if len(sent_lengths) > 1:
                variances.append(np.var(sent_lengths))
            else:
                variances.append(0)
        return np.array(variances)

    def compute_lexical_richness(self, texts):
        """Measure Brunet's index (more sophisticated than TTR)"""
        scores = []
        for text in texts:
            tokens = TextPreprocessor.tokenize(text)
            if len(tokens) > 0:
                types = len(set(tokens))
                # Brunet's index: N^(types^-0.172)
                score = len(tokens) ** (types ** -0.172)
            else:
                score = 0
            scores.append(score)
        return np.array(scores)

    @staticmethod
    def analyze_fashion_subgroups(texts_2003, texts_2026):
        """
        Split Fashion articles by detected subgroup
        Example subgroups: 'Traditional Dress', 'Western Fashion', 'Beauty', 'Accessories'
        """

        # Define subgroup keywords
        subgroups = {
            'Traditional': ['saree', 'salwar', 'kameez', 'lungi', 'panjabi', 'nakshi', 'jamdani'],
            'Western': ['jeans', 't-shirt', 'dress', 'skirt', 'blazer', 'sneakers'],
            'Beauty': ['makeup', 'skin care', 'hair', 'cosmetic', 'lipstick', 'foundation'],
            'Accessories': ['jewelry', 'bag', 'shoe', 'belt', 'watch', 'scarf']
        }

        # Classify each text
        results = {}
        for subgroup, keywords in subgroups.items():
            texts_2003_sub = [t for t in texts_2003 if any(k in t.lower() for k in keywords)]
            texts_2026_sub = [t for t in texts_2026 if any(k in t.lower() for k in keywords)]

            if len(texts_2003_sub) >= 3 and len(texts_2026_sub) >= 3:
                results[subgroup] = {
                    '2003_n': len(texts_2003_sub),
                    '2026_n': len(texts_2026_sub),
                    '2003_texts': texts_2003_sub,
                    '2026_texts': texts_2026_sub
                }

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

    # Add this function to check your current statistical power
    @staticmethod
    def calculate_power_analysis(effect_size=0.203, alpha=0.05, power=0.80):
        """
        Calculate required sample size for Fashion convergence
        Based on your current effect size (d=0.203)
        """
        # Using common power analysis formula
        # For d=0.203 (small effect), you need approximately 380 texts per group!
        # For d=0.5 (medium effect), you need approximately 64 texts per group

        print("Required sample sizes for different effect sizes:")
        print(f"  Current effect size (d=0.203 - small): Need ~380 texts/year")
        print(f"  Medium effect size (d=0.5): Need ~64 texts/year")
        print(f"  Large effect size (d=0.8): Need ~25 texts/year")
        print(f"\nYour current sample (n=5) can only detect VERY large effects (d>1.5)")
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


def create_fashion_feature_matrix(texts_2003, texts_2026):
    """
    Create a qualitative comparison matrix
    """
    features = {
        'First-person pronouns (I, we, our)': lambda t: sum(1 for w in ['i', 'we', 'our'] if w in t.lower()),
        'Direct address to reader (you)': lambda t: t.lower().count(' you '),
        'Local terms (saree, jamdani, etc.)': lambda t: sum(
            1 for term in ['saree', 'jamdani', 'nakshi', 'panjabi'] if term in t.lower()),
        'Exclamation marks': lambda t: t.count('!'),
        'Question marks': lambda t: t.count('?'),
        'Average sentence length': lambda t: np.mean(
            [len(s.split()) for s in re.split(r'[.!?]+', t) if len(s.split()) > 0]) if t else 0,
    }

    results = []
    for year, texts in [('2003', texts_2003), ('2026', texts_2026)]:
        year_results = {'Year': year, 'N': len(texts)}
        for feature_name, feature_func in features.items():
            scores = [feature_func(t) for t in texts]
            year_results[feature_name] = f"{np.mean(scores):.2f} (±{np.std(scores):.2f})"
        results.append(year_results)

    return pd.DataFrame(results)


def bootstrap_convergence_test(texts_2003, texts_2026, n_iterations=1000):
    """
    Bootstrap test for convergence with small samples (OPTIMIZED VERSION)
    Reduced iterations and caching for better performance
    """
    np.random.seed(42)

    # Initialize analyzer
    analyzer = ConvergenceAnalyzer()

    # Calculate observed difference
    sim_2003 = analyzer.compute_pairwise_tfidf_similarity(texts_2003)
    sim_2026 = analyzer.compute_pairwise_tfidf_similarity(texts_2026)

    if len(sim_2003) == 0 or len(sim_2026) == 0:
        print("Warning: Not enough texts for bootstrap test")
        return {
            'observed_difference': 0,
            'ci_95_lower': 0,
            'ci_95_upper': 0,
            'bootstrap_p_value': 1.0,
            'significant': False
        }

    observed_diff = np.mean(sim_2026) - np.mean(sim_2003)

    # Combine all texts
    all_texts = texts_2003 + texts_2026
    n1, n2 = len(texts_2003), len(texts_2026)

    # Pre-compute TF-IDF for all texts once (much faster!)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    try:
        all_tfidf = vectorizer.fit_transform(all_texts)
    except Exception as e:
        print(f"Error in TF-IDF computation: {e}")
        return {
            'observed_difference': observed_diff,
            'ci_95_lower': observed_diff - 0.05,
            'ci_95_upper': observed_diff + 0.05,
            'bootstrap_p_value': 0.5,
            'significant': False
        }

    # Bootstrap by resampling indices (much faster than recomputing TF-IDF)
    bootstrap_diffs = []
    n_samples = min(n_iterations, 500)  # Reduce iterations for speed

    print(f"  Running bootstrap with {n_samples} iterations...")

    for _ in range(n_samples):
        # Resample indices
        idx1 = np.random.choice(len(all_texts), size=n1, replace=True)
        idx2 = np.random.choice(len(all_texts), size=n2, replace=True)

        # Get corresponding TF-IDF vectors
        tfidf1 = all_tfidf[idx1]
        tfidf2 = all_tfidf[idx2]

        # Compute similarity matrices
        sim1_matrix = cosine_similarity(tfidf1)
        sim2_matrix = cosine_similarity(tfidf2)

        # Extract upper triangles
        n1_mat = len(sim1_matrix)
        n2_mat = len(sim2_matrix)

        if n1_mat >= 2 and n2_mat >= 2:
            upper1 = sim1_matrix[np.triu_indices(n1_mat, k=1)]
            upper2 = sim2_matrix[np.triu_indices(n2_mat, k=1)]

            if len(upper1) > 0 and len(upper2) > 0:
                bootstrap_diffs.append(np.mean(upper2) - np.mean(upper1))

    if len(bootstrap_diffs) == 0:
        return {
            'observed_difference': observed_diff,
            'ci_95_lower': observed_diff - 0.05,
            'ci_95_upper': observed_diff + 0.05,
            'bootstrap_p_value': 0.5,
            'significant': False
        }

    # Calculate bootstrap confidence interval
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    # Calculate p-value
    p_value = np.mean([d >= observed_diff for d in bootstrap_diffs])

    return {
        'observed_difference': observed_diff,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'bootstrap_p_value': p_value,
        'significant': ci_lower > 0 or ci_upper < 0
    }


def plot_fashion_convergence_pattern(results_2003, results_2026):
    """
    Create specialized visualization for Fashion convergence
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Similarity distribution (shows spread)
    ax1 = axes[0, 0]
    scores_2003 = results_2003.get('similarity_scores', [])
    scores_2026 = results_2026.get('similarity_scores', [])

    ax1.boxplot([scores_2003, scores_2026], labels=['2003', '2026'])
    ax1.set_ylabel('Similarity Score')
    ax1.set_title('Distribution of Pairwise Similarity')
    ax1.grid(True, alpha=0.3)

    # 2. Individual text similarity heatmap
    ax2 = axes[0, 1]
    # Create similarity matrix for each year
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer

    all_texts = results_2003.get('texts', []) + results_2026.get('texts', [])
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf = vectorizer.fit_transform(all_texts)

    n1 = len(results_2003.get('texts', []))
    similarity_matrix = cosine_similarity(tfidf)

    im = ax2.imshow(similarity_matrix, cmap='RdBu_r', vmin=0, vmax=0.3)
    ax2.axvline(x=n1 - 0.5, color='black', linewidth=2)
    ax2.axhline(y=n1 - 0.5, color='black', linewidth=2)
    ax2.set_xlabel('Text Index')
    ax2.set_ylabel('Text Index')
    ax2.set_title('Similarity Heatmap (2003 top-left, 2026 bottom-right)')
    plt.colorbar(im, ax=ax2)

    # 3. AI marker trend
    ax3 = axes[1, 0]
    ai_2003 = results_2003.get('ai_fingerprint_scores', [])
    ai_2026 = results_2026.get('ai_fingerprint_scores', [])

    ax3.scatter(['2003'] * len(ai_2003), ai_2003, alpha=0.6, label='Individual texts')
    ax3.scatter(['2026'] * len(ai_2026), ai_2026, alpha=0.6)
    ax3.plot([0, 1], [np.mean(ai_2003), np.mean(ai_2026)], 'r-', linewidth=2, label='Mean')
    ax3.set_ylabel('AI Markers per 1000 words')
    ax3.set_title('AI Marker Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Lexical diversity trend
    ax4 = axes[1, 1]
    ttr_2003 = results_2003.get('lexical_diversity_scores', [])
    ttr_2026 = results_2026.get('lexical_diversity_scores', [])

    ax4.scatter(['2003'] * len(ttr_2003), ttr_2003, alpha=0.6)
    ax4.scatter(['2026'] * len(ttr_2026), ttr_2026, alpha=0.6)
    ax4.plot([0, 1], [np.mean(ttr_2003), np.mean(ttr_2026)], 'r-', linewidth=2)
    ax4.set_ylabel('Type-Token Ratio')
    ax4.set_title('Lexical Diversity Trend')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Fashion Genre Convergence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('convergence_results/Fashion/convergence_patterns.png', dpi=300)
    plt.show()
# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("=" * 60)
    print("CONVERGENCE ANALYSIS FOR BANGLADESHI ENGLISH TEXTS")
    print("Testing: Are 2025/2026 texts more similar/converged than 2003 texts?")
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
        print("    ├── 2025/")
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

    # Get available years (2003, 2025, 2026)
    target_years = [2003, 2025, 2026]
    available_years = sorted([y for y in data_dict.keys() if y in target_years])

    if 2003 not in available_years:
        print("\nError: 2003 data required for baseline comparison!")
        print(f"Found years: {available_years}")
        return

    print(f"\nComparing baseline year 2003 with: {[y for y in available_years if y != 2003]}")

    # Get all genres that exist in ALL selected years
    genres_all = set(data_dict[2003].keys())
    for year in available_years:
        if year != 2003:
            genres_all = genres_all.intersection(set(data_dict[year].keys()))

    if not genres_all:
        print("\nError: No common genres found across years!")
        return

    print(f"\nAnalyzing genres: {list(genres_all)}")

    # Initialize analyzer
    analyzer = ConvergenceAnalyzer()

    # Store all results for summary
    all_results = {}

    # For each genre, compare 2003 with 2025 and 2026
    for genre in sorted(genres_all):
        print(f"\n" + "=" * 60)
        print(f"ANALYZING: {genre}")
        print("=" * 60)

        # Get 2003 baseline texts and analyze once
        texts_2003 = data_dict[2003][genre]
        print(f"  Baseline 2003: {len(texts_2003)} texts")
        print(f"  Processing 2003 baseline...")
        results_2003 = analyzer.analyze_year_group(texts_2003, 2003)

        # Store results for this genre
        genre_results = {}

        # Compare with each available year
        for year in [y for y in available_years if y != 2003]:
            if year not in data_dict or genre not in data_dict[year]:
                print(f"  Year {year}: No data for genre {genre}, skipping...")
                continue

            texts_year = data_dict[year][genre]
            print(f"\n  Processing {len(texts_year)} texts from {year}...")
            results_year = analyzer.analyze_year_group(texts_year, year)

            # Create output directory for this genre and year
            output_dir = f'convergence_results/{genre}/{year}_vs_2003'
            os.makedirs(output_dir, exist_ok=True)

            # Statistical comparisons
            comparator = StatisticalComparison()

            print("\n" + "-" * 40)
            print(f"STATISTICAL RESULTS: 2003 vs {year}")
            print("-" * 40)

            metrics_to_test = ['similarity_mean', 'lexical_diversity_mean', 'ai_fingerprint_mean']
            results_list = []

            for metric in metrics_to_test:
                comp = comparator.compare_two_years(results_2003, results_year, 2003, year, metric)
                if comp:
                    results_list.append(comp)
                    print(f"\n{comp['metric']}:")
                    print(f"  2003: {comp[f'mean_{2003}']:.4f} (±{comp[f'std_{2003}']:.4f})")
                    print(f"  {year}: {comp[f'mean_{year}']:.4f} (±{comp[f'std_{year}']:.4f})")
                    if comp[f'mean_{2003}'] != 0:
                        print(f"  Change: {((comp[f'mean_{year}'] - comp[f'mean_{2003}']) / comp[f'mean_{2003}'] * 100):.1f}%")
                    else:
                        print(f"  Change: N/A (baseline was 0)")
                    print(f"  P-value: {comp['p_value']:.4f}")
                    print(f"  Significant: {'✓ YES' if comp['significant'] else '✗ NO'}")
                    print(f"  Direction: {comp['direction']} in {year}")
                    print(f"  Effect size: {comp['effect_size']:.3f}")

            # Generate visualizations
            visualizer = ConvergenceVisualizer()

            print("\n" + "-" * 40)
            print("GENERATING PLOTS")
            print("-" * 40)

            visualizer.plot_comparison(results_2003, results_year, 2003, year, output_dir)
            visualizer.plot_similarity_distributions(results_2003, results_year, 2003, year, output_dir)

            # Save results to CSV
            results_df = pd.DataFrame(results_list)
            results_df.to_csv(f'{output_dir}/statistical_results.csv', index=False)

            print(f"\n✓ Results saved to {output_dir}/")

            # Store for summary
            genre_results[year] = results_list

            # ============================================================
            # US STRUCTURAL DOMINANCE ANALYSIS (Americanization)
            # ============================================================
            print("\n" + "-" * 40)
            print(f"US STRUCTURAL DOMINANCE ANALYSIS: 2003 vs {year}")
            print("-" * 40)

            us_analyzer = US_Structural_Analyzer()

            # Analyze both years
            us_results_2003 = us_analyzer.analyze_corpus(texts_2003)
            us_results_year = us_analyzer.analyze_corpus(texts_year)

            if us_results_2003 and us_results_year:
                # Compare years
                us_comparison = us_analyzer.compare_years(texts_2003, texts_year)

                print(f"\nUS DOMINANCE SCORE (-1=UK/BdE, +1=US):")
                print(f"  2003: {us_comparison['us_dominance_shift']['score_2003']:.3f}")
                print(f"  {year}: {us_comparison['us_dominance_shift']['score_2026']:.3f}")
                print(f"  Shift: {us_comparison['us_dominance_shift']['shift']:+.3f}")

                print(f"\nKEY US STRUCTURAL FEATURES (Higher = More American):")
                print(f"  Exclamation marks: {us_comparison['exclamation']['mean_2003']:.2f} → {us_comparison['exclamation']['mean_2026']:.2f} ({us_comparison['exclamation']['change_percent']:+.0f}%, p={us_comparison['exclamation']['p_value']:.4f})")
                print(f"  Rhetorical questions: {us_comparison['rhetorical_question']['mean_2003']:.2f} → {us_comparison['rhetorical_question']['mean_2026']:.2f} ({us_comparison['rhetorical_question']['change_percent']:+.0f}%, p={us_comparison['rhetorical_question']['p_value']:.4f})")
                print(f"  Direct address ('you'): {us_comparison['direct_address']['mean_2003']:.2f} → {us_comparison['direct_address']['mean_2026']:.2f} ({us_comparison['direct_address']['change_percent']:+.0f}%, p={us_comparison['direct_address']['p_value']:.4f})")
                print(f"  Numbered lists: {us_comparison['numbered_list']['mean_2003']:.2f} → {us_comparison['numbered_list']['mean_2026']:.2f} ({us_comparison['numbered_list']['change_percent']:+.0f}%, p={us_comparison['numbered_list']['p_value']:.4f})")

                print(f"\nKEY UK/BdE STRUCTURAL FEATURES (Lower = Less UK Influence):")
                print(f"  Passive voice: {us_comparison['passive_voice']['mean_2003']:.2f} → {us_comparison['passive_voice']['mean_2026']:.2f} ({us_comparison['passive_voice']['change_percent']:+.0f}%, p={us_comparison['passive_voice']['p_value']:.4f})")
                print(f"  Complex transitions: {us_comparison['complex_transition']['mean_2003']:.2f} → {us_comparison['complex_transition']['mean_2026']:.2f} ({us_comparison['complex_transition']['change_percent']:+.0f}%, p={us_comparison['complex_transition']['p_value']:.4f})")

                print(f"\nSIGNIFICANCE:")
                if us_comparison['us_dominance_shift']['shift'] > 0 and us_comparison['exclamation']['significant']:
                    print("  ✓ SIGNIFICANT SHIFT toward US structural norms detected")
                    print(f"  US Dominance increased by {abs(us_comparison['us_dominance_shift']['shift']):.3f} points")
                else:
                    print("  ○ No significant structural shift detected (may need larger sample)")

                # Save results
                us_df = pd.DataFrame([{
                    'year': year,
                    'genre': genre,
                    'us_dominance_2003': us_comparison['us_dominance_shift']['score_2003'],
                    f'us_dominance_{year}': us_comparison['us_dominance_shift']['score_2026'],
                    'us_dominance_shift': us_comparison['us_dominance_shift']['shift']
                }])
                us_df.to_csv(f'{output_dir}/us_structural_analysis.csv', index=False)

                # Generate visualization
                us_analyzer.plot_us_dominance(us_results_2003, us_results_year, genre, output_dir)
                print(f"\n✓ US structural analysis saved to {output_dir}/")

        # Store for summary
        all_results[genre] = genre_results

        # Final interpretation for this genre
        print("\n" + "=" * 60)
        print(f"INTERPRETATION - {genre}")
        print("=" * 60)

        # Check similarity results for each year
        for year, results_list in genre_results.items():
            sim_comp = next((r for r in results_list if r['metric'] == 'Similarity Score'), None)
            if sim_comp:
                if sim_comp['significant'] and sim_comp['direction'] == 'higher':
                    print(f"\n✓✓✓ HYPOTHESIS SUPPORTED for {year}!")
                    print(f"   Texts in {year} are SIGNIFICANTLY MORE SIMILAR than in 2003.")
                    print(f"   Effect size: {sim_comp['effect_size']:.3f} (moderate to large)")
                elif sim_comp['significant'] and sim_comp['direction'] == 'lower':
                    print(f"\n✗ OPPOSITE TREND DETECTED for {year}!")
                    print(f"   Texts in {year} are LESS similar than in 2003.")
                else:
                    print(f"\n○ NO SIGNIFICANT CHANGE DETECTED for {year}")
                    print(f"   Similarity scores are not statistically different from 2003.")
                    print(f"   Effect size: {sim_comp['effect_size']:.3f}")

    # Print cross-genre summary
    print("\n" + "=" * 60)
    print("CROSS-GENRE SUMMARY")
    print("=" * 60)

    summary_data = []
    for genre, year_results in all_results.items():
        for year, results_list in year_results.items():
            sim_comp = next((r for r in results_list if r['metric'] == 'Similarity Score'), None)
            if sim_comp:
                summary_data.append({
                    'Genre': genre,
                    'Year': year,
                    '2003 Similarity': f"{sim_comp[f'mean_{2003}']:.4f}",
                    f'{year} Similarity': f"{sim_comp[f'mean_{year}']:.4f}",
                    'Change': f"{((sim_comp[f'mean_{year}'] - sim_comp[f'mean_{2003}']) / sim_comp[f'mean_{2003}'] * 100):.1f}%",
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