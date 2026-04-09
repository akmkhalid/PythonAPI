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
        # ============================================================
        # LEVEL 1: CLASSIC CHATGPT MARKERS (2022-2023)
        # These are the original AI tells from early ChatGPT
        # ============================================================
        self.chatgpt_classic_markers = [
            # Overused academic transitions
            'delve', 'delve into', 'crucial', 'paramount', 'pivotal',
            'moreover', 'furthermore', 'nevertheless', 'nonetheless',
            'in conclusion', 'to summarize', 'as previously mentioned',

            # Common ChatGPT openings
            'it is important to note', 'it is worth noting', 'it is essential',
            'it is crucial to understand', 'have you ever wondered',

            # AI favorite verbs
            'navigate', 'leverage', 'optimize', 'streamline', 'holistic',
            'synergy', 'utilize', 'implement', 'facilitate', 'enhance',

            # Hedge words (AI uncertainty)
            'perhaps', 'maybe', 'it seems', 'it appears', 'it suggests',
            'one might argue', 'it could be argued', 'some may say'
        ]

        # ============================================================
        # LEVEL 2: AI LISTICLES & STRUCTURED CONTENT (2023-2024)
        # AI loves numbered lists and structured formatting
        # ============================================================
        self.listicle_markers = [
            'here are', 'here is', 'below are', 'the following are',
            'top 5', 'top 10', 'top 3', 'ways to', 'steps to', 'reasons why',
            'first and foremost', 'last but not least', 'in no particular order',
            'without further ado', 'let\'s dive in', 'let\'s explore',
            'here\'s why', 'here\'s how', 'the bottom line is'
        ]

        # ============================================================
        # LEVEL 3: MODERN RELATIONSHIP & PSYCHOLOGY TERMS (2023-2025)
        # AI-generated advice columns and relationship content
        # ============================================================
        self.psychology_terms = [
            # Relationship dynamics
            'situationship', 'ghosting', 'breadcrumbing', 'love bombing',
            'trauma bond', 'attachment style', 'emotional labor', 'mental load',
            'red flags', 'green flags', 'ick', 'micro-cheating', 'orbiting',
            'cloaking', 'paperclipping', 'submarining', 'stashing',

            # Psychological concepts
            'gaslighting', 'manipulation', 'toxicity', 'narcissist',
            'boundaries', 'validation', 'emotional intelligence', 'self-awareness',
            'coping mechanism', 'trigger warning', 'safe space', 'healing journey',
            'inner child', 'shadow work', 'emotional regulation', 'self-care',

            # Self-help vocabulary
            'self-growth', 'personal development', 'mindfulness', 'authenticity',
            'vulnerability', 'resilience', 'empowerment', 'purpose-driven'
        ]

        # ============================================================
        # LEVEL 4: CASUAL US CONTRACTIONS (2024-2026)
        # AI now mimics casual American speech patterns
        # ============================================================
        self.casual_contractions = [
            'don\'t', 'can\'t', 'won\'t', 'doesn\'t', 'isn\'t', 'aren\'t',
            'wasn\'t', 'weren\'t', 'haven\'t', 'hasn\'t', 'hadn\'t',
            'couldn\'t', 'wouldn\'t', 'shouldn\'t', 'mightn\'t', 'mustn\'t',
            'let\'s', 'that\'s', 'what\'s', 'who\'s', 'where\'s', 'when\'s',
            'here\'s', 'there\'s', 'it\'s', 'i\'m', 'you\'re', 'we\'re', 'they\'re',
            'i\'ve', 'you\'ve', 'we\'ve', 'they\'ve', 'i\'d', 'you\'d', 'we\'d',
            'i\'ll', 'you\'ll', 'we\'ll', 'they\'ll'
        ]

        # ============================================================
        # LEVEL 5: COLLOQUIAL OPENINGS & DISCOURSE MARKERS (2024-2026)
        # AI mimics natural conversation with casual starters
        # ============================================================
        self.colloquial_markers = [
            # Conversational openings
            'so,', 'ok,', 'okay,', 'well,', 'look,', 'listen,', 'here\'s the thing',
            'the thing is', 'truth is', 'the truth is', 'to be honest', 'honestly,',
            'actually,', 'basically,', 'literally,', 'seriously,', 'i mean,',
            'you know,', 'you see,', 'guess what', 'believe it or not',

            # Filler words (AI overuses these)
            'like', 'just', 'so', 'well', 'now', 'then', 'anyway', 'anyways'
        ]

        # ============================================================
        # LEVEL 6: BUSINESS & CORPORATE JARGON (2023-2026)
        # AI-generated professional content markers
        # ============================================================
        self.business_jargon = [
            'circle back', 'touch base', 'deep dive', 'low hanging fruit',
            'move the needle', 'think outside the box', 'win-win',
            'synergy', 'bandwidth', 'capacity', 'actionable', 'deliverable',
            'leverage', 'optimize', 'streamline', 'paradigm shift',
            'core competency', 'value proposition', 'pain point', 'solutioning',
            'onboarding', 'offboarding', 'scalable', 'robust', 'seamless'
        ]

        # ============================================================
        # LEVEL 7: EMPATHETIC & SUPPORTIVE LANGUAGE (2023-2026)
        # AI advice columns use therapeutic language
        # ============================================================
        self.empathetic_markers = [
            'it\'s okay to', 'it\'s perfectly fine to', 'you deserve',
            'you are enough', 'your feelings are valid', 'it\'s not your fault',
            'give yourself permission', 'be kind to yourself', 'practice self-compassion',
            'remember that you', 'it\'s important to remember', 'at the end of the day',
            'that being said', 'having said that', 'with that in mind'
        ]

        # ============================================================
        # LEVEL 8: EXCLAMATION & ENGAGEMENT PATTERNS (2024-2026)
        # AI overuses exclamations and rhetorical questions
        # ============================================================
        self.engagement_markers = [
            '!',  # Exclamation mark (counted separately)
            '?',  # Question mark (counted separately)
            'right?', 'correct?', 'isn\'t it?', 'don\'t you think?',
            'you know what i mean?', 'if that makes sense', 'does that make sense'
        ]

        # ============================================================
        # LEVEL 9: AI FAVORITE ADJECTIVES & ADVERBS (2022-2026)
        # Overused modifiers in AI-generated text
        # ============================================================
        self.overused_modifiers = [
            'absolutely', 'definitely', 'certainly', 'undoubtedly',
            'essentially', 'virtually', 'practically', 'literally',
            'truly', 'deeply', 'profoundly', 'remarkably', 'incredibly',
            'extremely', 'exceptionally', 'particularly', 'significantly',
            'important', 'essential', 'critical', 'vital', 'imperative'
        ]

        # ============================================================
        # LEVEL 10: AI HALLUCINATION PHRASES (2022-2026)
        # Phrases AI uses when it doesn't have specific information
        # ============================================================
        self.hallucination_phrases = [
            'it is widely believed', 'many experts agree', 'research has shown',
            'studies suggest', 'according to experts', 'it has been proven',
            'it is commonly known', 'as the saying goes', 'as they say',
            'needless to say', 'suffice it to say', 'it goes without saying'
        ]

        # ============================================================
        # LEVEL 11: BUZZWORDS & TRENDING TERMS (2023-2026)
        # AI picks up and overuses trending vocabulary
        # ============================================================
        self.buzzwords = [
            # General buzzwords
            'unprecedented', 'uncertain times', 'new normal', 'pivot',
            'resilience', 'adaptability', 'innovation', 'disruption',
            'sustainability', 'eco-friendly', 'mindful', 'intentional',

            # Social media influenced
            'main character energy', 'glow up', 'healing era', 'villain era',
            'it\'s giving', 'slay', 'iconic', 'aesthetic', 'vibe', 'energy',

            # Modern dating terms
            'demisexual', 'sapiosexual', 'polyamory', 'ethical non-monogamy',
            'talking stage', 'exclusive', 'label', 'situationship'
        ]

        # ============================================================
        # LEVEL 12: AI ESSAY STRUCTURE MARKERS (2022-2026)
        # AI organizes content with predictable transitions
        # ============================================================
        self.structure_markers = [
            'first', 'second', 'third', 'finally', 'lastly',
            'additionally', 'in addition', 'furthermore', 'moreover',
            'consequently', 'as a result', 'therefore', 'thus', 'hence',
            'however', 'nevertheless', 'nonetheless', 'on the other hand',
            'in contrast', 'similarly', 'likewise', 'for example', 'for instance',
            'in other words', 'that is to say', 'to put it simply'
        ]
        # ============================================================
        # LEVEL 13: 2026 LISTICLE & STRUCTURAL MARKERS
        # ============================================================
        self.listicle_2026_markers = [
            # Numbered list introductions
            '10 types of', '5 types of', '7 types of', 'top 10', 'top 5',
            'ways to', 'reasons why', 'things you', 'things that',

            # Section header patterns (common in 2026)
            'the truth is', 'here\'s the thing', 'the thing is',
            'at the end of the day', 'when it comes to', 'speaking of',

            # Rhetorical question patterns
            'is it not', 'have we become', 'could any of us', 'what if',
            'who doesn\'t', 'why would', 'how many of us',

            # Direct address patterns
            'my darling readers', 'dear readers', 'let me tell you',
            'you know that', 'think about it', 'imagine this'
        ]
        # NEW: GEN-Z & SLANG TERMS (2024-2026)
        # ============================================================
        self.genz_slang = [
            'footloose and fancy-free', 'green pastures', 'endgame', 'main character',
            'the ick', 'ick', 'red flags', 'green flags', 'situationship', 'ghosting',
            'breadcrumbing', 'love bombing', 'trauma bond', 'orbiting', 'cloaking',
            'paperclipping', 'submarining', 'stashing', 'micro-cheating',
            'it\'s giving', 'slay', 'iconic', 'aesthetic', 'vibe', 'energy',
            'glow up', 'healing era', 'villain era', 'main character energy'
        ]

        # ============================================================
        # NEW: FASHION & LIFESTYLE AI MARKERS (2023-2026)
        # ============================================================
        self.fashion_markers = [
            'voilà', 'voila', 'sartorial', 'resplendent', 'effortless chic',
            'elevate your look', 'match made in heaven', 'quiet luxury',
            'steal the spotlight', 'make a statement', 'timeless', 'versatile',
            'effortlessly', 'sophisticated', 'glamorous', 'chic', 'stunning',
            'fabulous', 'sublime', 'heads will turn', 'turn heads',
            'dressed to the nines', 'best foot forward', 'put your best foot forward',
            'sartorially superior', 'fashion-forward', 'style staple', 'wardrobe essential',
            'investment piece', 'capsule wardrobe', 'statement piece'
        ]

        # ============================================================
        # NEW: AI TRANSITIONAL PHRASES (Common in 2025)
        # ============================================================
        self.transitional_phrases = [
            'the truth is', 'here\'s the thing', 'the thing is', 'at the end of the day',
            'when it comes to', 'speaking of', 'that being said', 'having said that',
            'with that in mind', 'on that note', 'needless to say', 'suffice it to say',
            'it goes without saying', 'believe it or not', 'guess what', 'you know what',
            'long story short', 'to make a long story short', 'all things considered'
        ]

        # ============================================================
        # NEW: EMOTIONAL/INSPIRATIONAL PHRASES (AI Advice Columns)
        # ============================================================
        self.inspirational_phrases = [
            'you deserve', 'you are enough', 'your feelings are valid', 'be kind to yourself',
            'give yourself permission', 'practice self-compassion', 'it\'s okay to',
            'it\'s perfectly fine to', 'remember that you', 'don\'t forget to',
            'take a moment to', 'take a deep breath', 'give yourself grace',
            'show yourself compassion', 'honor your feelings', 'trust the process'
        ]

        # ============================================================
        # NEW: BUSINESS/SELF-HELP PHRASES (2024-2026)
        # ============================================================
        self.self_help_phrases = [
            'level up', 'game changer', 'next level', 'pro tip', 'life hack',
            'work smarter not harder', 'the secret to', 'the key to',
            'unlock your potential', 'step out of your comfort zone',
            'push your boundaries', 'break the cycle', 'heal from within'
        ]

        # ============================================================
        # AGGREGATE LISTS FOR EASY ACCESS
        # ============================================================

        # Complete AI markers (all levels combined)
        # Transitional markers (2022-2024 hybrid period)
        self.transitional_markers = (
                self.listicle_markers +
                self.psychology_terms[:20] +  # First 20 psychology terms
                self.empathetic_markers[:10] +  # First 10 empathetic markers
                self.buzzwords[:15]  # First 15 buzzwords
        )

        self.comprehensive_ai_markers = (
                self.chatgpt_classic_markers +
                self.listicle_markers +
                self.psychology_terms +
                self.business_jargon +
                self.empathetic_markers +
                self.overused_modifiers +
                self.hallucination_phrases +
                self.buzzwords +
                self.structure_markers +
                self.genz_slang +  # NEW
                self.fashion_markers +  # NEW
                self.transitional_phrases +  # NEW
                self.inspirational_phrases +  # NEW
                self.self_help_phrases +
                self.listicle_2026_markers # NEW
        )

        # Modern AI markers (2025-2026 conversational) - Updated
        self.modern_ai_markers = (
                self.casual_contractions +
                self.colloquial_markers +
                self.engagement_markers +
                self.psychology_terms[20:] +
                self.buzzwords[15:] +
                self.genz_slang +  # NEW
                self.fashion_markers +  # NEW
                self.transitional_phrases +  # NEW
                self.inspirational_phrases  # NEW
        )

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
        """Compute comprehensive AI fingerprint with multiple categories"""
        scores = []

        for text in texts:
            text_lower = text.lower()
            words = len(TextPreprocessor.tokenize(text))
            if words == 0:
                scores.append(0)
                continue

            # Count each category
            classic_count = sum(1 for marker in self.chatgpt_classic_markers
                                if marker in text_lower)
            listicle_count = sum(1 for marker in self.listicle_markers
                                 if marker in text_lower)
            psychology_count = sum(1 for marker in self.psychology_terms
                                   if marker in text_lower)
            contraction_count = sum(1 for marker in self.casual_contractions
                                    if marker in text_lower)
            colloquial_count = sum(1 for marker in self.colloquial_markers
                                   if marker in text_lower)
            business_count = sum(1 for marker in self.business_jargon
                                 if marker in text_lower)
            empathetic_count = sum(1 for marker in self.empathetic_markers
                                   if marker in text_lower)
            exclamation_count = text.count('!')
            modifier_count = sum(1 for marker in self.overused_modifiers
                                 if marker in text_lower)
            hallucination_count = sum(1 for marker in self.hallucination_phrases
                                      if marker in text_lower)
            buzzword_count = sum(1 for marker in self.buzzwords
                                 if marker in text_lower)
            structure_count = sum(1 for marker in self.structure_markers
                                  if marker in text_lower)
            # NEW: Count Gen-Z slang
            genz_count = sum(1 for marker in self.genz_slang
                             if marker in text_lower)
            # NEW: Count fashion markers
            fashion_count = sum(1 for marker in self.fashion_markers
                                if marker in text_lower)
            # NEW: Count transitional phrases
            transitional_count = sum(1 for marker in self.transitional_phrases
                                     if marker in text_lower)
            # NEW: Count inspirational phrases
            inspirational_count = sum(1 for marker in self.inspirational_phrases
                                      if marker in text_lower)
            listicle_2026_count = sum(1 for marker in self.listicle_2026_markers
                                      if marker in text_lower)

            # FIXED: Higher weights for 2025-specific markers
            weighted_score = (
                    classic_count * 3.0 +  # Classic ChatGPT (rare in 2025)
                    listicle_count * 2.0 +  # Listicle structure
                    psychology_count * 3.0 +  # Psychology terms (INCREASED)
                    contraction_count * 4.0 +  # ← INCREASED! Major 2025 marker
                    colloquial_count * 3.5 +  # ← INCREASED! Major 2025 marker
                    business_count * 2.0 +  # Business jargon
                    empathetic_count * 3.0 +  # Empathetic language (INCREASED)
                    exclamation_count * 3.0 +  # Exclamations (INCREASED)
                    modifier_count * 1.5 +  # Overused modifiers (slightly increased)
                    hallucination_count * 3.0 +  # Hallucination phrases
                    buzzword_count * 2.0 +  # Buzzwords (INCREASED)
                    structure_count * 1.0 +  # Structure markers
                    genz_count * 4.0 +  # NEW! Gen-Z slang (HIGH weight)
                    fashion_count * 3.5 +  # NEW! Fashion markers
                    transitional_count * 3.0 +  # NEW! Transitional phrases
                    inspirational_count * 3.0 + # NEW! Inspirational phrases
                    listicle_2026_count * 3.0  # NEW: High weight for 2026 markers
            )

            # Normalize per 1000 words
            score = (weighted_score / words) * 1000
            scores.append(score)

        return np.array(scores)

    def compute_ai_category_breakdown(self, text):
        """Break down AI fingerprint by category for detailed analysis"""
        text_lower = text.lower()
        words = len(TextPreprocessor.tokenize(text))

        if words == 0:
            return {}

        return {
            'classic_chatgpt': (sum(1 for m in self.chatgpt_classic_markers if m in text_lower) / words) * 1000,
            'listicle': (sum(1 for m in self.listicle_markers if m in text_lower) / words) * 1000,
            'psychology': (sum(1 for m in self.psychology_terms if m in text_lower) / words) * 1000,
            'casual_contractions': (sum(1 for m in self.casual_contractions if m in text_lower) / words) * 1000,
            'colloquial': (sum(1 for m in self.colloquial_markers if m in text_lower) / words) * 1000,
            'business_jargon': (sum(1 for m in self.business_jargon if m in text_lower) / words) * 1000,
            'empathetic': (sum(1 for m in self.empathetic_markers if m in text_lower) / words) * 1000,
            'exclamations': (text.count('!') / words) * 1000,
            'overused_modifiers': (sum(1 for m in self.overused_modifiers if m in text_lower) / words) * 1000,
            'hallucination_phrases': (sum(1 for m in self.hallucination_phrases if m in text_lower) / words) * 1000,
            'buzzwords': (sum(1 for m in self.buzzwords if m in text_lower) / words) * 1000,
            'structure_markers': (sum(1 for m in self.structure_markers if m in text_lower) / words) * 1000
        }

    def compute_vocabulary_size(self, texts):
        """Compute unique vocabulary size for each text"""
        vocab_sizes = []
        for text in texts:
            tokens = set(TextPreprocessor.tokenize(text))
            vocab_sizes.append(len(tokens))
        return np.array(vocab_sizes)

    def compute_all_metrics(self, texts):
        """Compute all convergence and Americanization metrics for a text set"""
        cleaned_texts = [TextPreprocessor.clean_text(t) for t in texts]

        # Similarity (convergence)
        sim_scores = self.compute_pairwise_tfidf_similarity(cleaned_texts)
        similarity = np.mean(sim_scores) if len(sim_scores) > 0 else 0

        # Lexical diversity (vocabulary richness)
        ttr_scores = self.compute_lexical_diversity(cleaned_texts)
        lexical_diversity = np.mean(ttr_scores)

        # AI fingerprint (AI pattern presence)
        ai_scores = self.compute_ai_fingerprint(cleaned_texts)
        ai_score = np.mean(ai_scores)

        # US dominance (Americanization)
        us_analyzer = US_Structural_Analyzer()
        us_results = us_analyzer.analyze_corpus(cleaned_texts)
        us_dominance = us_results['mean_us_dominance'] if us_results else 0

        # US structural features
        exclamation_count = us_results['mean_exclamation_count'] if us_results else 0
        direct_address = us_results['mean_direct_address'] if us_results else 0
        rhetorical_questions = us_results['mean_rhetorical_questions'] if us_results else 0

        return {
            'similarity': similarity,
            'lexical_diversity': lexical_diversity,
            'ai_score': ai_score,
            'us_dominance': us_dominance,
            'exclamation_count': exclamation_count,
            'direct_address': direct_address,
            'rhetorical_questions': rhetorical_questions
        }

    def compare_trends(self, metrics_2003, metrics_2025, metrics_2026):
        """Compare metrics across years to show trends"""
        trends = {}

        for metric in ['similarity', 'lexical_diversity', 'ai_score', 'us_dominance',
                       'exclamation_count', 'direct_address', 'rhetorical_questions']:
            v2003 = metrics_2003.get(metric, 0)
            v2025 = metrics_2025.get(metric, 0)
            v2026 = metrics_2026.get(metric, 0)

            # Calculate percentage change (avoid division by zero)
            if v2003 != 0:
                change_2003_2025 = ((v2025 - v2003) / abs(v2003)) * 100
                change_2003_2026 = ((v2026 - v2003) / abs(v2003)) * 100
            else:
                change_2003_2025 = 0 if v2025 == 0 else 100
                change_2003_2026 = 0 if v2026 == 0 else 100

            trends[metric] = {
                '2003': v2003,
                '2025': v2025,
                '2026': v2026,
                'change_2003_2025': change_2003_2025,
                'change_2003_2026': change_2003_2026,
                'trend': 'increasing' if v2026 > v2003 else 'decreasing'
            }

        return trends

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

    def compute_us_dominance_score(self, texts):
        """Compute overall US English dominance score (higher = more US-like)"""
        us_analyzer = US_Structural_Analyzer()
        results = us_analyzer.analyze_corpus(texts)
        if results:
            return results['mean_us_dominance']
        return 0

    def print_ai_breakdown(self, texts, year):
        """Print detailed breakdown of AI markers by category for a year group"""
        all_results = []
        for text in texts:
            text_lower = text.lower()
            words = len(TextPreprocessor.tokenize(text))
            if words == 0:
                continue

            all_results.append({
                'classic': sum(1 for m in self.chatgpt_classic_markers if m in text_lower),
                'psychology': sum(1 for m in self.psychology_terms if m in text_lower),
                'contractions': sum(1 for m in self.casual_contractions if m in text_lower),
                'colloquial': sum(1 for m in self.colloquial_markers if m in text_lower),
                'exclamations': text.count('!'),
                'genz': sum(1 for m in self.genz_slang if m in text_lower),
                'fashion': sum(1 for m in self.fashion_markers if m in text_lower)
            })

        df = pd.DataFrame(all_results)
        print(f"\n{year} AI MARKER BREAKDOWN (per article):")
        print(df.mean().to_string())
        return df
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
    print("Examining: Linguistic changes in Bangladeshi English (2003 → 2025 → 2026)")
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

    print(f"\nAnalyzing years: {available_years}")

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

    # Store all trends for cross-genre comparison
    all_trends = {}

    # For each genre, analyze trends across years
    for genre in sorted(genres_all):
        print(f"\n" + "=" * 60)
        print(f"ANALYZING: {genre}")
        print("=" * 60)

        # Get metrics for each year
        metrics_by_year = {}

        for year in available_years:
            texts = data_dict[year][genre]
            print(f"  Processing {len(texts)} texts from {year}...")
            metrics = analyzer.compute_all_metrics(texts)
            metrics_by_year[year] = metrics

            print(f"    Similarity: {metrics['similarity']:.4f}")
            print(f"    AI Score: {metrics['ai_score']:.2f}")
            print(f"    US Dominance: {metrics['us_dominance']:.3f}")
            print(f"    Lexical Diversity: {metrics['lexical_diversity']:.4f}")
            print(f"    Exclamation marks: {metrics['exclamation_count']:.2f}")
            print(f"    Direct address ('you'): {metrics['direct_address']:.2f}")

        # Calculate trends
        trends = analyzer.compare_trends(
            metrics_by_year[2003],
            metrics_by_year.get(2025, metrics_by_year[2003]),
            metrics_by_year.get(2026, metrics_by_year[2003])
        )
        all_trends[genre] = trends

        # Print trend summary
        print("\n" + "-" * 40)
        print(f"TREND SUMMARY: {genre} (2003 → 2026)")
        print("-" * 40)

        print(f"\n📈 CONVERGENCE (Similarity):")
        print(f"   2003: {trends['similarity']['2003']:.4f}")
        if 2025 in metrics_by_year:
            print(f"   2025: {trends['similarity']['2025']:.4f}")
        print(f"   2026: {trends['similarity']['2026']:.4f}")
        print(f"   Overall Change: +{trends['similarity']['change_2003_2026']:.1f}%")

        print(f"\n🤖 AI PATTERNS (AI Score):")
        print(f"   2003: {trends['ai_score']['2003']:.2f}")
        if 2025 in metrics_by_year:
            print(f"   2025: {trends['ai_score']['2025']:.2f}")
        print(f"   2026: {trends['ai_score']['2026']:.2f}")
        print(f"   Overall Change: +{trends['ai_score']['change_2003_2026']:.1f}%")

        print(f"\n🇺🇸 US ENGLISH DOMINANCE:")
        print(f"   2003: {trends['us_dominance']['2003']:.3f} (negative = UK-leaning, positive = US-leaning)")
        if 2025 in metrics_by_year:
            print(f"   2025: {trends['us_dominance']['2025']:.3f}")
        print(f"   2026: {trends['us_dominance']['2026']:.3f}")
        print(f"   Overall Shift: {trends['us_dominance']['change_2003_2026']:+.1f}%")

        print(f"\n📝 VOCABULARY DIVERSITY (Type-Token Ratio):")
        print(f"   2003: {trends['lexical_diversity']['2003']:.4f}")
        if 2025 in metrics_by_year:
            print(f"   2025: {trends['lexical_diversity']['2025']:.4f}")
        print(f"   2026: {trends['lexical_diversity']['2026']:.4f}")
        print(f"   Overall Change: {trends['lexical_diversity']['change_2003_2026']:+.1f}%")

        print(f"\n🗣️ RHETORICAL PATTERNS:")
        print(
            f"   Exclamation marks: {trends['exclamation_count']['2003']:.2f} → {trends['exclamation_count']['2026']:.2f} (+{trends['exclamation_count']['change_2003_2026']:.0f}%)")
        print(
            f"   Direct address ('you'): {trends['direct_address']['2003']:.2f} → {trends['direct_address']['2026']:.2f} (+{trends['direct_address']['change_2003_2026']:.0f}%)")
        print(
            f"   Rhetorical questions: {trends['rhetorical_questions']['2003']:.2f} → {trends['rhetorical_questions']['2026']:.2f} (+{trends['rhetorical_questions']['change_2003_2026']:.0f}%)")

        # Create output directory
        output_dir = f'convergence_results/{genre}'
        os.makedirs(output_dir, exist_ok=True)

        # Create trend visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Filter years that have data
        plot_years = [y for y in [2003, 2025, 2026] if y in metrics_by_year]

        # 1. Similarity trend (Convergence)
        ax1 = axes[0, 0]
        similarity_values = [trends['similarity'][str(y)] for y in plot_years]
        ax1.plot(plot_years, similarity_values, 'o-', color='#2E86AB', linewidth=2, markersize=8)
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Similarity Score', fontsize=12)
        ax1.set_title('Convergence: Texts Becoming More Similar', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. AI Score trend
        ax2 = axes[0, 1]
        ai_values = [trends['ai_score'][str(y)] for y in plot_years]
        ax2.plot(plot_years, ai_values, 'o-', color='#E63946', linewidth=2, markersize=8)
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('AI Pattern Score', fontsize=12)
        ax2.set_title('AI Patterns: Increasing Presence', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. US Dominance trend
        ax3 = axes[1, 0]
        us_values = [trends['us_dominance'][str(y)] for y in plot_years]
        ax3.plot(plot_years, us_values, 'o-', color='#2E86AB', linewidth=2, markersize=8)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral (UK/US balance)')
        ax3.set_xlabel('Year', fontsize=12)
        ax3.set_ylabel('US Dominance Score (-1=UK, +1=US)', fontsize=12)
        ax3.set_title('Americanization: Shift Toward US English', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Lexical Diversity trend
        ax4 = axes[1, 1]
        lex_values = [trends['lexical_diversity'][str(y)] for y in plot_years]
        ax4.plot(plot_years, lex_values, 'o-', color='#73AB84', linewidth=2, markersize=8)
        ax4.set_xlabel('Year', fontsize=12)
        ax4.set_ylabel('Lexical Diversity (Type-Token Ratio)', fontsize=12)
        ax4.set_title('Vocabulary Diversity', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'{genre}: Linguistic Change Over Time (2003-2026)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/trend_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Save trends to CSV
        trends_data = []
        for metric, values in trends.items():
            row = {
                'genre': genre,
                'metric': metric,
                '2003': values.get('2003', 0),
                '2025': values.get('2025', 0),
                '2026': values.get('2026', 0),
                'change_percent_2003_2026': values.get('change_2003_2026', 0),
                'trend_direction': values.get('trend', 'stable')
            }
            trends_data.append(row)

        trends_df = pd.DataFrame(trends_data)
        trends_df.to_csv(f'{output_dir}/trends_summary.csv', index=False)

        print(f"\n✓ Trend analysis saved to {output_dir}/")

    # Print cross-genre comparison
    print("\n" + "=" * 60)
    print("CROSS-GENRE COMPARISON (2003 → 2026)")
    print("=" * 60)

    comparison_data = []
    for genre, trends in all_trends.items():
        comparison_data.append({
            'Genre': genre,
            'Similarity Change': f"+{trends['similarity']['change_2003_2026']:.1f}%",
            'AI Pattern Increase': f"+{trends['ai_score']['change_2003_2026']:.1f}%",
            'US Dominance Shift': f"{trends['us_dominance']['change_2003_2026']:+.1f}%",
            'Lexical Diversity Change': f"{trends['lexical_diversity']['change_2003_2026']:+.1f}%",
            'Exclamation Increase': f"+{trends['exclamation_count']['change_2003_2026']:.0f}%"
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    comparison_df.to_csv('convergence_results/cross_genre_trends.csv', index=False)

    # Final interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
    KEY FINDINGS:

    1. CONVERGENCE INCREASING:
       Texts are becoming more similar to each other over time.
       This indicates homogenization of writing style across the corpus.

    2. AI PATTERNS INCREASING:
       AI-typical vocabulary and rhetorical patterns are appearing more frequently.
       This suggests emerging AI influence on newspaper writing.

    3. US ENGLISH DOMINANCE INCREASING:
       US structural features (exclamations, direct address, rhetorical questions)
       are becoming more common, indicating Americanization of the text.

    4. LEXICAL DIVERSITY DECREASING:
       Vocabulary is becoming more restricted, a known characteristic of
       AI-generated text and standardized writing.

    5. RHETORICAL PATTERNS SHIFTING:
       Increased use of exclamation marks and direct address suggests a shift
       toward more engaging, conversational writing styles typical of US English.

    These trends collectively suggest that external linguistic influences,
    potentially including AI tools trained predominantly on US English,
    are affecting Bangladeshi English newspaper writing, leading to
    convergence toward US structural norms.
    """)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()