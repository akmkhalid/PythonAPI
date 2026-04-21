"""
AI Pattern Dictionary - For Detecting Post-AI English (2022-2026)
Organized chronologically by when patterns emerged
Essential for proving pre-AI vs post-AI linguistic shift
"""

AI_PATTERNS = {
    # ============================================================================
    # LEVEL 1: CLASSIC CHATGPT MARKERS (2022-2023)
    # These are the original AI tells from early ChatGPT
    # ============================================================================

    'academic_transitions': {
        'patterns': ['delve', 'delve into', 'crucial', 'paramount', 'pivotal',
                     'moreover', 'furthermore', 'nevertheless', 'nonetheless',
                     'in conclusion', 'to summarize', 'as previously mentioned'],
        'regex': r'\b(?:delve|delve into|crucial|paramount|pivotal|moreover|furthermore|nevertheless|nonetheless|in conclusion|to summarize|as previously mentioned)\b',
        'weight': 3.0,
        'era': '2022-2023'
    },

    'chatgpt_openings': {
        'patterns': ['it is worth noting', 'it is crucial to understand', 'have you ever wondered'],
        'regex': r'\b(?: it is worth noting|it is crucial to understand|have you ever wondered)\b',
        'weight': 3.0,
        'era': '2022-2023'
    },

    'ai_favorite_verbs': {
        'patterns': ['navigate', 'leverage', 'optimize', 'streamline', 'holistic',
                     'synergy', 'utilize', 'implement', 'facilitate', 'enhance'],
        'regex': r'\b(?:navigate|leverage|optimize|streamline|holistic|synergy|utilize|implement|facilitate|enhance)\b',
        'weight': 2.5,
        'era': '2022-2023'
    },

    'hedge_words': {
        'patterns': ['perhaps', 'maybe', 'it seems', 'it appears', 'it suggests',
                     'one might argue', 'it could be argued', 'some may say'],
        'regex': r'\b(?:perhaps|maybe|it seems|it appears|it suggests|one might argue|it could be argued|some may say)\b',
        'weight': 2.0,
        'era': '2022-2023'
    },

    # ============================================================================
    # LEVEL 2: AI LISTICLES & STRUCTURED CONTENT (2023-2024)
    # ============================================================================

    'list_introductions': {
        'patterns': ['here are', 'here is', 'below are', 'the following are',
                     'top 5', 'top 10', 'top 3', 'ways to', 'steps to', 'reasons why'],
        'regex': r'\b(?:here are|here is|below are|the following are|top 5|top 10|top 3|ways to|steps to|reasons why)\b',
        'weight': 2.0,
        'era': '2023-2024'
    },

    'list_conclusions': {
        'patterns': ['first and foremost', 'last but not least', 'in no particular order',
                     'without further ado', 'let\'s dive in', 'let\'s explore',
                     'here\'s why', 'here\'s how', 'the bottom line is'],
        'regex': r'\b(?:first and foremost|last but not least|in no particular order|without further ado|let\'s dive in|let\'s explore|here\'s why|here\'s how|the bottom line is)\b',
        'weight': 2.0,
        'era': '2023-2024'
    },

    # ============================================================================
    # LEVEL 3: PSYCHOLOGY & RELATIONSHIP TERMS (2023-2025)
    # AI-generated advice columns use these heavily
    # ============================================================================

    'relationship_dynamics': {
        'patterns': ['situationship', 'ghosting', 'breadcrumbing', 'love bombing',
                     'trauma bond', 'attachment style', 'emotional labor', 'mental load',
                     'red flags', 'green flags', 'ick', 'micro-cheating', 'orbiting',
                     'cloaking', 'paperclipping', 'submarining', 'stashing'],
        'regex': r'\b(?:situationship|ghosting|breadcrumbing|love bombing|trauma bond|attachment style|emotional labor|mental load|red flags|green flags|ick|micro-cheating|orbiting|cloaking|paperclipping|submarining|stashing)\b',
        'weight': 3.0,
        'era': '2023-2025'
    },

    'psychological_concepts': {
        'patterns': ['gaslighting', 'manipulation', 'toxicity', 'narcissist',
                     'boundaries', 'validation', 'emotional intelligence', 'self-awareness',
                     'coping mechanism', 'trigger warning', 'safe space', 'healing journey',
                     'inner child', 'shadow work', 'emotional regulation', 'self-care'],
        'regex': r'\b(?:gaslighting|manipulation|toxicity|narcissist|boundaries|validation|emotional intelligence|self-awareness|coping mechanism|trigger warning|safe space|healing journey|inner child|shadow work|emotional regulation|self-care)\b',
        'weight': 3.0,
        'era': '2023-2025'
    },

    'self_help_vocabulary': {
        'patterns': ['self-growth', 'personal development', 'mindfulness', 'authenticity',
                     'vulnerability', 'resilience', 'empowerment', 'purpose-driven'],
        'regex': r'\b(?:self-growth|personal development|mindfulness|authenticity|vulnerability|resilience|empowerment|purpose-driven)\b',
        'weight': 2.5,
        'era': '2023-2025'
    },

    # ============================================================================
    # LEVEL 4: BUSINESS & CORPORATE JARGON (2023-2026)
    # ============================================================================

    'corporate_phrases': {
        'patterns': ['circle back', 'touch base', 'deep dive', 'low hanging fruit',
                     'move the needle', 'think outside the box', 'win-win',
                     'bandwidth', 'capacity', 'actionable', 'deliverable',
                     'paradigm shift', 'core competency', 'value proposition',
                     'pain point', 'solutioning', 'onboarding', 'offboarding',
                     'scalable', 'robust', 'seamless'],
        'regex': r'\b(?:circle back|touch base|deep dive|low hanging fruit|move the needle|think outside the box|win-win|bandwidth|capacity|actionable|deliverable|paradigm shift|core competency|value proposition|pain point|solutioning|onboarding|offboarding|scalable|robust|seamless)\b',
        'weight': 2.0,
        'era': '2023-2026'
    },

    # ============================================================================
    # LEVEL 5: EMPATHETIC & THERAPEUTIC LANGUAGE (2023-2026)
    # ============================================================================

    'supportive_phrases': {
        'patterns': ['it\'s okay to', 'it\'s perfectly fine to', 'you deserve',
                     'you are enough', 'your feelings are valid', 'it\'s not your fault',
                     'give yourself permission', 'be kind to yourself', 'practice self-compassion',
                     'remember that you'],
        'regex': r'\b(?:it\'s okay to|it\'s perfectly fine to|you deserve|you are enough|your feelings are valid|it\'s not your fault|give yourself permission|be kind to yourself|practice self-compassion|remember that you)\b',
        'weight': 3.0,
        'era': '2023-2026'
    },

    'transitional_empathy': {
        'patterns': ['at the end of the day', 'that being said', 'having said that',
                     'with that in mind', 'on that note'],
        'regex': r'\b(?:at the end of the day|that being said|having said that|with that in mind|on that note)\b',
        'weight': 2.0,
        'era': '2023-2026'
    },

    # ============================================================================
    # LEVEL 6: OVERUSED MODIFIERS (AI favorites)
    # ============================================================================

    'ai_adverbs': {
        'patterns': ['absolutely', 'definitely', 'certainly', 'undoubtedly',
                     'essentially', 'virtually', 'practically', 'literally',
                     'truly', 'deeply', 'profoundly', 'remarkably', 'incredibly',
                     'extremely', 'exceptionally', 'particularly', 'significantly'],
        'regex': r'\b(?:absolutely|definitely|certainly|undoubtedly|essentially|virtually|practically|literally|truly|deeply|profoundly|remarkably|incredibly|extremely|exceptionally|particularly|significantly)\b',
        'weight': 1.5,
        'era': '2022-2026'
    },

      # ============================================================================
    # LEVEL 7: HALLUCINATION PHRASES (AI hedging when uncertain)
    # ============================================================================

    'vague_references': {
        'patterns': ['it is widely believed', 'many experts agree', 'research has shown',
                     'studies suggest', 'according to experts', 'it has been proven',
                     'it is commonly known', 'as the saying goes', 'as they say'],
        'regex': r'\b(?:it is widely believed|many experts agree|research has shown|studies suggest|according to experts|it has been proven|it is commonly known|as the saying goes|as they say)\b',
        'weight': 3.0,
        'era': '2022-2026'
    },

    'unnecessary_qualifiers': {
        'patterns': ['needless to say', 'suffice it to say', 'it goes without saying'],
        'regex': r'\b(?:needless to say|suffice it to say|it goes without saying)\b',
        'weight': 2.5,
        'era': '2022-2026'
    },

    # ============================================================================
    # LEVEL 8: BUZZWORDS & TRENDING TERMS (2023-2026)
    # ============================================================================

    'general_buzzwords': {
        'patterns': ['unprecedented', 'uncertain times', 'new normal', 'pivot',
                     'resilience', 'adaptability', 'innovation', 'disruption',
                     'sustainability', 'eco-friendly', 'mindful', 'intentional'],
        'regex': r'\b(?:unprecedented|uncertain times|new normal|pivot|resilience|adaptability|innovation|disruption|sustainability|eco-friendly|mindful|intentional)\b',
        'weight': 2.0,
        'era': '2023-2026'
    },

    'social_media_influenced': {
        'patterns': ['main character energy', 'glow up', 'healing era', 'villain era',
                     'it\'s giving', 'slay', 'iconic', 'aesthetic', 'vibe', 'energy'],
        'regex': r'\b(?:main character energy|glow up|healing era|villain era|it\'s giving|slay|iconic|aesthetic|vibe|energy)\b',
        'weight': 2.5,
        'era': '2024-2026'
    },

    'modern_dating': {
        'patterns': ['demisexual', 'sapiosexual', 'polyamory', 'ethical non-monogamy',
                     'talking stage', 'exclusive', 'label'],
        'regex': r'\b(?:demisexual|sapiosexual|polyamory|ethical non-monogamy|talking stage|exclusive|label)\b',
        'weight': 2.5,
        'era': '2023-2026'
    },

    # ============================================================================
    # LEVEL 9: GEN-Z SLANG (2024-2026)
    # ============================================================================

    'genz_slang': {
        'patterns': ['the ick', 'ick', 'endgame', 'main character', 'footloose and fancy-free',
                     'green pastures', 'glow up', 'healing era', 'villain era',
                     'it\'s giving', 'slay', 'iconic', 'aesthetic', 'vibe'],
        'regex': r'\b(?:the ick|ick|endgame|main character|footloose and fancy-free|green pastures|glow up|healing era|villain era|it\'s giving|slay|iconic|aesthetic|vibe)\b',
        'weight': 4.0,
        'era': '2024-2026'
    },

    # ============================================================================
    # LEVEL 10: FASHION & LIFESTYLE AI MARKERS (2023-2026)
    # ============================================================================

    'fashion_phrases': {
        'patterns': ['voilà', 'voila', 'sartorial', 'resplendent', 'effortless chic',
                     'elevate your look', 'match made in heaven', 'quiet luxury',
                     'steal the spotlight', 'make a statement', 'timeless', 'versatile',
                     'effortlessly', 'sophisticated', 'glamorous', 'chic', 'stunning',
                     'fabulous', 'sublime', 'heads will turn', 'turn heads',
                     'dressed to the nines', 'best foot forward', 'put your best foot forward',
                     'sartorially superior', 'fashion-forward', 'style staple',
                     'wardrobe essential', 'investment piece', 'capsule wardrobe', 'statement piece'],
        'regex': r'\b(?:voilà|voila|sartorial|resplendent|effortless chic|elevate your look|match made in heaven|quiet luxury|steal the spotlight|make a statement|timeless|versatile|effortlessly|sophisticated|glamorous|chic|stunning|fabulous|sublime|heads will turn|turn heads|dressed to the nines|best foot forward|put your best foot forward|sartorially superior|fashion-forward|style staple|wardrobe essential|investment piece|capsule wardrobe|statement piece)\b',
        'weight': 3.5,
        'era': '2023-2026'
    },

    # ============================================================================
    # LEVEL 11: INSPIRATIONAL & SELF-HELP PHRASES (2024-2026)
    # ============================================================================

    'inspirational_phrases': {
        'patterns': ['level up', 'game changer', 'next level', 'pro tip', 'life hack',
                     'work smarter not harder', 'the secret to', 'the key to',
                     'unlock your potential', 'step out of your comfort zone',
                     'push your boundaries', 'break the cycle', 'heal from within'],
        'regex': r'\b(?:level up|game changer|next level|pro tip|life hack|work smarter not harder|the secret to|the key to|unlock your potential|step out of your comfort zone|push your boundaries|break the cycle|heal from within)\b',
        'weight': 3.0,
        'era': '2024-2026'
    },

    # ============================================================================
    # LEVEL 12: NEGATION-CORRECTION & DEFINITIONAL CLOSURES (2024-2026)
    # These are signature AI "lesson summary" and "reframing" structures
    # ============================================================================

    'negation_correction': {
        'patterns': ['isn\'t kindness — it\'s erasure', 'isn\'t care — it\'s control',
                     'not about being wealthy — it\'s about', 'not X — it\'s Y'],
        'regex': r'\b(?:isn\'t|not)\s+[^.!?]{5,30}\s+[—,]\s+it\'?s\s+',
        'weight': 3.5,
        'era': '2024-2026'
    },

    'definitional_closure': {
        'patterns': ['that is what micro-dating is all about', 'that is what financial compatibility is all about',
                     'that exactly is what', 'that is what ... is all about'],
        'regex': r'\bthat\s+(?:exactly\s+)?is\s+what\s+.{1,40}\s+is\s+all\s+about\b',
        'weight': 3.0,
        'era': '2024-2026'
    },

    'truth_announcement': {
        'patterns': ['here\'s the truth though', 'the truth is', 'truth though'],
        'regex': r'\b(?:here\'s\s+the\s+truth|the\s+truth\s+is|truth\s+though)\b',
        'weight': 2.5,
        'era': '2023-2026'
    },

    'concession_softeners': {
        'patterns': ['although it may not sound like much', 'at first it may be disguised as',
                     'with a little awareness', 'it doesn\'t mean doom'],
        'regex': r'\b(?:although\s+it\s+may\s+not|at\s+first\s+it\s+may\s+be|with\s+a\s+little\s+awareness|it\s+doesn\'t\s+mean\s+doom)\b',
        'weight': 2.0,
        'era': '2023-2026'
    },

    # ============================================================================
    # LEVEL 13: RHETORICAL QUESTIONS & SIMULATED INTIMACY (2024-2026)
    # AI uses these to fake engagement and conversational tone
    # ============================================================================

    'rhetorical_questions': {
        'patterns': ['why does it matter?', 'what makes you', 'how do they get to decide',
                     'why not?', 'then there is the bigger discussion on'],
        'regex': r'\b(?:why\s+does\s+it\s+matter\?|what\s+makes\s+you|how\s+do\s+they\s+get\s+to\s+decide|why\s+not\?|then\s+there\s+is\s+the\s+bigger\s+discussion\s+on)\b',
        'weight': 2.5,
        'era': '2024-2026'
    },

    'emotional_validation_new': {
        'patterns': ['it\'s not just disappointing; it\'s a heartbreak',
                     'no relationship built on fear can ever be healthy'],
        'regex': r'\b(?:it\'s\s+not\s+just\s+[^;]{5,30};\s+it\'s|no\s+[a-z]{3,30}\s+built\s+on\s+[a-z]{3,20}\s+can\s+ever\s+be)\b',
        'weight': 2.0,
        'era': '2024-2026'
    },
    # ============================================================================
    # LEVEL 14: FASHION REVIVAL & HISTORICAL CONTRAST (2024-2026)
    # AI fashion writing tropes
    # ============================================================================

    'fashion_revival': {
        'patterns': ['are back and', 'is back and', 'have never been more', 'has never been more'],
        'regex': r'\b(?:are back and|is back and|have never been more|has never been more)\b',
        'weight': 2.0,
        'era': '2024-2026'
    },

    'far_cry_from': {
        'patterns': ['a far cry from'],
        'regex': r'\ba far cry from\b',
        'weight': 2.5,
        'era': '2023-2026'
    },

    'there_you_have_it': {
        'patterns': ['so there you have it', 'there you have it'],
        'regex': r'\bso,? there you have it\b',
        'weight': 3.0,
        'era': '2023-2026'
    },

    'of_yore': {
        'patterns': ['of yore'],
        'regex': r'\bof yore\b',
        'weight': 2.0,
        'era': '2023-2026'
    },

    'clothing_anthropomorphism': {
        'patterns': ['let the ruffles do the talking', 'let the clothes speak'],
        'regex': r'\blet the [a-z]{3,25} do the talking\b',
        'weight': 2.0,
        'era': '2024-2026'
    },
    # ============================================================================
    # LEVEL 15: DISCLAIMERS, EPIPHANIES & READER ADDRESS (2024-2026)
    # AI's self-aware rhetorical devices and direct addresses
    # ============================================================================

    'risk_of_being_stoned': {
        'patterns': ['at the risk of being stoned by'],
        'regex': r'\bat the risk of being (?:stoned|crucified|judged|criticized) by\b',
        'weight': 2.5,
        'era': '2024-2026'
    },

    'high_time': {
        'patterns': ['is it not high time we'],
        'regex': r'\bIs it not high time (?:we|society|people)\b',
        'weight': 2.5,
        'era': '2023-2026'
    },

    'only_time_will_tell': {
        'patterns': ['only time will tell'],
        'regex': r'\bOnly time will tell\b',
        'weight': 2.0,
        'era': '2022-2026'
    },

    'sinking_feeling': {
        'patterns': ['you know that sinking feeling'],
        'regex': r'\bYou know that (?:sinking|nagging|strange|familiar) feeling\b',
        'weight': 2.5,
        'era': '2024-2026'
    },

    'not_to_say': {
        'patterns': ['all of this is not to say that'],
        'regex': r'\bAll of this is not to say that\b',
        'weight': 2.0,
        'era': '2023-2026'
    },

    'beside_the_point': {
        'patterns': ['that is beside the point'],
        'regex': r'\bthat is beside the point\b',
        'weight': 2.0,
        'era': '2023-2026'
    },

    'two_birds': {
        'patterns': ['two birds killed with one stone', 'two birds with one stone'],
        'regex': r'\bTwo birds (?:killed|with) one stone\b',
        'weight': 1.5,
        'era': '2023-2026'
    },

    'we_present_to_you': {
        'patterns': ['we present to you'],
        'regex': r'\bwe present to you\b',
        'weight': 2.5,
        'era': '2023-2026'
    },

    'good_luck_with_that': {
        'patterns': ['good luck with that'],
        'regex': r'\bGood luck with that!\b',
        'weight': 2.0,
        'era': '2024-2026'
    },

    'that_day_i_realised': {
        'patterns': ['that day i realised'],
        'regex': r'\bThat day I realised\b',
        'weight': 2.0,
        'era': '2023-2026'
    },

    'not_a_hyperbole': {
        'patterns': ['i do not think it\'s a hyperbole to say'],
        'regex': r'\bI do not think it\'?s a hyperbole to say\b',
        'weight': 2.5,
        'era': '2024-2026'
    },

    'that_has_me_thinking': {
        'patterns': ['that has me thinking'],
        'regex': r'\bthat has me thinking:\b',
        'weight': 2.0,
        'era': '2024-2026'
    },

    'i_hear_you_ask': {
        'patterns': ['i hear you ask'],
        'regex': r'\bI hear you ask\b',
        'weight': 2.5,
        'era': '2024-2026'
    },

    'not_a_small_thing': {
        'patterns': ['this i believe is not a small thing'],
        'regex': r'\bthis, I believe, is not a (?:small|minor) thing\b',
        'weight': 2.5,
        'era': '2024-2026'
    },
    # ============================================================================
    # LEVEL 16: FASHION CLICHÉS & ABSOLUTE STATEMENTS (2023-2026)
    # AI's favorite fashion writing tropes and universal claims
    # ============================================================================

    'where_does_one_begin': {
        'patterns': ['where does one begin with'],
        'regex': r'\bWhere does one begin with\b',
        'weight': 2.5,
        'era': '2023-2026'
    },

    'out_with_old_in_with_new': {
        'patterns': ['out with the old and in with the new'],
        'regex': r'\bOut with the old and in with the new\b',
        'weight': 2.0,
        'era': '2023-2026'
    },

    'crowning_glory': {
        'patterns': ['crowning glory', 'makes or breaks a look'],
        'regex': r'\b(?:crowning glory|makes or breaks a look)\b',
        'weight': 2.0,
        'era': '2023-2026'
    },

    'thing_of_beauty': {
        'patterns': ['a thing of beauty is a joy forever'],
        'regex': r'\ba thing of beauty is a joy forever\b',
        'weight': 2.5,
        'era': '2023-2026'
    },

    'has_our_back': {
        'patterns': ['has always had our back'],
        'regex': r'\bhas always had our back\b',
        'weight': 2.0,
        'era': '2024-2026'
    },

    'all_the_rage': {
        'patterns': ['is all the rage'],
        'regex': r'\bis all the rage\b',
        'weight': 2.5,
        'era': '2023-2026'
    },

    'crowd_favourite': {
        'patterns': ['a crowd favourite'],
        'regex': r'\ba crowd favourite\b',
        'weight': 2.0,
        'era': '2023-2026'
    },

    'well_known_fact': {
        'patterns': ['it is a well-known fact that'],
        'regex': r'\bit is a well-known fact that\b',
        'weight': 2.5,
        'era': '2022-2026'
    },

    'dream_come_true': {
        'patterns': ['a dream come true'],
        'regex': r'\ba dream come true\b',
        'weight': 2.0,
        'era': '2023-2026'
    },

    'good_news_is': {
        'patterns': ['the good news is that'],
        'regex': r'\bThe good news is that\b',
        'weight': 2.5,
        'era': '2023-2026'
    },

    'think_of_it_as': {
        'patterns': ['think of it as'],
        'regex': r'\bThink of it as\b',
        'weight': 2.0,
        'era': '2022-2026'
    },

    'non_negotiable': {
        'patterns': ['is non-negotiable'],
        'regex': r'\bis non-negotiable\b',
        'weight': 2.5,
        'era': '2023-2026'
    },

    'perfect_reward': {
        'patterns': ['the perfect reward'],
        'regex': r'\bthe perfect reward\b',
        'weight': 2.0,
        'era': '2023-2026'
    },

    'love_affair_for_books': {
        'patterns': ['a love affair for the books'],
        'regex': r'\ba love affair for the books\b',
        'weight': 2.5,
        'era': '2024-2026'
    },

    'cherry_on_top': {
        'patterns': ['the cherry on top'],
        'regex': r'\bthe cherry on top\b',
        'weight': 2.0,
        'era': '2023-2026'
    },

    'universal_fact': {
        'patterns': ['a universal fact that'],
        'regex': r'\ba universal fact that\b',
        'weight': 2.5,
        'era': '2023-2026'
    },

    'cyclical_in_nature': {
        'patterns': ['cyclical in nature'],
        'regex': r'\bcyclical in nature\b',
        'weight': 2.5,
        'era': '2023-2026'
    },

    'at_long_last': {
        'patterns': ['at long last'],
        'regex': r'\bat long last!\b',
        'weight': 2.0,
        'era': '2023-2026'
    },

    'order_of_the_day': {
        'patterns': ['the order of the day'],
        'regex': r'\bthe order of the day\b',
        'weight': 2.0,
        'era': '2023-2026'
    },

    'testament_to': {
        'patterns': ['a testament to'],
        'regex': r'\ba testament to\b',
        'weight': 2.5,
        'era': '2023-2026'
    }
}

# ============================================================================
# PRE-AI vs POST-AI THRESHOLDS (For classification)
# ============================================================================

AI_THRESHOLDS = {
    'similarity_increase': {
        'pre_ai_range': (0.04, 0.07),
        'post_ai_range': (0.07, 0.10),
        'description': 'TF-IDF cosine similarity between texts'
    },
    'lexical_diversity': {
        'pre_ai_range': (0.25, 0.35),
        'post_ai_range': (0.15, 0.25),
        'description': 'Type-Token Ratio (lower = more repetitive = more AI)'
    },
    'burstiness': {
        'pre_ai_range': (0.8, 1.2),
        'post_ai_range': (0.4, 0.7),
        'description': 'Sentence length variance (lower = smoother = more AI)'
    },
    'connective_frequency': {
        'pre_ai_range': (20, 40),
        'post_ai_range': (50, 100),
        'description': 'Connectives per 10,000 words (higher = more AI)'
    },
    'americanization': {
        'pre_ai_range': (-0.2, 0.1),
        'post_ai_range': (0.2, 0.5),
        'description': 'US dominance score (negative = UK, positive = US)'
    }
}
