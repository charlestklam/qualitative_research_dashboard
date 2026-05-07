import re
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

class TextAnalyzer:
    """Class for analyzing text data and extracting insights."""
    
    def __init__(self):
        """Initialize the TextAnalyzer with necessary NLTK resources."""
        # Download NLTK resources explicitly
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Warning: Could not load stopwords: {str(e)}")
            self.stop_words = set()
            
        # Add common stopwords manually as fallback
        common_stops = {"a", "an", "the", "and", "or", "but", "if", "because", "as", "what", 
                       "while", "of", "at", "by", "for", "with", "about", "against", "between",
                       "into", "through", "during", "before", "after", "above", "below", "to",
                       "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
                       "further", "then", "once", "here", "there", "when", "where", "why", "how",
                       "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
                       "only", "own", "same", "so", "than", "too", "very", "s",
                       "t", "don", "now", "d", "ll", "m", "o", "re",
                       "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
                       "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
                       "10", "1371", "journal", "pone", "t001", "table 1", "g001", "fig",
                       "1","2","3","4","5","6","7","8","9"
                       }
        
        # Add these to our stopwords regardless
        self.stop_words.update(common_stops)
    
    def preprocess_text(self, text, remove_stopwords=True, remove_punctuation=True, lowercase=True):
        """
        Preprocess text for analysis
        
        Args:
            text (str): Input text to process
            remove_stopwords (bool): Whether to remove stop words
            remove_punctuation (bool): Whether to remove punctuation
            lowercase (bool): Whether to convert to lowercase
            
        Returns:
            list: List of processed tokens
        """
        if not isinstance(text, str):
            return []
            
        # Convert to lowercase if specified
        if lowercase:
            text = text.lower()
            
        # Remove punctuation if specified
        if remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
            
        # Simple word tokenization without relying on NLTK's word_tokenize
        # Split on whitespace and filter out empty strings
        tokens = [t for t in text.split() if t]
        
        # Remove stop words if specified
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
            
        return tokens
    
    def extract_ngrams(self, text, n=2, remove_stopwords=True, min_frequency=1):
        """
        Extract n-grams from text
        
        Args:
            text (str): Input text
            n (int): Size of n-grams to extract
            remove_stopwords (bool): Whether to remove stop words
            min_frequency (int): Minimum frequency threshold
            
        Returns:
            Counter: Counter object with n-grams and their frequencies
        """
        tokens = self.preprocess_text(text, remove_stopwords=remove_stopwords)
        
        if len(tokens) < n:
            return Counter()
            
        # Generate n-grams
        n_grams = list(ngrams(tokens, n))
        
        # Count n-gram frequencies
        n_gram_freq = Counter(n_grams)
        
        # Filter by minimum frequency
        if min_frequency > 1:
            n_gram_freq = Counter({gram: freq for gram, freq in n_gram_freq.items() if freq >= min_frequency})
            
        return n_gram_freq
    
    def extract_multiple_ngrams(self, texts, n=2, remove_stopwords=True, min_frequency=1):
        """
        Extract n-grams from multiple texts
        
        Args:
            texts (list): List of input texts
            n (int): Size of n-grams to extract
            remove_stopwords (bool): Whether to remove stop words
            min_frequency (int): Minimum frequency threshold
            
        Returns:
            Counter: Combined Counter object with n-grams and their frequencies
        """
        all_ngrams = Counter()
        
        for text in texts:
            text_ngrams = self.extract_ngrams(text, n, remove_stopwords, min_frequency=1)
            all_ngrams.update(text_ngrams)
            
        # Filter by minimum frequency
        if min_frequency > 1:
            all_ngrams = Counter({gram: freq for gram, freq in all_ngrams.items() if freq >= min_frequency})
            
        return all_ngrams
    
    @staticmethod
    def format_ngram(ngram):
        """
        Format an n-gram tuple into a readable string
        
        Args:
            ngram (tuple): N-gram tuple
            
        Returns:
            str: Formatted n-gram string
        """
        return ' '.join(ngram)
        
    def find_collocations(self, texts, keyword, left_window_size=3, right_window_size=3, top_n=20, remove_stopwords=True):
        """
        Find frequent collocations of a keyword in texts
        
        Args:
            texts (list): List of text strings to search in
            keyword (str): The keyword to find collocations for
            left_window_size (int): Number of words to consider on the left side of the keyword
            right_window_size (int): Number of words to consider on the right side of the keyword
            top_n (int): Number of top collocations to return
            remove_stopwords (bool): Whether to filter out stopwords from collocations.
            
        Returns:
            dict: Dictionary containing 'left' and 'right' collocations as (word, freq) tuples
        """
        if not texts or not keyword:
            return {"left": [], "right": []}
            
        # Make keyword lowercase for case-insensitive matching
        keyword = keyword.lower()
        left_collocations = Counter()
        right_collocations = Counter()
        
        for text in texts:
            if not isinstance(text, str):
                continue
                
            tokens = self.preprocess_text(text, remove_stopwords=False, remove_punctuation=True, lowercase=True)
            
            # Find keyword occurrences
            for i, token in enumerate(tokens):
                if token == keyword:
                    # Get left context
                    start = max(0, i - left_window_size)
                    left_context = tokens[start:i]
                    
                    # Get right context
                    end = min(len(tokens), i + right_window_size + 1)
                    right_context = tokens[i+1:end]
                    
                    for word in left_context:
                        if remove_stopwords and word in self.stop_words:
                            continue
                        if word != keyword:  # Avoid counting the keyword itself
                            left_collocations[word] += 1
                            
                    for word in right_context:
                        if remove_stopwords and word in self.stop_words:
                            continue
                        if word != keyword:  # Avoid counting the keyword itself
                            right_collocations[word] += 1
        
        # Get top collocations
        top_left = left_collocations.most_common(top_n)
        top_right = right_collocations.most_common(top_n)
        
        # Return as a simple dictionary that can be accessed by key
        results = {}
        results["left"] = top_left
        results["right"] = top_right
        return results
        
    def generate_concordance(self, texts, keyword, window_size=5, max_results=100, 
                            case_sensitive=False, whole_word=False):
        """
        Generate KWIC (Key Word In Context) concordance for a keyword
        
        Args:
            texts (list): List of text strings to search in
            keyword (str): The keyword to find contexts for
            window_size (int): Number of words to include on each side
            max_results (int): Maximum number of concordance lines to return
            case_sensitive (bool): Whether search should be case-sensitive
            whole_word (bool): Whether to match whole words only
            
        Returns:
            list: List of dictionaries with 'left', 'keyword', 'right' contexts
        """
        import re
        concordance_lines = []
        
        # Prepare search pattern
        if whole_word:
            pattern = r'\b(' + re.escape(keyword) + r')\b'
        else:
            pattern = r'(' + re.escape(keyword) + r')'
            
        # Compile the pattern with appropriate flags
        if case_sensitive:
            regex = re.compile(pattern)
        else:
            regex = re.compile(pattern, re.IGNORECASE)
        
        # Process each text
        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                continue
                
            # Find all matches
            for match in regex.finditer(text):
                # Get the actual matched text (preserving case)
                matched_word = match.group(1)
                
                # Get start and end positions
                start_pos = match.start()
                end_pos = match.end()
                
                # Get text before and after keyword
                left_context = text[max(0, start_pos - 100):start_pos].strip()
                right_context = text[end_pos:min(len(text), end_pos + 100)].strip()
                
                # Add to concordance lines
                concordance_lines.append({
                    'text_id': idx,
                    'left': left_context,
                    'keyword': matched_word,
                    'right': right_context
                })
                
                # Stop if we've reached the maximum number of results
                if len(concordance_lines) >= max_results:
                    break
        
        return concordance_lines
    
    @staticmethod
    def get_basic_stats(texts, include_top_words=True):
        """
        Get basic statistics about a collection of texts
        
        Args:
            texts (list): List of text strings
            include_top_words (bool): Whether to include top frequent words analysis
            
        Returns:
            dict: Dictionary of basic statistics
        """
        if not texts:
            return {
                "total_texts": 0,
                "avg_length": 0,
                "unique_words": 0
            }
            
        # Calculate total word count
        word_counts = [len(text.split()) for text in texts]
        total_words = sum(word_counts)
        
        # Calculate unique words
        all_words = ' '.join(texts).split()
        unique_words = len(set(all_words))
        
        result = {
            "total_texts": len(texts),
            "total_words": total_words,
            "avg_words_per_text": total_words / len(texts) if texts else 0,
            "unique_words": unique_words,
            "lexical_diversity": unique_words / total_words if total_words > 0 else 0
        }
        
        if include_top_words:
            # Get most frequent words (without stopwords)
            try:
                # Basic stopwords list to use even if NLTK data is not available
                basic_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
                                  'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
                                  'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
                                  'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
                                  'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                                  'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                                  'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                                  'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
                                  'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                                  'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
                                  'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
                                  'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
                                  'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
                                  't', 'can', 'will', 'just', 'don', 'should', 'now'}
                
                # Tokenize and clean words
                all_words_clean = []
                for word in all_words:
                    word = word.lower()
                    # Remove punctuation from word
                    word = ''.join(c for c in word if c.isalnum())
                    if len(word) > 1 and word not in basic_stopwords and word.isalpha():
                        all_words_clean.append(word)
                
                # Get frequency distribution
                from collections import Counter
                word_counts = Counter(all_words_clean)
                result["most_frequent_words"] = word_counts.most_common(20)
            except Exception:
                # If word frequency calculation fails, continue without it
                result["most_frequent_words"] = []
                
        return result

    def get_corpus_word_counts(self, texts, remove_stopwords=True):
        """
        Generates a Counter of all words in the entire corpus.
        Uses the same preprocessing as other methods.
        
        Args:
            texts (list): List of text strings
            remove_stopwords (bool): Whether to filter out stopwords
        
        Returns:
            Counter: A Counter object of word frequencies
        """
        all_words = Counter()
        for text in texts:
            # Use the class's own preprocessing method for consistency
            tokens = self.preprocess_text(text, 
                                         remove_stopwords=remove_stopwords, 
                                         remove_punctuation=True, 
                                         lowercase=True)
            all_words.update(tokens)
        return all_words
    