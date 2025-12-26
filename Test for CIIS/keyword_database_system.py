import sqlite3
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
import logging

class KeywordDatabaseManager:
    """
    A comprehensive system to build and maintain a dynamic keyword database
    for detecting anti-India campaigns on digital platforms.
    """
    
    def __init__(self, db_path: str = "keyword_database.db"):
        self.db_path = db_path
        self.setup_database()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for tracking keyword additions and modifications."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('keyword_manager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main keywords table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                source TEXT,
                confidence_score REAL DEFAULT 0.5,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',
                usage_count INTEGER DEFAULT 0,
                verified BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Hashtags table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hashtags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hashtag TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                source TEXT,
                confidence_score REAL DEFAULT 0.5,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',
                usage_count INTEGER DEFAULT 0,
                verified BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Phrases table for multi-word expressions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS phrases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phrase TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                source TEXT,
                confidence_score REAL DEFAULT 0.5,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',
                usage_count INTEGER DEFAULT 0,
                verified BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Keyword relationships table for context analysis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS keyword_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                primary_keyword TEXT NOT NULL,
                related_keyword TEXT NOT NULL,
                relationship_type TEXT,
                strength REAL DEFAULT 1.0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Pattern templates for dynamic detection
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_regex TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT,
                weight REAL DEFAULT 1.0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def initialize_seed_keywords(self):
        """Initialize the database with seed keywords and hashtags."""
        
        # Seed keywords categorized by sentiment type
        seed_data = {
            'keywords': {
                'political_negative': [
                    'fascist', 'authoritarian', 'oppression', 'tyranny', 'dictatorship',
                    'persecution', 'genocide', 'apartheid', 'occupation', 'brutality'
                ],
                'religious_targeting': [
                    'hinduphobia', 'islamophobia', 'communalism', 'fundamentalism',
                    'extremism', 'terrorism', 'jihadi', 'hindutva', 'radical'
                ],
                'economic_negative': [
                    'poverty', 'corruption', 'scam', 'fraud', 'inequality',
                    'exploitation', 'underdeveloped', 'backward'
                ],
                'social_issues': [
                    'casteism', 'discrimination', 'violence', 'rape', 'harassment',
                    'misogyny', 'inequality', 'suppression'
                ],
                'geopolitical': [
                    'aggressor', 'invasion', 'violation', 'dispute', 'conflict',
                    'tension', 'threat', 'destabilizing'
                ]
            },
            'hashtags': {
                'campaign_tags': [
                    '#boycottindia', '#indiafails', '#stopindia', '#indiabad',
                    '#indiaterror', '#indiaoppression', '#kashmirsupport'
                ],
                'political_tags': [
                    '#modiout', '#bjpfails', '#indiadictatorship', '#savedemocracy',
                    '#indiaauthoritarian', '#rssextremism'
                ],
                'regional_tags': [
                    '#freekashmir', '#khalistan', '#independentkashmir',
                    '#kashmirgenocide', '#indiapakistan', '#chinaindia'
                ]
            },
            'phrases': {
                'propaganda_phrases': [
                    'india is fascist', 'hindu nationalism', 'indian occupation',
                    'modi regime', 'indian imperialism', 'brahmanical supremacy',
                    'indian terrorism', 'fake democracy', 'human rights violations'
                ],
                'regional_phrases': [
                    'kashmir under siege', 'free kashmir movement', 'indian army brutality',
                    'kashmir genocide', 'independent khalistan', 'sikh persecution'
                ]
            }
        }
        
        # Insert seed data
        for category, items in seed_data['keywords'].items():
            for keyword in items:
                self.add_keyword(keyword, category, 'seed_data', 0.7)
                
        for category, items in seed_data['hashtags'].items():
            for hashtag in items:
                self.add_hashtag(hashtag, category, 'seed_data', 0.7)
                
        for category, items in seed_data['phrases'].items():
            for phrase in items:
                self.add_phrase(phrase, category, 'seed_data', 0.7)
    
    def add_keyword(self, keyword: str, category: str, source: str = 'manual', 
                   confidence: float = 0.5, weight: float = 1.0) -> bool:
        """Add a new keyword to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO keywords 
                (keyword, category, weight, source, confidence_score, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (keyword.lower().strip(), category, weight, source, confidence, datetime.now()))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Added keyword: {keyword} (category: {category})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding keyword {keyword}: {str(e)}")
            return False
    
    def add_hashtag(self, hashtag: str, category: str, source: str = 'manual',
                   confidence: float = 0.5, weight: float = 1.0) -> bool:
        """Add a new hashtag to the database."""
        try:
            # Ensure hashtag starts with #
            if not hashtag.startswith('#'):
                hashtag = '#' + hashtag
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO hashtags 
                (hashtag, category, weight, source, confidence_score, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (hashtag.lower().strip(), category, weight, source, confidence, datetime.now()))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Added hashtag: {hashtag} (category: {category})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding hashtag {hashtag}: {str(e)}")
            return False
    
    def add_phrase(self, phrase: str, category: str, source: str = 'manual',
                  confidence: float = 0.5, weight: float = 1.0) -> bool:
        """Add a new phrase to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO phrases 
                (phrase, category, weight, source, confidence_score, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (phrase.lower().strip(), category, weight, source, confidence, datetime.now()))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Added phrase: {phrase} (category: {category})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding phrase {phrase}: {str(e)}")
            return False
    
    def extract_keywords_from_text(self, text: str, min_frequency: int = 3) -> List[str]:
        """
        Extract potential keywords from text using frequency analysis and patterns.
        """
        # Clean and tokenize text
        text = re.sub(r'[^\w\s#@]', ' ', text.lower())
        words = text.split()
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'may', 'might', 'can', 'this', 'that',
                     'these', 'those', 'a', 'an'}
        
        # Extract hashtags
        hashtags = re.findall(r'#\w+', text)
        
        # Extract mentions
        mentions = re.findall(r'@\w+', text)
        
        # Filter meaningful words (length > 3, not stop words)
        meaningful_words = [word for word in words 
                          if len(word) > 3 and word not in stop_words]
        
        # Count frequency
        word_freq = Counter(meaningful_words)
        
        # Extract frequent words
        frequent_words = [word for word, freq in word_freq.items() 
                         if freq >= min_frequency]
        
        return {
            'keywords': frequent_words,
            'hashtags': hashtags,
            'mentions': mentions
        }
    
    def auto_discover_keywords(self, posts: List[str], threshold: float = 0.6) -> Dict:
        """
        Automatically discover new keywords from a collection of posts.
        """
        all_extracted = {'keywords': [], 'hashtags': [], 'phrases': []}
        
        for post in posts:
            extracted = self.extract_keywords_from_text(post)
            all_extracted['keywords'].extend(extracted['keywords'])
            all_extracted['hashtags'].extend(extracted['hashtags'])
            
            # Extract potential phrases (2-3 word combinations)
            words = post.lower().split()
            for i in range(len(words) - 1):
                if len(words[i]) > 3 and len(words[i+1]) > 3:
                    phrase = f"{words[i]} {words[i+1]}"
                    all_extracted['phrases'].append(phrase)
                    
                if i < len(words) - 2 and len(words[i+2]) > 3:
                    phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                    all_extracted['phrases'].append(phrase)
        
        # Analyze frequency and suggest new keywords
        suggestions = {}
        
        for category in all_extracted:
            freq_analysis = Counter(all_extracted[category])
            # Suggest items that appear frequently but aren't in database
            suggestions[category] = []
            
            for item, count in freq_analysis.most_common(50):
                if not self.keyword_exists(item, category):
                    confidence = min(0.9, count / len(posts) + 0.1)
                    suggestions[category].append({
                        'term': item,
                        'frequency': count,
                        'confidence': confidence
                    })
        
        return suggestions
    
    def keyword_exists(self, term: str, table_type: str = 'keywords') -> bool:
        """Check if a keyword/hashtag/phrase already exists in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if table_type == 'hashtags':
            cursor.execute('SELECT COUNT(*) FROM hashtags WHERE hashtag = ?', (term,))
        elif table_type == 'phrases':
            cursor.execute('SELECT COUNT(*) FROM phrases WHERE phrase = ?', (term,))
        else:
            cursor.execute('SELECT COUNT(*) FROM keywords WHERE keyword = ?', (term,))
        
        exists = cursor.fetchone()[0] > 0
        conn.close()
        return exists
    
    def update_keyword_weights(self, performance_data: Dict):
        """
        Update keyword weights based on detection performance.
        performance_data: {keyword: {'hits': int, 'false_positives': int}}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for keyword, stats in performance_data.items():
            hits = stats.get('hits', 0)
            false_positives = stats.get('false_positives', 0)
            
            # Calculate new weight based on accuracy
            if hits + false_positives > 0:
                accuracy = hits / (hits + false_positives)
                new_weight = min(2.0, max(0.1, accuracy * 2))
                
                # Update in all relevant tables
                for table in ['keywords', 'hashtags', 'phrases']:
                    column = table[:-1] if table != 'phrases' else 'phrase'
                    cursor.execute(f'''
                        UPDATE {table} 
                        SET weight = ?, last_updated = ?, usage_count = usage_count + ?
                        WHERE {column} = ?
                    ''', (new_weight, datetime.now(), hits, keyword))
        
        conn.commit()
        conn.close()
        self.logger.info("Updated keyword weights based on performance data")
    
    def add_related_keywords(self, primary_keyword: str, related_terms: List[str],
                           relationship_type: str = 'similar'):
        """Add relationships between keywords for context analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for related_term in related_terms:
            cursor.execute('''
                INSERT OR REPLACE INTO keyword_relationships
                (primary_keyword, related_keyword, relationship_type, created_date)
                VALUES (?, ?, ?, ?)
            ''', (primary_keyword, related_term, relationship_type, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_contextual_keywords(self, base_keywords: List[str]) -> Set[str]:
        """Get related keywords based on existing relationships."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        related_keywords = set()
        for keyword in base_keywords:
            cursor.execute('''
                SELECT related_keyword FROM keyword_relationships 
                WHERE primary_keyword = ? AND strength > 0.5
            ''', (keyword,))
            
            related = cursor.fetchall()
            related_keywords.update([r[0] for r in related])
        
        conn.close()
        return related_keywords
    
    def analyze_emerging_patterns(self, recent_posts: List[str], 
                                days_back: int = 7) -> Dict:
        """
        Analyze recent posts to identify emerging keyword patterns.
        """
        # Extract all terms from recent posts
        all_terms = []
        for post in recent_posts:
            extracted = self.extract_keywords_from_text(post)
            all_terms.extend(extracted['keywords'])
            all_terms.extend(extracted['hashtags'])
        
        # Analyze frequency changes over time
        term_frequency = Counter(all_terms)
        
        # Get existing keywords for comparison
        existing_keywords = self.get_all_active_keywords()
        
        # Identify new emerging terms
        emerging_terms = []
        for term, freq in term_frequency.most_common(100):
            if term not in existing_keywords and freq > 2:
                emerging_terms.append({
                    'term': term,
                    'frequency': freq,
                    'emergence_score': freq / len(recent_posts),
                    'suggested_category': self.classify_term(term)
                })
        
        return {
            'emerging_terms': emerging_terms,
            'trend_analysis': self.analyze_trend_patterns(all_terms)
        }
    
    def classify_term(self, term: str) -> str:
        """Automatically classify a term into appropriate category."""
        term_lower = term.lower()
        
        # Political indicators
        political_indicators = ['modi', 'bjp', 'rss', 'government', 'regime', 'state']
        if any(indicator in term_lower for indicator in political_indicators):
            return 'political_negative'
        
        # Religious indicators
        religious_indicators = ['hindu', 'muslim', 'sikh', 'christian', 'religion']
        if any(indicator in term_lower for indicator in religious_indicators):
            return 'religious_targeting'
        
        # Regional indicators
        regional_indicators = ['kashmir', 'punjab', 'khalistan', 'china', 'pakistan']
        if any(indicator in term_lower for indicator in regional_indicators):
            return 'geopolitical'
        
        # Default category
        return 'general_negative'
    
    def analyze_trend_patterns(self, terms: List[str]) -> Dict:
        """Analyze trending patterns in keyword usage."""
        term_freq = Counter(terms)
        
        # Identify co-occurring terms
        co_occurrence = defaultdict(Counter)
        for i, term1 in enumerate(terms):
            for j in range(max(0, i-5), min(len(terms), i+6)):
                if i != j:
                    term2 = terms[j]
                    co_occurrence[term1][term2] += 1
        
        return {
            'most_frequent': term_freq.most_common(20),
            'co_occurrence_patterns': dict(co_occurrence)
        }
    
    def get_all_active_keywords(self) -> Set[str]:
        """Get all active keywords from all tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        all_keywords = set()
        
        # Get from keywords table
        cursor.execute('SELECT keyword FROM keywords WHERE status = "active"')
        all_keywords.update([r[0] for r in cursor.fetchall()])
        
        # Get from hashtags table
        cursor.execute('SELECT hashtag FROM hashtags WHERE status = "active"')
        all_keywords.update([r[0] for r in cursor.fetchall()])
        
        # Get from phrases table
        cursor.execute('SELECT phrase FROM phrases WHERE status = "active"')
        all_keywords.update([r[0] for r in cursor.fetchall()])
        
        conn.close()
        return all_keywords
    
    def get_weighted_keywords_by_category(self, category: str = None) -> Dict:
        """Get keywords with their weights, optionally filtered by category."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        result = {'keywords': {}, 'hashtags': {}, 'phrases': {}}
        
        for table, key in [('keywords', 'keyword'), ('hashtags', 'hashtag'), ('phrases', 'phrase')]:
            if category:
                cursor.execute(f'''
                    SELECT {key}, weight, confidence_score FROM {table} 
                    WHERE category = ? AND status = "active"
                ''', (category,))
            else:
                cursor.execute(f'''
                    SELECT {key}, weight, confidence_score FROM {table} 
                    WHERE status = "active"
                ''')
            
            for row in cursor.fetchall():
                result[table][row[0]] = {
                    'weight': row[1],
                    'confidence': row[2]
                }
        
        conn.close()
        return result
    
    def batch_import_keywords(self, file_path: str, source: str = 'import'):
        """Import keywords from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            imported_count = 0
            
            if 'keywords' in data:
                for category, keywords in data['keywords'].items():
                    for keyword_data in keywords:
                        if isinstance(keyword_data, str):
                            self.add_keyword(keyword_data, category, source)
                        else:
                            self.add_keyword(
                                keyword_data.get('term'),
                                category,
                                source,
                                keyword_data.get('confidence', 0.5),
                                keyword_data.get('weight', 1.0)
                            )
                        imported_count += 1
            
            if 'hashtags' in data:
                for category, hashtags in data['hashtags'].items():
                    for hashtag_data in hashtags:
                        if isinstance(hashtag_data, str):
                            self.add_hashtag(hashtag_data, category, source)
                        else:
                            self.add_hashtag(
                                hashtag_data.get('term'),
                                category,
                                source,
                                hashtag_data.get('confidence', 0.5),
                                hashtag_data.get('weight', 1.0)
                            )
                        imported_count += 1
            
            self.logger.info(f"Successfully imported {imported_count} terms from {file_path}")
            return imported_count
            
        except Exception as e:
            self.logger.error(f"Error importing from {file_path}: {str(e)}")
            return 0
    
    def export_keywords(self, file_path: str, include_stats: bool = True):
        """Export all keywords to JSON file."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            export_data = {'keywords': {}, 'hashtags': {}, 'phrases': {}}
            
            # Export keywords
            cursor.execute('''
                SELECT keyword, category, weight, confidence_score, usage_count, verified
                FROM keywords WHERE status = "active"
                ORDER BY category, weight DESC
            ''')
            
            for row in cursor.fetchall():
                category = row[1]
                if category not in export_data['keywords']:
                    export_data['keywords'][category] = []
                
                keyword_data = {
                    'term': row[0],
                    'weight': row[2],
                    'confidence': row[3]
                }
                
                if include_stats:
                    keyword_data.update({
                        'usage_count': row[4],
                        'verified': row[5]
                    })
                
                export_data['keywords'][category].append(keyword_data)
            
            # Export hashtags
            cursor.execute('''
                SELECT hashtag, category, weight, confidence_score, usage_count, verified
                FROM hashtags WHERE status = "active"
                ORDER BY category, weight DESC
            ''')
            
            for row in cursor.fetchall():
                category = row[1]
                if category not in export_data['hashtags']:
                    export_data['hashtags'][category] = []
                
                hashtag_data = {
                    'term': row[0],
                    'weight': row[2],
                    'confidence': row[3]
                }
                
                if include_stats:
                    hashtag_data.update({
                        'usage_count': row[4],
                        'verified': row[5]
                    })
                
                export_data['hashtags'][category].append(hashtag_data)
            
            # Export phrases
            cursor.execute('''
                SELECT phrase, category, weight, confidence_score, usage_count, verified
                FROM phrases WHERE status = "active"
                ORDER BY category, weight DESC
            ''')
            
            for row in cursor.fetchall():
                category = row[1]
                if category not in export_data['phrases']:
                    export_data['phrases'][category] = []
                
                phrase_data = {
                    'term': row[0],
                    'weight': row[2],
                    'confidence': row[3]
                }
                
                if include_stats:
                    phrase_data.update({
                        'usage_count': row[4],
                        'verified': row[5]
                    })
                
                export_data['phrases'][category].append(phrase_data)
            
            conn.close()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported keyword database to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to {file_path}: {str(e)}")
            return False
    
    def cleanup_database(self, days_inactive: int = 90):
        """Remove inactive or low-performing keywords."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_inactive)
        
        for table in ['keywords', 'hashtags', 'phrases']:
            # Mark low-weight, unused terms as inactive
            cursor.execute(f'''
                UPDATE {table} 
                SET status = 'inactive'
                WHERE weight < 0.2 AND usage_count = 0 
                AND last_updated < ? AND verified = FALSE
            ''', (cutoff_date,))
        
        conn.commit()
        
        # Log cleanup results
        cursor.execute('SELECT COUNT(*) FROM keywords WHERE status = "inactive"')
        inactive_keywords = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM hashtags WHERE status = "inactive"')
        inactive_hashtags = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM phrases WHERE status = "inactive"')
        inactive_phrases = cursor.fetchone()[0]
        
        conn.close()
        
        self.logger.info(f"Cleanup completed. Inactive items - Keywords: {inactive_keywords}, "
                        f"Hashtags: {inactive_hashtags}, Phrases: {inactive_phrases}")
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive statistics about the keyword database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        for table in ['keywords', 'hashtags', 'phrases']:
            cursor.execute(f'SELECT COUNT(*) FROM {table} WHERE status = "active"')
            active_count = cursor.fetchone()[0]
            
            cursor.execute(f'SELECT COUNT(*) FROM {table} WHERE verified = TRUE')
            verified_count = cursor.fetchone()[0]
            
            cursor.execute(f'SELECT AVG(weight) FROM {table} WHERE status = "active"')
            avg_weight = cursor.fetchone()[0] or 0
            
            cursor.execute(f'SELECT category, COUNT(*) FROM {table} WHERE status = "active" GROUP BY category')
            category_breakdown = dict(cursor.fetchall())
            
            stats[table] = {
                'active_count': active_count,
                'verified_count': verified_count,
                'average_weight': round(avg_weight, 3),
                'categories': category_breakdown
            }
        
        # Get relationship stats
        cursor.execute('SELECT COUNT(*) FROM keyword_relationships')
        relationship_count = cursor.fetchone()[0]
        
        stats['relationships'] = {
            'total_relationships': relationship_count
        }
        
        conn.close()
        return stats
    
    def search_keywords(self, search_term: str, search_type: str = 'all') -> List[Dict]:
        """Search for keywords in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        results = []
        search_pattern = f'%{search_term.lower()}%'
        
        if search_type in ['all', 'keywords']:
            cursor.execute('''
                SELECT keyword, category, weight, confidence_score, usage_count
                FROM keywords WHERE keyword LIKE ? AND status = "active"
                ORDER BY weight DESC, usage_count DESC
            ''', (search_pattern,))
            
            for row in cursor.fetchall():
                results.append({
                    'term': row[0],
                    'type': 'keyword',
                    'category': row[1],
                    'weight': row[2],
                    'confidence': row[3],
                    'usage_count': row[4]
                })
        
        if search_type in ['all', 'hashtags']:
            cursor.execute('''
                SELECT hashtag, category, weight, confidence_score, usage_count
                FROM hashtags WHERE hashtag LIKE ? AND status = "active"
                ORDER BY weight DESC, usage_count DESC
            ''', (search_pattern,))
            
            for row in cursor.fetchall():
                results.append({
                    'term': row[0],
                    'type': 'hashtag',
                    'category': row[1],
                    'weight': row[2],
                    'confidence': row[3],
                    'usage_count': row[4]
                })
        
        if search_type in ['all', 'phrases']:
            cursor.execute('''
                SELECT phrase, category, weight, confidence_score, usage_count
                FROM phrases WHERE phrase LIKE ? AND status = "active"
                ORDER BY weight DESC, usage_count DESC
            ''', (search_pattern,))
            
            for row in cursor.fetchall():
                results.append({
                    'term': row[0],
                    'type': 'phrase',
                    'category': row[1],
                    'weight': row[2],
                    'confidence': row[3],
                    'usage_count': row[4]
                })
        
        conn.close()
        return results
    
    def verify_keyword(self, term: str, table_type: str = 'keywords'):
        """Mark a keyword as manually verified."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        column = 'keyword' if table_type == 'keywords' else ('hashtag' if table_type == 'hashtags' else 'phrase')
        
        cursor.execute(f'''
            UPDATE {table_type} 
            SET verified = TRUE, last_updated = ?
            WHERE {column} = ?
        ''', (datetime.now(), term.lower()))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Verified {table_type[:-1]}: {term}")
    
    def add_pattern_template(self, regex_pattern: str, category: str, 
                           description: str, weight: float = 1.0):
        """Add regex pattern templates for dynamic keyword detection."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO pattern_templates
            (pattern_regex, category, description, weight)
            VALUES (?, ?, ?, ?)
        ''', (regex_pattern, category, description, weight))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Added pattern template for category: {category}")
    
    def detect_with_patterns(self, text: str) -> List[Dict]:
        """Use pattern templates to detect potential keywords in text."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT pattern_regex, category, weight FROM pattern_templates WHERE status = "active"')
        patterns = cursor.fetchall()
        conn.close()
        
        detections = []
        for pattern, category, weight in patterns:
            try:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    detections.append({
                        'matched_text': match.group(),
                        'category': category,
                        'weight': weight,
                        'start_pos': match.start(),
                        'end_pos': match.end()
                    })
            except re.error:
                self.logger.warning(f"Invalid regex pattern: {pattern}")
        
        return detections


class KeywordMaintenanceScheduler:
    """
    Automated maintenance and update scheduler for the keyword database.
    """
    
    def __init__(self, keyword_manager: KeywordDatabaseManager):
        self.keyword_manager = keyword_manager
        self.logger = logging.getLogger(__name__)
    
    def daily_maintenance(self, recent_posts: List[str]):
        """Run daily maintenance tasks."""
        self.logger.info("Starting daily keyword database maintenance")
        
        # 1. Analyze emerging patterns
        emerging_analysis = self.keyword_manager.analyze_emerging_patterns(recent_posts)
        
        # 2. Auto-suggest high-confidence new keywords
        for term_data in emerging_analysis['emerging_terms']:
            if term_data['emergence_score'] > 0.1 and len(term_data['term']) > 3:
                # Auto-add high-confidence terms
                if term_data['emergence_score'] > 0.3:
                    if term_data['term'].startswith('#'):
                        self.keyword_manager.add_hashtag(
                            term_data['term'], 
                            term_data['suggested_category'],
                            'auto_discovery',
                            term_data['emergence_score']
                        )
                    else:
                        self.keyword_manager.add_keyword(
                            term_data['term'], 
                            term_data['suggested_category'],
                            'auto_discovery',
                            term_data['emergence_score']
                        )
        
        # 3. Update keyword relationships based on co-occurrence
        self._update_relationships(emerging_analysis['trend_analysis']['co_occurrence_patterns'])
        
        self.logger.info("Daily maintenance completed")
    
    def weekly_maintenance(self):
        """Run weekly maintenance tasks."""
        self.logger.info("Starting weekly keyword database maintenance")
        
        # 1. Cleanup inactive keywords
        self.keyword_manager.cleanup_database(days_inactive=30)
        
        # 2. Generate performance report
        stats = self.keyword_manager.get_database_stats()
        self.logger.info(f"Database stats: {json.dumps(stats, indent=2)}")
        
        # 3. Export backup
        backup_file = f"keyword_backup_{datetime.now().strftime('%Y%m%d')}.json"
        self.keyword_manager.export_keywords(backup_file)
        
        self.logger.info("Weekly maintenance completed")
    
    def _update_relationships(self, co_occurrence_data: Dict):
        """Update keyword relationships based on co-occurrence patterns."""
        for primary_term, related_terms in co_occurrence_data.items():
            if len(primary_term) > 3:  # Only process meaningful terms
                top_related = [term for term, count in related_terms.most_common(5) 
                             if count > 2 and len(term) > 3]
                
                if top_related:
                    self.keyword_manager.add_related_keywords(
                        primary_term, top_related, 'co_occurrence'
                    )


# Example usage and testing
def main():
    """Example usage of the keyword database system."""
    
    # Initialize the system
    keyword_manager = KeywordDatabaseManager()
    
    # Initialize with seed data
    print("Initializing with seed keywords...")
    keyword_manager.initialize_seed_keywords()
    
    # Add some pattern templates
    keyword_manager.add_pattern_template(
        r'\b(india|indian)\s+(terror|terrorist|terrorism)\b',
        'anti_india_terrorism',
        'Patterns linking India with terrorism'
    )
    
    keyword_manager.add_pattern_template(
        r'\b(boycott|ban|stop)\s+(india|indian)\b',
        'boycott_campaigns',
        'Boycott campaign patterns'
    )
    
    # Example: Discover keywords from sample posts
    sample_posts = [
        "India is becoming increasingly authoritarian under Modi regime",
        "#BoycottIndia trending again after kashmir incident",
        "Human rights violations in Kashmir need international attention",
        "Modi government suppressing democracy and press freedom",
        "Hindu nationalism is rising in India, minorities at risk"
    ]
    
    print("\nAnalyzing sample posts for new keywords...")
    suggestions = keyword_manager.auto_discover_keywords(sample_posts)
    
    print("Suggested new keywords:")
    for category, items in suggestions.items():
        print(f"\n{category.upper()}:")
        for item in items[:5]:  # Show top 5
            print(f"  - {item['term']} (confidence: {item['confidence']:.2f})")
    
    # Get database statistics
    print("\nDatabase Statistics:")
    stats = keyword_manager.get_database_stats()
    for table, data in stats.items():
        print(f"\n{table.upper()}:")
        for key, value in data.items():
            print(f"  {key}: {value}")
    
    # Example search
    print("\nSearching for 'kashmir' related terms:")
    search_results = keyword_manager.search_keywords('kashmir')
    for result in search_results[:5]:
        print(f"  {result['term']} ({result['type']}) - Weight: {result['weight']}")
    
    # Setup maintenance scheduler
    scheduler = KeywordMaintenanceScheduler(keyword_manager)
    
    print("\nSystem initialized successfully!")
    print("The keyword database is ready for integration with your X monitoring system.")

if __name__ == "__main__":
    main()
                    