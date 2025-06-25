from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import logging
import time
from typing import List, Dict, Optional, Tuple
import traceback
from dataclasses import asdict
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import pymysql
from pymysql.err import Error as PyMySQLError

# Import our modified PSO algorithm
from modified_pso_algorithm import (
    Item, CustomerPreference, RecommendationResult,
    CustomerRecommendationPSO, create_customer_preference
)

# Flask App Configuration
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommendation_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# MySQL Configuration
MYSQL_CONFIG = {
    'host': 'jgn28.h.filess.io',
    'database': 'aspoo_youholdmap',
    'user': 'aspoo_youholdmap',
    'password': '5a2c3dbad3cbe97e83ac711947653544f0f3d48a',
    'port': 3307
}

def load_items_from_mysql() -> List[Item]:
    """Load items from MySQL database"""
    items = []
    try:
        connection = pymysql.connect(
            host=MYSQL_CONFIG['host'],
            database=MYSQL_CONFIG['database'],
            user=MYSQL_CONFIG['user'],
            password=MYSQL_CONFIG['password'],
            port=MYSQL_CONFIG['port'],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        with connection.cursor() as cursor:
            query = """
                SELECT id, nama_barang, kategori_umum, harga_umum, berat, produsen, 
                       bahan_dasar, basah_kering, rasa
                FROM barang
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            
            for row in rows:
                items.append(Item(
                    id=row['id'],
                    name=row['nama_barang'],
                    category=row['kategori_umum'] or 'Unknown',
                    price=float(row['harga_umum']),
                    weight=float(row['berat']),
                    store=row['produsen'] or '',
                    bahan_dasar=row['bahan_dasar'] or '',
                    basah_kering=row['basah_kering'] or '',
                    rasa=row['rasa'] or '',
                    value=float(row['harga_umum']) / max(float(row['berat']), 1e-6)  # Calculate value dynamically
                ))
        
        connection.close()
        return items
        
    except PyMySQLError as e:
        logger.error(f"Error connecting to MySQL: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading items from MySQL: {str(e)}")
        return []

class AdvancedDiversitySelector:
    """Advanced diversity selection with multiple algorithms"""
    
    def __init__(self, diversity_weight: float = 0.6):
        self.diversity_weight = diversity_weight
        
    def select_diverse_recommendations(self, 
                                     candidates: List[RecommendationResult],
                                     target_count: int = 10) -> List[RecommendationResult]:
        if len(candidates) <= target_count:
            return candidates
        
        candidates_sorted = sorted(candidates, key=lambda x: x.fitness_score, reverse=True)
        selected = [candidates_sorted.pop(0)]
        
        while len(selected) < target_count and candidates_sorted:
            best_candidate = None
            best_score = -float('inf')
            best_idx = -1
            
            for idx, candidate in enumerate(candidates_sorted):
                diversity_score = self._calculate_multi_diversity_score(candidate, selected)
                combined_score = (1 - self.diversity_weight) * candidate.fitness_score + \
                               self.diversity_weight * diversity_score * 1000
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
                    best_idx = idx
            
            if best_candidate:
                selected.append(best_candidate)
                candidates_sorted.pop(best_idx)
            else:
                break
        
        return selected
    
    def _calculate_multi_diversity_score(self, 
                                       candidate: RecommendationResult,
                                       selected: List[RecommendationResult]) -> float:
        if not selected:
            return 1.0
        
        candidate_items = set(item.id for item in candidate.items)
        diversity_scores = []
        
        for existing in selected:
            existing_items = set(item.id for item in existing.items)
            jaccard_diversity = self._jaccard_distance(candidate_items, existing_items)
            category_diversity = self._category_diversity(candidate, existing)
            price_diversity = self._price_range_diversity(candidate, existing)
            weight_diversity = self._weight_range_diversity(candidate, existing)
            store_diversity = self._store_diversity(candidate, existing)
            
            combined_diversity = (
                0.4 * jaccard_diversity +
                0.2 * category_diversity +
                0.15 * price_diversity +
                0.15 * weight_diversity +
                0.1 * store_diversity
            )
            diversity_scores.append(combined_diversity)
        
        return min(diversity_scores)
    
    def _jaccard_distance(self, set1: set, set2: set) -> float:
        if not set1 or not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return 1 - (intersection / union) if union > 0 else 1.0
    
    def _category_diversity(self, rec1: RecommendationResult, rec2: RecommendationResult) -> float:
        cat1 = set(item.category for item in rec1.items)
        cat2 = set(item.category for item in rec2.items)
        return self._jaccard_distance(cat1, cat2)
    
    def _price_range_diversity(self, rec1: RecommendationResult, rec2: RecommendationResult) -> float:
        if not rec1.items or not rec2.items:
            return 1.0
        avg_price1 = rec1.total_price / len(rec1.items)
        avg_price2 = rec2.total_price / len(rec2.items)
        max_price1 = max(item.price for item in rec1.items)
        max_price2 = max(item.price for item in rec2.items)
        max_possible_diff = max(max_price1, max_price2)
        if max_possible_diff == 0:
            return 0.0
        price_diff = abs(avg_price1 - avg_price2)
        return min(price_diff / max_possible_diff, 1.0)
    
    def _weight_range_diversity(self, rec1: RecommendationResult, rec2: RecommendationResult) -> float:
        if not rec1.items or not rec2.items:
            return 1.0
        avg_weight1 = rec1.total_weight / len(rec1.items)
        avg_weight2 = rec2.total_weight / len(rec2.items)
        max_weight1 = max(item.weight for item in rec1.items)
        max_weight2 = max(item.weight for item in rec2.items)
        max_possible_diff = max(max_weight1, max_weight2)
        if max_possible_diff == 0:
            return 0.0
        weight_diff = abs(avg_weight1 - avg_weight2)
        return min(weight_diff / max_possible_diff, 1.0)
    
    def _store_diversity(self, rec1: RecommendationResult, rec2: RecommendationResult) -> float:
        stores1 = set(item.store for item in rec1.items if hasattr(item, 'store'))
        stores2 = set(item.store for item in rec2.items if hasattr(item, 'store'))
        return self._jaccard_distance(stores1, stores2)

class RecommendationService:
    def __init__(self):
        self.items = []
        self.diversity_selector = AdvancedDiversitySelector(diversity_weight=0.6)
        self.optimizer = None
        self.load_data()
        
    def load_data(self):
        try:
            logger.info("Loading data from MySQL database (barang table)")
            self.items = load_items_from_mysql()
            if not self.items:
                logger.error("No items loaded from database")
                return
            logger.info(f"Loaded {len(self.items)} items successfully")
            categories = set(item.category for item in self.items if item.category != 'Unknown')
            stores = set(item.store for item in self.items if item.store)
            logger.info(f"Categories: {len(categories)}")
            logger.info(f"Produsen: {len(stores)}")
            logger.info(f"Price range: Rp {min(item.price for item in self.items):,.0f} - "
                       f"Rp {max(item.price for item in self.items):,.0f}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            self.items = []
    
    def generate_recommendations(self, customer_request: Dict) -> Dict:
        try:
            validation_error = self._validate_request(customer_request)
            if validation_error:
                return {"error": validation_error, "status": "failed"}
            
            customer_pref = create_customer_preference(customer_request)
            desired_filters = customer_request.get('desired_filters', {})
            unwanted_filters = customer_request.get('unwanted_filters', {})
            
            filtered_items = self._filter_items(customer_pref, desired_filters, unwanted_filters)
            
            if not filtered_items:
                return {
                    "error": "No items found matching your criteria",
                    "status": "failed",
                    "suggestions": "Try relaxing desired filters or removing unwanted filters"
                }
            
            logger.info(f"Filtered to {len(filtered_items)} items for optimization")
            
            self.optimizer = CustomerRecommendationPSO(
                n_particles=60,
                max_iterations=300,
                target_recommendations=25,
                diversity_threshold=0.4
            )
            
            start_time = time.time()
            candidate_recommendations = self._generate_candidate_pool(filtered_items, customer_pref, desired_filters)
            final_recommendations = self.diversity_selector.select_diverse_recommendations(
                candidate_recommendations, target_count=10
            )
            end_time = time.time()

            # Verify recommendations match filters
            valid_recommendations = self._verify_recommendations(final_recommendations, desired_filters, unwanted_filters)
            
            if not valid_recommendations:
                return {
                    "error": "No valid recommendations generated matching all filter criteria",
                    "status": "failed",
                    "suggestions": "Try relaxing desired filters or removing unwanted filters"
                }
            
            response = {
                "status": "success",
                "generation_time": round(end_time - start_time, 2),
                "total_candidates_generated": len(candidate_recommendations),
                "recommendations_returned": len(valid_recommendations),
                "customer_preferences": {
                    "budget": customer_pref.budget,
                    "weight_capacity": customer_pref.weight_capacity,
                    "preferred_categories": customer_pref.preferred_categories
                },
                "dataset_info": {
                    "total_items_available": len(self.items),
                    "items_matching_criteria": len(filtered_items)
                },
                "recommendations": [rec.to_dict() for rec in valid_recommendations]
            }
            
            logger.info(f"Generated {len(valid_recommendations)} valid recommendations in {end_time - start_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "error": f"Internal server error: {str(e)}",
                "status": "failed"
            }
    
    def _validate_request(self, request: Dict) -> Optional[str]:
        required_fields = ['budget', 'weight_capacity']
        for field in required_fields:
            if field not in request:
                return f"Missing required field: {field}"
            try:
                value = float(request[field])
                if value <= 0:
                    return f"{field} must be positive"
            except (ValueError, TypeError):
                return f"{field} must be a valid number"
        if 'preferred_categories' in request and not isinstance(request['preferred_categories'], list):
            return "preferred_categories must be a list"
        return None
    
    def _filter_items(self, customer_pref: CustomerPreference, desired_filters: Dict, unwanted_filters: Dict) -> List[Item]:
        filtered = []
        # Track item IDs that match each desired filter type
        desired_filter_matches = {
            'kategori_umum': set(),
            'bahan_dasar': set(),
            'basah_kering': set(),
            'rasa': set(),
            'produsen': set(),
            'nama_barang': set()
        }
        
        for item in self.items:
            if item.price > customer_pref.budget or item.weight > customer_pref.weight_capacity:
                continue
            
            # Unwanted filters: NONE should match
            if unwanted_filters.get('kategori_umum') and item.category in unwanted_filters['kategori_umum']:
                continue
            if unwanted_filters.get('bahan_dasar') and item.bahan_dasar in unwanted_filters['bahan_dasar']:
                continue
            if unwanted_filters.get('basah_kering') and item.basah_kering in unwanted_filters['basah_kering']:
                continue
            if unwanted_filters.get('rasa') and item.rasa in unwanted_filters['rasa']:
                continue
            if unwanted_filters.get('produsen') and item.store in unwanted_filters['produsen']:
                continue
            if unwanted_filters.get('nama_barang') and item.name in unwanted_filters['nama_barang']:
                continue
            
            # Track matches for desired filters using item IDs
            if desired_filters.get('kategori_umum') and item.category in desired_filters['kategori_umum']:
                desired_filter_matches['kategori_umum'].add(item.id)
            if desired_filters.get('bahan_dasar') and item.bahan_dasar in desired_filters['bahan_dasar']:
                desired_filter_matches['bahan_dasar'].add(item.id)
            if desired_filters.get('basah_kering') and item.basah_kering in desired_filters['basah_kering']:
                desired_filter_matches['basah_kering'].add(item.id)
            if desired_filters.get('rasa') and item.rasa in desired_filters['rasa']:
                desired_filter_matches['rasa'].add(item.id)
            if desired_filters.get('produsen') and item.store in desired_filters['produsen']:
                desired_filter_matches['produsen'].add(item.id)
            if desired_filters.get('nama_barang') and item.name in desired_filters['nama_barang']:
                desired_filter_matches['nama_barang'].add(item.id)
            
            filtered.append(item)
        
        # Ensure at least one item exists for each non-empty desired filter
        for filter_type, matches in desired_filters.items():
            if matches and not desired_filter_matches[filter_type]:
                logger.warning(f"No items found matching desired filter {filter_type}: {matches}")
                return []
        
        return filtered
    
    def _verify_recommendations(self, recommendations: List[RecommendationResult], desired_filters: Dict, unwanted_filters: Dict) -> List[RecommendationResult]:
        """Verify that recommendations strictly adhere to desired and unwanted filters"""
        valid_recommendations = []
        
        for rec in recommendations:
            # Check unwanted filters
            has_unwanted = False
            for item in rec.items:
                if unwanted_filters.get('kategori_umum') and item.category in unwanted_filters['kategori_umum']:
                    has_unwanted = True
                    break
                if unwanted_filters.get('bahan_dasar') and item.bahan_dasar in unwanted_filters['bahan_dasar']:
                    has_unwanted = True
                    break
                if unwanted_filters.get('basah_kering') and item.basah_kering in unwanted_filters['basah_kering']:
                    has_unwanted = True
                    break
                if unwanted_filters.get('rasa') and item.rasa in unwanted_filters['rasa']:
                    has_unwanted = True
                    break
                if unwanted_filters.get('produsen') and item.store in unwanted_filters['produsen']:
                    has_unwanted = True
                    break
                if unwanted_filters.get('nama_barang') and item.name in unwanted_filters['nama_barang']:
                    has_unwanted = True
                    break
            
            if has_unwanted:
                continue
            
            # Check that each desired filter type is represented
            desired_filter_coverage = {
                'kategori_umum': False,
                'bahan_dasar': False,
                'basah_kering': False,
                'rasa': False,
                'produsen': False,
                'nama_barang': False
            }
            
            for item in rec.items:
                if desired_filters.get('kategori_umum') and item.category in desired_filters['kategori_umum']:
                    desired_filter_coverage['kategori_umum'] = True
                if desired_filters.get('bahan_dasar') and item.bahan_dasar in desired_filters['bahan_dasar']:
                    desired_filter_coverage['bahan_dasar'] = True
                if desired_filters.get('basah_kering') and item.basah_kering in desired_filters['basah_kering']:
                    desired_filter_coverage['basah_kering'] = True
                if desired_filters.get('rasa') and item.rasa in desired_filters['rasa']:
                    desired_filter_coverage['rasa'] = True
                if desired_filters.get('produsen') and item.store in desired_filters['produsen']:
                    desired_filter_coverage['produsen'] = True
                if desired_filters.get('nama_barang') and item.name in desired_filters['nama_barang']:
                    desired_filter_coverage['nama_barang'] = True
            
            # Ensure all non-empty desired filters are covered
            all_filters_covered = True
            for filter_type, covered in desired_filter_coverage.items():
                if desired_filters.get(filter_type) and not covered:
                    all_filters_covered = False
                    break
            
            if all_filters_covered:
                valid_recommendations.append(rec)
        
        return valid_recommendations
    
    def _generate_candidate_pool(self, 
                               items: List[Item], 
                               customer_pref: CustomerPreference,
                               desired_filters: Dict) -> List[RecommendationResult]:
        candidates = []
        max_candidates = 25
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(max_candidates):
                modified_pref = CustomerPreference(
                    budget=customer_pref.budget,
                    weight_capacity=customer_pref.weight_capacity,
                    preferred_categories=customer_pref.preferred_categories,
                    category_weight=customer_pref.category_weight + np.random.uniform(-0.05, 0.05)
                )
                future = executor.submit(
                    self.optimizer.optimize_single_recommendation,
                    items, modified_pref, desired_filters
                )
                futures.append(future)
            
            for future in futures:
                try:
                    recommendation = future.result(timeout=30)
                    if recommendation and len(recommendation.items) > 0:
                        candidates.append(recommendation)
                except Exception as e:
                    logger.warning(f"Failed to generate candidate: {str(e)}")
        
        unique_candidates = self._remove_duplicate_recommendations(candidates)
        logger.info(f"Generated {len(unique_candidates)} unique candidates from {len(candidates)} total")
        return unique_candidates
    
    def _remove_duplicate_recommendations(self, 
                                        candidates: List[RecommendationResult]) -> List[RecommendationResult]:
        unique = []
        similarity_threshold = 0.8
        for candidate in candidates:
            is_unique = True
            candidate_items = set(item.id for item in candidate.items)
            for existing in unique:
                existing_items = set(item.id for item in existing.items)
                if candidate_items and existing_items:
                    intersection = len(candidate_items & existing_items)
                    union = len(candidate_items | existing_items)
                    similarity = intersection / union if union > 0 else 0
                    if similarity > similarity_threshold:
                        is_unique = False
                        break
            if is_unique:
                unique.append(candidate)
        return unique

recommendation_service = None

def init_service():
    global recommendation_service
    try:
        recommendation_service = RecommendationService()
        logger.info("Recommendation service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "PSO Product Recommendation API",
        "version": "2.0",
        "timestamp": time.time()
    })

@app.route('/api/recommend', methods=['POST'])
def recommend_products():
    try:
        if not recommendation_service:
            return jsonify({
                "error": "Service not initialized",
                "status": "failed"
            }), 500
        
        if not request.is_json:
            return jsonify({
                "error": "Request must be JSON",
                "status": "failed"
            }), 400
        
        customer_request = request.get_json()
        if not customer_request:
            return jsonify({
                "error": "Empty request body",
                "status": "failed"
            }), 400
        
        result = recommendation_service.generate_recommendations(customer_request)
        if result.get("status") == "failed":
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "status": "failed",
            "details": str(e)
        }), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    try:
        if not recommendation_service or not recommendation_service.items:
            return jsonify({
                "error": "No data available",
                "status": "failed"
            }), 500
        
        categories = list(set(
            item.category for item in recommendation_service.items 
            if item.category and item.category != 'Unknown'
        ))
        categories.sort()
        
        return jsonify({
            "status": "success",
            "categories": categories,
            "total_count": len(categories)
        })
        
    except Exception as e:
        logger.error(f"Categories API error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "status": "failed"
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_dataset_stats():
    try:
        if not recommendation_service or not recommendation_service.items:
            return jsonify({
                "error": "No data available",
                "status": "failed"
            }), 500
        
        items = recommendation_service.items
        categories = set(item.category for item in items if item.category != 'Unknown')
        stores = set(item.store for item in items if item.store)
        
        price_stats = {
            "min": min(item.price for item in items),
            "max": max(item.price for item in items),
            "avg": sum(item.price for item in items) / len(items)
        }
        
        weight_stats = {
            "min": min(item.weight for item in items),
            "max": max(item.weight for item in items),
            "avg": sum(item.weight for item in items) / len(items)
        }
        
        return jsonify({
            "status": "success",
            "dataset_stats": {
                "total_items": len(items),
                "categories_count": len(categories),
                "stores_count": len(stores),
                "price_range": price_stats,
                "weight_range": weight_stats
            }
        })
        
    except Exception as e:
        logger.error(f"Stats API error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "status": "failed"
        }), 500

@app.route('/api/docs', methods=['GET'])
def api_documentation():
    docs = {
        "API_Documentation": {
            "base_url": request.host_url,
            "version": "2.0",
            "endpoints": {
                "/health": {
                    "method": "GET",
                    "description": "Health check",
                    "response": "Service status"
                },
                "/api/recommend": {
                    "method": "POST",
                    "description": "Generate product recommendations",
                    "required_fields": ["budget", "weight_capacity"],
                    "optional_fields": ["preferred_categories", "category_weight", "desired_filters", "unwanted_filters"],
                    "example_request": {
                        "budget": 100000,
                        "weight_capacity": 500,
                        "preferred_categories": ["Abon", "Keripik"],
                        "category_weight": 0.15,
                        "desired_filters": {
                            "kategori_umum": ["Abon", "Jenang"],
                            "bahan_dasar": ["Bakso"],
                            "basah_kering": ["Kering"],
                            "rasa": ["Gurih", "Asin"],
                            "produsen": ["Gemini"],
                            "nama_barang": ["Wingko Cap Jago"]
                        },
                        "unwanted_filters": {
                            "kategori_umum": ["Sirup"],
                            "bahan_dasar": ["Bandeng"],
                            "basah_kering": ["Cair"],
                            "rasa": ["Pahit"],
                            "produsen": ["Jaya Snack"],
                            "nama_barang": ["Moci Rahma"]
                        }
                    }
                },
                "/api/categories": {
                    "method": "GET",
                    "description": "Get available product categories",
                    "response": "List of categories"
                },
                "/api/stats": {
                    "method": "GET",
                    "description": "Get dataset statistics",
                    "response": "Dataset overview"
                }
            }
        }
    }
    return jsonify(docs)

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "status": "failed",
        "available_endpoints": ["/health", "/api/recommend", "/api/categories", "/api/stats", "/api/docs"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "status": "failed"
    }), 500

if __name__ == '__main__':
    init_service()
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    logger.info("Starting PSO Product Recommendation API with MySQL...")
    app.run(
        host='127.0.0.1',
        port=5001,
        debug=True,
        threaded=True
    )