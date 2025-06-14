import numpy as np
from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
import logging
import time
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Item:
    id: int
    name: str
    category: str
    price: float
    weight: float
    store: str
    value: float = 0.0
    bahan_dasar: str = ''
    basah_kering: str = ''
    rasa: str = ''

    def __post_init__(self):
        self.value_weight_ratio = self.value / max(self.weight, 1e-6)
        self.value_price_ratio = self.value / max(self.price, 1e-6)
        self.efficiency_score = (self.value_weight_ratio + self.value_price_ratio) / 2

@dataclass
class CustomerPreference:
    budget: float
    weight_capacity: float
    preferred_categories: Optional[List[str]] = None
    category_weight: float = 0.1

@dataclass
class RecommendationResult:
    items: List[Item]
    total_value: float
    total_price: float
    total_weight: float
    budget_utilization: float
    weight_utilization: float
    fitness_score: float
    category_diversity: int
    store_diversity: int
    
    def to_dict(self) -> Dict:
        return {
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "category": item.category,
                    "price": item.price,
                    "weight": item.weight,
                    "value": item.value,
                    "store": item.store,
                    "bahan_dasar": item.bahan_dasar,
                    "basah_kering": item.basah_kering,
                    "rasa": item.rasa
                } for item in self.items
            ],
            "summary": {
                "total_value": self.total_value,
                "total_price": self.total_price,
                "total_weight": self.total_weight,
                "budget_utilization": self.budget_utilization,
                "weight_utilization": self.weight_utilization,
                "fitness_score": self.fitness_score,
                "item_count": len(self.items),
                "category_diversity": self.category_diversity,
                "store_diversity": self.store_diversity
            }
        }

class SingleKnapsackParticle:
    def __init__(self, num_items: int):
        self.num_items = num_items
        self.position = np.random.uniform(0.0, 1.0, num_items)
        self.velocity = np.random.uniform(-0.1, 0.1, num_items)
        self.best_position = self.position.copy()
        self.best_fitness = -np.inf
        self.current_fitness = -np.inf
        
    def update_velocity(self, global_best_position: np.ndarray, w: float, c1: float, c2: float):
        r1 = np.random.random(self.num_items)
        r2 = np.random.random(self.num_items)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social
        self.velocity = np.clip(self.velocity, -0.2, 0.2)
    
    def update_position(self):
        self.position += self.velocity
        self.position = np.clip(self.position, 0.0, 1.0)
    
    def get_binary_selection(self, threshold: float = 0.5) -> np.ndarray:
        return (self.position > threshold).astype(int)

@jit(nopython=True, cache=True)
def evaluate_single_knapsack_fitness(binary_selection: np.ndarray, 
                                    items_array: np.ndarray,
                                    budget: float, 
                                    weight_capacity: float,
                                    category_preferences: np.ndarray,
                                    category_weight: float) -> float:
    selected_items = binary_selection == 1
    if not np.any(selected_items):
        return 0.0
    
    total_price = np.sum(items_array[selected_items, 0])
    total_weight = np.sum(items_array[selected_items, 1])
    total_value = np.sum(items_array[selected_items, 2])
    
    if total_price > budget:
        price_violation = (total_price - budget) / budget
        return -1000.0 * price_violation
    
    if total_weight > weight_capacity:
        weight_violation = (total_weight - weight_capacity) / weight_capacity
        return -1000.0 * weight_violation
    
    fitness = total_value
    budget_utilization = total_price / budget if budget > 0 else 0
    weight_utilization = total_weight / weight_capacity if weight_capacity > 0 else 0
    
    if 0.8 <= budget_utilization <= 0.95:
        fitness += total_value * 0.1
    if 0.8 <= weight_utilization <= 0.95:
        fitness += total_value * 0.1
    
    if len(category_preferences) > 0:
        selected_categories = items_array[selected_items, 3]
        category_bonus = 0.0
        for cat_id in category_preferences:
            if cat_id in selected_categories:
                category_bonus += np.sum(selected_categories == cat_id) * 10
        fitness += category_bonus * category_weight
    
    unique_categories = len(np.unique(items_array[selected_items, 3]))
    fitness += unique_categories * 5
    return fitness

@jit(nopython=True, cache=True)
def repair_solution(binary_selection: np.ndarray, 
                   items_array: np.ndarray,
                   budget: float, 
                   weight_capacity: float) -> np.ndarray:
    repaired = binary_selection.copy()
    selected_indices = np.where(repaired == 1)[0]
    if len(selected_indices) == 0:
        return repaired
    
    current_price = np.sum(items_array[selected_indices, 0])
    current_weight = np.sum(items_array[selected_indices, 1])
    
    while (current_price > budget or current_weight > weight_capacity) and len(selected_indices) > 0:
        efficiencies = items_array[selected_indices, 4]
        worst_idx_pos = np.argmin(efficiencies)
        worst_item_idx = selected_indices[worst_idx_pos]
        repaired[worst_item_idx] = 0
        current_price -= items_array[worst_item_idx, 0]
        current_weight -= items_array[worst_item_idx, 1]
        selected_indices = np.delete(selected_indices, worst_idx_pos)
    
    return repaired

class CustomerRecommendationPSO:
    def __init__(self, 
                 n_particles: int = 50,
                 max_iterations: int = 200,
                 target_recommendations: int = 10,
                 diversity_threshold: float = 0.3):
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.target_recommendations = target_recommendations
        self.diversity_threshold = diversity_threshold
        self.w_start = 0.9
        self.w_end = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        self.stagnation_limit = 50
        self.repair_probability = 0.2
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def _preprocess_data(self, items: List[Item], customer_pref: CustomerPreference) -> Tuple[np.ndarray, Dict, np.ndarray]:
        n_items = len(items)
        categories = list(set(item.category for item in items))
        category_map = {cat: i for i, cat in enumerate(categories)}
        items_array = np.zeros((n_items, 5))
        for i, item in enumerate(items):
            cat_id = category_map.get(item.category, 0)
            items_array[i] = [
                item.price,
                item.weight,
                item.value,
                cat_id,
                item.efficiency_score
            ]
        
        preferred_category_ids = np.array([
            category_map[cat] for cat in customer_pref.preferred_categories
            if cat in category_map
        ]) if customer_pref.preferred_categories else np.array([])
        
        return items_array, category_map, preferred_category_ids
    
    def _create_recommendation_result(self,
                                    binary_selection: np.ndarray,
                                    items: List[Item],
                                    items_array: np.ndarray,
                                    customer_pref: CustomerPreference) -> RecommendationResult:
        selected_indices = np.where(binary_selection == 1)[0]
        selected_items = [items[i] for i in selected_indices]
        
        total_price = np.sum(items_array[selected_indices, 0]) if len(selected_indices) > 0 else 0.0
        total_weight = np.sum(items_array[selected_indices, 1]) if len(selected_indices) > 0 else 0.0
        total_value = np.sum(items_array[selected_indices, 2]) if len(selected_indices) > 0 else 0.0
        fitness = evaluate_single_knapsack_fitness(
            binary_selection,
            items_array,
            customer_pref.budget,
            customer_pref.weight_capacity,
            np.array([i for i in range(len(set(item.category for item in items)))]),
            customer_pref.category_weight
        )
        
        budget_utilization = total_price / customer_pref.budget if customer_pref.budget > 0 else 0.0
        weight_utilization = total_weight / customer_pref.weight_capacity if customer_pref.weight_capacity > 0 else 0.0
        category_diversity = len(set(item.category for item in selected_items))
        store_diversity = len(set(item.store for item in selected_items))
        
        return RecommendationResult(
            items=selected_items,
            total_value=total_value,
            total_price=total_price,
            total_weight=total_weight,
            budget_utilization=budget_utilization,
            weight_utilization=weight_utilization,
            fitness_score=fitness,
            category_diversity=category_diversity,
            store_diversity=store_diversity
        )
    
    def optimize_single_recommendation(self, items: List[Item], customer_pref: CustomerPreference) -> RecommendationResult:
        items_array, category_map, preferred_category_ids = self._preprocess_data(items, customer_pref)
        num_items = len(items)
        
        particles = [SingleKnapsackParticle(num_items) for _ in range(self.n_particles)]
        global_best_position = particles[0].position.copy()
        global_best_fitness = -np.inf
        global_best_result = None
        
        stagnation_counter = 0
        previous_best_fitness = -np.inf
        
        for iteration in range(self.max_iterations):
            w = self.w_start - (self.w_start - self.w_end) * (iteration / self.max_iterations)
            
            for particle in particles:
                binary_selection = particle.get_binary_selection()
                
                if np.random.random() < self.repair_probability:
                    binary_selection = repair_solution(
                        binary_selection,
                        items_array,
                        customer_pref.budget,
                        customer_pref.weight_capacity
                    )
                
                fitness = evaluate_single_knapsack_fitness(
                    binary_selection,
                    items_array,
                    customer_pref.budget,
                    customer_pref.weight_capacity,
                    preferred_category_ids,
                    customer_pref.category_weight
                )
                
                particle.current_fitness = fitness
                
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = binary_selection.copy()
                
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = binary_selection.copy()
                    global_best_result = self._create_recommendation_result(
                        binary_selection,
                        items,
                        items_array,
                        customer_pref
                    )
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                particle.update_velocity(global_best_position, w, self.c1, self.c2)
                particle.update_position()
            
            if stagnation_counter >= self.stagnation_limit:
                self.logger.info(f"Stagnation reached at iteration {iteration}")
                break
            
            if global_best_fitness == previous_best_fitness:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                previous_best_fitness = global_best_fitness
        
        return global_best_result if global_best_result else self._create_recommendation_result(
            np.zeros(num_items, dtype=int),
            items,
            items_array,
            customer_pref
        )
    
    def optimize(self, items: List[Item], customer_pref: CustomerPreference) -> List[RecommendationResult]:
        results = []
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = [
                executor.submit(self.optimize_single_recommendation, items, customer_pref)
                for _ in range(self.target_recommendations)
            ]
            for future in futures:
                result = future.result()
                if result and result.items:
                    results.append(result)
        
        unique_results = []
        seen_selections = set()
        for result in sorted(results, key=lambda x: x.fitness_score, reverse=True):
            item_ids = tuple(sorted(item.id for item in result.items))
            if item_ids not in seen_selections:
                jaccard_similarities = []
                for existing in unique_results:
                    existing_ids = set(item.id for item in existing.items)
                    current_ids = set(item.id for item in result.items)
                    intersection = len(existing_ids & current_ids)
                    union = len(existing_ids | current_ids)
                    jaccard = intersection / union if union > 0 else 0
                    jaccard_similarities.append(jaccard)
                
                if not jaccard_similarities or max(jaccard_similarities) < self.diversity_threshold:
                    unique_results.append(result)
                    seen_selections.add(item_ids)
        
        return unique_results[:self.target_recommendations]

def create_customer_preference(customer_request: Dict) -> CustomerPreference:
    return CustomerPreference(
        budget=float(customer_request.get('budget', 0)),
        weight_capacity=float(customer_request.get('weight_capacity', 0)),
        preferred_categories=customer_request.get('preferred_categories', []),
        category_weight=float(customer_request.get('category_weight', 0.1))
    )