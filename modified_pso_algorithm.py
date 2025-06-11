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
    value: float = 0.0  # Default value

    def __post_init__(self):
        self.value_weight_ratio = self.value / max(self.weight, 1e-6)
        self.value_price_ratio = self.value / max(self.price, 1e-6)
        # Combined efficiency score
        self.efficiency_score = (self.value_weight_ratio + self.value_price_ratio) / 2

@dataclass
class CustomerPreference:
    """Customer preference specification"""
    budget: float
    weight_capacity: float
    preferred_categories: Optional[List[str]] = None
    category_weight: float = 0.1  # Weight for category preference in fitness

@dataclass
class RecommendationResult:
    """Single knapsack recommendation result"""
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
        """Convert to dictionary for JSON serialization"""
        return {
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "category": item.category,
                    "price": item.price,
                    "weight": item.weight,
                    "value": item.value,
                    "store": item.store
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
    """Particle for single knapsack recommendation optimization"""
    
    def __init__(self, num_items: int):
        self.num_items = num_items
        # Position: probability of including each item (0.0 to 1.0)
        self.position = np.random.uniform(0.0, 1.0, num_items)
        # Velocity for continuous optimization
        self.velocity = np.random.uniform(-0.1, 0.1, num_items)
        # Best personal position and fitness
        self.best_position = self.position.copy()
        self.best_fitness = -np.inf
        self.current_fitness = -np.inf
        
    def update_velocity(self, global_best_position: np.ndarray, w: float, c1: float, c2: float):
        """Update particle velocity with PSO formula"""
        r1 = np.random.random(self.num_items)
        r2 = np.random.random(self.num_items)
        
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive + social
        
        # Velocity clamping
        self.velocity = np.clip(self.velocity, -0.2, 0.2)
    
    def update_position(self):
        """Update particle position and ensure bounds"""
        self.position += self.velocity
        self.position = np.clip(self.position, 0.0, 1.0)
    
    def get_binary_selection(self, threshold: float = 0.5) -> np.ndarray:
        """Convert continuous position to binary item selection"""
        return (self.position > threshold).astype(int)

# Numba-optimized fitness evaluation
@jit(nopython=True, cache=True)
def evaluate_single_knapsack_fitness(binary_selection: np.ndarray, 
                                    items_array: np.ndarray,
                                    budget: float, 
                                    weight_capacity: float,
                                    category_preferences: np.ndarray,
                                    category_weight: float) -> float:
    """
    Optimized fitness evaluation for single knapsack
    items_array: [price, weight, value, category_id, efficiency_score]
    """
    selected_items = binary_selection == 1
    
    if not np.any(selected_items):
        return 0.0
    
    # Calculate totals for selected items
    total_price = np.sum(items_array[selected_items, 0])
    total_weight = np.sum(items_array[selected_items, 1])
    total_value = np.sum(items_array[selected_items, 2])
    
    # Hard constraint violations (heavy penalties)
    if total_price > budget:
        price_violation = (total_price - budget) / budget
        return -1000.0 * price_violation  # Heavy penalty
    
    if total_weight > weight_capacity:
        weight_violation = (total_weight - weight_capacity) / weight_capacity
        return -1000.0 * weight_violation  # Heavy penalty
    
    # Base fitness is total value
    fitness = total_value
    
    # Utilization bonuses (encourage efficient use of capacity)
    budget_utilization = total_price / budget if budget > 0 else 0
    weight_utilization = total_weight / weight_capacity if weight_capacity > 0 else 0
    
    # Bonus for high utilization (target 80-95%)
    if 0.8 <= budget_utilization <= 0.95:
        fitness += total_value * 0.1
    if 0.8 <= weight_utilization <= 0.95:
        fitness += total_value * 0.1
    
    # Category preference bonus
    if len(category_preferences) > 0:
        selected_categories = items_array[selected_items, 3]
        category_bonus = 0.0
        for cat_id in category_preferences:
            if cat_id in selected_categories:
                category_bonus += np.sum(selected_categories == cat_id) * 10
        fitness += category_bonus * category_weight
    
    # Diversity bonuses
    unique_categories = len(np.unique(items_array[selected_items, 3]))
    fitness += unique_categories * 5  # Encourage category diversity
    
    return fitness

@jit(nopython=True, cache=True)
def repair_solution(binary_selection: np.ndarray, 
                   items_array: np.ndarray,
                   budget: float, 
                   weight_capacity: float) -> np.ndarray:
    """Smart repair mechanism for constraint violations"""
    repaired = binary_selection.copy()
    selected_indices = np.where(repaired == 1)[0]
    
    if len(selected_indices) == 0:
        return repaired
    
    # Calculate current totals
    current_price = np.sum(items_array[selected_indices, 0])
    current_weight = np.sum(items_array[selected_indices, 1])
    
    # Remove items if constraints are violated
    while (current_price > budget or current_weight > weight_capacity) and len(selected_indices) > 0:
        # Find least efficient item to remove
        efficiencies = items_array[selected_indices, 4]  # efficiency scores
        worst_idx_pos = np.argmin(efficiencies)
        worst_item_idx = selected_indices[worst_idx_pos]
        
        # Remove the item
        repaired[worst_item_idx] = 0
        current_price -= items_array[worst_item_idx, 0]
        current_weight -= items_array[worst_item_idx, 1]
        
        # Update selected indices
        selected_indices = np.delete(selected_indices, worst_idx_pos)
    
    return repaired

class CustomerRecommendationPSO:
    """PSO optimizer for customer product recommendations"""
    
    def __init__(self, 
                 n_particles: int = 50,
                 max_iterations: int = 200,
                 target_recommendations: int = 10,
                 diversity_threshold: float = 0.3):
        
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.target_recommendations = target_recommendations
        self.diversity_threshold = diversity_threshold
        
        # PSO parameters (adaptive)
        self.w_start = 0.9
        self.w_end = 0.4
        self.c1 = 2.0
        self.c2 = 2.0
        
        # Convergence parameters
        self.stagnation_limit = 50
        self.repair_probability = 0.2
        
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def _preprocess_data(self, items: List[Item], customer_pref: CustomerPreference) -> Tuple[np.ndarray, Dict, np.ndarray]:
        """Preprocess items for vectorized operations"""
        n_items = len(items)
        
        # Create category mapping
        categories = list(set(item.category for item in items))
        category_map = {cat: i for i, cat in enumerate(categories)}
        
        # Convert items to numpy array: [price, weight, value, category_id, efficiency_score]
        items_array = np.zeros((n_items, 5))
        for i, item in enumerate(items):
            cat_id = category_map.get(item.category, 0)
            items_array[i] = [item.price, item.weight, item.value, cat_id, item.efficiency_score]
        
        # Process category preferences
        preferred_cat_ids = np.array([])
        if customer_pref.preferred_categories:
            preferred_cat_ids = np.array([
                category_map[cat] for cat in customer_pref.preferred_categories 
                if cat in category_map
            ])
        
        return items_array, category_map, preferred_cat_ids 
    
    def _initialize_population(self, n_items: int) -> List[SingleKnapsackParticle]:
        """Initialize particle population with diverse strategies"""
        particles = []
        
        # Strategy 1: Random initialization (50%)
        n_random = int(0.5 * self.n_particles)
        for _ in range(n_random):
            particles.append(SingleKnapsackParticle(n_items))
        
        # Strategy 2: Greedy initialization (30%)
        n_greedy = int(0.3 * self.n_particles)
        for _ in range(n_greedy):
            particle = SingleKnapsackParticle(n_items)
            # Bias toward high-efficiency items
            particle.position = np.random.beta(2, 1, n_items)  # Skewed toward 1
            particles.append(particle)
        
        # Strategy 3: Sparse initialization (20%)
        n_sparse = self.n_particles - n_random - n_greedy
        for _ in range(n_sparse):
            particle = SingleKnapsackParticle(n_items)
            # Bias toward fewer items
            particle.position = np.random.beta(1, 3, n_items)  # Skewed toward 0
            particles.append(particle)
        
        return particles
    
    def optimize_single_recommendation(self, 
                                     items: List[Item], 
                                     customer_pref: CustomerPreference) -> RecommendationResult:
        """Optimize a single recommendation for customer"""
        start_time = time.time()
        
        # Preprocess data
        items_array, category_map, preferred_cat_ids = self._preprocess_data(items, customer_pref)
        n_items = len(items)
        
        # Initialize population
        particles = self._initialize_population(n_items)
        
        # Global best tracking
        global_best_position = None
        global_best_fitness = -np.inf
        stagnation_counter = 0
        
        self.logger.info(f"Starting optimization for {n_items} items...")
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Adaptive inertia weight
            progress = iteration / self.max_iterations
            w = self.w_start - (self.w_start - self.w_end) * progress
            
            # Evaluate all particles
            for particle in particles:
                # Convert to binary selection
                binary_selection = particle.get_binary_selection(0.5)
                
                # Apply repair mechanism probabilistically
                if np.random.random() < self.repair_probability:
                    binary_selection = repair_solution(
                        binary_selection, items_array, 
                        customer_pref.budget, customer_pref.weight_capacity
                    )
                
                # Evaluate fitness
                fitness = evaluate_single_knapsack_fitness(
                    binary_selection, items_array,
                    customer_pref.budget, customer_pref.weight_capacity,
                    preferred_cat_ids, customer_pref.category_weight
                )
                
                particle.current_fitness = fitness
                
                # Update personal best
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particle.position.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            
            # Update particles
            if global_best_position is not None:
                for particle in particles:
                    particle.update_velocity(global_best_position, w, self.c1, self.c2)
                    particle.update_position()
            
            # Log progress
            if iteration % 50 == 0:
                avg_fitness = np.mean([p.current_fitness for p in particles])
                self.logger.info(f"Iteration {iteration}: Best={global_best_fitness:.2f}, Avg={avg_fitness:.2f}")
            
            # Early stopping
            if stagnation_counter > self.stagnation_limit:
                self.logger.info(f"Early stopping at iteration {iteration}")
                break
        
        # Create final recommendation
        if global_best_position is not None:
            final_selection = (global_best_position > 0.5).astype(int)
            final_selection = repair_solution(
                final_selection, items_array,
                customer_pref.budget, customer_pref.weight_capacity
            )
        else:
            final_selection = np.zeros(n_items, dtype=int)
        
        recommendation = self._create_recommendation_result(
            final_selection, items, customer_pref, global_best_fitness
        )
        
        end_time = time.time()
        self.logger.info(f"Optimization completed in {end_time - start_time:.2f}s")
        
        return recommendation
    
    def _create_recommendation_result(self, 
                                    binary_selection: np.ndarray,
                                    items: List[Item],
                                    customer_pref: CustomerPreference,
                                    fitness_score: float) -> RecommendationResult:
        """Create detailed recommendation result"""
        selected_items = [items[i] for i in range(len(items)) if binary_selection[i] == 1]
        
        total_price = sum(item.price for item in selected_items)
        total_weight = sum(item.weight for item in selected_items)
        total_value = sum(item.value for item in selected_items)
        
        budget_utilization = (total_price / customer_pref.budget * 100) if customer_pref.budget > 0 else 0
        weight_utilization = (total_weight / customer_pref.weight_capacity * 100) if customer_pref.weight_capacity > 0 else 0
        
        categories = set(item.category for item in selected_items)
        stores = set(item.store for item in selected_items) if selected_items and hasattr(selected_items[0], 'store') else set()
        
        return RecommendationResult(
            items=selected_items,
            total_value=total_value,
            total_price=total_price,
            total_weight=total_weight,
            budget_utilization=budget_utilization,
            weight_utilization=weight_utilization,
            fitness_score=fitness_score,
            category_diversity=len(categories),
            store_diversity=len(stores)
        )
    
    def generate_diverse_recommendations(self, 
                                       items: List[Item],
                                       customer_pref: CustomerPreference) -> List[RecommendationResult]:
        """Generate multiple diverse recommendations"""
        recommendations = []
        used_item_sets = []
        
        self.logger.info(f"Generating {self.target_recommendations} diverse recommendations...")
        
        for i in range(self.target_recommendations):
            max_attempts = 5
            best_recommendation = None
            best_diversity_score = -1
            
            for attempt in range(max_attempts):
                # Generate a recommendation
                recommendation = self.optimize_single_recommendation(items, customer_pref)
                
                # Calculate diversity score against existing recommendations
                diversity_score = self._calculate_diversity_score(recommendation, recommendations)
                
                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_recommendation = recommendation
                
                # If diversity is sufficient, use this recommendation
                if diversity_score >= self.diversity_threshold:
                    break
            
            if best_recommendation and len(best_recommendation.items) > 0:
                recommendations.append(best_recommendation)
                used_item_sets.append(set(item.id for item in best_recommendation.items))
                self.logger.info(f"Generated recommendation {i+1}/{self.target_recommendations} "
                               f"(diversity: {best_diversity_score:.3f})")
            else:
                self.logger.warning(f"Failed to generate recommendation {i+1}")
        
        return recommendations
    
    def _calculate_diversity_score(self, 
                                 new_recommendation: RecommendationResult,
                                 existing_recommendations: List[RecommendationResult]) -> float:
        """Calculate diversity score for a new recommendation"""
        if not existing_recommendations:
            return 1.0
        
        new_item_set = set(item.id for item in new_recommendation.items)
        
        if not new_item_set:
            return 0.0
        
        diversity_scores = []
        for existing in existing_recommendations:
            existing_set = set(item.id for item in existing.items)
            
            if not existing_set:
                continue
            
            # Jaccard distance (1 - Jaccard similarity)
            intersection = len(new_item_set & existing_set)
            union = len(new_item_set | existing_set)
            jaccard_similarity = intersection / union if union > 0 else 0
            diversity_scores.append(1 - jaccard_similarity)
        
        return np.mean(diversity_scores) if diversity_scores else 1.0

def create_customer_preference(request_data: Dict) -> CustomerPreference:
    """Create customer preference from API request data"""
    return CustomerPreference(
        budget=float(request_data['budget']),
        weight_capacity=float(request_data['weight_capacity']),
        preferred_categories=request_data.get('preferred_categories', []),
        category_weight=float(request_data.get('category_weight', 0.1))
    )

# Example usage and testing
def test_recommendation_system():
    """Test the recommendation system"""
    print("Testing Customer Recommendation System...")
    
    # Create sample items
    sample_items = [
        Item(1, "Abon", "Abon Ayam A", 25000, 100, "Store A", 80),
        Item(2, "Abon", "Abon Ayam B", 30000, 120, "Store B", 90),
        Item(3, "Kripik", "Kripik Pisang A", 15000, 80, "Store A", 70),
        Item(4, "Kripik", "Kripik Pisang B", 18000, 90, "Store C", 75),
        Item(5, "Bakpia", "Bakpia A", 20000, 150, "Store B", 85),
        Item(6, "Bakpia", "Bakpia B", 22000, 160, "Store A", 88),
        Item(7, "Emping", "Emping A", 12000, 60, "Store C", 65),
        Item(8, "Emping", "Emping B", 14000, 70, "Store B", 68),
    ]
    
    # Create customer preference
    customer_pref = CustomerPreference(
        budget=100000,
        weight_capacity=500,
        preferred_categories=["Abon", "Kripik"],
        category_weight=0.15
    )
    
    # Initialize optimizer
    optimizer = CustomerRecommendationPSO(
        n_particles=30,
        max_iterations=100,
        target_recommendations=3,
        diversity_threshold=0.3
    )
    
    # Generate recommendations
    recommendations = optimizer.generate_diverse_recommendations(sample_items, customer_pref)
    
    # Display results
    print(f"\nGenerated {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n--- Recommendation {i} ---")
        print(f"Items: {len(rec.items)}")
        print(f"Total Value: {rec.total_value:.2f}")
        print(f"Total Price: Rp {rec.total_price:,.0f} ({rec.budget_utilization:.1f}%)")
        print(f"Total Weight: {rec.total_weight:.0f}g ({rec.weight_utilization:.1f}%)")
        print(f"Categories: {rec.category_diversity}, Stores: {rec.store_diversity}")
        print("Items:")
        for item in rec.items:
            print(f"  - {item.name} (Rp {item.price:,.0f}, {item.weight}g)")
    
    return recommendations

if __name__ == "__main__":
    test_recommendation_system()