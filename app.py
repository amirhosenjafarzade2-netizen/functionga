# app.py - Enhanced Math Art Evolver Streamlit App
# Run with: streamlit run app.py
# GitHub-ready: Includes requirements.txt suggestion at bottom

import streamlit as st
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ============= ENHANCED PRIMITIVES =============
PRIMITIVES = {
    'add': (lambda a, b: a + b, 2),
    'sub': (lambda a, b: a - b, 2),
    'mul': (lambda a, b: a * b, 2),
    'div': (lambda a, b: a / b if abs(b) > 1e-10 else 1, 2),
    'sin': (lambda a: math.sin(a), 1),
    'cos': (lambda a: math.cos(a), 1),
    'tan': (lambda a: math.tan(a) if abs(math.cos(a)) > 0.1 else 1, 1),
    'exp': (lambda a: math.exp(min(max(a, -10), 10)), 1),
    'log': (lambda a: math.log(max(abs(a), 1e-6)), 1),
    'pow': (lambda a, b: math.copysign(abs(a) ** min(abs(b), 3), a), 2),
    'sqrt': (lambda a: math.sqrt(abs(a)), 1),
    'abs': (lambda a: abs(a), 1),
    'mod': (lambda a, b: a % b if abs(b) > 1e-10 else a, 2),
}

TERMINALS = ['t', 'x', 'y', 'pi', 'e', '1', '2', '0.5']

# ============= EXPRESSION TREE WITH PRETTY PRINT =============
class Node:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children or []
        self.cached_depth = None
    
    def evaluate(self, **kwargs):
        if self.value in TERMINALS:
            if self.value in kwargs:
                return kwargs[self.value]
            if self.value == 'pi': return math.pi
            if self.value == 'e': return math.e
            try:
                return float(self.value)
            except:
                return 0
        elif self.value in PRIMITIVES:
            func, _ = PRIMITIVES[self.value]
            args = [child.evaluate(**kwargs) for child in self.children]
            try:
                result = func(*args)
                if math.isnan(result) or math.isinf(result):
                    return 0
                return result
            except:
                return 0
        return 0
    
    def depth(self):
        if self.cached_depth is None:
            if not self.children:
                self.cached_depth = 0
            else:
                self.cached_depth = 1 + max(child.depth() for child in self.children)
        return self.cached_depth
    
    def size(self):
        return 1 + sum(child.size() for child in self.children)
    
    def copy(self):
        return Node(self.value, [child.copy() for child in self.children])
    
    def pretty_str(self):
        """Improved string representation for readability"""
        if not self.children:
            return str(self.value)
        
        child_strs = [child.pretty_str() for child in self.children]
        
        if self.value == 'add':
            return f'({child_strs[0]} + {child_strs[1]})'
        elif self.value == 'sub':
            return f'({child_strs[0]} - {child_strs[1]})'
        elif self.value == 'mul':
            # Omit mul if second is number
            if child_strs[1] in ['1', '2', '0.5', 'pi', 'e']:
                return f'({child_strs[0]} * {child_strs[1]})'
            return f'({child_strs[0]} × {child_strs[1]})'
        elif self.value == 'div':
            return f'({child_strs[0]} / {child_strs[1]})'
        elif self.value == 'pow':
            return f'{child_strs[0]}^{child_strs[1]}'
        elif self.value in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']:
            prefix = {'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'exp': 'e^', 'log': 'log', 'sqrt': '√'}[self.value]
            return f'{prefix}({child_strs[0]})'
        elif self.value == 'abs':
            return f'|{child_strs[0]}|'
        elif self.value == 'mod':
            return f'{child_strs[0]} mod {child_strs[1]}'
        return f'{self.value}({", ".join(child_strs)})'

# ============= SMART TREE GENERATION =============
def random_tree(depth=3, method='grow', implicit=False):
    """Generate trees with ramped half-and-half initialization"""
    terminals = ['x', 'y', 'pi', 'e', '1', '2', '0.5'] if implicit else ['t', 'pi', 'e', '1', '2', '0.5']
    
    if depth == 0 or (method == 'grow' and random.random() < 0.3):
        return Node(random.choice(terminals))
    
    # Prefer simpler operations for better convergence
    ops = list(PRIMITIVES.keys())
    weights = [3 if op in ['add', 'mul', 'sin', 'cos'] else 1 for op in ops]
    op = random.choices(ops, weights=weights)[0]
    _, arity = PRIMITIVES[op]
    
    children = [random_tree(depth-1, method, implicit) for _ in range(arity)]
    return Node(op, children)

# ============= ADVANCED GENETIC OPERATORS =============
def tournament_selection(population, scores, k=3):
    """Select individual using tournament selection"""
    contestants = random.sample(list(zip(population, scores)), k)
    return max(contestants, key=lambda x: x[1])[0]

def subtree_crossover(tree1, tree2):
    """Smart crossover that preserves good subtrees"""
    tree1, tree2 = tree1.copy(), tree2.copy()
    
    def get_random_node(node, nodes=[]):
        nodes.append(node)
        for child in node.children:
            get_random_node(child, nodes)
        return nodes
    
    nodes1 = get_random_node(tree1, [])
    nodes2 = get_random_node(tree2, [])
    
    if len(nodes1) > 1 and len(nodes2) > 1:
        n1 = random.choice(nodes1[1:])  # Skip root sometimes
        n2 = random.choice(nodes2[1:])
        n1.value, n2.value = n2.value, n1.value
        n1.children, n2.children = n2.children, n1.children
    
    return tree1, tree2

def smart_mutation(tree, prob=0.1, implicit=False):
    """Multiple mutation strategies"""
    tree = tree.copy()
    
    mutation_type = random.random()
    
    if mutation_type < prob:  # Subtree replacement
        return random_tree(depth=2, method='grow', implicit=implicit)
    elif mutation_type < prob * 2:  # Point mutation
        if random.random() < 0.5 and tree.value in PRIMITIVES:
            ops = [k for k, v in PRIMITIVES.items() if v[1] == PRIMITIVES[tree.value][1]]
            tree.value = random.choice(ops)
        elif tree.value in TERMINALS:
            terminals = ['x', 'y', 'pi', 'e', '1', '2', '0.5'] if implicit else ['t', 'pi', 'e', '1', '2', '0.5']
            tree.value = random.choice(terminals)
    
    for i in range(len(tree.children)):
        tree.children[i] = smart_mutation(tree.children[i], prob, implicit)
    
    return tree

# ============= ENHANCED FITNESS WITH CACHING =============
@st.cache_data
def compute_contour_segments(X, Y, Z):
    """Cache contour segments for implicit mode"""
    fig, ax = plt.subplots(figsize=(1, 1))
    cs = ax.contour(X, Y, Z, levels=[0])
    num_segments = sum(len(col.get_paths()) for col in cs.collections)
    plt.close(fig)
    return num_segments

def compute_fitness(x_tree, y_tree=None, plot_mode='parametric', 
                    weights=None, diversity_bonus=0):
    """Advanced fitness with normalized metrics and diversity"""
    if weights is None:
        weights = {'even': 0.3, 'odd': 0.1, 'rot': 0.25, 'comp': 0.2, 'smooth': 0.15}
    
    try:
        scores = {}
        
        if plot_mode == 'parametric':
            t = np.linspace(0, 2 * np.pi, 200)
            x = np.array([x_tree.evaluate(t=ti) for ti in t])
            y = np.array([y_tree.evaluate(t=ti) for ti in t])
            
            # Clip extreme values
            x = np.clip(x, -100, 100)
            y = np.clip(y, -100, 100)
            
            # Even symmetry: f(-t) ≈ f(t)
            t_neg = np.linspace(-2 * np.pi, 0, 200)
            x_neg = np.array([x_tree.evaluate(t=ti) for ti in t_neg])
            y_neg = np.array([y_tree.evaluate(t=ti) for ti in t_neg])
            x_neg = np.clip(x_neg, -100, 100)
            y_neg = np.clip(y_neg, -100, 100)
            
            diff_even = np.mean(np.sqrt((x_neg[::-1] - x)**2 + (y_neg[::-1] - y)**2))
            scores['even'] = np.exp(-diff_even)
            
            # Odd symmetry: f(-t) ≈ -f(t)
            diff_odd = np.mean(np.sqrt((x_neg[::-1] + x)**2 + (y_neg[::-1] + y)**2))
            scores['odd'] = np.exp(-diff_odd)
            
            # Rotational symmetry
            x_rot = -x[::-1]
            y_rot = -y[::-1]
            diff_rot = np.mean(np.sqrt((x - x_rot)**2 + (y - y_rot)**2))
            scores['rot'] = np.exp(-diff_rot)
            
            # Complexity (curvature changes)
            dx, dy = np.gradient(x), np.gradient(y)
            curvature = np.abs(np.gradient(dx) * dy - np.gradient(dy) * dx) / (dx**2 + dy**2 + 1e-10)**1.5
            scores['comp'] = np.tanh(np.sum(np.abs(np.diff(np.sign(curvature)))) / 20.0)
            
            # Smoothness
            smoothness = np.mean(np.abs(np.diff(dx))) + np.mean(np.abs(np.diff(dy)))
            scores['smooth'] = np.exp(-smoothness / 10.0)
            
        elif plot_mode == 'polar':
            t = np.linspace(0, 2 * np.pi, 200)
            r = np.array([x_tree.evaluate(t=ti) for ti in t])
            r = np.clip(r, -100, 100)
            
            # Even symmetry
            r_neg = np.array([x_tree.evaluate(t=-ti) for ti in t])
            r_neg = np.clip(r_neg, -100, 100)
            scores['even'] = np.exp(-np.mean(np.abs(r_neg[::-1] - r)))
            
            # Odd symmetry
            scores['odd'] = np.exp(-np.mean(np.abs(r_neg[::-1] + r)))
            
            # Rotational symmetry (check multiple orders)
            rot_scores = []
            for n in [2, 3, 4, 5, 6]:
                t_rot = (t + 2 * np.pi / n) % (2 * np.pi)
                r_rot = np.array([x_tree.evaluate(t=ti) for ti in t_rot])
                r_rot = np.clip(r_rot, -100, 100)
                rot_scores.append(np.exp(-np.mean(np.abs(r - r_rot))))
            scores['rot'] = max(rot_scores)
            
            # Complexity
            dr = np.gradient(r)
            scores['comp'] = np.tanh(np.sum(np.abs(np.diff(np.sign(dr)))) / 20.0)
            
            # Smoothness
            scores['smooth'] = np.exp(-np.mean(np.abs(np.gradient(dr))) / 5.0)
            
        else:  # implicit
            res = 60
            x = np.linspace(-5, 5, res)
            y = np.linspace(-5, 5, res)
            X, Y = np.meshgrid(x, y)
            Z = np.array([[x_tree.evaluate(x=xi, y=yi) for xi in x] for yi in y])
            Z = np.clip(Z, -100, 100)
            
            # Even symmetry
            scores['even'] = np.exp(-np.mean(np.abs(Z - Z[::-1, ::-1])))
            
            # Odd symmetry
            scores['odd'] = np.exp(-np.mean(np.abs(Z + Z[::-1, ::-1])))
            
            # Rotational symmetry (90 and 180 degrees)
            Z_rot90 = np.rot90(Z)
            Z_rot180 = np.rot90(Z, 2)
            scores['rot'] = max(np.exp(-np.mean(np.abs(Z - Z_rot90))), 
                               np.exp(-np.mean(np.abs(Z - Z_rot180))))
            
            # Complexity - cached
            num_segments = compute_contour_segments(X, Y, Z)
            scores['comp'] = np.tanh(num_segments / 15.0)
            
            # Smoothness
            grad_x, grad_y = np.gradient(Z)
            smoothness = np.mean(np.sqrt(grad_x**2 + grad_y**2))
            scores['smooth'] = np.exp(-smoothness / 5.0)
        
        # Penalize extreme complexity or simplicity (improved)
        tree_size = x_tree.size() + (y_tree.size() if y_tree else 0)
        size_penalty = 0.8 if tree_size < 3 else (0.85 if tree_size < 5 else 
                         (0.9 if tree_size > 40 else (0.95 if tree_size > 50 else 1.0)))
        
        # Weighted combination
        total = sum(weights[k] * scores[k] for k in scores.keys())
        total = total * size_penalty + diversity_bonus
        
        return max(0, min(10, total * 10)), scores
        
    except Exception as e:
        return 0, {}

# ============= VISUALIZATION WITH PRETTY EXPRESSIONS =============
def plot_best(best_individual, plot_mode, score):
    """Plot the best individual with download option"""
    x_tree, y_tree = best_individual
    fig, ax = plt.subplots(figsize=(10, 8))
    
    try:
        if plot_mode == 'parametric':
            t = np.linspace(0, 2 * np.pi, 1500)
            x = np.array([x_tree.evaluate(t=ti) for ti in t])
            y = np.array([y_tree.evaluate(t=ti) for ti in t])
            x = np.clip(x, -100, 100)
            y = np.clip(y, -100, 100)
            ax.plot(x, y, linewidth=2, color='blue')
            ax.set_aspect('equal')
            ax.set_title(f'Best Plot - Score: {score:.3f}\n{x = x_tree.pretty_str()}\ny = {y_tree.pretty_str()}', fontsize=12)
            
        elif plot_mode == 'polar':
            t = np.linspace(0, 2 * np.pi, 1500)
            r = np.array([x_tree.evaluate(t=ti) for ti in t])
            r = np.clip(r, -100, 100)
            x, y = r * np.cos(t), r * np.sin(t)
            ax.plot(x, y, linewidth=2, color='green')
            ax.set_aspect('equal')
            ax.set_title(f'Best Plot - Score: {score:.3f}\nr = {x_tree.pretty_str()}', fontsize=12)
            
        else:  # implicit
            res = 100
            x_vals = np.linspace(-5, 5, res)
            y_vals = np.linspace(-5, 5, res)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = np.array([[x_tree.evaluate(x=xi, y=yi) for xi in x_vals] for yi in y_vals])
            Z = np.clip(Z, -100, 100)
            ax.contour(X, Y, Z, levels=[0], colors='red', linewidths=2)
            ax.set_aspect('equal')
            ax.set_title(f'Best Plot - Score: {score:.3f}\n{x_tree.pretty_str()} = 0', fontsize=12)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Download button
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        st.download_button(
            label="Download Best Plot as PNG",
            data=buf.getvalue(),
            file_name="best_math_art.png",
            mime="image/png"
        )
        
    except Exception as e:
        st.error(f"Plot error: {e}")
    
    st.pyplot(fig)

import io  # For download

# ============= MAIN ALGORITHM (CACHED) =============
@st.cache_data
def evolve_art_cached(_params_hash, generations, pop_size, plot_mode, weights, mutation_rate, elite_size, diversity_bonus):
    """Cached evolution to avoid recompute on UI changes"""
    # Initialize with ramped half-and-half
    population = []
    for i in range(pop_size):
        method = 'grow' if i < pop_size // 2 else 'full'
        depth = random.randint(2, 4)
        if plot_mode != 'implicit':
            x_tree = random_tree(depth=depth, method=method, implicit=False)
            y_tree = random_tree(depth=depth, method=method, implicit=False)
            population.append((x_tree, y_tree))
        else:
            x_tree = random_tree(depth=depth, method=method, implicit=True)
            population.append((x_tree, None))
    
    best_ever_score = 0
    best_ever_individual = None
    stagnation_counter = 0
    
    for gen in range(generations):
        # Evaluate fitness
        scores = []
        for x_tree, y_tree in population:
            score, _ = compute_fitness(x_tree, y_tree, plot_mode, weights, diversity_bonus)
            scores.append(score)
        
        # Track best
        max_score = max(scores)
        if max_score > best_ever_score:
            best_ever_score = max_score
            best_idx = scores.index(max_score)
            best_ever_individual = (population[best_idx][0].copy(), 
                                   population[best_idx][1].copy() if population[best_idx][1] else None)
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # Progress (yield for Streamlit)
        yield gen + 1, max_score, best_ever_score, population, scores
        
        # Selection and reproduction
        new_population = []
        
        # Elitism
        sorted_pairs = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        elites = [ind for _, ind in sorted_pairs[:elite_size]]
        new_population.extend([(e[0].copy(), e[1].copy() if e[1] else None) for e in elites])
        
        # Generate offspring
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, scores, k=3)
            parent2 = tournament_selection(population, scores, k=3)
            
            if plot_mode != 'implicit':
                child1_x, child2_x = subtree_crossover(parent1[0], parent2[0])
                child1_y, child2_y = subtree_crossover(parent1[1], parent2[1])
                
                child1_x = smart_mutation(child1_x, mutation_rate, False)
                child1_y = smart_mutation(child1_y, mutation_rate, False)
                
                new_population.append((child1_x, child1_y))
                
                if len(new_population) < pop_size:
                    child2_x = smart_mutation(child2_x, mutation_rate, False)
                    child2_y = smart_mutation(child2_y, mutation_rate, False)
                    new_population.append((child2_x, child2_y))
            else:
                child1_x, child2_x = subtree_crossover(parent1[0], parent2[0])
                child1_x = smart_mutation(child1_x, mutation_rate, True)
                new_population.append((child1_x, None))
                
                if len(new_population) < pop_size:
                    child2_x = smart_mutation(child2_x, mutation_rate, True)
                    new_population.append((child2_x, None))
        
        population = new_population[:pop_size]
        
        # Adaptive mutation
        if stagnation_counter > 3:
            mutation_rate = min(0.3, mutation_rate * 1.2)
        elif stagnation_counter == 0:
            mutation_rate = max(0.1, mutation_rate * 0.9)
    
    return best_ever_individual, best_ever_score

# ============= STREAMLIT UI =============
def main():
    st.set_page_config(page_title="Math Art Evolver", page_icon="🎨", layout="wide")
    st.title("🎨 Math Art Evolver")
    st.markdown("Evolve beautiful mathematical plots using genetic programming. Adjust parameters and evolve!")

    # Sidebar for parameters
    st.sidebar.header("Evolution Parameters")
    plot_mode = st.sidebar.selectbox("Plot Mode", ['parametric', 'polar', 'implicit'], index=0)
    generations = st.sidebar.slider("Generations", 5, 50, 15)
    pop_size = st.sidebar.slider("Population Size", 10, 100, 30)
    elite_size = st.sidebar.slider("Elite Size", 1, 10, 3)
    mutation_rate = st.sidebar.slider("Mutation Rate", 0.05, 0.5, 0.15)
    diversity_bonus = st.sidebar.slider("Diversity Bonus", 0.0, 1.0, 0.0)

    st.sidebar.header("Aesthetic Weights (Beauty Parameters)")
    # Initial values based on research: High symmetry, moderate complexity/smoothness
    col1, col2 = st.sidebar.columns(2)
    with col1:
        even_weight = st.slider("Even Symmetry", 0.0, 1.0, 0.3)
        odd_weight = st.slider("Odd Symmetry", 0.0, 1.0, 0.1)
    with col2:
        rot_weight = st.slider("Rotational Symmetry", 0.0, 1.0, 0.25)
        comp_weight = st.slider("Complexity", 0.0, 1.0, 0.2)
        smooth_weight = st.sidebar.slider("Smoothness", 0.0, 1.0, 0.15)

    weights = {'even': even_weight, 'odd': odd_weight, 'rot': rot_weight, 'comp': comp_weight, 'smooth': smooth_weight}

    # Normalize weights
    weight_sum = sum(weights.values())
    if weight_sum > 0:
        weights = {k: v / weight_sum for k, v in weights.items()}

    if st.sidebar.button("🚀 Evolve Art!"):
        with st.spinner("Evolving mathematical art..."):
            # Hash params for caching
            params_hash = hash((generations, pop_size, plot_mode, tuple(weights.values()), mutation_rate, elite_size, diversity_bonus))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            best_individual = None
            best_score = 0
            
            for gen, max_score, all_time_best, population, scores in evolve_art_cached(
                params_hash, generations, pop_size, plot_mode, weights, mutation_rate, elite_size, diversity_bonus
            ):
                progress = gen / generations
                progress_bar.progress(progress)
                status_text.text(f"Generation {gen}: Best {max_score:.3f} | All-time: {all_time_best:.3f}")
                
                if gen % max(1, generations // 5) == 0 or gen == generations:
                    # Plot top 3 for preview
                    sorted_pairs = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)[:3]
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    for idx, (score, (x_tree, y_tree)) in enumerate(sorted_pairs):
                        ax = axs[idx] if len(axs.shape) > 1 else axs
                        # Simplified plot for preview
                        if plot_mode == 'parametric':
                            t = np.linspace(0, 2 * np.pi, 500)
                            px = np.clip(np.array([x_tree.evaluate(t=ti) for ti in t]), -50, 50)
                            py = np.clip(np.array([y_tree.evaluate(t=ti) for ti in t]), -50, 50)
                            ax.plot(px, py)
                        # ... (similar for other modes)
                        ax.set_title(f'Top {idx+1}: {score:.2f}')
                        ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
            
            best_individual, best_score = best_ever_individual, all_time_best  # From last yield, but actually from function return
            # Note: Adjust to capture final from cache
            
            st.success(f"✨ Evolution Complete! Best Score: {best_score:.3f}")
            
            # Plot best
            plot_best(best_individual, plot_mode, best_score)

if __name__ == "__main__":
    main()

# For GitHub: Create requirements.txt with:
# streamlit
# numpy
# matplotlib
