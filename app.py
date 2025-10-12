import streamlit as st
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import io
import json
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

# ============= EXPRESSION TREE WITH SERIALIZATION =============
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
        if not self.children:
            return str(self.value)
        
        child_strs = [child.pretty_str() for child in self.children]
        
        if self.value == 'add':
            return f'({child_strs[0]} + {child_strs[1]})'
        elif self.value == 'sub':
            return f'({child_strs[0]} - {child_strs[1]})'
        elif self.value == 'mul':
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

    def to_dict(self):
        return {
            'value': self.value,
            'children': [child.to_dict() for child in self.children]
        }

    @classmethod
    def from_dict(cls, data):
        node = cls(value=data['value'])
        node.children = [cls.from_dict(child) for child in data.get('children', [])]
        return node

# ============= SMART TREE GENERATION =============
def random_tree(depth=3, method='grow', implicit=False, max_nodes=20, current_nodes=1):
    terminals = ['x', 'y', 'pi', 'e', '1', '2', '0.5'] if implicit else ['t', 'pi', 'e', '1', '2', '0.5']
    
    if depth == 0 or current_nodes >= max_nodes or (method == 'grow' and random.random() < 0.3):
        return Node(random.choice(terminals)), current_nodes + 1
    
    ops = list(PRIMITIVES.keys())
    weights = [3 if op in ['add', 'mul', 'sin', 'cos'] else 1 for op in ops]
    op = random.choices(ops, weights=weights)[0]
    _, arity = PRIMITIVES[op]
    
    children = []
    nodes_used = current_nodes + 1  # Count the current node
    for _ in range(arity):
        if nodes_used < max_nodes:
            child, child_nodes = random_tree(depth-1, method, implicit, max_nodes, nodes_used)
            children.append(child)
            nodes_used = child_nodes
        else:
            children.append(Node(random.choice(terminals)))
            nodes_used += 1
    
    return Node(op, children), nodes_used

# ============= ADVANCED GENETIC OPERATORS =============
def tournament_selection(population, scores, k=3):
    contestants = random.sample(list(zip(population, scores)), k)
    return max(contestants, key=lambda x: x[1])[0]

def subtree_crossover(tree1, tree2, max_nodes=20):
    tree1, tree2 = tree1.copy(), tree2.copy()
    
    def get_random_node(node, nodes=[]):
        nodes.append(node)
        for child in node.children:
            get_random_node(child, nodes)
        return nodes
    
    nodes1 = get_random_node(tree1, [])
    nodes2 = get_random_node(tree2, [])
    
    if len(nodes1) > 1 and len(nodes2) > 1:
        n1 = random.choice(nodes1[1:])
        n2 = random.choice(nodes2[1:])
        # Check if swapping subtrees keeps tree sizes within max_nodes
        size1 = tree1.size() - n1.size() + n2.size()
        size2 = tree2.size() - n2.size() + n1.size()
        if size1 <= max_nodes and size2 <= max_nodes:
            n1.value, n2.value = n2.value, n1.value
            n1.children, n2.children = n2.children, n1.children
    
    return tree1, tree2

def smart_mutation(tree, prob=0.2, implicit=False, max_nodes=20):
    tree = tree.copy()
    
    mutation_type = random.random()
    
    if mutation_type < prob:
        new_tree, _ = random_tree(depth=3, method='grow', implicit=implicit, max_nodes=max_nodes)
        return new_tree
    elif mutation_type < prob * 1.5:
        if random.random() < 0.5 and tree.value in PRIMITIVES:
            ops = [k for k, v in PRIMITIVES.items() if v[1] == PRIMITIVES[tree.value][1]]
            tree.value = random.choice(ops)
        elif tree.value in TERMINALS:
            terminals = ['x', 'y', 'pi', 'e', '1', '2', '0.5'] if implicit else ['t', 'pi', 'e', '1', '2', '0.5']
            tree.value = random.choice(terminals)
    
    for i in range(len(tree.children)):
        if tree.size() < max_nodes:
            tree.children[i] = smart_mutation(tree.children[i], prob, implicit, max_nodes)
    
    return tree

# ============= ENHANCED FITNESS WITH CACHING =============
@st.cache_data
def compute_contour_segments(_X, _Y, _Z):
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        cs = ax.contour(_X, _Y, _Z, levels=[0])
        num_segments = sum(len(col.get_paths()) for col in cs.collections)
        plt.close(fig)
        return num_segments
    except:
        plt.close(fig)
        return 0

def compute_fitness(x_tree, y_tree=None, plot_mode='parametric', 
                    weights=None, diversity_bonus=0):
    if weights is None:
        weights = {'even': 0.3, 'odd': 0.1, 'rot': 0.25, 'comp': 0.2, 'smooth': 0.15}
    
    try:
        scores = {}
        
        if plot_mode == 'parametric':
            t = np.linspace(0, 2 * np.pi, 200)
            x = np.array([x_tree.evaluate(t=ti) for ti in t])
            y = np.array([y_tree.evaluate(t=ti) for ti in t])
            
            x = np.clip(x, -100, 100)
            y = np.clip(y, -100, 100)
            
            t_neg = np.linspace(-2 * np.pi, 0, 200)
            x_neg = np.array([x_tree.evaluate(t=ti) for ti in t_neg])
            y_neg = np.array([y_tree.evaluate(t=ti) for ti in t_neg])
            x_neg = np.clip(x_neg, -100, 100)
            y_neg = np.clip(y_neg, -100, 100)
            
            diff_even = np.mean(np.sqrt((x_neg[::-1] - x)**2 + (y_neg[::-1] - y)**2))
            scores['even'] = np.exp(-diff_even / 10)
            
            diff_odd = np.mean(np.sqrt((x_neg[::-1] + x)**2 + (y_neg[::-1] + y)**2))
            scores['odd'] = np.exp(-diff_odd / 10)
            
            x_rot = -x[::-1]
            y_rot = -y[::-1]
            diff_rot = np.mean(np.sqrt((x - x_rot)**2 + (y - y_rot)**2))
            scores['rot'] = np.exp(-diff_rot / 10)
            
            dx, dy = np.gradient(x), np.gradient(y)
            curvature = np.abs(np.gradient(dx) * dy - np.gradient(dy) * dx) / (dx**2 + dy**2 + 1e-10)**1.5
            scores['comp'] = np.tanh(np.sum(np.abs(np.diff(np.sign(curvature)))) / 30.0)
            
            smoothness = np.mean(np.abs(np.diff(dx))) + np.mean(np.abs(np.diff(dy)))
            scores['smooth'] = np.exp(-smoothness / 15.0)
            
        elif plot_mode == 'polar':
            t = np.linspace(0, 2 * np.pi, 200)
            r = np.array([x_tree.evaluate(t=ti) for ti in t])
            r = np.clip(r, -100, 100)
            
            r_neg = np.array([x_tree.evaluate(t=-ti) for ti in t])
            r_neg = np.clip(r_neg, -100, 100)
            scores['even'] = np.exp(-np.mean(np.abs(r_neg[::-1] - r)) / 10)
            
            scores['odd'] = np.exp(-np.mean(np.abs(r_neg[::-1] + r)) / 10)
            
            rot_scores = []
            for n in [2, 3, 4, 5, 6]:
                t_rot = (t + 2 * np.pi / n) % (2 * np.pi)
                r_rot = np.array([x_tree.evaluate(t=ti) for ti in t_rot])
                r_rot = np.clip(r_rot, -100, 100)
                rot_scores.append(np.exp(-np.mean(np.abs(r - r_rot)) / 10))
            scores['rot'] = max(rot_scores)
            
            dr = np.gradient(r)
            scores['comp'] = np.tanh(np.sum(np.abs(np.diff(np.sign(dr)))) / 30.0)
            
            scores['smooth'] = np.exp(-np.mean(np.abs(np.gradient(dr))) / 7.5)
            
        else:  # implicit
            res = 40
            x = np.linspace(-5, 5, res)
            y = np.linspace(-5, 5, res)
            X, Y = np.meshgrid(x, y)
            Z = np.array([[x_tree.evaluate(x=xi, y=yi) for xi in x] for yi in y])
            Z = np.clip(Z, -100, 100)
            
            has_zero_crossing = np.min(Z) < 0 < np.max(Z)
            
            scores['even'] = np.exp(-np.mean(np.abs(Z - Z[::-1, ::-1])) / 10)
            scores['odd'] = np.exp(-np.mean(np.abs(Z + Z[::-1, ::-1])) / 10)
            
            Z_rot90 = np.rot90(Z)
            Z_rot180 = np.rot90(Z, 2)
            scores['rot'] = max(np.exp(-np.mean(np.abs(Z - Z_rot90)) / 10), 
                               np.exp(-np.mean(np.abs(Z - Z_rot180)) / 10))
            
            num_segments = compute_contour_segments(X, Y, Z)
            scores['comp'] = np.tanh(num_segments / 20.0) if num_segments > 0 else 0
            
            grad_x, grad_y = np.gradient(Z)
            smoothness = np.mean(np.sqrt(grad_x**2 + grad_y**2))
            scores['smooth'] = np.exp(-smoothness / 7.5)
            
            if not has_zero_crossing:
                scores['comp'] = 0
    
        tree_size = x_tree.size() + (y_tree.size() if y_tree else 0)
        size_penalty = 0.7 if tree_size < 3 else (0.8 if tree_size < 5 else 
                         (0.9 if tree_size > 40 else (0.95 if tree_size > 50 else 1.0)))
        
        total = sum(weights[k] * scores[k] for k in scores.keys())
        total = total * size_penalty + diversity_bonus * 2
        
        return max(0, min(15, total * 7)), scores
        
    except Exception as e:
        st.warning(f"Fitness calculation error: {e}")
        return 0, {}

# ============= VISUALIZATION WITH PRETTY EXPRESSIONS =============
def plot_generation(population, scores, gen, plot_mode, top_n=3, line_width=1.5, line_color='blue'):
    sorted_pairs = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)[:top_n]
    
    figs = []
    for idx, (score, (x_tree, y_tree)) in enumerate(sorted_pairs):
        fig, ax = plt.subplots(figsize=(6, 6))  # Individual figure for each plot
        try:
            if plot_mode == 'parametric':
                t = np.linspace(0, 2 * np.pi, 2000)  # Increased resolution
                x = np.array([x_tree.evaluate(t=ti) for ti in t])
                y = np.array([y_tree.evaluate(t=ti) for ti in t])
                x = np.clip(x, -100, 100)
                y = np.clip(y, -100, 100)
                # Filter out NaN or infinite values
                valid_mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isinf(x)) & (~np.isinf(y))
                x, y = x[valid_mask], y[valid_mask]
                if len(x) > 1 and not (np.all(np.isnan(x)) or np.all(np.isnan(y))):
                    ax.plot(x, y, linewidth=line_width, color=line_color)
                    ax.set_aspect('equal')
                    ax.set_title(f'Top {idx+1} - Score: {score:.3f}\nx = {x_tree.pretty_str()}\ny = {y_tree.pretty_str()}\nNodes: {x_tree.size() + y_tree.size()}', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'Invalid Plot', ha='center', va='center', transform=ax.transAxes)
            elif plot_mode == 'polar':
                t = np.linspace(0, 2 * np.pi, 300)
                r = np.array([x_tree.evaluate(t=ti) for ti in t])
                r = np.clip(r, -100, 100)
                x, y = r * np.cos(t), r * np.sin(t)
                if not (np.all(np.isnan(x)) or np.all(np.isnan(y))):
                    ax.plot(x, y, linewidth=line_width, color=line_color)
                    ax.set_aspect('equal')
                    ax.set_title(f'Top {idx+1} - Score: {score:.3f}\nr = {x_tree.pretty_str()}\nNodes: {x_tree.size()}', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'Invalid Plot', ha='center', va='center', transform=ax.transAxes)
            else:  # implicit
                res = 50
                x_vals = np.linspace(-5, 5, res)
                y_vals = np.linspace(-5, 5, res)
                X, Y = np.meshgrid(x_vals, y_vals)
                Z = np.array([[x_tree.evaluate(x=xi, y=yi) for xi in x_vals] for yi in y_vals])
                Z = np.clip(Z, -50, 50)
                cs = ax.contour(X, Y, Z, levels=[0], colors=line_color, linewidths=line_width)
                if cs.collections:
                    ax.set_aspect('equal')
                    ax.set_title(f'Top {idx+1} - Score: {score:.3f}\n{x_tree.pretty_str()} = 0\nNodes: {x_tree.size()}', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'No Contours', ha='center', va='center', transform=ax.transAxes)
            
            ax.grid(True, alpha=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            
            # Add download button for this individual plot
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            st.download_button(
                label=f"Download Top {idx+1} Plot as PNG",
                data=buf.getvalue(),
                file_name=f"math_art_top_{idx+1}_gen_{gen}.png",
                mime="image/png",
                key=f"download_top_{idx+1}_gen_{gen}"
            )
            
            figs.append(fig)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center', transform=ax.transAxes)
            plt.close(fig)
            figs.append(None)
        
        st.pyplot(fig)
        plt.close(fig)
    
    return figs

def plot_best(best_individual, plot_mode, score, line_width=2, line_color='blue'):
    x_tree, y_tree = best_individual
    fig, ax = plt.subplots(figsize=(10, 8))
    
    try:
        if plot_mode == 'parametric':
            t = np.linspace(0, 2 * np.pi, 5000)  # Increased resolution
            x = np.array([x_tree.evaluate(t=ti) for ti in t])
            y = np.array([y_tree.evaluate(t=ti) for ti in t])
            x = np.clip(x, -100, 100)
            y = np.clip(y, -100, 100)
            # Filter out NaN or infinite values
            valid_mask = (~np.isnan(x)) & (~np.isnan(y)) & (~np.isinf(x)) & (~np.isinf(y))
            x, y = x[valid_mask], y[valid_mask]
            if len(x) > 1 and not (np.all(np.isnan(x)) or np.all(np.isnan(y))):
                ax.plot(x, y, linewidth=line_width, color=line_color)
                ax.set_aspect('equal')
                ax.set_title(f'Best Plot - Score: {score:.3f}\nx = {x_tree.pretty_str()}\ny = {y_tree.pretty_str()}\nNodes: {x_tree.size() + y_tree.size()}', fontsize=12)
            else:
                ax.text(0.5, 0.5, 'Invalid Plot', ha='center', va='center', transform=ax.transAxes)
        elif plot_mode == 'polar':
            t = np.linspace(0, 2 * np.pi, 1500)
            r = np.array([x_tree.evaluate(t=ti) for ti in t])
            r = np.clip(r, -100, 100)
            x, y = r * np.cos(t), r * np.sin(t)
            if not (np.all(np.isnan(x)) or np.all(np.isnan(y))):
                ax.plot(x, y, linewidth=line_width, color=line_color)
                ax.set_aspect('equal')
                ax.set_title(f'Best Plot - Score: {score:.3f}\nr = {x_tree.pretty_str()}\nNodes: {x_tree.size()}', fontsize=12)
            else:
                ax.text(0.5, 0.5, 'Invalid Plot', ha='center', va='center', transform=ax.transAxes)
        else:  # implicit
            res = 100
            x_vals = np.linspace(-5, 5, res)
            y_vals = np.linspace(-5, 5, res)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = np.array([[x_tree.evaluate(x=xi, y=yi) for xi in x_vals] for yi in y_vals])
            Z = np.clip(Z, -50, 50)
            cs = ax.contour(X, Y, Z, levels=[0], colors=line_color, linewidths=line_width)
            if cs.collections:
                ax.set_aspect('equal')
                ax.set_title(f'Best Plot - Score: {score:.3f}\n{x_tree.pretty_str()} = 0\nNodes: {x_tree.size()}', fontsize=12)
            else:
                ax.text(0.5, 0.5, 'No Contours', ha='center', va='center', transform=ax.transAxes)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
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
    return fig

# ============= MAIN ALGORITHM =============
def evolve_art(generations, pop_size, plot_mode, weights, mutation_rate, elite_size, diversity_bonus, max_nodes):
    population = []
    for i in range(pop_size):
        method = 'grow' if i < pop_size // 2 else 'full'
        depth = random.randint(2, 5)
        if plot_mode != 'implicit':
            x_tree, _ = random_tree(depth=depth, method=method, implicit=False, max_nodes=max_nodes)
            y_tree, _ = random_tree(depth=depth, method=method, implicit=False, max_nodes=max_nodes)
            population.append((x_tree, y_tree))
        else:
            x_tree, _ = random_tree(depth=depth, method=method, implicit=True, max_nodes=max_nodes)
            population.append((x_tree, None))
    
    best_ever_score = 0
    best_ever_individual = None
    stagnation_counter = 0
    
    for gen in range(generations):
        scores = []
        for x_tree, y_tree in population:
            score, _ = compute_fitness(x_tree, y_tree, plot_mode, weights, diversity_bonus)
            scores.append(score)
        
        max_score = max(scores)
        if max_score > best_ever_score:
            best_ever_score = max_score
            best_idx = scores.index(max_score)
            best_ever_individual = (population[best_idx][0].copy(), 
                                   population[best_idx][1].copy() if population[best_idx][1] else None)
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        yield gen + 1, max_score, best_ever_score, population, scores
        
        new_population = []
        sorted_pairs = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        elites = [ind for _, ind in sorted_pairs[:elite_size]]
        new_population.extend([(e[0].copy(), e[1].copy() if e[1] else None) for e in elites])
        
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, scores, k=3)
            parent2 = tournament_selection(population, scores, k=3)
            
            if plot_mode != 'implicit':
                child1_x, child2_x = subtree_crossover(parent1[0], parent2[0], max_nodes)
                child1_y, child2_y = subtree_crossover(parent1[1], parent2[1], max_nodes)
                
                child1_x = smart_mutation(child1_x, mutation_rate, False, max_nodes)
                child1_y = smart_mutation(child1_y, mutation_rate, False, max_nodes)
                new_population.append((child1_x, child1_y))
                
                if len(new_population) < pop_size:
                    child2_x = smart_mutation(child2_x, mutation_rate, False, max_nodes)
                    child2_y = smart_mutation(child2_y, mutation_rate, False, max_nodes)
                    new_population.append((child2_x, child2_y))
            else:
                child1_x, child2_x = subtree_crossover(parent1[0], parent2[0], max_nodes)
                child1_x = smart_mutation(child1_x, mutation_rate, True, max_nodes)
                new_population.append((child1_x, None))
                
                if len(new_population) < pop_size:
                    child2_x = smart_mutation(child2_x, mutation_rate, True, max_nodes)
                    new_population.append((child2_x, None))
        
        population = new_population[:pop_size]
        
        if stagnation_counter > 5:
            mutation_rate = min(0.4, mutation_rate * 1.3)
        elif stagnation_counter == 0:
            mutation_rate = max(0.1, mutation_rate * 0.8)
    
    st.session_state['best_individual'] = best_ever_individual
    st.session_state['best_score'] = best_ever_score
    st.session_state['best_individual_dict'] = (
        best_ever_individual[0].to_dict(),
        best_ever_individual[1].to_dict() if best_ever_individual[1] else None
    )
    return best_ever_individual, best_ever_score

# ============= STREAMLIT UI =============
def main():
    st.set_page_config(page_title="Math Art Evolver", page_icon="🎨", layout="wide")
    st.title("🎨 Math Art Evolver")
    st.markdown("Evolve beautiful mathematical plots using genetic programming. Adjust parameters to create stunning art!")

    if 'running' not in st.session_state:
        st.session_state.running = False
        st.session_state.best_individual = None
        st.session_state.best_score = 0
        st.session_state.best_individual_dict = None

    st.sidebar.header("Evolution Parameters")
    plot_mode = st.sidebar.selectbox("Plot Mode", ['parametric', 'polar', 'implicit'], index=0,
                                     help="Parametric: x(t), y(t); Polar: r(t); Implicit: f(x,y)=0")
    generations = st.sidebar.number_input("Generations", min_value=1, max_value=500, value=15,
                                         help="Number of evolution cycles (higher = longer but better results)")
    pop_size = st.sidebar.number_input("Population Size", min_value=10, max_value=500, value=30,
                                       help="Number of individuals per generation")
    elite_size = st.sidebar.slider("Elite Size", 1, min(10, pop_size), 3,
                                   help="Number of top individuals preserved each generation")
    mutation_rate = st.sidebar.slider("Mutation Rate", 0.05, 0.5, 0.15,
                                      help="Probability of random changes in expressions")
    diversity_bonus = st.sidebar.slider("Diversity Bonus", 0.0, 1.0, 0.1,
                                        help="Encourages diverse patterns")
    max_nodes = st.sidebar.number_input("Max Nodes per Tree", min_value=2, max_value=100, value=20,
                                        help="Maximum number of nodes (subfunctions) in each expression tree")

    st.sidebar.header("Aesthetic Weights")
    st.sidebar.markdown("Adjust weights to prioritize visual qualities:")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        even_weight = st.slider("Even Symmetry", 0.0, 1.0, 0.3, help="Favors mirror-like symmetry (e.g., ellipses)")
        odd_weight = st.slider("Odd Symmetry", 0.0, 1.0, 0.1, help="Favors point reflection symmetry")
    with col2:
        rot_weight = st.slider("Rotational Symmetry", 0.0, 1.0, 0.25, help="Favors star-like or floral patterns")
        comp_weight = st.slider("Complexity", 0.0, 1.0, 0.2, help="Encourages intricate patterns")
    smooth_weight = st.sidebar.slider("Smoothness", 0.0, 1.0, 0.15, help="Promotes smooth curves")

    weights = {'even': even_weight, 'odd': odd_weight, 'rot': rot_weight, 'comp': comp_weight, 'smooth': smooth_weight}
    weight_sum = sum(weights.values())
    if weight_sum > 0:
        weights = {k: v / weight_sum for k, v in weights.items()}

    st.sidebar.header("Plot Style")
    line_width = st.sidebar.slider("Line Width", 0.5, 5.0, 1.5, help="Thickness of plot lines")
    line_color = st.sidebar.color_picker("Line Color", "#0000FF", help="Color of plot lines")

    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        evolve_button = st.button("🚀 Evolve Art!")
    with col2:
        stop_button = st.button("🛑 Stop Evolution")
    with col3:
        save_button = st.button("💾 Save Best")

    if save_button and st.session_state.best_individual_dict:
        save_data = {
            'plot_mode': plot_mode,
            'best_score': st.session_state.best_score,
            'x_tree': st.session_state.best_individual_dict[0],
            'y_tree': st.session_state.best_individual_dict[1]
        }
        buf = io.BytesIO()
        buf.write(json.dumps(save_data, indent=2).encode('utf-8'))
        buf.seek(0)
        st.download_button(
            label="Download Evolution State as JSON",
            data=buf.getvalue(),
            file_name="math_art_state.json",
            mime="application/json"
        )

    if stop_button and st.session_state.running:
        st.session_state.running = False
        st.success("Evolution stopped. Displaying best result so far.")
        if st.session_state.best_individual:
            fig = plot_best(st.session_state.best_individual, plot_mode, st.session_state.best_score, line_width, line_color)
            plt.close(fig)
        return

    if evolve_button and not st.session_state.running:
        st.session_state.running = True
        with st.spinner("Evolving mathematical art..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for gen, max_score, best_ever_score, population, scores in evolve_art(
                generations, pop_size, plot_mode, weights, mutation_rate, elite_size, diversity_bonus, max_nodes
            ):
                if not st.session_state.running:
                    break
                
                progress = gen / generations
                progress_bar.progress(progress)
                status_text.text(f"Generation {gen}/{generations}: Best {max_score:.3f} | All-time Best: {best_ever_score:.3f}")
                
                if gen % max(1, generations // 5) == 0 or gen == generations:
                    figs = plot_generation(population, scores, gen, plot_mode, top_n=3, line_width=line_width, line_color=line_color)
                    for fig in figs:
                        if fig:
                            plt.close(fig)
            
            if st.session_state.running:
                st.session_state.running = False
                st.success(f"✨ Evolution Complete! Best Score: {st.session_state.best_score:.3f}")
                if st.session_state.best_individual:
                    fig = plot_best(st.session_state.best_individual, plot_mode, st.session_state.best_score, line_width, line_color)
                    plt.close(fig)

if __name__ == "__main__":
    main()
