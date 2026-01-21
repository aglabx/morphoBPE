from tokenizer import Tokenizer
from typing import Dict, List, Tuple, Set
import argparse
from collections import defaultdict
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
from graphviz import Digraph
import os

class MergeStructureComparator:
    def __init__(self, tokenizer1: Tokenizer, tokenizer2: Tokenizer):
        self.t1 = tokenizer1
        self.t2 = tokenizer2
        self.graph1 = self._build_merge_graph(tokenizer1)
        self.graph2 = self._build_merge_graph(tokenizer2)
        
    def _build_merge_graph(self, tokenizer: Tokenizer) -> nx.DiGraph:
        """Build a directed graph representing merge operations."""
        G = nx.DiGraph()
        
        # Add all basic characters first
        for token in tokenizer.vocab:
            if len(token) == 1:
                G.add_node(token, type='char')
                
        # Add merge operations as edges
        for merge in tokenizer.merges:
            first, second = merge.split()
            result = first + second
            G.add_node(result, type='merge')
            G.add_edge(first, result)
            G.add_edge(second, result)
            
        return G
    
    def get_merge_paths(self, graph: nx.DiGraph) -> Dict[str, List[List[str]]]:
        """Get all possible merge paths for each token."""
        paths = defaultdict(list)
        
        for node in graph.nodes():
            if graph.out_degree(node) == 0:  # Terminal nodes
                for predecessor in graph.predecessors(node):
                    path = nx.shortest_path(graph, predecessor, node)
                    paths[node].append(path)
                    
        return paths
    
    def compare_structures(self) -> dict:
        """Compare merge structures between tokenizers."""
        paths1 = self.get_merge_paths(self.graph1)
        paths2 = self.get_merge_paths(self.graph2)
        
        result = {
            'common_tokens': set(paths1.keys()) & set(paths2.keys()),
            'unique_tokens1': set(paths1.keys()) - set(paths2.keys()),
            'unique_tokens2': set(paths2.keys()) - set(paths1.keys()),
            'common_paths': defaultdict(list),
            'different_paths': defaultdict(dict)
        }
        
        # Analyze common tokens
        for token in result['common_tokens']:
            paths_t1 = set(tuple(path) for path in paths1[token])
            paths_t2 = set(tuple(path) for path in paths2[token])
            
            common_paths = paths_t1 & paths_t2
            if common_paths:
                result['common_paths'][token] = list(common_paths)
            
            diff_t1 = paths_t1 - paths_t2
            diff_t2 = paths_t2 - paths_t1
            if diff_t1 or diff_t2:
                result['different_paths'][token] = {
                    'tokenizer1': list(diff_t1),
                    'tokenizer2': list(diff_t2)
                }
                
        return result
    
    def get_statistics(self) -> dict:
        """Get statistical comparison of merge structures."""
        return {
            'tokenizer1': {
                'total_tokens': len(self.graph1.nodes()),
                'total_merges': len(self.t1.merges),
                'chars': len([n for n, d in self.graph1.nodes(data=True) if d['type'] == 'char']),
                'max_depth': max(len(nx.ancestors(self.graph1, n)) for n in self.graph1.nodes() if self.graph1.out_degree(n) == 0)
            },
            'tokenizer2': {
                'total_tokens': len(self.graph2.nodes()),
                'total_merges': len(self.t2.merges),
                'chars': len([n for n, d in self.graph2.nodes(data=True) if d['type'] == 'char']),
                'max_depth': max(len(nx.ancestors(self.graph2, n)) for n in self.graph2.nodes() if self.graph2.out_degree(n) == 0)
            }
        }
    
    def print_comparison(self, result: dict, stats: dict):
        """Pretty print the comparison results."""
        print("\nTokenizer Statistics:")
        print(f"Tokenizer 1: {stats['tokenizer1']}")
        print(f"Tokenizer 2: {stats['tokenizer2']}")
        
        print(f"\nCommon tokens: {len(result['common_tokens'])}")
        print(f"Unique to tokenizer 1: {len(result['unique_tokens1'])}")
        print(f"Unique to tokenizer 2: {len(result['unique_tokens2'])}")
        
        print("\nSample of different merge paths:")
        sample_size = min(5, len(result['different_paths']))
        for token in list(result['different_paths'].keys())[:sample_size]:
            print(f"\nToken: {token}")
            print("Tokenizer 1 paths:")
            for path in result['different_paths'][token]['tokenizer1'][:2]:
                print(f"  {' -> '.join(path)}")
            print("Tokenizer 2 paths:")
            for path in result['different_paths'][token]['tokenizer2'][:2]:
                print(f"  {' -> '.join(path)}")
    
    def analyze_merge_complexity(self, graph: nx.DiGraph) -> dict:
        """Analyze the complexity of merge paths in the graph."""
        analysis = {
            'branching_factors': [],  # количество разных путей образования токена
            'merge_depths': [],       # глубина цепочек мерджей
            'reuse_counts': {},       # как часто токен используется в разных мерджах
            'complex_tokens': []      # токены с множественными путями образования
        }
        
        # Find all leaf nodes (nodes with no outgoing edges)
        leaf_nodes = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        # Find all root nodes (nodes with no incoming edges)
        root_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        
        for node in graph.nodes():
            # Count all paths from any root to this node
            all_paths = []
            for source in root_nodes:
                try:
                    paths = list(nx.all_simple_paths(graph, source, node))
                    all_paths.extend(paths)
                except nx.NetworkXNoPath:
                    continue
            
            if len(all_paths) > 1:
                analysis['complex_tokens'].append((node, len(all_paths)))
                analysis['branching_factors'].append(len(all_paths))
            
            # Calculate depth (length of longest path)
            if all_paths:
                max_depth = max(len(path) for path in all_paths)
                analysis['merge_depths'].append(max_depth)
            
            # Count token reuse
            reuse = graph.out_degree(node)
            if reuse > 0:
                analysis['reuse_counts'][node] = reuse
        
        return analysis
    
    def visualize_full_graph(self, graph: nx.DiGraph, filename: str):
        """Create a detailed visualization of the merge graph using graphviz."""
        dot = Digraph(comment='Merge Graph')
        dot.attr(rankdir='LR')
        
        # Добавляем узлы с разными стилями
        for node in graph.nodes():
            attrs = {}
            if len(node) == 1:  # базовые символы
                attrs['shape'] = 'circle'
                attrs['color'] = 'blue'
            else:  # составные токены
                attrs['shape'] = 'box'
                attrs['color'] = 'green'
            
            # Добавляем информацию о частоте использования
            reuse_count = graph.out_degree(node)
            if reuse_count > 0:
                attrs['label'] = f"{node}\n(reuse: {reuse_count})"
            else:
                attrs['label'] = node
                
            dot.node(node, **attrs)
        
        # Добавляем ребра
        for edge in graph.edges():
            dot.edge(edge[0], edge[1])
        
        # Сохраняем в разных форматах
        dot.render(filename, format='png', cleanup=True)
        dot.render(filename, format='svg', cleanup=True)
    
    def visualize_token_formation(self, token: str, max_paths: int = 5):
        """Visualize different formation paths for a specific token."""
        plt.figure(figsize=(15, 8))
        
        paths1 = list(nx.all_simple_paths(self.graph1, source=None, target=token))
        paths2 = list(nx.all_simple_paths(self.graph2, source=None, target=token))
        
        plt.subplot(121)
        self._plot_paths(paths1[:max_paths], "Tokenizer 1")
        
        plt.subplot(122)
        self._plot_paths(paths2[:max_paths], "Tokenizer 2")
        
        plt.tight_layout()
        return plt.gcf()
    
    def _plot_paths(self, paths: List[List[str]], title: str):
        """Helper method to plot paths in a readable format."""
        plt.title(title)
        for i, path in enumerate(paths):
            y_positions = [len(paths) - i] * len(path)
            plt.plot(range(len(path)), y_positions, 'o-', label=f'Path {i+1}')
            
            # Добавляем метки токенов
            for j, token in enumerate(path):
                plt.text(j, len(paths) - i, token, 
                        ha='right', va='bottom')
        
        plt.grid(True)
        plt.yticks([])
        plt.xlabel('Merge Steps')
    
    def print_extended_analysis(self):
        """Print detailed analysis of both tokenizers."""
        analysis1 = self.analyze_merge_complexity(self.graph1)
        analysis2 = self.analyze_merge_complexity(self.graph2)
        
        print("\nDetailed Merge Analysis:")
        print("\nTokenizer 1:")
        print(f"Average branching factor: {sum(analysis1['branching_factors'])/len(analysis1['branching_factors']):.2f}")
        print(f"Average merge depth: {sum(analysis1['merge_depths'])/len(analysis1['merge_depths']):.2f}")
        print(f"Most reused tokens: {sorted(analysis1['reuse_counts'].items(), key=lambda x: x[1], reverse=True)[:5]}")
        print(f"Most complex tokens: {sorted(analysis1['complex_tokens'], key=lambda x: x[1], reverse=True)[:5]}")
        
        print("\nTokenizer 2:")
        print(f"Average branching factor: {sum(analysis2['branching_factors'])/len(analysis2['branching_factors']):.2f}")
        print(f"Average merge depth: {sum(analysis2['merge_depths'])/len(analysis2['merge_depths']):.2f}")
        print(f"Most reused tokens: {sorted(analysis2['reuse_counts'].items(), key=lambda x: x[1], reverse=True)[:5]}")
        print(f"Most complex tokens: {sorted(analysis2['complex_tokens'], key=lambda x: x[1], reverse=True)[:5]}")

def main():
    parser = argparse.ArgumentParser(description='Compare merge structures of two BPE tokenizers')
    parser.add_argument('vocab1', help='Path to first vocabulary file')
    parser.add_argument('vocab2', help='Path to second vocabulary file')
    parser.add_argument('--output-dir', default='analysis_output', help='Directory for output files')
    parser.add_argument('--token', help='Specific token to analyze', default=None)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer1 = Tokenizer(args.vocab1)
    tokenizer2 = Tokenizer(args.vocab2)
    
    comparator = MergeStructureComparator(tokenizer1, tokenizer2)
    
    # Basic comparison
    result = comparator.compare_structures()
    stats = comparator.get_statistics()
    comparator.print_comparison(result, stats)
    
    # Extended analysis
    comparator.print_extended_analysis()
    
    # Visualize full graphs
    comparator.visualize_full_graph(comparator.graph1, 
                                  f"{args.output_dir}/tokenizer1_graph")
    comparator.visualize_full_graph(comparator.graph2, 
                                  f"{args.output_dir}/tokenizer2_graph")
    
    # Analyze specific token if provided
    if args.token:
        fig = comparator.visualize_token_formation(args.token)
        fig.savefig(f"{args.output_dir}/token_{args.token}_analysis.png")
        plt.close(fig)

if __name__ == "__main__":
    main()
