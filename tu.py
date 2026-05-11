"""
CSI Feedback领域图模型可视化
CSI Feedback Domain Graph Model Visualization
Using NetworkX and Matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

# Use sans-serif font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def create_csi_feedback_graph():
    """Create CSI Feedback domain graph structure"""
    
    G = nx.DiGraph()
    
    # ==================== Node Definitions ====================
    
    # Core Challenges Layer (C1-C7)
    challenges = {
        'C1\nGeneralization\nBottleneck': {
            'layer': 0, 'type': 'challenge',
            'desc': 'DL models degrade\nin unseen environments',
            'papers': '[19,21,23,28,32,34]'
        },
        'C2\nUE Compute\nConstraint': {
            'layer': 0, 'type': 'challenge',
            'desc': 'Mobile terminals cannot\ngrun complex DNN inference',
            'papers': '[14,16]'
        },
        'C3\nLimited Bit\nBudget': {
            'layer': 0, 'type': 'challenge',
            'desc': 'Feedback bandwidth\nis severely limited',
            'papers': '[15,26]'
        },
        'C4\nDistribution\nMismatch': {
            'layer': 0, 'type': 'challenge',
            'desc': 'Train/test channel\ndistributions differ',
            'papers': '[23,28,34]'
        },
        'C5\nTemporal\nCorrelation': {
            'layer': 0, 'type': 'challenge',
            'desc': 'Channel varies over\ntime, correlated across t',
            'papers': '[24,31]'
        },
        'C6\nMulti-User\nCorrelation': {
            'layer': 0, 'type': 'challenge',
            'desc': 'Nearby users\' channels\nhave spatial correlation',
            'papers': '[13,27]'
        },
        'C7\nNovel System\nAdaptation': {
            'layer': 0, 'type': 'challenge',
            'desc': 'RIS, XL-MIMO bring\nnew challenges',
            'papers': '[12,29,30,36]'
        },
    }
    
    # Technical Methods Layer (M1-M6)
    methods = {
        'M1\nCNN/\nEncoder': {
            'layer': 1, 'type': 'method',
            'desc': 'CsiNet, CRNet,\nlightweight networks',
            'papers': '[16]'
        },
        'M2\nTransformer': {
            'layer': 1, 'type': 'method',
            'desc': 'TransNet,\nself-attention',
            'papers': '[10,11]'
        },
        'M3\nLLM-Driven': {
            'layer': 1, 'type': 'method',
            'desc': 'GPT-2 finetuning,\nprompting, offline codebook',
            'papers': '[1-8,38,39]'
        },
        'M4\nCompressed\nSensing': {
            'layer': 1, 'type': 'method',
            'desc': 'Sparse recovery,\nOMP, TVAL3',
            'papers': '[17,18]'
        },
        'M5\nHybrid/\nFusion': {
            'layer': 1, 'type': 'method',
            'desc': 'CNN-Transformer mix,\nmulti-domain, sensing-assisted',
            'papers': '[10,11,24,31]'
        },
        'M6\nFoundation\nModel': {
            'layer': 1, 'type': 'method',
            'desc': 'WiFo-CF,\nS-R MoE',
            'papers': '[21]'
        },
    }
    
    # Performance Metrics Layer (P1-P6)
    metrics = {
        'P1\nNMSE': {
            'layer': 2, 'type': 'metric',
            'desc': 'Normalized MSE between\nreconstructed & true CSI'
        },
        'P2\nCompression\nRatio': {
            'layer': 2, 'type': 'metric',
            'desc': 'Original CSI dim /\nfeedback code dim'
        },
        'P3\nComputational\nComplexity': {
            'layer': 2, 'type': 'metric',
            'desc': 'FLOPs at UE/BS'
        },
        'P4\nGeneralization\nError': {
            'layer': 2, 'type': 'metric',
            'desc': 'Performance drop in\nunseen environments'
        },
        'P5\nCodebook\nDistortion': {
            'layer': 2, 'type': 'metric',
            'desc': 'Quantization loss\nfrom codebook'
        },
        'P6\nFLOPs/\nParams': {
            'layer': 2, 'type': 'metric',
            'desc': 'Model size\nquantitative metric'
        },
    }
    
    # Application Scenarios Layer (S1-S4)
    scenarios = {
        'S1\nFDD Massive\nMIMO': {
            'layer': 3, 'type': 'scenario',
            'desc': 'Traditional CSI feedback,\nmany antennas'
        },
        'S2\nRIS-Assisted': {
            'layer': 3, 'type': 'scenario',
            'desc': 'Reconfigurable\nIntelligent Surface'
        },
        'S3\nXL-MIMO/\nNear-Field': {
            'layer': 3, 'type': 'scenario',
            'desc': 'Very large array,\nnear-field effects'
        },
        'S4\nWi-Fi/\nLAN': {
            'layer': 3, 'type': 'scenario',
            'desc': 'IEEE 802.11,\nstrong temporal correlation'
        },
    }
    
    # Application Objectives Layer (A1-A3)
    applications = {
        'A1\nPrecoding\nAccuracy': {
            'layer': 4, 'type': 'application',
            'desc': 'Downlink precoding\nmatrix accuracy'
        },
        'A2\nSystem\nThroughput': {
            'layer': 4, 'type': 'application',
            'desc': 'Achievable rate in\npractical scenarios'
        },
        'A3\nResource\nAllocation': {
            'layer': 4, 'type': 'application',
            'desc': 'Power/spectrum\nresource optimization'
        },
    }
    
    # Merge all nodes
    all_nodes = {**challenges, **methods, **metrics, **scenarios, **applications}
    
    for node, attrs in all_nodes.items():
        G.add_node(node, **attrs)
    
    # ==================== Edge Definitions ====================
    
    # Solve Relation (Challenge -> Method)
    solve_edges = [
        ('C1\nGeneralization\nBottleneck', 'M3\nLLM-Driven'),
        ('C1\nGeneralization\nBottleneck', 'M6\nFoundation\nModel'),
        ('C1\nGeneralization\nBottleneck', 'M5\nHybrid/\nFusion'),
        ('C2\nUE Compute\nConstraint', 'M1\nCNN/\nEncoder'),
        ('C2\nUE Compute\nConstraint', 'M5\nHybrid/\nFusion'),
        ('C3\nLimited Bit\nBudget', 'M4\nCompressed\nSensing'),
        ('C3\nLimited Bit\nBudget', 'M5\nHybrid/\nFusion'),
        ('C4\nDistribution\nMismatch', 'M6\nFoundation\nModel'),
        ('C5\nTemporal\nCorrelation', 'M5\nHybrid/\nFusion'),
        ('C6\nMulti-User\nCorrelation', 'M5\nHybrid/\nFusion'),
        ('C7\nNovel System\nAdaptation', 'M5\nHybrid/\nFusion'),
    ]
    
    # Improve Relation (Method -> Method)
    improve_edges = [
        ('M1\nCNN/\nEncoder', 'M2\nTransformer'),
        ('M1\nCNN/\nEncoder', 'M5\nHybrid/\nFusion'),
        ('M2\nTransformer', 'M3\nLLM-Driven'),
        ('M2\nTransformer', 'M6\nFoundation\nModel'),
        ('M4\nCompressed\nSensing', 'M5\nHybrid/\nFusion'),
        ('M3\nLLM-Driven', 'M6\nFoundation\nModel'),
    ]
    
    # Evaluation Relation (Method -> Metric)
    eval_edges = [
        ('M1\nCNN/\nEncoder', 'P1\nNMSE'),
        ('M2\nTransformer', 'P1\nNMSE'),
        ('M3\nLLM-Driven', 'P1\nNMSE'),
        ('M3\nLLM-Driven', 'P4\nGeneralization\nError'),
        ('M1\nCNN/\nEncoder', 'P3\nComputational\nComplexity'),
        ('M5\nHybrid/\nFusion', 'P2\nCompression\nRatio'),
        ('M5\nHybrid/\nFusion', 'P5\nCodebook\nDistortion'),
        ('M6\nFoundation\nModel', 'P4\nGeneralization\nError'),
    ]
    
    # Metric to Application (Metric -> Application)
    metric_app_edges = [
        ('P1\nNMSE', 'A1\nPrecoding\nAccuracy'),
        ('P1\nNMSE', 'A2\nSystem\nThroughput'),
        ('P2\nCompression\nRatio', 'A2\nSystem\nThroughput'),
        ('P3\nComputational\nComplexity', 'A2\nSystem\nThroughput'),
        ('P4\nGeneralization\nError', 'A2\nSystem\nThroughput'),
        ('P5\nCodebook\nDistortion', 'A1\nPrecoding\nAccuracy'),
    ]
    
    # Scenario to Challenge (Scenario -> Challenge)
    scenario_challenge_edges = [
        ('S1\nFDD Massive\nMIMO', 'C1\nGeneralization\nBottleneck'),
        ('S1\nFDD Massive\nMIMO', 'C3\nLimited Bit\nBudget'),
        ('S2\nRIS-Assisted', 'C7\nNovel System\nAdaptation'),
        ('S2\nRIS-Assisted', 'C6\nMulti-User\nCorrelation'),
        ('S3\nXL-MIMO/\nNear-Field', 'C7\nNovel System\nAdaptation'),
        ('S3\nXL-MIMO/\nNear-Field', 'C1\nGeneralization\nBottleneck'),
        ('S4\nWi-Fi/\nLAN', 'C5\nTemporal\nCorrelation'),
        ('S4\nWi-Fi/\nLAN', 'C2\nUE Compute\nConstraint'),
    ]
    
    # Add all edges
    G.add_edges_from(solve_edges, relation='solve')
    G.add_edges_from(improve_edges, relation='improve')
    G.add_edges_from(eval_edges, relation='eval')
    G.add_edges_from(metric_app_edges, relation='metric_app')
    G.add_edges_from(scenario_challenge_edges, relation='scenario')
    
    return G, all_nodes


def draw_graph(G, all_nodes, save_path='csi_feedback_graph.png'):
    """Draw the graph model"""
    
    fig, ax = plt.subplots(1, 1, figsize=(28, 20))
    
    # Layer Y coordinates
    layer_y = {
        4: 0.93,  # Application layer
        3: 0.79,  # Scenario layer
        2: 0.63,  # Metric layer
        1: 0.42,  # Method layer
        0: 0.13,  # Challenge layer
    }
    
    pos = {}
    
    # Challenge layer (C1-C7)
    challenge_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'challenge']
    challenge_x = np.linspace(0.05, 0.95, len(challenge_nodes))
    for i, node in enumerate(sorted(challenge_nodes)):
        pos[node] = (challenge_x[i], layer_y[0])
    
    # Method layer (M1-M6)
    method_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'method']
    method_x_coords = [0.10, 0.30, 0.50, 0.70, 0.50, 0.90]
    for i, node in enumerate(sorted(method_nodes)):
        if i == 4:  # M5 in center
            pos[node] = (0.50, layer_y[1] - 0.06)
        else:
            pos[node] = (method_x_coords[i], layer_y[1])
    
    # Metric layer (P1-P6)
    metric_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'metric']
    metric_x = np.linspace(0.10, 0.90, len(metric_nodes))
    for i, node in enumerate(sorted(metric_nodes)):
        pos[node] = (metric_x[i], layer_y[2])
    
    # Scenario layer (S1-S4)
    scenario_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'scenario']
    scenario_x = np.linspace(0.12, 0.88, len(scenario_nodes))
    for i, node in enumerate(sorted(scenario_nodes)):
        pos[node] = (scenario_x[i], layer_y[3])
    
    # Application layer (A1-A3)
    app_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'application']
    app_x = [0.20, 0.50, 0.80]
    for i, node in enumerate(sorted(app_nodes)):
        pos[node] = (app_x[i], layer_y[4])
    
    # ==================== Draw Edges ====================
    
    edge_colors = {
        'solve': '#E74C3C',        # Red - Solve relation
        'improve': '#3498DB',      # Blue - Improve relation
        'eval': '#27AE60',         # Green - Evaluation relation
        'metric_app': '#9B59B6',   # Purple - Metric-Application
        'scenario': '#F39C12',     # Orange - Scenario-Challenge
    }
    
    edge_styles = {
        'solve': 'solid',
        'improve': 'dashed',
        'eval': 'dotted',
        'metric_app': 'dashdot',
        'scenario': 'solid',
    }
    
    # Draw edges
    for u, v, data in G.edges(data=True):
        relation = data.get('relation', 'solve')
        ax.annotate("",
                    xy=pos[v], xytext=pos[u],
                    arrowprops=dict(
                        arrowstyle='->',
                        color=edge_colors[relation],
                        lw=1.8,
                        connectionstyle="arc3,rad=0.08",
                        linestyle=edge_styles[relation],
                    ))
    
    # ==================== Draw Nodes ====================
    
    node_colors = {
        'challenge': '#FDEBD0',    # Light orange
        'method': '#D5E8D4',       # Light green
        'metric': '#DAE8FC',       # Light blue
        'scenario': '#E1D5E7',      # Light purple
        'application': '#FFE6CC',  # Light orange-yellow
    }
    
    node_borders = {
        'challenge': '#E67E22',
        'method': '#27AE60',
        'metric': '#2980B9',
        'scenario': '#8E44AD',
        'application': '#D35400',
    }
    
    # Draw each node
    for node, (x, y) in pos.items():
        node_type = G.nodes[node].get('type', 'challenge')
        
        if node_type == 'method':
            width, height = 0.11, 0.09
        elif node_type == 'challenge':
            width, height = 0.10, 0.09
        elif node_type == 'metric':
            width, height = 0.10, 0.09
        else:
            width, height = 0.09, 0.08
        
        # Draw rectangle node
        rect = mpatches.FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.01",
            facecolor=node_colors[node_type],
            edgecolor=node_borders[node_type],
            linewidth=2.5,
            alpha=0.95,
            zorder=3
        )
        ax.add_patch(rect)
        
        # Add node label
        ax.text(x, y, node, ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='#2C3E50', zorder=4)
        
        # Add description
        desc = G.nodes[node].get('desc', '')
        if desc:
            ax.text(x, y - height/2 - 0.022, desc,
                    ha='center', va='top', fontsize=5.5,
                    color='#7F8C8D', zorder=4,
                    style='italic')
        
        # Add paper references
        papers = G.nodes[node].get('papers', '')
        if papers:
            ax.text(x, y - height/2 - 0.058, papers,
                    ha='center', va='top', fontsize=5,
                    color='#95A5A6', zorder=4)
    
    # ==================== Draw Layer Labels ====================
    
    layer_labels = {
        4: ('Application Layer', '#D35400'),
        3: ('Scenario Layer', '#8E44AD'),
        2: ('Metric Layer', '#2980B9'),
        1: ('Method Layer', '#27AE60'),
        0: ('Challenge Layer', '#E67E22'),
    }
    
    for layer, (label, color) in layer_labels.items():
        ax.text(-0.03, layer_y[layer], label,
                ha='right', va='center',
                fontsize=12, fontweight='bold',
                color=color)
        
        ax.axhline(y=layer_y[layer] + 0.065, color=color,
                   linestyle='--', alpha=0.3, linewidth=1.2, xmin=0.05, xmax=0.98)
    
    # ==================== Legend ====================
    
    legend_elements = [
        Line2D([0], [0], color=edge_colors['solve'], lw=2.5, linestyle='solid',
               label='Solve (Challenge->Method)'),
        Line2D([0], [0], color=edge_colors['improve'], lw=2.5, linestyle='dashed',
               label='Improve (Method->Method)'),
        Line2D([0], [0], color=edge_colors['eval'], lw=2.5, linestyle='dotted',
               label='Evaluate (Method->Metric)'),
        Line2D([0], [0], color=edge_colors['metric_app'], lw=2.5, linestyle='dashdot',
               label='Apply (Metric->App)'),
        Line2D([0], [0], color=edge_colors['scenario'], lw=2.5, linestyle='solid',
               label='Scenario (Scenario->Challenge)'),
    ]
    
    node_legend = [
        mpatches.Patch(facecolor=node_colors['challenge'], edgecolor=node_borders['challenge'],
                       label='Challenge [C1-C7]', linewidth=2.5),
        mpatches.Patch(facecolor=node_colors['method'], edgecolor=node_borders['method'],
                       label='Method [M1-M6]', linewidth=2.5),
        mpatches.Patch(facecolor=node_colors['metric'], edgecolor=node_borders['metric'],
                       label='Metric [P1-P6]', linewidth=2.5),
        mpatches.Patch(facecolor=node_colors['scenario'], edgecolor=node_borders['scenario'],
                       label='Scenario [S1-S4]', linewidth=2.5),
        mpatches.Patch(facecolor=node_colors['application'], edgecolor=node_borders['application'],
                       label='Application [A1-A3]', linewidth=2.5),
    ]
    
    leg1 = ax.legend(handles=legend_elements, loc='upper right',
                    title='Edge Types', title_fontsize=11,
                    fontsize=9, framealpha=0.95,
                    bbox_to_anchor=(0.99, 0.99))
    ax.add_artist(leg1)
    
    ax.legend(handles=node_legend, loc='upper right',
             title='Node Types', title_fontsize=11,
             fontsize=9, framealpha=0.95,
             bbox_to_anchor=(0.99, 0.78))
    
    # ==================== Title and Settings ====================
    
    ax.set_title('Heterogeneous Graph Model for CSI Feedback\nCSI Feedback',
                fontsize=18, fontweight='bold', pad=25, color='#2C3E50')
    
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.02, 1.02)
    ax.axis('off')
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#BDC3C7')
        spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Graph saved to: {save_path}")
    print(f"PDF saved to: {save_path.replace('.png', '.pdf')}")
    plt.show()


def print_graph_statistics(G):
    """Print graph statistics"""
    print("\n" + "="*60)
    print("CSI Feedback Graph Model - Statistics")
    print("="*60)
    
    print(f"\nTotal Nodes: {G.number_of_nodes()}")
    print(f"Total Edges: {G.number_of_edges()}")
    
    node_types = {}
    for node in G.nodes():
        ntype = G.nodes[node].get('type', 'unknown')
        node_types[ntype] = node_types.get(ntype, 0) + 1
    
    print("\nNode Type Distribution:")
    for ntype, count in sorted(node_types.items()):
        print(f"  - {ntype}: {count}")
    
    edge_types = {}
    for u, v, data in G.edges(data=True):
        rel = data.get('relation', 'unknown')
        edge_types[rel] = edge_types.get(rel, 0) + 1
    
    print("\nEdge Relation Distribution:")
    rel_names = {
        'solve': 'Solve relation',
        'improve': 'Improve relation',
        'eval': 'Evaluation relation',
        'metric_app': 'Application relation',
        'scenario': 'Scenario relation'
    }
    for rel, count in sorted(edge_types.items()):
        print(f"  - {rel_names.get(rel, rel)}: {count}")
    
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    print("\nTop 10 Nodes by Degree (in+out):")
    total_degrees = {node: in_degrees[node] + out_degrees[node] 
                    for node in G.nodes()}
    sorted_nodes = sorted(total_degrees.items(), key=lambda x: x[1], reverse=True)
    for node, deg in sorted_nodes[:10]:
        print(f"  - {node.replace(chr(10), ' ')}: degree={deg}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    G, all_nodes = create_csi_feedback_graph()
    print_graph_statistics(G)
    save_path = r"/home/wutong/SemCQI/SemCSI_rrn-main/SemCSI_rrn-main/TransCQA/trash/CSI_Feedback_Graph.png"
    draw_graph(G, all_nodes, save_path)
