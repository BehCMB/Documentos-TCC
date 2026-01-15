import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import matplotlib.patches as mpatches
import numpy as np

# ==================== FUNÇÕES AUXILIARES ====================
def parse_log_to_df(log_content):
    """Converte string de log em DataFrame"""
    from io import StringIO
    df = pd.read_csv(StringIO(log_content))
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

def extract_cases(df):
    """Extrai sequências de atividades por caso"""
    cases = {}
    for case_id, group in df.groupby('CaseID'):
        # Ordenar por timestamp e extrair atividades
        activities = group.sort_values('Timestamp')['Activity'].tolist()
        cases[case_id] = activities
    return cases

def build_directly_follows_graph(df):
    """Constrói DFG a partir do DataFrame"""
    # Agrupar por caso e ordenar por timestamp
    df_sorted = df.sort_values(['CaseID', 'Timestamp'])
    
    # Calcular relações diretamente-segue
    dfg = Counter()
    total_flows = 0
    
    for case_id, group in df_sorted.groupby('CaseID'):
        activities = group['Activity'].tolist()
        
        # Para cada par consecutivo
        for i in range(len(activities) - 1):
            from_act = activities[i]
            to_act = activities[i + 1]
            dfg[(from_act, to_act)] += 1
            total_flows += 1
    
    return dfg, total_flows

def build_dfg_with_metrics(df):
    """Constrói DFG com métricas adicionais"""
    df_sorted = df.sort_values(['CaseID', 'Timestamp'])
    
    dfg_counts = Counter()
    dfg_times = defaultdict(list)
    activity_durations = defaultdict(list)
    
    for case_id, group in df_sorted.groupby('CaseID'):
        group = group.sort_values('Timestamp')
        activities = group['Activity'].tolist()
        timestamps = group['Timestamp'].tolist()
        
        # Calcular durações das atividades
        for i in range(len(activities)):
            if i < len(activities) - 1:
                duration = (timestamps[i + 1] - timestamps[i]).total_seconds() / 60  # em minutos
                dfg_times[(activities[i], activities[i + 1])].append(duration)
        
        # Contar transições
        for i in range(len(activities) - 1):
            dfg_counts[(activities[i], activities[i + 1])] += 1
    
    return dfg_counts, dfg_times

# ==================== VISUALIZAÇÃO DFG ====================

def visualize_dfg_simple(dfg, total_flows, title="DFG - Directly Follows Graph"):
    """Visualização do DFG usando networkx"""
    
    G = nx.DiGraph()
    
    # Adicionar arestas com pesos
    for (from_act, to_act), count in dfg.items():
        weight = count
        frequency = (count / total_flows) * 100
        
        G.add_edge(from_act, to_act, 
                   weight=weight, 
                   frequency=frequency,
                   label=f"{count} ({frequency:.1f}%)")
    
    # Layout melhorado
    plt.figure(figsize=(14, 10))
    
    # Tentar layout hierárquico (para processos sequenciais)
    try:
        # Criar ordem hierárquica baseada na frequência de ocorrência
        nodes_by_importance = sorted(G.nodes(), key=lambda x: G.in_degree(x) + G.out_degree(x), reverse=True)
        pos = {}
        
        # Organizar em camadas (layers) baseado na ordem temporal típica
        layers = {
            'start': ['Ticket Received'],
            'middle1': ['Initial Triage'],
            'middle2': ['Assign to Specialist', 'Escalate to Manager'],
            'middle3': ['Technical Analysis', 'Manager Review', 'Waiting for User'],
            'end': ['Solution Provided', 'Ticket Closed']
        }
        
        # Mapear nós para camadas
        node_to_layer = {}
        for layer_name, nodes_in_layer in layers.items():
            for node in nodes_in_layer:
                if node in G.nodes():
                    node_to_layer[node] = layer_name
        
        # Definir posições para camadas
        layer_positions = {
            'start': 0,
            'middle1': 1,
            'middle2': 2,
            'middle3': 3,
            'end': 4
        }
        
        # Posicionar nós
        for node in G.nodes():
            layer = node_to_layer.get(node, 'middle3')
            layer_idx = layer_positions[layer]
            
            # Contar quantos nós já estão nesta camada
            nodes_in_same_layer = [n for n in G.nodes() if node_to_layer.get(n) == layer]
            position_in_layer = nodes_in_same_layer.index(node) if node in nodes_in_same_layer else 0
            
            # Posição vertical baseada na posição na camada
            y = position_in_layer - (len(nodes_in_same_layer)-1)/2 if nodes_in_same_layer else 0
            
            pos[node] = (layer_idx, y)
            
    except Exception as e:
        print(f"Usando layout spring devido a: {e}")
        # Fallback para spring layout com parâmetros otimizados
        pos = nx.spring_layout(G, k=3, iterations=200, seed=42)
    
    # Aumentar tamanho dos nós baseado no comprimento do texto
    node_sizes = []
    for node in G.nodes():
        # Tamanho base + ajuste pelo comprimento do texto
        base_size = 3000
        text_length = len(str(node))
        adjusted_size = base_size + (text_length * 100)
        node_sizes.append(adjusted_size)
    
    # Cores para diferentes tipos de atividades
    activity_types = {
        'Início': ['Ticket Received'],
        'Triagem': ['Initial Triage'],
        'Atribuição': ['Assign to Specialist'],
        'Análise': ['Technical Analysis', 'Manager Review'],
        'Espera': ['Waiting for User'],
        'Escalonamento': ['Escalate to Manager'],
        'Resolução': ['Solution Provided'],
        'Fechamento': ['Ticket Closed']
    }
    
    # Mapear cores
    colors = plt.cm.tab20(np.linspace(0, 1, len(activity_types)))
    color_map = {}
    for (activity_type, activities), color in zip(activity_types.items(), colors):
        for activity in activities:
            if activity in G.nodes():
                color_map[activity] = color
    
    node_colors = [color_map.get(node, '#4285F4') for node in G.nodes()]
    
    # Desenhar nós COM TRANSPARÊNCIA para não cobrir as setas
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.85,  # Leve transparência
                          edgecolors='black',
                          linewidths=2,
                          node_shape='o')
    
    # Desenhar rótulos dos nós CENTRALIZADOS e ajustados
    node_labels = {}
    for node in G.nodes():
        # Quebrar texto longo em várias linhas
        if len(node) > 15:
            parts = node.split()
            if len(parts) > 1:
                mid = len(parts) // 2
                label = '\n'.join([' '.join(parts[:mid]), ' '.join(parts[mid:])])
            else:
                label = node
        else:
            label = node
        
        node_labels[node] = label
    
    # Desenhar labels com fundo branco para melhor legibilidade
    text_items = nx.draw_networkx_labels(G, pos, 
                                        labels=node_labels,
                                        font_size=9,
                                        font_weight='bold',
                                        font_family='sans-serif',
                                        bbox=dict(boxstyle='round,pad=0.5',
                                                facecolor='white',
                                                edgecolor='none',
                                                alpha=0.8))
    
    # Ajustar posição dos textos para ficarem melhor centralizados
    for text_item in text_items.values():
        text_item.set_verticalalignment('center')
        text_item.set_horizontalalignment('center')
    
    # Desenhar arestas COM SETAS VISÍVEIS
    edges = list(G.edges())
    widths = [G[u][v]['weight'] * 0.8 for u, v in edges]  # Espessura baseada no peso
    
    # Usar FancyArrowPatch para melhor controle das setas
    nx.draw_networkx_edges(G, pos, 
                          edgelist=edges,
                          edge_color='#555555',
                          width=widths,
                          alpha=0.9,
                          arrows=True,
                          arrowsize=25,  # Aumentar tamanho das setas
                          arrowstyle='-|>',
                          min_source_margin=20,  # Distância da origem
                          min_target_margin=25,  # Distância do destino
                          connectionstyle='arc3,rad=0.15',  # Curvatura suave
                          node_size=node_sizes)  # Passar node_sizes para calcular margens
    
    # Adicionar rótulos nas arestas
    edge_labels = nx.get_edge_attributes(G, 'label')
    
    # Posicionar labels das arestas em posições otimizadas
    edge_label_pos = {}
    for (u, v), label in edge_labels.items():
        # Calcular ponto médio da aresta com ajuste
        x_u, y_u = pos[u]
        x_v, y_v = pos[v]
        
        # Ponto médio
        x_mid = (x_u + x_v) / 2
        y_mid = (y_u + y_v) / 2
        
        # Pequeno deslocamento perpendicular para não sobrepor a aresta
        dx = x_v - x_u
        dy = y_v - y_u
        
        # Normalizar e rotacionar 90 graus
        length = np.sqrt(dx*dx + dy*dy)
        if length > 0:
            offset_x = -dy / length * 0.1  # Ajuste o 0.1 para controlar distância
            offset_y = dx / length * 0.1
        else:
            offset_x, offset_y = 0.1, 0.1
        
        edge_label_pos[(u, v)] = (x_mid + offset_x, y_mid + offset_y)
    
    # Desenhar labels das arestas
    for (u, v), label in edge_labels.items():
        plt.annotate(label,
                    xy=edge_label_pos[(u, v)],
                    xytext=edge_label_pos[(u, v)],
                    fontsize=8,
                    color='red',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='white',
                             edgecolor='gray',
                             alpha=0.9))
    
    # Adicionar legenda de cores
    legend_patches = []
    for activity_type, color in zip(activity_types.keys(), colors):
        patch = mpatches.Patch(color=color, label=activity_type)
        legend_patches.append(patch)
    
    plt.legend(handles=legend_patches, 
               loc='upper left', 
               bbox_to_anchor=(1, 1),
               fontsize=9,
               title='Tipo de Atividade')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Adicionar grade de fundo sutil para referência
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    
    plt.show()
    
    # Retornar métricas
    return G, pos

def visualize_dfg_advanced(dfg_counts, dfg_times, title="DFG Avançado - Métricas de Tempo"):
    """Visualização avançada do DFG com métricas de tempo"""
    
    G = nx.DiGraph()
    
    # Calcular métricas agregadas
    edge_metrics = {}
    for (from_act, to_act), times in dfg_times.items():
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            count = dfg_counts.get((from_act, to_act), 0)
            
            edge_metrics[(from_act, to_act)] = {
                'count': count,
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time
            }
    
    # Adicionar arestas ao grafo
    for (from_act, to_act), metrics in edge_metrics.items():
        label = f"Freq: {metrics['count']}\n"
        label += f"Tempo: {metrics['avg_time']:.1f}m\n"
        label += f"({metrics['min_time']:.0f}-{metrics['max_time']:.0f}m)"
        
        G.add_edge(from_act, to_act,
                   weight=metrics['count'],
                   avg_time=metrics['avg_time'],
                   label=label,
                   min_time=metrics['min_time'],
                   max_time=metrics['max_time'])
    
    # Layout hierárquico manual
    plt.figure(figsize=(16, 12))
    
    # Definir camadas do processo (layers) baseado no fluxo típico
    layers = {
        'Layer1': ['Ticket Received'],
        'Layer2': ['Initial Triage'],
        'Layer3': ['Assign to Specialist', 'Escalate to Manager'],
        'Layer4': ['Technical Analysis', 'Waiting for User', 'Manager Review'],
        'Layer5': ['Solution Provided'],
        'Layer6': ['Ticket Closed']
    }
    
    # Filtrar apenas atividades que existem no grafo
    filtered_layers = {}
    for layer_name, activities in layers.items():
        existing_activities = [act for act in activities if act in G.nodes()]
        if existing_activities:
            filtered_layers[layer_name] = existing_activities
    
    # Calcular posições manualmente
    pos = {}
    layer_gap = 2.5  # Distância horizontal entre camadas
    node_gap = 2.0   # Distância vertical entre nós na mesma camada
    
    # Ordenar camadas numericamente
    layer_order = sorted(filtered_layers.keys())
    
    for layer_idx, layer_name in enumerate(layer_order):
        activities_in_layer = filtered_layers[layer_name]
        x_pos = layer_idx * layer_gap
        
        # Centralizar verticalmente os nós na camada
        total_height = (len(activities_in_layer) - 1) * node_gap
        start_y = -total_height / 2
        
        for i, activity in enumerate(activities_in_layer):
            y_pos = start_y + i * node_gap
            pos[activity] = (x_pos, y_pos)
    
    # Ajuste específico para evitar sobreposição no Layer4
    if 'Technical Analysis' in pos and 'Waiting for User' in pos and 'Manager Review' in pos:
        # Reorganizar verticalmente para dar mais espaço
        ta_x, ta_y = pos['Technical Analysis']
        wu_x, wu_y = pos['Waiting for User']
        mr_x, mr_y = pos['Manager Review']
        
        # Ordenar por Y atual
        nodes_in_layer = [
            ('Technical Analysis', ta_y),
            ('Waiting for User', wu_y),
            ('Manager Review', mr_y)
        ]
        nodes_in_layer.sort(key=lambda x: x[1])
        
        # Redistribuir com mais espaço
        new_y_positions = [-node_gap, 0, node_gap]
        for (node_name, _), new_y in zip(nodes_in_layer, new_y_positions):
            pos[node_name] = (ta_x, new_y)
    
    # Para qualquer nó não mapeado, usar posição padrão
    for node in G.nodes():
        if node not in pos:
            pos[node] = (0, 0)
    
    # ==================== CALCULAR TAMANHOS DOS NÓS ====================
    node_sizes = {}
    for node in G.nodes():
        base_size = 4000
        text_length = len(node)
        if text_length > 20:
            adjusted_size = base_size + (text_length * 200)
        else:
            adjusted_size = base_size + (text_length * 150)
        
        degree_factor = G.degree(node) * 300
        node_sizes[node] = adjusted_size + degree_factor
    
    node_size_list = [node_sizes[node] for node in G.nodes()]
    
    # ==================== CORES DOS NÓS ====================
    activity_categories = {
        'Recepção': ['Ticket Received'],
        'Triagem': ['Initial Triage'],
        'Atribuição': ['Assign to Specialist'],
        'Análise': ['Technical Analysis'],
        'Espera': ['Waiting for User'],
        'Escalonamento': ['Escalate to Manager'],
        'Revisão': ['Manager Review'],
        'Solução': ['Solution Provided'],
        'Fechamento': ['Ticket Closed']
    }
    
    cmap = plt.cm.tab20
    colors = [cmap(i) for i in range(len(activity_categories))]
    color_map = {}
    
    for i, (category, activities) in enumerate(activity_categories.items()):
        for activity in activities:
            if activity in G.nodes():
                color_map[activity] = colors[i]
    
    node_colors = [color_map.get(node, '#CCCCCC') for node in G.nodes()]
    
    # ==================== ESTRATÉGIA DE POSICIONAMENTO DE LABELS DE ARESTAS ====================
    
    # 1. Primeiro identificar arestas que podem ter problemas de sobreposição
    problem_edges = []
    for u, v in G.edges():
        # Arestas que envolvem Waiting for User, Technical Analysis, Manager Review
        if ('Waiting for User' in [u, v] or 
            'Technical Analysis' in [u, v] or 
            'Manager Review' in [u, v]):
            problem_edges.append((u, v))
    
    # 2. Função para calcular posição do label com controle de posição ao longo da aresta
    def calculate_edge_label_position(u, v, position_ratio=0.5, offset_factor=0.15):
        """Calcula posição do label ao longo da aresta com offset perpendicular"""
        x_u, y_u = pos[u]
        x_v, y_v = pos[v]
        
        # Ponto ao longo da aresta (0.0 = início, 1.0 = fim)
        x_mid = x_u + (x_v - x_u) * position_ratio
        y_mid = y_u + (y_v - y_u) * position_ratio
        
        # Vetor da aresta
        dx = x_v - x_u
        dy = y_v - y_u
        
        # Normalizar
        length = np.sqrt(dx*dx + dy*dy)
        if length > 0:
            # Deslocamento perpendicular (normal à aresta)
            # O sinal controla para qual lado da aresta o label vai
            offset_x = -dy / length * offset_factor
            offset_y = dx / length * offset_factor
            
            # Para arestas específicas, ajustar o lado do offset
            if (u == 'Waiting for User' and v == 'Technical Analysis') or \
               (u == 'Technical Analysis' and v == 'Waiting for User'):
                # Inverter offset para ficar do lado oposto a Manager Review
                offset_x = -offset_x
                offset_y = -offset_y
        else:
            offset_x, offset_y = offset_factor, offset_factor
        
        return (x_mid + offset_x, y_mid + offset_y), (x_mid, y_mid)
    
    # 3. Mapeamento de posições específicas para arestas problemáticas
    edge_label_positions = {}
    edge_original_midpoints = {}
    
    for u, v in G.edges():
        # Posições padrão (no meio da aresta)
        position_ratio = 0.5
        offset_factor = 0.15
        
        # Ajustes específicos para evitar sobreposição
        if (u == 'Waiting for User' and v == 'Technical Analysis'):
            # Colocar label mais próximo do início (Waiting for User)
            position_ratio = 0.3
            offset_factor = 0.2  # Mais afastado da aresta
            
        elif (u == 'Technical Analysis' and v == 'Waiting for User'):
            # Colocar label mais próximo do início (Technical Analysis)
            position_ratio = 0.3
            offset_factor = 0.2
            
        elif (u == 'Technical Analysis' and v == 'Escalate to Manager'):
            # Colocar label mais próximo do fim (Escalate to Manager)
            position_ratio = 0.7
            offset_factor = 0.15
            
        elif (u == 'Escalate to Manager' and v == 'Manager Review'):
            # Colocar label mais próximo do início
            position_ratio = 0.3
            offset_factor = 0.15
            
        elif (u == 'Manager Review' and v == 'Solution Provided'):
            # Colocar label mais próximo do início
            position_ratio = 0.3
            offset_factor = 0.15
        
        # Calcular posição
        label_pos, midpoint = calculate_edge_label_position(u, v, position_ratio, offset_factor)
        edge_label_positions[(u, v)] = label_pos
        edge_original_midpoints[(u, v)] = midpoint
    
    # ==================== DESENHAR ARESTAS PRIMEIRO ====================
    edges = list(G.edges())
    edge_colors = []
    edge_widths = []
    
    for u, v in edges:
        avg_time = G[u][v]['avg_time']
        
        # Cor baseada no tempo
        if avg_time > 120:  # Mais de 2 horas
            color = '#FF6B6B'  # Vermelho
        elif avg_time > 60:   # 1-2 horas
            color = '#FFA726'  # Laranja
        else:
            color = '#66BB6A'  # Verde
        
        edge_colors.append(color)
        
        # Largura baseada na frequência (com ajuste visual)
        frequency = G[u][v]['weight']
        edge_widths.append(max(1.5, frequency * 0.4))
    
    # Desenhar arestas COM SETAS VISÍVEIS
    nx.draw_networkx_edges(G, pos,
                          edgelist=edges,
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=0.85,
                          arrows=True,
                          arrowsize=30,
                          arrowstyle='-|>',
                          min_source_margin=35,
                          min_target_margin=40,
                          connectionstyle='arc3,rad=0.15',
                          node_size=node_size_list)
    
    # ==================== DESENHAR NÓS ====================
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=node_size_list,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=2,
                          node_shape='o')
    
    # ==================== RÓTULOS DOS NÓS ====================
    node_labels = {}
    for node in G.nodes():
        if len(node) > 15:
            parts = node.split()
            if len(parts) > 1:
                mid = len(parts) // 2
                label = '\n'.join([' '.join(parts[:mid]), ' '.join(parts[mid:])])
            else:
                mid = len(node) // 2
                label = f"{node[:mid]}-\n{node[mid:]}"
        else:
            label = node
        
        node_labels[node] = label
    
    # Desenhar labels com fundo branco
    label_items = nx.draw_networkx_labels(G, pos,
                                         labels=node_labels,
                                         font_size=10,
                                         font_weight='bold',
                                         font_family='sans-serif',
                                         verticalalignment='center',
                                         bbox=dict(boxstyle='round,pad=0.6',
                                                  facecolor='white',
                                                  edgecolor='gray',
                                                  linewidth=1,
                                                  alpha=0.95))
    
    for node, text_item in label_items.items():
        text_item.set_zorder(10)
        if '\n' in node_labels[node]:
            text_item.set_linespacing(0.8)
    
    # ==================== RÓTULOS DAS ARESTAS ====================
    edge_labels = nx.get_edge_attributes(G, 'label')
    
    # Primeiro, desenhar labels de arestas NÃO problemáticas
    non_problem_edges = [(u, v) for (u, v) in edges if (u, v) not in problem_edges]
    
    for (u, v) in non_problem_edges:
        label = edge_labels[(u, v)]
        x, y = edge_label_positions.get((u, v), edge_original_midpoints[(u, v)])
        
        # Cor da borda baseada no tempo
        avg_time = G[u][v]['avg_time']
        if avg_time > 120:
            edge_color = '#FF6B6B'
        elif avg_time > 60:
            edge_color = '#FFA726'
        else:
            edge_color = '#66BB6A'
        
        plt.annotate(label,
                    xy=(x, y),
                    xytext=(0, 0),
                    textcoords='offset points',
                    fontsize=8,
                    fontweight='normal',
                    color='black',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.4',
                             facecolor='white',
                             edgecolor=edge_color,
                             linewidth=1.5,
                             alpha=0.95))
    
    # Agora, desenhar labels de arestas problemáticas com tratamento especial
    for (u, v) in problem_edges:
        if (u, v) in edge_labels:
            label = edge_labels[(u, v)]
            x, y = edge_label_positions.get((u, v), edge_original_midpoints[(u, v)])
            
            # Cor da borda baseada no tempo
            avg_time = G[u][v]['avg_time']
            if avg_time > 120:
                edge_color = '#FF6B6B'
            elif avg_time > 60:
                edge_color = '#FFA726'
            else:
                edge_color = '#66BB6A'
            
            # Para arestas muito problemáticas, usar seta apontando para o label
            if (u == 'Waiting for User' and v == 'Technical Analysis') or \
               (u == 'Technical Analysis' and v == 'Waiting for User'):
                
                # Adicionar uma pequena linha conectando o label à aresta
                mid_x, mid_y = edge_original_midpoints[(u, v)]
                plt.plot([mid_x, x], [mid_y, y], 'k-', lw=0.5, alpha=0.5, zorder=1)
                
                # Label com fundo mais destacado
                plt.annotate(label,
                            xy=(x, y),
                            xytext=(0, 0),
                            textcoords='offset points',
                            fontsize=8,
                            fontweight='bold',
                            color='black',
                            ha='center',
                            va='center',
                            bbox=dict(boxstyle='round,pad=0.5',
                                     facecolor='white',
                                     edgecolor=edge_color,
                                     linewidth=2,
                                     alpha=1.0))
            else:
                # Label normal
                plt.annotate(label,
                            xy=(x, y),
                            xytext=(0, 0),
                            textcoords='offset points',
                            fontsize=8,
                            fontweight='normal',
                            color='black',
                            ha='center',
                            va='center',
                            bbox=dict(boxstyle='round,pad=0.4',
                                     facecolor='white',
                                     edgecolor=edge_color,
                                     linewidth=1.5,
                                     alpha=0.95))
    
    # ==================== LEGENDA ====================
    time_legend = [
        plt.Line2D([0], [0], color='#66BB6A', lw=4, label='Rápido (< 60 min)'),
        plt.Line2D([0], [0], color='#FFA726', lw=4, label='Moderado (60-120 min)'),
        plt.Line2D([0], [0], color='#FF6B6B', lw=4, label='Lento (> 120 min)')
    ]
    
    node_legend_patches = []
    for category, activities in activity_categories.items():
        has_activity = any(act in G.nodes() for act in activities)
        if has_activity:
            color = color_map.get(activities[0], '#CCCCCC')
            patch = mpatches.Patch(color=color, label=category)
            node_legend_patches.append(patch)
    
    legend1 = plt.legend(handles=time_legend, 
                        loc='upper left', 
                        bbox_to_anchor=(0.02, 0.98),
                        fontsize=9,
                        title='Tempo de Transição',
                        framealpha=0.9)
    
    legend2 = plt.legend(handles=node_legend_patches, 
                        loc='upper left', 
                        bbox_to_anchor=(0.02, 0.75),
                        fontsize=9,
                        title='Tipo de Atividade',
                        framealpha=0.9)
    
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    
    # ==================== INFORMAÇÕES ADICIONAIS ====================
    total_transitions = sum(dfg_counts.values())
    total_time = sum(metrics['avg_time'] * metrics['count'] 
                     for metrics in edge_metrics.values())
    avg_process_time = total_time / total_transitions if total_transitions > 0 else 0
    
    info_text = f"Total de transições: {total_transitions}\n"
    info_text += f"Tempo médio por transição: {avg_process_time:.1f} minutos\n"
    info_text += f"Arestas coloridas por tempo médio"
    
    plt.figtext(0.02, 0.02, info_text, 
                fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.title(title, fontsize=16, fontweight='bold', pad=25)
    plt.axis('off')
    plt.tight_layout()
    
    # Adicionar grade de fundo sutil
    plt.gca().set_facecolor('#f8f9fa')
    
    plt.show()
    
    return G, edge_metrics

# ==================== VERSÃO COM CONTORNO DE DETECÇÃO DE COLISÃO ====================

def visualize_dfg_advanced_with_collision_detection(dfg_counts, dfg_times, title="DFG Avançado - Métricas de Tempo"):
    """Versão com detecção de colisão para evitar sobreposição de labels"""
    
    G = nx.DiGraph()
    
    # Calcular métricas agregadas
    edge_metrics = {}
    for (from_act, to_act), times in dfg_times.items():
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            count = dfg_counts.get((from_act, to_act), 0)
            
            edge_metrics[(from_act, to_act)] = {
                'count': count,
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time
            }
    
    # Adicionar arestas ao grafo
    for (from_act, to_act), metrics in edge_metrics.items():
        # Label simplificado para arestas curtas
        if metrics['avg_time'] < 30:
            label = f"{metrics['count']}x\n{metrics['avg_time']:.0f}m"
        else:
            label = f"Freq: {metrics['count']}\n"
            label += f"Tempo: {metrics['avg_time']:.1f}m\n"
            label += f"({metrics['min_time']:.0f}-{metrics['max_time']:.0f}m)"
        
        G.add_edge(from_act, to_act,
                   weight=metrics['count'],
                   avg_time=metrics['avg_time'],
                   label=label)
    
    # Layout hierárquico manual
    plt.figure(figsize=(16, 12))
    
    # Definir camadas do processo com mais espaçamento
    layers = {
        'Layer1': ['Ticket Received'],
        'Layer2': ['Initial Triage'],
        'Layer3': ['Assign to Specialist', 'Escalate to Manager'],
        'Layer4': ['Technical Analysis', 'Waiting for User', 'Manager Review'],
        'Layer5': ['Solution Provided'],
        'Layer6': ['Ticket Closed']
    }
    
    # Posicionamento manual com mais espaço
    pos = {
        'Ticket Received': (0, 0),
        'Initial Triage': (2, 0),
        'Assign to Specialist': (4, 1),
        'Escalate to Manager': (4, -1),
        'Technical Analysis': (6, 1.5),
        'Waiting for User': (6, 0),
        'Manager Review': (6, -1.5),
        'Solution Provided': (8, 0),
        'Ticket Closed': (10, 0)
    }
    
    # Ajustar posições para nós que não existem
    for node in list(pos.keys()):
        if node not in G.nodes():
            del pos[node]
    
    # ==================== FUNÇÃO DE POSICIONAMENTO INTELIGENTE ====================
    
    def smart_edge_label_position(u, v, pos_dict, node_radius=0.3):
        """Calcula posição inteligente para label evitando sobreposição"""
        x1, y1 = pos_dict[u]
        x2, y2 = pos_dict[v]
        
        # Distância entre nós
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return (x1, y1)
        
        # Direção normalizada
        ux = dx / length
        uy = dy / length
        
        # Vetor perpendicular
        px = -uy
        py = ux
        
        # Tentar diferentes posições ao longo da aresta
        positions_to_try = []
        
        # Posições ao longo da aresta (de 0.2 a 0.8)
        for ratio in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            x = x1 + dx * ratio
            y = y1 + dy * ratio
            
            # Tentar ambos os lados da aresta
            for side in [-1, 1]:
                offset = 0.15 * side
                positions_to_try.append((x + px * offset, y + py * offset))
        
        # Para arestas específicas, priorizar posições
        if (u == 'Waiting for User' and v == 'Technical Analysis'):
            # Priorizar posição perto do início, lado esquerdo
            x = x1 + dx * 0.25
            y = y1 + dy * 0.25
            return (x + px * 0.2, y + py * 0.2)
        
        elif (u == 'Technical Analysis' and v == 'Waiting for User'):
            # Priorizar posição perto do início, lado direito
            x = x1 + dx * 0.25
            y = y1 + dy * 0.25
            return (x - px * 0.2, y - py * 0.2)
        
        elif (u == 'Manager Review' and v == 'Solution Provided'):
            # Priorizar posição perto do início
            x = x1 + dx * 0.3
            y = y1 + dy * 0.3
            return (x + px * 0.15, y + py * 0.15)
        
        # Para outras arestas, usar a primeira posição
        return positions_to_try[0]
    
    # ==================== DESENHAR O GRAFO ====================
    
    # Tamanho dos nós
    node_sizes = [3500 for _ in G.nodes()]
    
    # Cores dos nós
    node_color_map = {
        'Ticket Received': '#4CAF50',
        'Initial Triage': '#2196F3',
        'Assign to Specialist': '#FF9800',
        'Escalate to Manager': '#795548',
        'Technical Analysis': '#9C27B0',
        'Waiting for User': '#FF5722',
        'Manager Review': '#3F51B5',
        'Solution Provided': '#00BCD4',
        'Ticket Closed': '#607D8B'
    }
    
    node_colors = [node_color_map.get(node, '#CCCCCC') for node in G.nodes()]
    
    # Desenhar arestas primeiro
    edges = list(G.edges())
    edge_colors = []
    edge_widths = []
    
    for u, v in edges:
        avg_time = G[u][v]['avg_time']
        
        if avg_time > 120:
            color = '#FF6B6B'
        elif avg_time > 60:
            color = '#FFA726'
        else:
            color = '#66BB6A'
        
        edge_colors.append(color)
        edge_widths.append(max(1.5, G[u][v]['weight'] * 0.4))
    
    nx.draw_networkx_edges(G, pos,
                          edgelist=edges,
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=0.85,
                          arrows=True,
                          arrowsize=30,
                          arrowstyle='-|>',
                          min_source_margin=35,
                          min_target_margin=40,
                          node_size=node_sizes)
    
    # Desenhar nós
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=2)
    
    # Desenhar labels dos nós
    node_labels = {node: node.replace(' ', '\n') if len(node) > 15 else node 
                   for node in G.nodes()}
    
    nx.draw_networkx_labels(G, pos,
                           labels=node_labels,
                           font_size=9,
                           font_weight='bold',
                           verticalalignment='center',
                           bbox=dict(boxstyle='round,pad=0.4',
                                    facecolor='white',
                                    edgecolor='gray',
                                    alpha=0.95))
    
    # Desenhar labels das arestas com posicionamento inteligente
    edge_labels = nx.get_edge_attributes(G, 'label')
    
    for (u, v), label in edge_labels.items():
        # Calcular posição inteligente
        x, y = smart_edge_label_position(u, v, pos)
        
        # Cor da borda
        avg_time = G[u][v]['avg_time']
        if avg_time > 120:
            edge_color = '#FF6B6B'
        elif avg_time > 60:
            edge_color = '#FFA726'
        else:
            edge_color = '#66BB6A'
        
        # Tamanho da fonte baseado no comprimento do label
        fontsize = 7 if len(label) > 20 else 8
        
        plt.annotate(label,
                    xy=(x, y),
                    xytext=(0, 0),
                    textcoords='offset points',
                    fontsize=fontsize,
                    color='black',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='white',
                             edgecolor=edge_color,
                             linewidth=1.5,
                             alpha=0.95))
    
    # ==================== LEGENDA E FINALIZAÇÃO ====================
    
    legend_elements = [
        plt.Line2D([0], [0], color='#66BB6A', lw=4, label='Rápido (< 60 min)'),
        plt.Line2D([0], [0], color='#FFA726', lw=4, label='Moderado (60-120 min)'),
        plt.Line2D([0], [0], color='#FF6B6B', lw=4, label='Lento (> 120 min)')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=25)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return G, edge_metrics


# Executar teste

    G1, G2, metrics1, metrics2 = test_problematic_edges()
    
    # Salvar as figuras para comparação
    plt.figure(1)
    plt.savefig('dfg_advanced_corrigido.png', dpi=300, bbox_inches='tight')
    
    plt.figure(2)
    plt.savefig('dfg_advanced_colision_detection.png', dpi=300, bbox_inches='tight')
    
    print("\nFiguras salvas:")
    print("- dfg_advanced_corrigido.png")
    print("- dfg_advanced_colision_detection.png")
def generate_process_summary(dfg, total_flows, cases):
    """Gera um resumo textual do processo baseado no DFG"""
    
    print("=" * 60)
    print("ANÁLISE DO PROCESSO - RESUMO DO DFG")
    print("=" * 60)
    
    # 1. Estatísticas básicas
    print(f"\n ESTATÍSTICAS BÁSICAS:")
    print(f"   • Total de casos: {len(cases)}")
    print(f"   • Total de transições: {total_flows}")
    print(f"   • Transições únicas: {len(dfg)}")
    
    # 2. Atividades mais frequentes
    activity_counts = Counter()
    for (from_act, _), count in dfg.items():
        activity_counts[from_act] += count
    
    # Adicionar última atividade
    for case_activities in cases.values():
        if case_activities:
            activity_counts[case_activities[-1]] += 1
    
    print(f"\n🎯 ATIVIDADES MAIS FREQUENTES:")
    for activity, count in activity_counts.most_common(5):
        percentage = (count / (total_flows + len(cases))) * 100
        print(f"   • {activity}: {count} ocorrências ({percentage:.1f}%)")
    
    # 3. Fluxos mais comuns
    print(f"\n FLUXOS MAIS COMUNS:")
    for (from_act, to_act), count in dfg.most_common(5):
        percentage = (count / total_flows) * 100
        print(f"   • {from_act} → {to_act}: {count} vezes ({percentage:.1f}%)")
    
    # 4. Pontos de decisão
    print(f"\n PONTOS DE DECISÃO (RAMIFICAÇÕES):")
    from_activities = defaultdict(list)
    for (from_act, to_act) in dfg.keys():
        from_activities[from_act].append(to_act)
    
    for activity, next_activities in from_activities.items():
        if len(next_activities) > 1:
            print(f"   • {activity} pode levar para: {', '.join(next_activities)}")
    
    # 5. Início e fim do processo
    start_activities = Counter([activities[0] for activities in cases.values() if activities])
    end_activities = Counter([activities[-1] for activities in cases.values() if activities])
    
    print(f"\n ATIVIDADES DE INÍCIO:")
    for activity, count in start_activities.most_common(3):
        percentage = (count / len(cases)) * 100
        print(f"   • {activity}: {count} casos ({percentage:.1f}%)")
    
    print(f"\n ATIVIDADES DE FIM:")
    for activity, count in end_activities.most_common(3):
        percentage = (count / len(cases)) * 100
        print(f"   • {activity}: {count} casos ({percentage:.1f}%)")
    
    print("\n" + "=" * 60)

# ==================== ANÁLISE DE CONFORMIDADE COM DFG ====================
def check_conformance_dfg(dfg, expected_paths):
    """Verifica conformidade com caminhos esperados no DFG"""
    
    print("=" * 60)
    print("ANÁLISE DE CONFORMIDADE - DFG")
    print("=" * 60)
    
    deviations = []
    
    for (from_act, to_act), count in dfg.items():
        # Verificar se esta transição é esperada
        is_expected = False
        for expected_path in expected_paths:
            for i in range(len(expected_path) - 1):
                if expected_path[i] == from_act and expected_path[i + 1] == to_act:
                    is_expected = True
                    break
        
        if not is_expected:
            deviations.append((from_act, to_act, count))
    
    if deviations:
        print(f"\n  DESVIOS ENCONTRADOS: {len(deviations)} transições não esperadas")
        for from_act, to_act, count in deviations:
            print(f"   • {from_act} → {to_act}: {count} ocorrências")
    else:
        print(f"\n TODAS AS TRANSIÇÕES SÃO CONFORMES!")
    
    # Calcular taxa de conformidade
    total_transitions = sum(dfg.values())
    unexpected_count = sum(count for _, _, count in deviations)
    conformance_rate = ((total_transitions - unexpected_count) / total_transitions) * 100
    
    print(f"\n TAXA DE CONFORMIDADE: {conformance_rate:.1f}%")
    print("=" * 60)
    
    return conformance_rate, deviations

# ==================== FUNÇÃO PRINCIPAL ====================
def main():
    """Função principal para demonstração didática"""
    
    # 1. Carregar dados (usando o log do helpdesk fornecido anteriormente)
    log_content = """CaseID,Activity,Resource,Timestamp,Status
1001,Ticket Received,System,2024-01-01 09:00:00,Open
1001,Initial Triage,Agent_John,2024-01-01 09:15:00,In Progress
1001,Assign to Specialist,Agent_John,2024-01-01 09:30:00,In Progress
1001,Technical Analysis,Specialist_Maria,2024-01-01 10:00:00,In Progress
1001,Solution Provided,Specialist_Maria,2024-01-01 11:30:00,Resolved
1001,Ticket Closed,Agent_John,2024-01-01 12:00:00,Closed
1002,Ticket Received,System,2024-01-01 09:05:00,Open
1002,Initial Triage,Agent_Sarah,2024-01-01 09:20:00,In Progress
1002,Assign to Specialist,Agent_Sarah,2024-01-01 09:35:00,In Progress
1002,Technical Analysis,Specialist_Maria,2024-01-01 10:30:00,In Progress
1002,Waiting for User,Specialist_Maria,2024-01-01 11:00:00,Waiting
1002,Technical Analysis,Specialist_Maria,2024-01-01 14:00:00,In Progress
1002,Solution Provided,Specialist_Maria,2024-01-01 15:30:00,Resolved
1002,Ticket Closed,Agent_Sarah,2024-01-02 09:00:00,Closed
1003,Ticket Received,System,2024-01-01 09:10:00,Open
1003,Initial Triage,Agent_John,2024-01-01 09:25:00,In Progress
1003,Escalate to Manager,Agent_John,2024-01-01 09:45:00,In Progress
1003,Manager Review,Manager_David,2024-01-01 11:00:00,In Progress
1003,Solution Provided,Manager_David,2024-01-01 12:30:00,Resolved
1003,Ticket Closed,Agent_John,2024-01-01 13:00:00,Closed
1004,Ticket Received,System,2024-01-02 10:00:00,Open
1004,Initial Triage,Agent_Sarah,2024-01-02 10:15:00,In Progress
1004,Assign to Specialist,Agent_Sarah,2024-01-02 10:30:00,In Progress
1004,Technical Analysis,Specialist_Maria,2024-01-02 11:00:00,In Progress
1004,Solution Provided,Specialist_Maria,2024-01-02 12:00:00,Resolved
1004,Ticket Closed,Agent_Sarah,2024-01-02 12:30:00,Closed
1005,Ticket Received,System,2024-01-02 14:00:00,Open
1005,Initial Triage,Agent_John,2024-01-02 14:15:00,In Progress
1005,Assign to Specialist,Agent_John,2024-01-02 14:30:00,In Progress
1005,Technical Analysis,Specialist_Carlos,2024-01-02 15:00:00,In Progress
1005,Escalate to Manager,Specialist_Carlos,2024-01-02 16:00:00,In Progress
1005,Manager Review,Manager_David,2024-01-03 09:30:00,In Progress
1005,Solution Provided,Manager_David,2024-01-03 11:00:00,Resolved
1005,Ticket Closed,Agent_John,2024-01-03 11:30:00,Closed"""
    
    df = parse_log_to_df(log_content)
    
    # 2. Extrair casos
    cases = extract_cases(df)
    
    # 3. Construir DFG básico
    dfg, total_flows = build_directly_follows_graph(df)
    
    # 4. Visualizar DFG simples
    print(" GERANDO VISUALIZAÇÃO DFG SIMPLES...")
    G_simple = visualize_dfg_simple(dfg, total_flows)
    
    # 5. Gerar resumo do processo
    generate_process_summary(dfg, total_flows, cases)
    
    # 6. Construir DFG com métricas de tempo
    print("\n  GERANDO DFG COM MÉTRICAS DE TEMPO...")
    dfg_counts, dfg_times = build_dfg_with_metrics(df)
    G_advanced, edge_metrics = visualize_dfg_advanced(dfg_counts, dfg_times)
    
    # 7. Análise de conformidade
    expected_paths = [
        ['Ticket Received', 'Initial Triage', 'Assign to Specialist', 
         'Technical Analysis', 'Solution Provided', 'Ticket Closed'],
        ['Ticket Received', 'Initial Triage', 'Escalate to Manager',
         'Manager Review', 'Solution Provided', 'Ticket Closed']
    ]
    
    conformance_rate, deviations = check_conformance_dfg(dfg, expected_paths)
    
    # 8. Análise de bottlenecks (com base nos tempos)
    print("\n IDENTIFICANDO BOTTLENECKS...")
    bottlenecks = []
    for (from_act, to_act), metrics in edge_metrics.items():
        if metrics['avg_time'] > 60:  # Mais de 1 hora
            bottlenecks.append((from_act, to_act, metrics['avg_time'], metrics['count']))
    
    if bottlenecks:
        print(f" {len(bottlenecks)} BOTTLENECKS IDENTIFICADOS:")
        for from_act, to_act, avg_time, count in sorted(bottlenecks, key=lambda x: x[2], reverse=True):
            print(f"   • {from_act} → {to_act}: {avg_time:.1f} min (média) em {count} ocorrências")
    else:
        print(" Nenhum bottleneck significativo identificado!")
    
    # 9. Exportar resultados
    export_results(dfg, edge_metrics, cases)

def export_results(dfg, edge_metrics, cases):
    """Exporta resultados para análise posterior"""
    
    # Criar DataFrame com fluxos
    dfg_list = []
    for (from_act, to_act), count in dfg.items():
        metrics = edge_metrics.get((from_act, to_act), {})
        dfg_list.append({
            'From': from_act,
            'To': to_act,
            'Frequency': count,
            'Avg_Time_Min': metrics.get('avg_time', 0),
            'Min_Time': metrics.get('min_time', 0),
            'Max_Time': metrics.get('max_time', 0)
        })
    
    dfg_df = pd.DataFrame(dfg_list)
    
    # Exportar para CSV
    dfg_df.to_csv('dfg_analysis.csv', index=False, encoding='utf-8')
    
    # Exportar casos
    cases_df = pd.DataFrame([
        {'CaseID': case_id, 'Path': ' → '.join(activities)}
        for case_id, activities in cases.items()
    ])
    cases_df.to_csv('cases_paths.csv', index=False, encoding='utf-8')
    
    print(f"\n Resultados exportados:")
    print(f"   • dfg_analysis.csv: {len(dfg_df)} transições analisadas")
    print(f"   • cases_paths.csv: {len(cases_df)} casos processados")

# ==================== EXECUÇÃO ====================
if __name__ == "__main__":
    print(" PROCESS MINING - DEMONSTRAÇÃO DIDÁTICA COM DFG")
    print("=" * 60)
    main()