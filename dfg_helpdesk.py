import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter, defaultdict
from datetime import datetime, timedelta

# ==================== FUNCOES AUXILIARES ====================
def parse_log_to_df(log_content):
    """Converte string de log em DataFrame"""
    from io import StringIO
    df = pd.read_csv(StringIO(log_content))
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

def extract_cases(df):
    """Extrai sequencias de atividades por caso"""
    cases = {}
    for case_id, group in df.groupby('CaseID'):
        # Ordenar por timestamp e extrair atividades
        activities = group.sort_values('Timestamp')['Activity'].tolist()
        cases[case_id] = activities
    return cases

def build_directly_follows_graph(df):
    """Constroi DFG a partir do DataFrame"""
    # Agrupar por caso e ordenar por timestamp
    df_sorted = df.sort_values(['CaseID', 'Timestamp'])
    
    # Calcular relacoes diretamente-segue
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
    """Constroi DFG com metricas adicionais"""
    df_sorted = df.sort_values(['CaseID', 'Timestamp'])
    
    dfg_counts = Counter()
    dfg_times = defaultdict(list)
    activity_durations = defaultdict(list)
    
    for case_id, group in df_sorted.groupby('CaseID'):
        group = group.sort_values('Timestamp')
        activities = group['Activity'].tolist()
        timestamps = group['Timestamp'].tolist()
        
        # Calcular duracoes das atividades
        for i in range(len(activities)):
            if i < len(activities) - 1:
                duration = (timestamps[i + 1] - timestamps[i]).total_seconds() / 60  # em minutos
                dfg_times[(activities[i], activities[i + 1])].append(duration)
        
        # Contar transicoes
        for i in range(len(activities) - 1):
            dfg_counts[(activities[i], activities[i + 1])] += 1
    
    return dfg_counts, dfg_times

# ==================== VISUALIZACAO DFG ====================
def visualize_dfg_simple(dfg, total_flows, title="DFG - Directly Follows Graph"):
    """Visualizacao simples do DFG usando networkx"""
    G = nx.DiGraph()
    
    # Adicionar arestas com pesos
    for (from_act, to_act), count in dfg.items():
        weight = count
        frequency = (count / total_flows) * 100
        
        G.add_edge(from_act, to_act, 
                   weight=weight, 
                   frequency=frequency,
                   label=f"{count}\n({frequency:.1f}%)")
    
    # Layout
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Desenhar nos
    node_colors = ['#4285F4' for _ in G.nodes()]  # Azul Google
    node_sizes = [2000 + G.degree(node) * 300 for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=2)
    
    # Desenhar rotulos dos nos
    nx.draw_networkx_labels(G, pos, 
                           font_size=10,
                           font_weight='bold',
                           font_color='white')
    
    # Desenhar arestas com espessura proporcional a frequencia
    edges = G.edges()
    widths = [G[u][v]['weight'] * 0.5 for u, v in edges]
    edge_labels = nx.get_edge_attributes(G, 'label')
    
    nx.draw_networkx_edges(G, pos, 
                          edge_color='#555555',
                          width=widths,
                          alpha=0.7,
                          arrowsize=20,
                          arrowstyle='-|>',
                          connectionstyle='arc3,rad=0.1')
    
    # Adicionar rotulos nas arestas
    nx.draw_networkx_edge_labels(G, pos, 
                                edge_labels=edge_labels,
                                font_size=8,
                                font_color='red',
                                bbox=dict(alpha=0.7, boxstyle='round'))
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Retornar metricas
    return G

def visualize_dfg_advanced(dfg_counts, dfg_times):
    """Visualizacao avancada do DFG com metricas de tempo"""
    G = nx.DiGraph()
    
    # Calcular metricas agregadas
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
                   label=label)
    
    # Layout melhorado
    plt.figure(figsize=(14, 10))
    
    # Usar layout hierarquico para processos sequenciais
    try:
        pos = nx.multipartite_layout(G, subset_key='subset', align='horizontal')
    except:
        pos = nx.spring_layout(G, k=3, iterations=100)
    
    # Desenhar nos
    nodes = list(G.nodes())
    node_colors = plt.cm.Set3(range(len(nodes)))
    
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=2500,
                          alpha=0.8,
                          edgecolors='black',
                          linewidths=2,
                          node_shape='s')
    
    # Rotulos dos nos
    nx.draw_networkx_labels(G, pos,
                           font_size=9,
                           font_weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='white',
                                   alpha=0.8))
    
    # Desenhar arestas com cores baseadas no tempo
    edges = G.edges()
    edge_colors = []
    edge_widths = []
    
    for u, v in edges:
        avg_time = G[u][v]['avg_time']
        # Cor baseada no tempo (vermelho = lento, verde = rapido)
        if avg_time > 120:  # Mais de 2 horas
            color = '#FF6B6B'  # Vermelho
        elif avg_time > 60:  # 1-2 horas
            color = '#FFA726'  # Laranja
        else:
            color = '#66BB6A'  # Verde
        
        edge_colors.append(color)
        edge_widths.append(G[u][v]['weight'] * 0.3)
    
    nx.draw_networkx_edges(G, pos,
                          edge_color=edge_colors,
                          width=edge_widths,
                          alpha=0.7,
                          arrowsize=25,
                          arrowstyle='fancy',
                          connectionstyle='arc3,rad=0.2')
    
    # Rotulos das arestas
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos,
                                edge_labels=edge_labels,
                                font_size=7,
                                bbox=dict(boxstyle='round,pad=0.5',
                                        facecolor='white',
                                        alpha=0.9))
    
    # Legenda de cores
    legend_elements = [
        plt.Line2D([0], [0], color='#66BB6A', lw=4, label='Rapido (< 60 min)'),
        plt.Line2D([0], [0], color='#FFA726', lw=4, label='Moderado (60-120 min)'),
        plt.Line2D([0], [0], color='#FF6B6B', lw=4, label='Lento (> 120 min)')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=9)
    plt.title('DFG com Metricas de Tempo e Frequencia', fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return G, edge_metrics

def generate_process_summary(dfg, total_flows, cases):
    """Gera um resumo textual do processo baseado no DFG"""
    
    print("=" * 60)
    print("ANALISE DO PROCESSO - RESUMO DO DFG")
    print("=" * 60)
    
    # 1. Estatisticas basicas
    print(f"\n ESTATISTICAS BASICAS:")
    print(f"   Total de casos: {len(cases)}")
    print(f"   Total de transicoes: {total_flows}")
    print(f"   Transicoes unicas: {len(dfg)}")
    
    # 2. Atividades mais frequentes
    activity_counts = Counter()
    for (from_act, _), count in dfg.items():
        activity_counts[from_act] += count
    
    # Adicionar ultima atividade
    for case_activities in cases.values():
        if case_activities:
            activity_counts[case_activities[-1]] += 1
    
    print(f"\n ATIVIDADES MAIS FREQUENTES:")
    for activity, count in activity_counts.most_common(5):
        percentage = (count / (total_flows + len(cases))) * 100
        print(f"   {activity}: {count} ocorrencias ({percentage:.1f}%)")
    
    # 3. Fluxos mais comuns
    print(f"\n FLUXOS MAIS COMUNS:")
    for (from_act, to_act), count in dfg.most_common(5):
        percentage = (count / total_flows) * 100
        print(f"   {from_act} -> {to_act}: {count} vezes ({percentage:.1f}%)")
    
    # 4. Pontos de decisao
    print(f"\n PONTOS DE DECISAO (RAMIFICACOES):")
    from_activities = defaultdict(list)
    for (from_act, to_act) in dfg.keys():
        from_activities[from_act].append(to_act)
    
    for activity, next_activities in from_activities.items():
        if len(next_activities) > 1:
            print(f"   {activity} pode levar para: {', '.join(next_activities)}")
    
    # 5. Inicio e fim do processo
    start_activities = Counter([activities[0] for activities in cases.values() if activities])
    end_activities = Counter([activities[-1] for activities in cases.values() if activities])
    
    print(f"\n ATIVIDADES DE INICIO:")
    for activity, count in start_activities.most_common(3):
        percentage = (count / len(cases)) * 100
        print(f"   {activity}: {count} casos ({percentage:.1f}%)")
    
    print(f"\n ATIVIDADES DE FIM:")
    for activity, count in end_activities.most_common(3):
        percentage = (count / len(cases)) * 100
        print(f"   {activity}: {count} casos ({percentage:.1f}%)")
    
    print("\n" + "=" * 60)

# ==================== ANALISE DE CONFORMIDADE COM DFG ====================
def check_conformance_dfg(dfg, expected_paths):
    """Verifica conformidade com caminhos esperados no DFG"""
    
    print("=" * 60)
    print("ANALISE DE CONFORMIDADE - DFG")
    print("=" * 60)
    
    deviations = []
    
    for (from_act, to_act), count in dfg.items():
        # Verificar se esta transicao e esperada
        is_expected = False
        for expected_path in expected_paths:
            for i in range(len(expected_path) - 1):
                if expected_path[i] == from_act and expected_path[i + 1] == to_act:
                    is_expected = True
                    break
        
        if not is_expected:
            deviations.append((from_act, to_act, count))
    
    if deviations:
        print(f"\n  DESVIOS ENCONTRADOS: {len(deviations)} transicoes nao esperadas")
        for from_act, to_act, count in deviations:
            print(f"   {from_act} -> {to_act}: {count} ocorrencias")
    else:
        print(f"\n TODAS AS TRANSICOES SAO CONFORMES!")
    
    # Calcular taxa de conformidade
    total_transitions = sum(dfg.values())
    unexpected_count = sum(count for _, _, count in deviations)
    conformance_rate = ((total_transitions - unexpected_count) / total_transitions) * 100
    
    print(f"\n TAXA DE CONFORMIDADE: {conformance_rate:.1f}%")
    print("=" * 60)
    
    return conformance_rate, deviations

# ==================== FUNCAO PRINCIPAL ====================
def main():
    """Funcao principal para demonstracao didatica"""
    
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
    
    # 3. Construir DFG basico
    dfg, total_flows = build_directly_follows_graph(df)
    
    # 4. Visualizar DFG simples
    print(" GERANDO VISUALIZACAO DFG SIMPLES...")
    G_simple = visualize_dfg_simple(dfg, total_flows)
    
    # 5. Gerar resumo do processo
    generate_process_summary(dfg, total_flows, cases)
    
    # 6. Construir DFG com metricas de tempo
    print("\n  GERANDO DFG COM METRICAS DE TEMPO...")
    dfg_counts, dfg_times = build_dfg_with_metrics(df)
    G_advanced, edge_metrics = visualize_dfg_advanced(dfg_counts, dfg_times)
    
    # 7. Analise de conformidade
    expected_paths = [
        ['Ticket Received', 'Initial Triage', 'Assign to Specialist', 
         'Technical Analysis', 'Solution Provided', 'Ticket Closed'],
        ['Ticket Received', 'Initial Triage', 'Escalate to Manager',
         'Manager Review', 'Solution Provided', 'Ticket Closed']
    ]
    
    conformance_rate, deviations = check_conformance_dfg(dfg, expected_paths)
    
    # 8. Analise de bottlenecks (com base nos tempos)
    print("\n IDENTIFICANDO BOTTLENECKS...")
    bottlenecks = []
    for (from_act, to_act), metrics in edge_metrics.items():
        if metrics['avg_time'] > 60:  # Mais de 1 hora
            bottlenecks.append((from_act, to_act, metrics['avg_time'], metrics['count']))
    
    if bottlenecks:
        print(f" {len(bottlenecks)} BOTTLENECKS IDENTIFICADOS:")
        for from_act, to_act, avg_time, count in sorted(bottlenecks, key=lambda x: x[2], reverse=True):
            print(f"   {from_act} -> {to_act}: {avg_time:.1f} min (media) em {count} ocorrencias")
    else:
        print(" Nenhum bottleneck significativo identificado!")
    
    # 9. Exportar resultados
    export_results(dfg, edge_metrics, cases)

def export_results(dfg, edge_metrics, cases):
    """Exporta resultados para analise posterior"""
    
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
        {'CaseID': case_id, 'Path': ' -> '.join(activities)}
        for case_id, activities in cases.items()
    ])
    cases_df.to_csv('cases_paths.csv', index=False, encoding='utf-8')
    
    print(f"\n Resultados exportados:")
    print(f"   dfg_analysis.csv: {len(dfg_df)} transicoes analisadas")
    print(f"   cases_paths.csv: {len(cases_df)} casos processados")

# ==================== EXECUCAO ====================
if __name__ == "__main__":
    print(" PROCESS MINING - DEMONSTRACAO DIDATICA COM DFG")
    print("=" * 60)
    main()
