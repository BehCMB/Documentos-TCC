import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pm4py
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
# importar funcao customizada de DFG (caso queira comparar)
try:
    from dfg_helpdesk import build_directly_follows_graph
except Exception:
    build_directly_follows_graph = None
from io import StringIO

# ==================== FUNCOES PM4Py ====================
def create_pm4py_log(df):
    """Cria log no formato PM4Py a partir de DataFrame"""
    # Renomear colunas para padrao PM4Py
    df = df.rename(columns={
        'CaseID': 'case:concept:name',
        'Activity': 'concept:name', 
        'Timestamp': 'time:timestamp',
        'Resource': 'org:resource',
        'Status': 'lifecycle:transition'
    })
    
    # Converter para log PM4Py
    log = log_converter.apply(df)
    return log, df

def discover_dfg_pm4py(log):
    """Descobre DFG usando PM4Py"""
    from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
    
    # Descobrir DFG (frequencia)
    # O retorno de dfg_discovery.apply pode variar entre versoes do PM4Py
    # (ex.: retornar 3 ou 4 elementos). Tratamos de forma robusta.
    res = dfg_discovery.apply(log)
    if isinstance(res, tuple) or isinstance(res, list):
        if len(res) >= 3:
            dfg_freq = res[0]
            start_activities = res[1]
            end_activities = res[2]
        elif len(res) == 2:
            dfg_freq = res[0]
            start_activities = res[1]
            end_activities = {}
        else:
            dfg_freq = res[0]
            start_activities = {}
            end_activities = {}
    else:
        # se retornou apenas um dicionario
        dfg_freq = res
        start_activities = {}
        end_activities = {}
    
    # Descobrir DFG (desempenho)
    perf_res = dfg_discovery.apply(log, variant=dfg_discovery.Variants.PERFORMANCE)
    if isinstance(perf_res, tuple) or isinstance(perf_res, list):
        dfg_perf = perf_res[0]
    else:
        dfg_perf = perf_res
    
    return dfg_freq, dfg_perf, start_activities, end_activities

def visualize_dfg_pm4py(dfg_freq, dfg_perf, log):
    """Visualiza DFG usando visualizacao nativa do PM4Py"""
    from pm4py.visualization.dfg import visualizer as dfg_visualizer
    
    print("GERANDO VISUALIZACOES COM PM4Py...")
    
    # Visualizacao de frequencia
    print("\n1. DFG por Frequencia:")
    gviz_freq = dfg_visualizer.apply(dfg_freq, 
                                     log=log,
                                     variant=dfg_visualizer.Variants.FREQUENCY)
    dfg_visualizer.view(gviz_freq)
    
    # Visualizacao de desempenho
    print("\n2. DFG por Desempenho (tempo):")
    gviz_perf = dfg_visualizer.apply(dfg_perf, 
                                     log=log,
                                     variant=dfg_visualizer.Variants.PERFORMANCE)
    dfg_visualizer.view(gviz_perf)
    
    return gviz_freq, gviz_perf

def discover_process_model_pm4py(log):
    """Descobre modelo de processo usando PM4Py"""
    from pm4py.algo.discovery.alpha import algorithm as alpha_miner
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    
    print("\nDESCOBRINDO MODELOS DE PROCESSO...")
    
    # 1. Alpha Miner (basico)
    print("1. Alpha Miner (classico):")
    net_alpha, initial_marking_alpha, final_marking_alpha = alpha_miner.apply(log)
    gviz_alpha = pn_visualizer.apply(net_alpha, initial_marking_alpha, final_marking_alpha)
    pn_visualizer.view(gviz_alpha)
    
    # 2. Heuristics Miner (robusto a ruidos)
    print("\n2. Heuristics Miner (robusto):")
    heu_net = heuristics_miner.apply_heu(log)
    from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
    gviz_heu = hn_visualizer.apply(heu_net)
    hn_visualizer.view(gviz_heu)
    
    # 3. Inductive Miner (moderno)
    print("\n3. Inductive Miner (moderno):")
    net_inductive, initial_marking_inductive, final_marking_inductive = inductive_miner.apply(log)
    gviz_inductive = pn_visualizer.apply(net_inductive, initial_marking_inductive, final_marking_inductive)
    pn_visualizer.view(gviz_inductive)
    
    return {
        'alpha': (net_alpha, initial_marking_alpha, final_marking_alpha),
        'heuristics': heu_net,
        'inductive': (net_inductive, initial_marking_inductive, final_marking_inductive)
    }

def analyze_process_stats_pm4py(log, df):
    """Analise de estatisticas do processo usando PM4Py"""
    from pm4py.statistics.traces.generic.log import case_statistics
    from pm4py.statistics.start_activities.log import get as start_activities_get
    from pm4py.statistics.end_activities.log import get as end_activities_get
    from pm4py.statistics.attributes.log import get as attributes_get
    
    print("\nESTATISTICAS DO PROCESSO (PM4Py):")
    print("=" * 60)
    
    # 1. Estatisticas basicas
    print(f"1. ESTATISTICAS BASICAS:")
    print(f"   - Total de casos: {len(log)}")
    print(f"   - Total de eventos: {sum(len(trace) for trace in log)}")
    
    # 2. Atividades de inicio
    start_activities = start_activities_get.get_start_activities(log)
    print(f"\n2. ATIVIDADES DE INICIO:")
    for activity, count in start_activities.items():
        percentage = (count / len(log)) * 100
        print(f"   - {activity}: {count} casos ({percentage:.1f}%)")
    
    # 3. Atividades de fim
    end_activities = end_activities_get.get_end_activities(log)
    print(f"\n3. ATIVIDADES DE FIM:")
    for activity, count in end_activities.items():
        percentage = (count / len(log)) * 100
        print(f"   - {activity}: {count} casos ({percentage:.1f}%)")
    
    # 4. Atividades mais frequentes
    activities = attributes_get.get_attribute_values(log, "concept:name")
    print(f"\n4. ATIVIDADES MAIS FREQUENTES:")
    for activity, count in sorted(activities.items(), key=lambda x: x[1], reverse=True)[:5]:
        total_events = sum(activities.values())
        percentage = (count / total_events) * 100
        print(f"   - {activity}: {count} ocorrencias ({percentage:.1f}%)")
    
    # 5. Duracao dos casos
    case_durations = case_statistics.get_all_case_durations(log, 
                                                           parameters={case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"})
    if case_durations:
        avg_duration = sum(case_durations) / len(case_durations)
        min_duration = min(case_durations)
        max_duration = max(case_durations)
        print(f"\n5. DURACAO DOS CASOS:")
        print(f"   - Media: {avg_duration/3600:.2f} horas")
        print(f"   - Minima: {min_duration/3600:.2f} horas")
        print(f"   - Maxima: {max_duration/3600:.2f} horas")
    
    print("=" * 60)

def discover_variants_pm4py(log):
    """Descobre variantes de processo usando PM4Py"""
    from pm4py.algo.filtering.log.variants import variants_filter
    
    print("\nVARIANTES DE PROCESSO (PM4Py):")
    print("=" * 60)
    
    # Obter variantes
    variants = variants_filter.get_variants(log)

    def variant_to_str(variant):
        # variant can be a string, a tuple/list of event dicts, or a tuple/list of names
        if isinstance(variant, str):
            return variant
        try:
            parts = []
            for ev in variant:
                if isinstance(ev, dict):
                    parts.append(ev.get("concept:name", str(ev)))
                else:
                    parts.append(str(ev))
            return " -> ".join(parts)
        except Exception:
            return str(variant)
    
    print(f"Total de variantes unicas: {len(variants)}")
    print(f"Total de casos: {len(log)}\n")
    
    # Mostrar variantes ordenadas por frequencia
    for i, (variant, cases) in enumerate(sorted(variants.items(), 
                                                key=lambda x: len(x[1]), 
                                                reverse=True)):
        count = len(cases)
        percentage = (count / len(log)) * 100
        
        # Simplificar visualizacao da variante
        variant_str = variant_to_str(variant)
        
        print(f"Variante {i+1}: {percentage:.1f}% dos casos ({count} casos)")
        print(f"  Sequencia: {variant_str}")
        print(f"  IDs dos casos: {list(cases)[:5]}{'...' if len(cases) > 5 else ''}")
        print()
    
    print("=" * 60)
    return variants

def analyze_conformance_pm4py(log):
    """Analise de conformidade usando PM4Py"""
    from pm4py.algo.discovery.alpha import algorithm as alpha_miner
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    
    print("\nANALISE DE CONFORMIDADE (PM4Py):")
    print("=" * 60)
    
    # 1. Descobrir modelo
    net, initial_marking, final_marking = alpha_miner.apply(log)
    
    # 2. Executar token replay
    replayed_traces = token_replay.apply(log, net, initial_marking, final_marking)
    
    # 3. Calcular metricas
    total_traces = len(replayed_traces)
    fitting_traces = sum(1 for trace in replayed_traces if trace['trace_is_fit'])
    missing_tokens = sum(trace['missing_tokens'] for trace in replayed_traces)
    remaining_tokens = sum(trace['remaining_tokens'] for trace in replayed_traces)
    produced_tokens = sum(trace['produced_tokens'] for trace in replayed_traces)
    consumed_tokens = sum(trace['consumed_tokens'] for trace in replayed_traces)
    
    # Calcular fitness
    fitness = 0.5 * (1 - missing_tokens/consumed_tokens if consumed_tokens > 0 else 1) + \
              0.5 * (1 - remaining_tokens/produced_tokens if produced_tokens > 0 else 1)
    
    print(f"1. METRICAS DE CONFORMIDADE:")
    print(f"   - Fitness: {fitness:.3f} (0-1, onde 1 eh perfeito)")
    print(f"   - Traces conformes: {fitting_traces}/{total_traces} ({(fitting_traces/total_traces)*100:.1f}%)")
    print(f"   - Tokens faltando: {missing_tokens}")
    print(f"   - Tokens restantes: {remaining_tokens}")
    
    # 4. Detalhar problemas por caso
    print(f"\n2. CASOS COM PROBLEMAS:")
    problematic_cases = []
    for i, trace in enumerate(replayed_traces):
        if not trace['trace_is_fit'] or trace['missing_tokens'] > 0 or trace['remaining_tokens'] > 0:
            case_id = log[i].attributes['concept:name'] if i < len(log) else f"Case_{i}"
            problematic_cases.append({
                'case': case_id,
                'is_fit': trace['trace_is_fit'],
                'missing': trace['missing_tokens'],
                'remaining': trace['remaining_tokens']
            })
    
    if problematic_cases:
        for case in problematic_cases[:3]:  # Mostrar apenas 3
            status = "OK" if case['is_fit'] else "NOT_OK"
            print(f"   {status} {case['case']}: {case['missing']} tokens faltando, {case['remaining']} tokens restantes")
        if len(problematic_cases) > 3:
            print(f"   ... e mais {len(problematic_cases) - 3} casos")
    else:
        print("   Todos os casos sao conformes!")
    
    print("=" * 60)
    
    return {
        'fitness': fitness,
        'fitting_traces': fitting_traces,
        'total_traces': total_traces,
        'problematic_cases': problematic_cases
    }

def analyze_bottlenecks_pm4py(log):
    """Identifica bottlenecks usando PM4Py"""
    import importlib
    spec = importlib.util.find_spec("pm4py.statistics.sojourn_time.log")
    if spec is not None:
        try:
            from pm4py.statistics.sojourn_time.log import get as sojourn_time
        except Exception:
            sojourn_time = None
    else:
        sojourn_time = None
    # importar workflow_graph se disponivel
    try:
        import importlib
        spec_wf = importlib.util.find_spec("pm4py.algo.analysis.workflow_graph")
        if spec_wf is not None:
            from pm4py.algo.analysis.workflow_graph import algorithm as wf_graph
        else:
            wf_graph = None
    except Exception:
        wf_graph = None
    
    print("\nIDENTIFICACAO DE BOTTLENECKS (PM4Py):")
    print("=" * 60)
    
    # 1. Tempo de espera por atividade
    if sojourn_time is not None:
        try:
            sojourn_times = sojourn_time.apply(log)
        except Exception:
            sojourn_times = None
    else:
        sojourn_times = None

    # Se nao houver funcao disponivel, calcular tempos manualmente a partir do log
    if sojourn_times is None:
        from collections import defaultdict
        sojourn_times = defaultdict(list)
        try:
            for trace in log:
                # trace is a list of events
                for i in range(len(trace) - 1):
                    e_curr = trace[i]
                    e_next = trace[i + 1]
                    t_curr = e_curr.get("time:timestamp") if isinstance(e_curr, dict) else getattr(e_curr, "time:timestamp", None)
                    t_next = e_next.get("time:timestamp") if isinstance(e_next, dict) else getattr(e_next, "time:timestamp", None)
                    name_curr = e_curr.get("concept:name") if isinstance(e_curr, dict) else getattr(e_curr, "concept:name", None)
                    if t_curr is None or t_next is None or name_curr is None:
                        continue
                    try:
                        delta_min = (t_next - t_curr).total_seconds() / 60.0
                    except Exception:
                        continue
                    sojourn_times[name_curr].append(delta_min)
        except Exception:
            sojourn_times = {}
    
    print("1. TEMPO DE ESPERA POR ATIVIDADE:")
    bottlenecks = []
    for activity, times in sojourn_times.items():
        if times:
            avg_time = sum(times) / len(times)
            if avg_time > 60:  # Mais de 1 hora
                bottlenecks.append((activity, avg_time, len(times)))
    
    if bottlenecks:
        for activity, avg_time, count in sorted(bottlenecks, key=lambda x: x[1], reverse=True):
            print(f"   - {activity}: {avg_time:.1f} min (media) em {count} ocorrencias")
    else:
        print("   Nenhum bottleneck significativo encontrado!")
    
    # 2. Analise de fluxo de trabalho
    print(f"\n2. ANALISE DE FLUXO DE TRABALHO:")
    if wf_graph is not None:
        try:
            workflow_graph = wf_graph.apply(log)
            # Aqui poderiamos extrair mais metricas do grafo de workflow
            print(f"   - Grafo de workflow gerado com {len(workflow_graph.nodes())} nos")
        except Exception:
            print("   - Analise de workflow falhou ao aplicar o algoritmo")
    else:
        print("   - Analise de workflow nao disponivel (pm4py.algo.analysis.workflow_graph ausente)")
    
    print("=" * 60)
    
    return bottlenecks

def export_results_pm4py(log, dfg_freq, dfg_perf, variants, conformance_results, bottlenecks):
    """Exporta resultados do PM4Py"""
    import json
    
    print("\nEXPORTANDO RESULTADOS PM4Py...")
    
    # 1. Exportar DFG para CSV
    dfg_data = []
    for (from_act, to_act), freq in dfg_freq.items():
        perf = dfg_perf.get((from_act, to_act), 0)
        dfg_data.append({
            'From': from_act,
            'To': to_act,
            'Frequency': freq,
            'Avg_Time_Seconds': perf
        })
    
    dfg_df = pd.DataFrame(dfg_data)
    dfg_df.to_csv('pm4py_dfg_analysis.csv', index=False, encoding='utf-8')
    
    # 2. Exportar variantes
    # Normalizar variantes para string (lida com varios formatos retornados pelo PM4Py)
    def variant_to_str_local(variant):
        if isinstance(variant, str):
            return variant
        try:
            parts = []
            for ev in variant:
                if isinstance(ev, dict):
                    parts.append(ev.get("concept:name", str(ev)))
                else:
                    parts.append(str(ev))
                    return " -> ".join(parts)
        except Exception:
            return str(variant)

    variants_data = []
    for variant, cases in variants.items():
        variant_str = variant_to_str_local(variant)
        variants_data.append({
            'Variant': variant_str,
            'Case_Count': len(cases),
            'Percentage': (len(cases) / len(log)) * 100,
            'Cases': list(cases)
        })
    
    variants_df = pd.DataFrame(variants_data)
    variants_df.to_csv('pm4py_variants.csv', index=False, encoding='utf-8')
    
    # 3. Exportar metricas de conformidade
    conformance_data = {
        'fitness': conformance_results['fitness'],
        'fitting_traces': conformance_results['fitting_traces'],
        'total_traces': conformance_results['total_traces'],
        'problematic_cases_count': len(conformance_results['problematic_cases'])
    }
    
    with open('pm4py_conformance.json', 'w') as f:
        json.dump(conformance_data, f, indent=2)
    
    # 4. Exportar bottlenecks
    bottlenecks_data = [{'Activity': a, 'Avg_Time_Min': t, 'Occurrences': c} 
                        for a, t, c in bottlenecks]
    bottlenecks_df = pd.DataFrame(bottlenecks_data)
    bottlenecks_df.to_csv('pm4py_bottlenecks.csv', index=False, encoding='utf-8')
    
    print(f"   - pm4py_dfg_analysis.csv: {len(dfg_df)} transicoes")
    print(f"   - pm4py_variants.csv: {len(variants_df)} variantes")
    print(f"   - pm4py_conformance.json: metricas de conformidade")
    print(f"   - pm4py_bottlenecks.csv: {len(bottlenecks_df)} bottlenecks")

# ==================== FUNCAO PRINCIPAL PM4Py ====================
def main_pm4py():
    """Funcao principal usando PM4Py"""
    
    print("PROCESS MINING COM PM4Py")
    print("=" * 60)
    
    # 1. Carregar dados
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
    
    # Converter para DataFrame
    from io import StringIO
    df = pd.read_csv(StringIO(log_content))
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # 2. Criar log PM4Py
    print("1. Criando log PM4Py...")
    log, df_formatted = create_pm4py_log(df)
    
    # 3. Descobrir DFG
    print("2. Descobrindo DFG...")
    dfg_freq, dfg_perf, start_activities, end_activities = discover_dfg_pm4py(log)
    
    # 4. Visualizar DFG
    print("3. Visualizando DFG...")
    gviz_freq, gviz_perf = visualize_dfg_pm4py(dfg_freq, dfg_perf, log)
    
    # 5. Estatisticas do processo
    analyze_process_stats_pm4py(log, df_formatted)
    
    # 6. Descobrir variantes
    variants = discover_variants_pm4py(log)
    
    # 7. Analise de conformidade
    conformance_results = analyze_conformance_pm4py(log)
    
    # 8. Identificar bottlenecks
    bottlenecks = analyze_bottlenecks_pm4py(log)
    
    # 9. Descobrir modelos de processo (opcional - descomente se quiser ver)
    # models = discover_process_model_pm4py(log)
    
    # 10. Exportar resultados
    export_results_pm4py(log, dfg_freq, dfg_perf, variants, conformance_results, bottlenecks)
    
    print("\n" + "=" * 60)
    print("ANALISE COM PM4Py CONCLUIDA!")
    print("=" * 60)

# ==================== FUNCAO DE COMPARACAO ====================
def compare_pm4py_vs_custom():
    """Compara resultados PM4Py vs implementacao customizada"""
    
    print("COMPARACAO: PM4Py vs IMPLEMENTACAO CUSTOMIZADA")
    print("=" * 60)
    
    # Dados
    log_content = """CaseID,Activity,Timestamp
1001,Ticket Received,2024-01-01 09:00:00
1001,Initial Triage,2024-01-01 09:15:00
1001,Assign to Specialist,2024-01-01 09:30:00
1001,Technical Analysis,2024-01-01 10:00:00
1001,Solution Provided,2024-01-01 11:30:00
1001,Ticket Closed,2024-01-01 12:00:00
1002,Ticket Received,2024-01-01 09:05:00
1002,Initial Triage,2024-01-01 09:20:00
1002,Assign to Specialist,2024-01-01 09:35:00
1002,Technical Analysis,2024-01-01 10:30:00
1002,Solution Provided,2024-01-01 11:30:00
1002,Ticket Closed,2024-01-01 12:00:00"""
    
    df = pd.read_csv(StringIO(log_content))
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # PM4Py
    print("\nPM4Py:")
    log_pm4py, _ = create_pm4py_log(df)
    dfg_freq_pm4py, _, _, _ = discover_dfg_pm4py(log_pm4py)
    print(f"   - Transicoes encontradas: {len(dfg_freq_pm4py)}")
    
    # Customizado
    print("\nCustomizado:")
    dfg_custom, total_flows = build_directly_follows_graph(df)
    print(f"   - Transicoes encontradas: {len(dfg_custom)}")
    print(f"   - Total de fluxos: {total_flows}")
    
    # Comparar
    print("\nCOMPARACAO:")
    
    # Converter para conjuntos para comparacao
    pm4py_set = set(dfg_freq_pm4py.keys())
    custom_set = set(dfg_custom.keys())
    
    print(f"   - Transicoes iguais: {len(pm4py_set.intersection(custom_set))}")
    print(f"   - Apenas no PM4Py: {len(pm4py_set - custom_set)}")
    print(f"   - Apenas no Custom: {len(custom_set - pm4py_set)}")
    
    if pm4py_set == custom_set:
        print("   Resultados identicos!")
    else:
        print("   Diferencas encontradas:")
        for trans in pm4py_set - custom_set:
            print(f"      - PM4Py tem: {trans}")
        for trans in custom_set - pm4py_set:
            print(f"      - Custom tem: {trans}")
    
    print("=" * 60)

# ==================== EXECUCAO ====================
if __name__ == "__main__":
    import sys
    
    print("Selecione a opcao:")
    print("1. Analise completa com PM4Py")
    print("2. Comparacao PM4Py vs Custom")
    print("3. Sair")
    
    choice = input("\nDigite sua escolha (1-3): ")
    
    if choice == "1":
        main_pm4py()
    elif choice == "2":
        compare_pm4py_vs_custom()
    elif choice == "3":
        sys.exit()
    else:
        print("Opcao invalida!")
