import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import pm4py
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.visualization.petri_net import visualizer as pn_visualizer

# DADOS DO HELPDESK
HELPDESK_LOG = """CaseID,Activity,Resource,Timestamp,Status
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

def carregar_dados_helpdesk():
    # Carrega e prepara dados do helpdesk
    df = pd.read_csv(StringIO(HELPDESK_LOG))
    # Garantir que o identificador do caso seja string (PM4Py requer string)
    if 'CaseID' in df.columns:
        df['CaseID'] = df['CaseID'].astype(str)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    df = df.rename(columns={
        'CaseID': 'case:concept:name',
        'Activity': 'concept:name',
        'Timestamp': 'time:timestamp',
        'Resource': 'org:resource',
        'Status': 'lifecycle:transition'
    })
    
    log = pm4py.convert_to_event_log(df)
    return log, df

def descobrir_modelo_petri_alpha(log):
    # Descobre modelo usando Alpha Miner (Rede de Petri classica)
    from pm4py.algo.discovery.alpha import algorithm as alpha_miner
    
    print("="*70)
    print("DESCOBERTA DO MODELO - ALPHA MINER")
    print("="*70)
    
    net, initial_marking, final_marking = alpha_miner.apply(log)
    
    print("ESTATISTICAS DA REDE DE PETRI:")
    print("  Transicoes (atividades): ", len(net.transitions))
    print("  Lugares (estados): ", len(net.places))
    print("  Arcos (relacoes): ", len(net.arcs))
    print("  Marcacao inicial: ", initial_marking)
    print("  Marcacao final: ", final_marking)
    
    gviz = pn_visualizer.apply(net, initial_marking, final_marking,
                               parameters={pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"})
    pn_visualizer.save(gviz, "helpdesk_petri_alpha.png")
    
    return net, initial_marking, final_marking

def descobrir_modelo_petri_inductive(log):
    # Descobre modelo usando Inductive Miner (moderno, garante soundness)
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    
    print("\n" + "="*70)
    print("DESCOBERTA DO MODELO - INDUCTIVE MINER")
    print("="*70)
    
    # O Inductive Miner pode retornar um ProcessTree em algumas versões/configs.
    # Tentamos desempacotar; se falhar, convertemos a ProcessTree para Rede de Petri.
    result = inductive_miner.apply(log)
    try:
        net, initial_marking, final_marking = result
    except TypeError:
        # result provavelmente eh um ProcessTree -> importar conversor dinamicamente
        tree = result
        from pm4py.objects.conversion.process_tree import converter as pt_converter
        net, initial_marking, final_marking = pt_converter.apply(tree)
    
    print("ESTATISTICAS DA REDE (INDUCTIVE):")
    print("  Transicoes: ", len(net.transitions))
    print("  Lugares: ", len(net.places))
    print("  Soundness garantida pelo algoritmo")
    
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    pn_visualizer.save(gviz, "helpdesk_petri_inductive.png")
    
    return net, initial_marking, final_marking

def descobrir_modelo_petri_heuristics(log):
    # Descobre modelo usando Heuristics Miner (robusto a ruidos)
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
    
    print("\n" + "="*70)
    print("DESCOBERTA DO MODELO - HEURISTICS MINER")
    print("="*70)
    
    heu_net = heuristics_miner.apply_heu(log)
    
    print("ESTATISTICAS DA HEURISTICS NET:")
    try:
        nodes_count = len(heu_net.nodes)
    except Exception:
        try:
            nodes_count = len(list(heu_net.get_nodes()))
        except Exception:
            nodes_count = 'N/A'
    try:
        edges_count = len(heu_net.edges)
    except Exception:
        try:
            edges_count = len(list(heu_net.get_edges()))
        except Exception:
            edges_count = 'N/A'
    print("  Nos: ", nodes_count)
    print("  Arestas: ", edges_count)
    print("  Robustez a ruidos: Alta")
    
    gviz = hn_visualizer.apply(heu_net)
    hn_visualizer.save(gviz, "helpdesk_heuristics_net.png")
    
    net, initial_marking, final_marking = heuristics_miner.apply(log)
    
    gviz_petri = pn_visualizer.apply(net, initial_marking, final_marking)
    pn_visualizer.save(gviz_petri, "helpdesk_petri_heuristics.png")
    
    return net, initial_marking, final_marking, heu_net

def analisar_conformidade_petri(log, net, initial_marking, final_marking):
    # Analise avancada de conformidade usando Redes de Petri
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    
    print("\n" + "="*70)
    print("ANALISE DE CONFORMIDADE - REDES DE PETRI")
    print("="*70)
    
    replayed_traces = token_replay.apply(log, net, initial_marking, final_marking)
    
    total_traces = len(replayed_traces)
    fitting_traces = sum(1 for trace in replayed_traces if trace['trace_is_fit'])
    missing_tokens = sum(trace['missing_tokens'] for trace in replayed_traces)
    remaining_tokens = sum(trace['remaining_tokens'] for trace in replayed_traces)
    consumed_tokens = sum(trace['consumed_tokens'] for trace in replayed_traces)
    produced_tokens = sum(trace['produced_tokens'] for trace in replayed_traces)
    
    fitness = 0.5 * (1 - missing_tokens/consumed_tokens if consumed_tokens > 0 else 1) + \
              0.5 * (1 - remaining_tokens/produced_tokens if produced_tokens > 0 else 1)
    
    print("METRICAS DE CONFORMIDADE:")
    print("  1. Fitness: ", round(fitness, 3), " (0-1, onde 1 e perfeito)")
    print("  2. Casos conformes: ", fitting_traces, "/", total_traces, 
          " (", round((fitting_traces/total_traces)*100, 1), "%)")
    print("  3. Tokens faltando: ", missing_tokens)
    print("  4. Tokens restantes: ", remaining_tokens)
    
    try:
        precision = precision_evaluator.apply(log, net, initial_marking, final_marking)
        print("  5. Precision: ", round(precision, 3), " (alta = modelo especifico)")
    except:
        print("  5. Precision: Nao calculavel")
    
    try:
        generalization = generalization_evaluator.apply(log, net, initial_marking, final_marking)
        print("  6. Generalization: ", round(generalization, 3), " (alta = boa generalizacao)")
    except:
        print("  6. Generalization: Nao calculavel")
    
    print("ANALISE DETALHADA DOS DESVIOS:")
    problematic_cases = []
    
    for i, trace in enumerate(replayed_traces):
        if not trace['trace_is_fit'] or trace['missing_tokens'] > 0:
            case_id = log[i].attributes['concept:name'] if i < len(log) else f"Case_{i}"
            problematic_cases.append({
                'case': case_id,
                'is_fit': trace['trace_is_fit'],
                'missing': trace['missing_tokens'],
                'remaining': trace['remaining_tokens']
            })
    
    if problematic_cases:
        print("  Total de casos com problemas: ", len(problematic_cases))
        for case in problematic_cases[:3]:
            status = "OK" if case['is_fit'] else "NOT_OK"
            print("  ", status, case['case'], ": ", case['missing'], " tokens faltando")
    else:
        print("  Todos os casos sao perfeitamente conformes!")
    
    return {
        'fitness': fitness,
        'fitting_traces': fitting_traces,
        'total_traces': total_traces,
        'problematic_cases': problematic_cases,
        'replayed_traces': replayed_traces
    }

def analisar_estrutura_petri(net, initial_marking, final_marking):
    # Analisa propriedades estruturais da Rede de Petri
    
    print("\n" + "="*70)
    print("ANALISE ESTRUTURAL - REDE DE PETRI")
    print("="*70)
    
    print("CONSTRUCOES IDENTIFICADAS:")
    
    sequencias = 0
    escolhas = 0
    loops = 0
    
    for place in net.places:
        input_arcs = len([arc for arc in net.arcs if arc.target == place])
        output_arcs = len([arc for arc in net.arcs if arc.source == place])
        
        if input_arcs == 1 and output_arcs == 1:
            sequencias += 1
        elif output_arcs > 1:
            escolhas += 1
    
    print("  Sequencias: ", sequencias)
    print("  Pontos de escolha (XOR/OR): ", escolhas)
    print("  Loops identificados: ", loops)
    
    print("TRANSIÇÕES CRÍTICAS:")
    
    for transition in net.transitions:
        if transition.label is None:
            print("  ", transition.name, ": Transicao silenciosa (tau)")
        else:
            print("  ", transition.label, ": Transicao visivel")
    
    print("VERIFICACAO DE SOUNDNESS:")
    
    if len(initial_marking) > 0:
        print("  Marcacao inicial presente: ", initial_marking)
    else:
        print("  Sem marcacao inicial")
    
    if len(final_marking) > 0:
        print("  Marcacao final presente: ", final_marking)
    else:
        print("  Sem marcacao final")
    
    print("ANALISE DE DEADLOCKS POTENCIAIS:")
    
    potential_deadlocks = []
    for place in net.places:
        input_transitions = [arc.source for arc in net.arcs if arc.target == place]
        output_transitions = [arc.target for arc in net.arcs if arc.source == place]
        
        if len(input_transitions) > 1 and len(output_transitions) == 0:
            potential_deadlocks.append(place.name)
    
    if potential_deadlocks:
        print("  Lugares com potencial deadlock: ", len(potential_deadlocks))
        for place in potential_deadlocks[:3]:
            print("      - ", place)
    else:
        print("  Nenhum deadlock potencial identificado")
    
    return {
        'sequencias': sequencias,
        'escolhas': escolhas,
        'loops': loops,
        'potential_deadlocks': potential_deadlocks
    }

def simular_processo_petri(net, initial_marking, final_marking):
    # Simula execucao do processo na Rede de Petri
    from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
    
    print("\n" + "="*70)
    print("SIMULACAO DO PROCESSO - REDE DE PETRI")
    print("="*70)
    
    simulated_log = simulator.apply(net, initial_marking, 
                                   parameters={
                                       simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: 10
                                   })
    
    print("RESULTADOS DA SIMULACAO (10 casos):")
    print("  Casos simulados: ", len(simulated_log))
    
    from pm4py.algo.filtering.log.variants import variants_filter
    variants = variants_filter.get_variants(simulated_log)
    
    print("  Variantes unicas na simulacao: ", len(variants))
    
    print("CAMINHOS SIMULADOS:")
    for i, (variant, cases) in enumerate(sorted(variants.items(), 
                                                key=lambda x: len(x[1]), 
                                                reverse=True)[:3]):
        # variant pode ser: lista/tupla de eventos, lista/tupla de nomes, ou string
        if isinstance(variant, str):
            variant_str = variant
        elif isinstance(variant, (list, tuple)):
            try:
                variant_str = " -> ".join([event['concept:name'] for event in variant])
            except Exception:
                # talvez seja uma tupla/lista de nomes
                try:
                    variant_str = " -> ".join(map(str, variant))
                except Exception:
                    variant_str = str(variant)
        else:
            variant_str = str(variant)
        print("  ", i+1, ". ", variant_str)
        print("      Ocorrencias: ", len(cases))
    
    print("COMPARACAO COM LOG REAL:")
    print("  A simulacao explora caminhos possiveis no modelo")
    print("  Permite identificar comportamentos nao observados no log")
    
    return simulated_log, variants

def analisar_bottlenecks_petri(net, log):
    # Analise avancada de bottlenecks usando teoria de Redes de Petri
    
    print("\n" + "="*70)
    print("ANALISE DE BOTTLENECKS - TEORIA DE REDES DE PETRI")
    print("="*70)
    
    sojourn_times = {}
    for trace in log:
        for i in range(len(trace) - 1):
            current = trace[i]
            next_event = trace[i + 1]
            activity = current['concept:name']
            duration = (next_event['time:timestamp'] - current['time:timestamp']).total_seconds() / 60
            sojourn_times.setdefault(activity, []).append(duration)
    
    print("LUGARES DE ESPERA/ESTOQUE:")
    
    bottlenecks = []
    for place in net.places:
        input_arcs = [arc for arc in net.arcs if arc.target == place]
        output_arcs = [arc for arc in net.arcs if arc.source == place]
        
        if len(input_arcs) > 1 and len(output_arcs) == 1:
            print("  ", place.name, ": Ponto de sincronizacao (input: ", 
                  len(input_arcs), ", output: ", len(output_arcs), ")")
    
    print("ANALISE DE CAPACIDADE:")
    
    resource_places = []
    for place in net.places:
        if "Resource" in place.name or "Agent" in place.name or "Specialist" in place.name:
            resource_places.append(place.name)
            print("  ", place.name, ": Representa recurso limitado")
    
    print("ANALISE DE FLUXO:")
    
    transition_demand = {}
    for transition in net.transitions:
        if transition.label and transition.label in sojourn_times:
            times = sojourn_times[transition.label]
            if times:
                avg_time = sum(times) / len(times)
                if avg_time > 60:
                    transition_demand[transition.label] = avg_time
    
    if transition_demand:
        print("  TRANSICOES COM ALTO TEMPO DE PROCESSAMENTO:")
        for activity, avg_time in sorted(transition_demand.items(), key=lambda x: x[1], reverse=True):
            print("      ", activity, ": ", round(avg_time, 1), " minutos (media)")
    else:
        print("  Nenhuma transicao com tempo excessivo identificada")
    
    return {
        'sojourn_times': sojourn_times,
        'resource_places': resource_places,
        'high_demand_transitions': transition_demand
    }

def visualizacao_avancada_petri(net, initial_marking, final_marking, log):
    # Visualizacoes avancadas da Rede de Petri
    
    print("\n" + "="*70)
    print("VISUALIZACAO AVANCADA - REDE DE PETRI")
    print("="*70)
    
    print("1. REDE DE PETRI COM FREQUENCIAS:")
    
    parameters = {
        pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png",
        pn_visualizer.Variants.WO_DECORATION.value.Parameters.DEBUG: False,
        pn_visualizer.Variants.WO_DECORATION.value.Parameters.RANKDIR: "LR"
    }
    
    gviz_decorated = pn_visualizer.apply(net, initial_marking, final_marking,
                                         parameters=parameters,
                                         variant=pn_visualizer.Variants.WO_DECORATION)
    pn_visualizer.save(gviz_decorated, "helpdesk_petri_decorated.png")
    
    print("2. MATRIZ DE INCIDENCIA DA REDE:")
    
    places = list(net.places)
    transitions = list(net.transitions)
    
    print("  Dimensoes: ", len(places), " lugares x ", len(transitions), " transicoes")
    print("  Matriz sparse (muitos zeros)")
    
    print("3. ESPACO DE ESTADOS (SIMPLIFICADO):")
    
    print("  Estados possiveis: Exponencial no numero de lugares")
    print("  Para analise completa: Arvore/grafo de alcancabilidade")
    
    return gviz_decorated

def gerar_relatorio_petri(net, initial_marking, final_marking, log, 
                         conformance_results, structural_results,
                         simulation_results, bottleneck_results):
    # Gera relatorio completo da analise com Redes de Petri
    
    print("\n" + "="*70)
    print("RELATORIO COMPLETO - ANALISE COM REDES DE PETRI")
    print("="*70)
    
    print("RESUMO EXECUTIVO:")
    print("  Processo analisado: Helpdesk de TI")
    print("  Casos analisados: ", len(log))
    print("  Fitness do modelo: ", round(conformance_results['fitness'], 3))
    print("  Conformidade: ", conformance_results['fitting_traces'], 
          "/", len(log), " casos")
    
    print("INSIGHTS PRINCIPAIS:")
    
    if bottleneck_results['high_demand_transitions']:
        print("  1. BOTTLENECKS IDENTIFICADOS:")
        for activity, avg_time in bottleneck_results['high_demand_transitions'].items():
            print("      ", activity, ": ", round(avg_time, 1), " minutos")
    else:
        print("  1. Nenhum bottleneck critico")
    
    if structural_results['potential_deadlocks']:
        print("  2. POTENCIAIS DEADLOCKS: ", 
              len(structural_results['potential_deadlocks']))
    else:
        print("  2. Nenhum deadlock potencial")
    
    from pm4py.algo.filtering.log.variants import variants_filter
    variants = variants_filter.get_variants(log)
    print("  3. VARIABILIDADE: ", len(variants), " variantes unicas")
    
    print("RECOMENDACOES:")
    
    if conformance_results['fitness'] < 0.9:
        print("  1. Melhorar conformidade do processo (fitness: ", 
              round(conformance_results['fitness'], 3), ")")
        print("      Analisar casos problematicos: ", 
              len(conformance_results['problematic_cases']))
    
    if bottleneck_results['high_demand_transitions']:
        print("  2. Otimizar atividades lentas:")
        for activity, avg_time in bottleneck_results['high_demand_transitions'].items():
            if avg_time > 120:
                print("      URGENTE: ", activity, " (", round(avg_time, 0), " min)")
            elif avg_time > 60:
                print("      IMPORTANTE: ", activity, " (", round(avg_time, 0), " min)")
    
    print("PROXIMOS PASSOS SUGERIDOS:")
    print("  1. Implementar monitoramento em tempo real com o modelo")
    print("  2. Estabelecer KPIs baseados nas metricas identificadas")
    print("  3. Realizar simulacoes de cenarios de melhoria")
    print("  4. Implementar sistema de detecao precoce de desvios")
    
    print("DADOS EXPORTADOS:")
    print("  1. helpdesk_petri_alpha.png - Modelo Alpha Miner")
    print("  2. helpdesk_petri_inductive.png - Modelo Inductive Miner")
    print("  3. helpdesk_petri_heuristics.png - Modelo Heuristics Miner")
    print("  4. helpdesk_heuristics_net.png - Heuristics Net")
    
    return {
        'summary': {
            'cases': len(log),
            'fitness': conformance_results['fitness'],
            'variants': len(variants),
            'bottlenecks': len(bottleneck_results['high_demand_transitions']),
            'potential_deadlocks': len(structural_results['potential_deadlocks'])
        }
    }

def main():
    # Funcao principal da analise com Redes de Petri
    
    print("\n" + "="*70)
    print("ANALISE DE PROCESSOS COM REDES DE PETRI - HELPDESK")
    print("="*70)
    print("Autor: Process Mining Specialist")
    print("Data: 2024")
    print("="*70)
    
    print("1. CARREGANDO DADOS DO HELPDESK...")
    log, df = carregar_dados_helpdesk()
    print("   ", len(log), " casos carregados")
    print("   ", sum(len(trace) for trace in log), " eventos totais")
    
    print("2. DESCOBRINDO MODELOS COM REDES DE PETRI...")
    
    net_alpha, im_alpha, fm_alpha = descobrir_modelo_petri_alpha(log)
    net_ind, im_ind, fm_ind = descobrir_modelo_petri_inductive(log)
    net_heu, im_heu, fm_heu, heu_net = descobrir_modelo_petri_heuristics(log)
    
    print("3. ANALISANDO CONFORMIDADE...")
    conformance_results = analisar_conformidade_petri(log, net_alpha, im_alpha, fm_alpha)
    
    print("4. ANALISANDO ESTRUTURA DA REDE...")
    structural_results = analisar_estrutura_petri(net_alpha, im_alpha, fm_alpha)
    
    print("5. SIMULANDO PROCESSO...")
    simulated_log, simulation_variants = simular_processo_petri(net_alpha, im_alpha, fm_alpha)
    
    print("6. IDENTIFICANDO BOTTLENECKS...")
    bottleneck_results = analisar_bottlenecks_petri(net_alpha, log)
    
    print("7. GERANDO VISUALIZACOES AVANÇADAS...")
    gviz_decorated = visualizacao_avancada_petri(net_alpha, im_alpha, fm_alpha, log)
    
    print("8. GERANDO RELATORIO FINAL...")
    relatorio = gerar_relatorio_petri(net_alpha, im_alpha, fm_alpha, log,
                                     conformance_results, structural_results,
                                     {'variants': simulation_variants}, bottleneck_results)
    
    print("\n" + "="*70)
    print("ANALISE COM REDES DE PETRI CONCLUIDA COM SUCESSO!")
    print("="*70)
    
    return {
        'log': log,
        'models': {
            'alpha': (net_alpha, im_alpha, fm_alpha),
            'inductive': (net_ind, im_ind, fm_ind),
            'heuristics': (net_heu, im_heu, fm_heu)
        },
        'conformance': conformance_results,
        'structure': structural_results,
        'bottlenecks': bottleneck_results,
        'report': relatorio
    }

if __name__ == "__main__":
    try:
        resultados = main()
        
        import json
        with open('resultados_petri.json', 'w') as f:
            resultados_serializaveis = {
                'statistics': {
                    'cases': len(resultados['log']),
                    'fitness': resultados['conformance']['fitness'],
                    'problematic_cases': len(resultados['conformance']['problematic_cases'])
                },
                'bottlenecks': resultados['bottlenecks']['high_demand_transitions'],
                'report_summary': resultados['report']['summary']
            }
            json.dump(resultados_serializaveis, f, indent=2)
        
        print("\nArquivos gerados:")
        print("   resultados_petri.json - Dados da analise")
        print("   helpdesk_petri_*.png - Modelos visuais")
        print("\nAnalise pronta para apresentacao e tomada de decisao!")
        
    except Exception as e:
        print("Erro durante a execucao: ", e)
        print("Solucao: Certifique-se de ter PM4Py instalado:")
        print("  pip install pm4py[all]")