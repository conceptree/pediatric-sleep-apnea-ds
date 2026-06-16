# Sleep AI Orchestrator - Registo Técnico Detalhado (MVP v1)

Data de execução: 2026-04-18

## 1) Enquadramento metodológico

Objetivo científico do ciclo v1:
- Construir uma base reproduzível de clinical decision support para investigação em apneia pediátrica.
- Evitar diagnóstico automático e evitar overengineering inicial.
- Estabelecer baseline comparável para iteração futura (DL e orquestração mais avançada) sem refazer pipeline.

Princípios adotados:
- Execução incremental com validação em cada passo.
- Contrato de dados estável entre etapas.
- Guardrails explícitos contra leakage.
- Seleção de baseline com regra de decisão fixa: métrica primária test balanced accuracy, desempate por test F1.

---

## 2) Resumo executivo do resultado final

Decisão final do MVP v1:
- Baseline selecionado: xgboost.
- Variante selecionada após calibração: uncalibrated.

Métricas chave no teste (comparação multímodelo):
- majority: balanced_accuracy 0.500000, f1 0.000000
- threshold_oxygen_desat: balanced_accuracy 0.819531, f1 0.484848
- logreg: balanced_accuracy 0.765263, f1 0.521739
- random_forest: balanced_accuracy 0.825325, f1 0.452174
- xgboost: balanced_accuracy 0.839669, f1 0.481481

Fontes:
- [outputs/baselines/baseline_comparison_step18_all_models_with_xgboost.txt](outputs/baselines/baseline_comparison_step18_all_models_with_xgboost.txt)
- [outputs/baselines/baseline_metrics_step20_xgboost_calibration.txt](outputs/baselines/baseline_metrics_step20_xgboost_calibration.txt)
- [outputs/baselines/mvp_v1_summary_step19.md](outputs/baselines/mvp_v1_summary_step19.md)

---

## 3) Diário técnico passo a passo (1-20)

### Step 1
Script:
- [pre-stage-scripts/01_dataset_inventory.py](pre-stage-scripts/01_dataset_inventory.py)

Output:
- [outputs/inventory/dataset_inventory_step1.txt](outputs/inventory/dataset_inventory_step1.txt)

Resultado:
- Inventário inicial do dataset com árvore superficial e distribuição de extensões.
- Primeira fotografia operacional do corpus (sem processamento de sinal).

Razão da abordagem:
- Criar baseline de observabilidade do dataset antes de qualquer transformação.
- Reduzir risco de decisões cegas sobre estrutura real dos ficheiros.

### Step 2
Script:
- [pre-stage-scripts/02_pairing_check.py](pre-stage-scripts/02_pairing_check.py)

Output:
- [outputs/inventory/pairing_check_step2.txt](outputs/inventory/pairing_check_step2.txt)

Resultado:
- Sleep_Data: 1834 EDF, 1833 TSV, 1833 pares, 1 EDF órfão (10345_25318).
- Health_Data: 18 CSV.

Razão da abordagem:
- Verificar integridade mínima de pares EDF-TSV por nome-base antes de modelação.
- Evitar falhas tardias no pipeline devido a pares incompletos.

### Step 3
Script:
- [pre-stage-scripts/03_build_paired_manifest.py](pre-stage-scripts/03_build_paired_manifest.py)

Outputs:
- [outputs/inventory/paired_sleep_records_step3.csv](outputs/inventory/paired_sleep_records_step3.csv)
- [outputs/inventory/paired_sleep_records_step3_qc.txt](outputs/inventory/paired_sleep_records_step3_qc.txt)

Resultado:
- Manifest explícito de pares válidos EDF/TSV para consumo posterior.
- QC com contagem de pares e amostras de órfãos.

Razão da abordagem:
- Formalizar contrato de dados para as etapas seguintes.
- Substituir varreduras ad-hoc por artefacto versionável e reexecutável.

### Step 4
Script:
- [pre-stage-scripts/04_create_grouped_splits.py](pre-stage-scripts/04_create_grouped_splits.py)

Outputs:
- [outputs/inventory/paired_sleep_records_step4_splits.csv](outputs/inventory/paired_sleep_records_step4_splits.csv)
- [outputs/inventory/paired_sleep_records_step4_splits_qc.txt](outputs/inventory/paired_sleep_records_step4_splits_qc.txt)

Resultado:
- Split agrupado por paciente com seed fixa 42.
- Totais: records train 1291, val 269, test 273; patients 1694.

Razão da abordagem:
- Guardrail principal contra leakage inter-split por identidade de paciente.
- Preservar validade metodológica dos resultados de generalização.

### Step 5
Script:
- [pre-stage-scripts/05_health_data_schema_inventory.py](pre-stage-scripts/05_health_data_schema_inventory.py)

Output:
- [outputs/inventory/health_data_schema_step5.txt](outputs/inventory/health_data_schema_step5.txt)

Resultado:
- Inventário de schema (colunas, contagens, id-like columns) das tabelas clínicas.
- Correções incorporadas durante execução:
- Fallback de encoding para CSV heterogéneo.
- Exclusão de ficheiros macOS resource-fork com prefixo ._.

Razão da abordagem:
- Conhecer fontes de labels e chaves clínicas sem tocar ainda em processamento PSG bruto.
- Resolver cedo variabilidade de encoding e artefactos de sistema operativo.

### Step 6
Script:
- [pre-stage-scripts/06_linkage_coverage_check.py](pre-stage-scripts/06_linkage_coverage_check.py)

Output:
- [outputs/inventory/linkage_coverage_step6.txt](outputs/inventory/linkage_coverage_step6.txt)

Resultado:
- Cobertura de ligação 100% entre manifest e tabelas clínicas:
- patient in demographic 1694/1694
- patient in sleep_study 1694/1694
- sleep_study_id in sleep_study 1833/1833

Razão da abordagem:
- Validar viabilidade de integração entre eventos de sono e metadados clínicos.
- Evitar entrar em modelação sem garantia de join consistente.

### Step 7
Script:
- [pre-stage-scripts/07_tsv_schema_probe.py](pre-stage-scripts/07_tsv_schema_probe.py)

Output:
- [outputs/inventory/tsv_schema_step7.txt](outputs/inventory/tsv_schema_step7.txt)

Resultado:
- Schema único em amostra: onset, duration, description.
- Não foram encontrados targets no header.

Razão da abordagem:
- Confirmar que o sinal supervisionado está no conteúdo de description e não no nome de coluna.
- Definir estratégia correta para extração de features/labels de eventos.

### Step 8
Script:
- [pre-stage-scripts/08_tsv_description_profile.py](pre-stage-scripts/08_tsv_description_profile.py)

Outputs:
- [outputs/inventory/tsv_description_profile_step8.txt](outputs/inventory/tsv_description_profile_step8.txt)
- [outputs/inventory/tsv_description_top_values_step8.csv](outputs/inventory/tsv_description_top_values_step8.csv)

Resultado:
- Perfil de taxonomia de eventos em description.
- Evidência de eventos respiratórios relevantes (obstructive apnea/hypopnea, oxygen desaturation, etc.).

Razão da abordagem:
- Basear desenho de labels e features em distribuição real do corpus.
- Evitar suposições clínicas sem suporte empírico nos dados.

### Step 9
Script:
- [pre-stage-scripts/09_build_event_features.py](pre-stage-scripts/09_build_event_features.py)

Outputs:
- [outputs/inventory/record_event_features_step9.csv](outputs/inventory/record_event_features_step9.csv)
- [outputs/inventory/record_event_features_step9_qc.txt](outputs/inventory/record_event_features_step9_qc.txt)

Resultado:
- Tabela por registo com contagens/taxas de eventos.
- Labels candidatos derivados para baseline inicial.
- Cobertura total: 1833 registos processados, 0 unreadable.

Razão da abordagem:
- Transformar eventos textuais em matriz tabular clássica para ML baseline.
- Manter interpretação transparente e auditável do que entra no modelo.

### Step 10
Script:
- [pre-stage-scripts/10_build_leakage_safe_baseline_matrix.py](pre-stage-scripts/10_build_leakage_safe_baseline_matrix.py)

Outputs:
- [outputs/inventory/baseline_matrix_step10.csv](outputs/inventory/baseline_matrix_step10.csv)
- [outputs/inventory/baseline_matrix_step10_guardrails.txt](outputs/inventory/baseline_matrix_step10_guardrails.txt)

Resultado:
- Construção de matriz leakage-safe para comparação de modelos.
- Remoção de colunas tautológicas derivadas diretamente de burden respiratório.

Razão da abordagem:
- Garantir avaliação justa de capacidade preditiva sem atalho de target leakage.
- Formalizar guardrails por escrito para rastreabilidade científica.

### Step 11
Script:
- [pre-stage-scripts/11_train_rule_baselines.py](pre-stage-scripts/11_train_rule_baselines.py)

Outputs:
- [outputs/baselines/baseline_metrics_step11.txt](outputs/baselines/baseline_metrics_step11.txt)
- [outputs/baselines/baseline_predictions_step11.csv](outputs/baselines/baseline_predictions_step11.csv)

Resultado:
- Baseline A majority e baseline B threshold em oxygen desaturation.
- Threshold mostrou ganho relevante face ao majority em balanced accuracy e recall.

Razão da abordagem:
- Criar referência mínima de desempenho com interpretabilidade máxima.
- Estabelecer benchmark simples antes de modelos mais complexos.

### Step 12
Script:
- [pre-stage-scripts/12_train_logreg_baseline.py](pre-stage-scripts/12_train_logreg_baseline.py)

Outputs:
- [outputs/baselines/baseline_metrics_step12_logreg.txt](outputs/baselines/baseline_metrics_step12_logreg.txt)
- [outputs/baselines/baseline_predictions_step12_logreg.csv](outputs/baselines/baseline_predictions_step12_logreg.csv)

Resultado:
- Logistic Regression com class imbalance handling e threshold tuning em validação.
- Melhoria de F1 de teste vs threshold, com trade-off em balanced accuracy.

Razão da abordagem:
- Primeiro baseline de ML clássico linear e robusto.
- Medir ganho real sobre heurística mantendo protocolo idêntico.

### Step 13
Script:
- [pre-stage-scripts/13_compare_baselines.py](pre-stage-scripts/13_compare_baselines.py)

Output:
- [outputs/baselines/baseline_comparison_step13.txt](outputs/baselines/baseline_comparison_step13.txt)

Resultado:
- Comparação consolidada majority vs threshold vs logreg.
- Seleção automática pelo critério definido à data.

Razão da abordagem:
- Eliminar arbitrariedade na escolha de baseline.
- Fixar regra de decisão reproduzível.

### Step 14
Script:
- [pre-stage-scripts/14_write_baseline_decision.py](pre-stage-scripts/14_write_baseline_decision.py)

Output:
- [outputs/baselines/baseline_decision_step14.md](outputs/baselines/baseline_decision_step14.md)

Resultado:
- Artefacto narrativo da decisão de baseline intermédia.

Razão da abordagem:
- Suportar componente escrita com decisão técnica documentada.
- Manter continuidade entre execução técnica e texto académico.

### Step 15
Script:
- [pre-stage-scripts/15_train_random_forest_baseline.py](pre-stage-scripts/15_train_random_forest_baseline.py)

Outputs:
- [outputs/baselines/baseline_metrics_step15_rf.txt](outputs/baselines/baseline_metrics_step15_rf.txt)
- [outputs/baselines/baseline_predictions_step15_rf.csv](outputs/baselines/baseline_predictions_step15_rf.csv)

Resultado:
- Baseline não linear com forte recall e ganho em balanced accuracy.

Razão da abordagem:
- Testar capacidade de modelar relações não lineares com o mesmo contrato de dados.
- Comparar com logreg e heurísticas sem mudar protocolo.

### Step 16
Script:
- [pre-stage-scripts/16_compare_all_baselines.py](pre-stage-scripts/16_compare_all_baselines.py)

Output:
- [outputs/baselines/baseline_comparison_step16_all_models.txt](outputs/baselines/baseline_comparison_step16_all_models.txt)

Resultado:
- Consolidação de majority, threshold, logreg, random forest.

Razão da abordagem:
- Uniformizar comparação antes de incluir XGBoost.

### Step 17
Script:
- [pre-stage-scripts/17_train_xgboost_baseline.py](pre-stage-scripts/17_train_xgboost_baseline.py)

Outputs:
- [outputs/baselines/baseline_metrics_step17_xgboost.txt](outputs/baselines/baseline_metrics_step17_xgboost.txt)
- [outputs/baselines/baseline_predictions_step17_xgboost.csv](outputs/baselines/baseline_predictions_step17_xgboost.csv)

Resultado:
- Melhor balanced accuracy de teste entre candidatos até então.
- Correção operacional necessária em macOS: instalação de libomp para runtime XGBoost.

Razão da abordagem:
- Incluir algoritmo boosting clássico de referência industrial.
- Verificar ganho incremental sem alterar pipeline base.

### Step 18
Script:
- [pre-stage-scripts/18_compare_all_baselines_with_xgboost.py](pre-stage-scripts/18_compare_all_baselines_with_xgboost.py)

Output:
- [outputs/baselines/baseline_comparison_step18_all_models_with_xgboost.txt](outputs/baselines/baseline_comparison_step18_all_models_with_xgboost.txt)

Resultado:
- Comparação final com cinco candidatos.
- Baseline selecionado: xgboost (pela regra definida).

Razão da abordagem:
- Fechar seleção com todos os modelos relevantes no mesmo quadro comparativo.

### Step 19
Script:
- [pre-stage-scripts/19_write_mvp_v1_summary.py](pre-stage-scripts/19_write_mvp_v1_summary.py)

Output:
- [outputs/baselines/mvp_v1_summary_step19.md](outputs/baselines/mvp_v1_summary_step19.md)

Resultado:
- Sumário executivo do ciclo v1.
- Ajustes aplicados para consistência de referências e baseline selecionado.

Razão da abordagem:
- Criar documento de fecho técnico para suporte direto da escrita.

### Step 20
Script:
- [pre-stage-scripts/20_calibrate_xgboost_baseline.py](pre-stage-scripts/20_calibrate_xgboost_baseline.py)

Outputs:
- [outputs/baselines/baseline_metrics_step20_xgboost_calibration.txt](outputs/baselines/baseline_metrics_step20_xgboost_calibration.txt)
- [outputs/baselines/baseline_predictions_step20_xgboost_calibration.csv](outputs/baselines/baseline_predictions_step20_xgboost_calibration.csv)

Resultado:
- Comparadas variantes uncalibrated, platt e isotonic.
- Pela regra de seleção operacional, selected_variant: uncalibrated.
- Observação: calibração melhorou Brier em alguns pontos, mas não venceu na métrica primária de seleção.

Razão da abordagem:
- Separar qualidade probabilística de performance de classificação operacional.
- Medir se calibração altera decisão de deployment baseline sob regra fixa.

---

## 4) Problemas encontrados e mitigação

1. Encoding heterogéneo em CSV clínicos.
- Mitigação: fallback de encoding no inventário de schema.

2. Ficheiros macOS ._ a contaminar inspeção.
- Mitigação: filtro explícito de hidden/resource-fork files.

3. XGBoost não carregava libxgboost.dylib por ausência de OpenMP.
- Mitigação: instalação de libomp no macOS.

4. Deriva de numeração de passos durante iteração rápida.
- Mitigação: renomeação para sequência final coerente 17 -> 18 -> 19 -> 20.

---

## 5) Decisões metodológicas justificadas

1. Split por paciente em vez de split por registo.
- Justificação: protege validade externa e minimiza leakage.

2. Seleção por balanced accuracy no teste, com F1 como desempate.
- Justificação: dataset desbalanceado e contexto de suporte clínico com sensibilidade/especificidade relevantes.

3. Exclusão de features leakage-prone na matriz baseline.
- Justificação: evitar inflacionamento artificial de performance.

4. Comparação incremental de modelos no mesmo protocolo.
- Justificação: comparabilidade justa entre heurística, linear, árvore e boosting.

---

## 6) Estado final do MVP v1

Estado:
- Ciclo MVP v1 concluído com baseline selecionado e calibração analisada.
- Pipeline reproduzível com artefactos persistidos em outputs.
- Documentação técnica de suporte à componente escrita disponível.

Baseline em vigor (regra atual):
- Modelo: xgboost.
- Variante: uncalibrated.

Artefactos nucleares:
- [outputs/inventory/baseline_matrix_step10.csv](outputs/inventory/baseline_matrix_step10.csv)
- [outputs/baselines/baseline_metrics_step17_xgboost.txt](outputs/baselines/baseline_metrics_step17_xgboost.txt)
- [outputs/baselines/baseline_metrics_step20_xgboost_calibration.txt](outputs/baselines/baseline_metrics_step20_xgboost_calibration.txt)
- [outputs/baselines/baseline_comparison_step18_all_models_with_xgboost.txt](outputs/baselines/baseline_comparison_step18_all_models_with_xgboost.txt)
- [outputs/baselines/mvp_v1_summary_step19.md](outputs/baselines/mvp_v1_summary_step19.md)

---

## 7) Nota para escrita académica

Este registo foi estruturado para poder ser reutilizado quase diretamente na secção de metodologia e na secção de resultados preliminares da dissertação, incluindo:
- rastreabilidade do processo,
- justificação de escolhas,
- evidência quantitativa por etapa,
- e ligação explícita entre objetivos científicos e decisões de engenharia.
