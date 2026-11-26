# ML Challenge — FutebolAI

## Problema
Estimar o resultado de uma partida da Premier League: vitória do time mandante (H), empate (D) ou vitória do visitante (A), utilizando nomes dos times e odds pré-jogo.

## Dataset público
Dados históricos da Premier League disponibilizados por Football-Data.co.uk:
https://www.football-data.co.uk/englandm.php
O script baixa automaticamente temporadas históricas concluídas (2020/21 a 2024/25, quando disponíveis).

## Algoritmos
- Regressão Logística.
- Random Forest.

## Metodologia
Os dados são limpos, explorados e divididos em treino e teste (80/20), com estratificação. Nomes dos times são codificados e odds são padronizadas. As métricas são calculadas apenas no conjunto de teste.

## Como executar no Windows
1. Instale Python 3.10 ou superior.
2. Abra o Prompt de Comando nesta pasta.
3. Instale: `py -m pip install -r requirements.txt`
4. Treine: `py train.py`
5. Abra a aplicação: `py -m streamlit run app.py`

## Resultados
O script gera as métricas reais em `reports/metricas.json`, gráficos em `reports/figures/` e o modelo treinado em `models/`. Nenhuma métrica é inventada ou pré-preenchida.

## Métricas
Acurácia, precisão macro, revocação macro, F1 macro e matriz de confusão.

## Limitações
O modelo utiliza odds pré-jogo, que representam estimativas de mercado e não informações completas sobre escalação, lesões, clima ou acontecimentos da partida. Não é garantia de resultado e não deve ser usado como recomendação de aposta.

## Uso de IA
Ferramentas de IA podem ter sido usadas como apoio à estruturação e explicação do código. O estudante deve executar, revisar e compreender os resultados antes da entrega.
