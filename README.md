# Invest IA: Análise Avançada de Ações com Previsão de Preços usando Machine Learning

> Turma: B | Curso: CIÊNCIA DA COMPUTAÇÃO | Período: NOTURNO | Ano: 2025

## Equipe e Papéis
| Integrante | RA | Papel principal | Principais entregas (commits/arquivos) |
|------------|----|------------------|----------------------------------------|
| Simone Gomes Marque | 2225106336 | Apresentação | Vídeo, roteiro, slides |
| Henriq Bernard Oliveira Novais | 2225108340 | Engenharia de Dados / Modelagem / Documentação | MultipleFiles/data_preprocessing.py, MultipleFiles/model_training.py, README.md |
| Radamés Marcellino Ferreira | 2225100831 | Engenharia de Dados / Modelagem | MultipleFiles/data_preprocessing.py, MultipleFiles/model_training.py |
| Guilherme Araújo do Carmo | 2225102897 | Engenharia de Dados / Avaliação & Gráficos / Edição | MultipleFiles/InvestIA.py (seção de gráficos), reports/figures/* |

---

## 1. Problema
O mercado financeiro é volátil e complexo, tornando a tomada de decisão de investimento um desafio para muitos. Investidores buscam ferramentas que possam auxiliar na previsão de movimentos de preços de ações para otimizar seus retornos e minimizar riscos. A dor do usuário reside na dificuldade de interpretar grandes volumes de dados históricos e indicadores técnicos para fazer previsões assertivas. O objetivo deste projeto é desenvolver uma aplicação que utilize Machine Learning para prever o preço futuro de ações, fornecendo insights claros e acionáveis. O sistema funcionará se for capaz de prever o preço das ações com uma margem de erro aceitável, auxiliando na decisão de compra ou venda.

**Métrica(s) alvo:**
*   **Principal:** \$RMSE\$ (Root Mean Squared Error) - Mede a magnitude média dos erros.
*   **Secundária:** \$MAE\$ (Mean Absolute Error) - Mede a magnitude média dos erros sem considerar a direção.

---

## 2. Abordagem de IA
*   **Tipo de IA/ML**: Regressão com modelos de Ensemble (Random Forest Regressor, Gradient Boosting Regressor) e modelos lineares (Linear Regression, Ridge, Lasso).
*   **Justificativa técnica**: Modelos de regressão são adequados para prever valores contínuos como preços de ações. Modelos de ensemble, como Random Forest e Gradient Boosting, são robustos, lidam bem com dados ruidosos e são capazes de capturar relações não-lineares complexas nos dados financeiros. Modelos lineares são incluídos para comparação de desempenho e como baseline. A utilização de TimeSeriesSplit para validação cruzada garante que a ordem temporal dos dados seja respeitada, o que é crucial para séries temporais.
*   **Semente aleatória**: `42`

---

## 3. Dados
*   **Origem**: Os dados são obtidos em tempo real através da biblioteca `yfinance`, que acessa dados históricos de ações do Yahoo Finance.
    *   Link: [https://pypi.org/project/yfinance/](https://pypi.org/project/yfinance/)
    *   Licença: BSD 3-Clause License (implícita pelo uso da biblioteca yfinance, que é de código aberto).
*   **Esquema**:
    *   `Date` (datetime): Data da cotação.
    *   `Open` (float): Preço de abertura.
    *   `High` (float): Preço máximo.
    *   `Low` (float): Preço mínimo.
    *   `Close` (float): Preço de fechamento.
    *   `Volume` (int): Volume de negociações.
    *   `Dividends` (float): Dividendos pagos.
    *   `Stock Splits` (float): Desdobramentos de ações.
    *   `SMA_5` (float): Média Móvel Simples de 5 dias.
    *   `SMA_20` (float): Média Móvel Simples de 20 dias.
    *   `SMA_50` (float): Média Móvel Simples de 50 dias.
    *   `RSI` (float): Índice de Força Relativa.
    *   `MACD` (float): Convergência e Divergência de Médias Móveis.
    *   `MACD_Signal` (float): Linha de sinal do MACD.
    *   `MACD_Histogram` (float): Histograma do MACD.
    *   `BB_Middle` (float): Banda de Bollinger Média.
    *   `BB_Upper` (float): Banda de Bollinger Superior.
    *   `BB_Lower` (float): Banda de Bollinger Inferior.
    *   `Daily_Return` (float): Retorno diário percentual.
    *   `Volatility` (float): Volatilidade (desvio padrão do retorno diário).
    *   `Volume_MA` (float): Média Móvel do Volume de 20 dias.
    *   `Target` (float): Preço de fechamento futuro (variável alvo).
*   **Cuidados éticos/privacidade**: Os dados são públicos e não contêm informações pessoais, portanto, não há preocupações éticas ou de privacidade diretas.
*   **Tamanho aproximado**: Varia conforme o período selecionado (ex: 1 ano, 5 anos, max). Para 5 anos de dados diários, aproximadamente \$1250\$ linhas \$ \times \$ \$22\$ colunas.

---

## 4. Estrutura do Projeto
PROJ_IA_2025_TurmaB_InvestIA/
├─ README.md
├─ requirements.txt
├─ MultipleFiles/
│  ├─ InvestIA.py (Aplicação Streamlit principal)
│  ├─ data_preprocessing.py (Funções de pré-processamento e cálculo de indicadores)
│  ├─ model_training.py (Funções de treinamento e previsão do modelo)
│  ├─ email_utils.py (Funções para envio de e-mail)
│  ├─ .env.example
├─ .gitignore
├─ notebooks/ (Não utilizado neste projeto, mas pode conter explorações futuras)
├─ data/ (Não utilizado para armazenamento de dados brutos, pois são obtidos em tempo real)
├─ models/ (Não utilizado para armazenamento de modelos persistidos, pois são treinados na execução)
├─ reports/ (Não utilizado para figuras/tabelas estáticas, pois são geradas dinamicamente no Streamlit)
├─ tests/ (Não implementado para esta entrega)
└─ docs/ (Não implementado para esta entrega)

---

## 5. Como Reproduzir

### 5.1 Ambiente
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 Variáveis de Ambiente
Crie um arquivo `.env` na raiz do projeto (baseado em `.env.example`) e preencha com suas credenciais de e-mail, caso deseje utilizar a funcionalidade de alertas:
```
GMAIL_USER="seu_email@gmail.com"
GMAIL_APP_PASSWORD="sua_senha_de_aplicativo_gmail"
```
**Importante:** Para Gmail, é necessário gerar uma "senha de aplicativo" se você tiver a verificação em duas etapas ativada.

### 5.3 Execução da Aplicação
Para iniciar a aplicação Streamlit:
```bash
streamlit run MultipleFiles/InvestIA.py
```
A aplicação será aberta em seu navegador padrão.

## 6. Resultados

**Métricas de Desempenho (Exemplo para Random Forest):**

| Métrica | Valor Médio |
|---------|-------------|
| MAE     | R\$ 2.50    |
| RMSE    | R\$ 3.20    |
| R²      | 0.85        |

*Nota: Os valores acima são exemplos. As métricas reais variam conforme a ação, período e modelo selecionado.*

**Gráficos:**

*   **Preços e Médias Móveis:**
    *   Este gráfico exibe a série histórica do preço de fechamento da ação juntamente com as Médias Móveis Simples de 20 e 50 dias. Permite visualizar tendências de curto e médio prazo, bem como pontos de cruzamento que podem indicar sinais de compra ou venda.
*   **Indicadores Técnicos (RSI):**
    *   Mostra o Índice de Força Relativa (RSI), um oscilador de momentum que mede a velocidade e a mudança dos movimentos de preços. As linhas horizontais em 30 e 70 indicam regiões de sobrevenda e sobrecompra, respectivamente.
*   **Comparação: Previsões vs Valores Reais:**
    *   Este gráfico compara os valores de preço reais com as previsões geradas pelo modelo para o último "fold" da validação cruzada. Ajuda a avaliar visualmente a acurácia do modelo e identificar desvios significativos.

## 7. Decisões Técnicas

*   **Pré-processamento**:
    *   **Nulos**: Linhas com valores nulos (resultantes do cálculo de indicadores técnicos) são removidas usando `dropna()`.
    *   **Normalização**: Os dados de entrada (\$X\$) são padronizados usando `StandardScaler` antes do treinamento do modelo. Isso garante que as diferentes escalas das features não influenciem indevidamente o modelo.
    *   **Features**: Além do preço de fechamento, foram incluídos diversos indicadores técnicos como SMA (5, 20, 50), RSI, MACD (e seus componentes), Bandas de Bollinger (superior e inferior), Retorno Diário, Volatilidade e Média Móvel do Volume. A seleção dessas features visa capturar diferentes aspectos do comportamento do preço e do volume.
*   **Arquitetura/Hiperparâmetros**:
    *   **Modelos**: Random Forest Regressor (com `n_estimators=100`), Gradient Boosting Regressor, Linear Regression, Ridge e Lasso. A escolha de múltiplos modelos permite ao usuário comparar o desempenho e selecionar o mais adequado para a ação e período analisados.
    *   **Validação Cruzada**: `TimeSeriesSplit` com `n_splits=5` é utilizada para garantir que o modelo seja avaliado em dados futuros, simulando um cenário de uso real e evitando vazamento de dados.
*   **Limitações conhecidas e possíveis melhorias**:
    *   **Dados Históricos**: O modelo é treinado apenas com dados históricos e indicadores técnicos, não incorporando notícias, eventos macroeconômicos ou análise fundamentalista.
    *   **Previsão de Curto Prazo**: A previsão é focada em um horizonte de poucos dias (\$1\$ a \$30\$ dias), e a acurácia pode diminuir significativamente para previsões de longo prazo.
    *   **Modelos Mais Complexos**: Poderiam ser explorados modelos de Deep Learning (LSTMs, GRUs) para séries temporais, que podem capturar padrões sequenciais mais complexos.
    *   **Otimização de Hiperparâmetros**: A otimização dos hiperparâmetros dos modelos (ex: usando GridSearchCV ou RandomizedSearchCV) poderia melhorar o desempenho.
    *   **Análise de Sentimento**: A inclusão de análise de sentimento de notícias e redes sociais poderia enriquecer as features do modelo.

## 8. Execução do Vídeo (YouTube — não listado)

Link: (A ser inserido)

Roteiro seguido: problema → dados → IA → execução ao vivo → resultados → conclusão.

## 9. Créditos e Licença

*   **Fontes de dados/código de terceiros**:
    *   `yfinance`: [https://pypi.org/project/yfinance/](https://pypi.org/project/yfinance/)
    *   `Streamlit`: [https://streamlit.io/](https://streamlit.io/)
    *   `Plotly`: [https://plotly.com/](https://plotly.com/)
    *   `scikit-learn`: [https://scikit-learn.org/](https://scikit-learn.org/)
*   **Licença escolhida**: MIT License
*   **Autores**:
    *   Simone Gomes Marque (RA: 2225106336)
    *   Henric Bernard Oliveira Novais (RA: 2225108340)
    *   Radamés Marcellino Ferreira (RA: 2225100831)
    *   Guilherme Araújo do Carmo (RA: 2225102897)

## 10. Changelog (opcional)

*   **v1.0** — Entrega final do projeto, incluindo aplicação Streamlit, modelos de ML, cálculo de indicadores e funcionalidade de alerta por e-mail.
*   **v0.9** — Implementação dos modelos de Machine Learning e cálculo de métricas de avaliação.
*   **v0.8** — Desenvolvimento das funções de pré-processamento de dados e cálculo de indicadores técnicos.
*   **v0.7** — Estruturação inicial do projeto e integração com `yfinance`.
