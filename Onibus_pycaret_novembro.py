# Importando bibliotecas necessárias
import pandas as pd
from pycaret.time_series import TSForecastingExperiment
import streamlit as st

# Configuração da página do Streamlit
st.set_page_config(
    page_title='Forecast Licenciamentos Ônibus',
    layout='wide',
    initial_sidebar_state='auto'
)

# Layout principal
col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.header('***Forecast Licenciamentos Ônibus por meio do PyCaret***', divider='green')
    st.markdown(
        '''
        O segmento de licenciamentos de ônibus é uma parte importante do setor automotivo no Brasil, monitorado pela 
        Anfavea (Associação Nacional dos Fabricantes de Veículos Automotores). A Anfavea coleta e divulga dados 
        estatísticos sobre a produção, licenciamento e exportação de veículos, incluindo ônibus. Esses dados são 
        essenciais para entender o desempenho do mercado, identificar tendências e planejar estratégias de negócios.

        Em 2022, a produção e exportação de veículos superaram as previsões, enquanto as vendas se mantiveram estáveis. 
        Para 2023, a Anfavea projeta um aumento de 3% nos licenciamentos e de 2,2% na produção.

        O módulo PyCaret Time Series é uma ferramenta avançada para analisar e prever dados de séries temporais usando aprendizado de máquina e técnicas estatísticas clássicas. Esse módulo permite que os usuários executem facilmente tarefas complexas de previsão de séries temporais, automatizando todo o processo, desde a preparação dos dados até a implantação do modelo. 

O módulo PyCaret Time Series Forecasting oferece suporte a uma ampla gama de métodos de previsão, como ARIMA, Prophet e LSTM. Ele também oferece vários recursos para lidar com valores ausentes, decomposição de séries temporais e visualizações de dados. 

        '''
    )

with col2:
    st.image(
        'https://www.eduardopaes.com.br/wp-content/uploads/2023/03/Transcarioca-7-scaled.jpg',
        use_container_width=True
    )

# Carregando o arquivo Excel local
try:
    data = pd.read_excel(r"C:\Tablets\Onibus_1990.xlsx")
    data['Mês'] = pd.to_datetime(data['Mês'])  # Ajustar o nome da coluna de data, se necessário
    data.set_index('Mês', inplace=True)  # Definir a coluna de data como índice
    st.success("Base de dados carregada com sucesso.")
except Exception as e:
    st.error(f"Erro ao carregar a base de dados: {e}")
    st.stop()

# Visualizar a base de dados no Streamlit
col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.write('***Base de Dados Anfavea***')
    st.dataframe(data, height=500)


with col2:
    # Configuração inicial do experimento
    s = TSForecastingExperiment()
    s.setup(data=data, target='Onibus', session_id=123)
    st.write("**Configuração inicial do PyCaret concluída.**")

    # Comparar modelos
    best = s.compare_models()

    # Obter a tabela de comparação
    comparison_df = s.pull()
    st.write("### Comparação de Modelos")
    st.dataframe(comparison_df)

    # Botão para download da tabela de comparação
    csv = comparison_df.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar Comparação", data=csv, file_name="model_comparison.csv", mime='text/csv')

col1, col2= st.columns([1, 1], gap='large')

with col1:
    # Plotar previsões
    st.write("**Previsão com horizonte de 36 períodos:**")
    forecast_plot = s.plot_model(best, plot='forecast', data_kwargs={'fh': 36})
    st.image(r"C:\Tablets\onibus.png", use_container_width=True)

with col2:
    # Finalizar o modelo
    final_best = s.finalize_model(best)
    st.write("**Modelo finalizado:**")
    st.write(final_best)

col1, col2=st.columns([1,1], gap='large')

with col1:

    # Realizar previsões
    predictions = s.predict_model(final_best, fh=36)
    st.write("**Previsões:**")
    st.dataframe(predictions, height=800)
# Botão para download da tabela de comparação
    csv = predictions.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar Previsão", data=csv, file_name="predictions.csv", mime='text/csv')
    
with col2:   
    st.markdown('''1. **LGBMRegressor**: É um modelo de regressão baseado em LightGBM (Light Gradient Boosting Machine). LightGBM é uma biblioteca de aprendizado de máquina que usa algoritmos de boosting baseados em árvores de decisão. É conhecido por sua eficiência e velocidade, especialmente em grandes conjuntos de dados.

2. **n_jobs=-1**: Este parâmetro define o número de threads a serem usadas para o treinamento do modelo. Quando definido como `-1`, ele utiliza todos os núcleos disponíveis do processador, acelerando o processo de treinamento.

3. **random_state=123**: Este parâmetro é usado para garantir a reprodutibilidade dos resultados. Ao definir um valor específico para `random_state`, você garante que o modelo produza os mesmos resultados toda vez que for treinado com os mesmos dados.

4. **sp=12**: Este parâmetro define a periodicidade sazonal dos dados. No contexto de séries temporais, `sp=12` indica uma sazonalidade anual (mensal), ou seja, os dados têm um padrão que se repete a cada 12 períodos (meses).

5. **window_length=12**: Este parâmetro define o comprimento da janela de tempo usada para criar as features de lag. Aqui, é definido como 12, indicando que o modelo usará os 12 períodos anteriores (meses) para fazer previsões.
''')
