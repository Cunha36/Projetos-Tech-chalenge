import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Função para carregar os dados
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Data'])
    return data

# Função para gerar textos informativos
def generate_insights(mae, mse, rmse):
    insights = (
        f"A análise de previsão dos preços do petróleo nos mostra algumas métricas importantes para avaliar a precisão do modelo.\n"
        f"- O **Erro Absoluto Médio (MAE)** é de {mae:.2f}, o que indica a média dos erros em termos absolutos.\n"
        f"- O **Erro Quadrático Médio (MSE)** é de {mse:.2f}, que mede a variabilidade dos erros do modelo.\n"
        f"- A **Raiz do Erro Quadrático Médio (RMSE)** é de {rmse:.2f}, oferecendo uma interpretação mais clara da escala dos erros.\n"
        f"Essas métricas sugerem que o modelo é razoavelmente eficaz na previsão dos preços, mas sempre há espaço para melhorias."
    )
    return insights

# Carregar os dados
data = load_data("Brent.csv")

# Filtrar dados a partir de uma data específica
start_date = '2020-01-01'
filtered_data = data[data['Data'] >= start_date]
data = filtered_data

# Renomear colunas para atender aos requisitos do Prophet
data = data.rename(columns={'Data': 'ds', 'Soma de Preço - petróleo bruto - Brent (FOB)': 'y'})

# Limpar a coluna 'y' para remover caracteres não numéricos
data['y'] = data['y'].str.replace('$', '').str.replace('.', '').str.replace(',', '.')
data['y'] = pd.to_numeric(data['y'], errors='coerce')

# Visualizar a média anual dos preços do petróleo
data['year'] = data['ds'].dt.year
annual_mean = data.groupby('year')['y'].mean().reset_index()

# Dividir os dados em treino e teste (80% treino, 20% teste)
train_size = int(len(data) * 0.8)
train = data[:train_size]
test = data[train_size:]

# Treinar o modelo Prophet
model = Prophet(daily_seasonality=True)
model.fit(train)

# Fazer previsões para o período de teste
future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

# Filtrar previsões para o período de teste
forecast_test = forecast[-len(test):]

# Calcular métricas de desempenho
y_true = test['y'].values
y_pred = forecast_test['yhat'].values

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

# Interface Streamlit
st.title('Previsão do Preço do Petróleo')

# Texto informativo gerado pela IA
st.write(generate_insights(mae, mse, rmse))

# Slider para selecionar intervalo de anos
year_range = st.slider('Selecione o intervalo de anos', int(data['year'].min()), int(data['year'].max()), (int(data['year'].min()), int(data['year'].max())))

# Filtrar os dados com base no intervalo de anos selecionado
filtered_annual_mean = annual_mean[(annual_mean['year'] >= year_range[0]) & (annual_mean['year'] <= year_range[1])]
filtered_train = train[(train['year'] >= year_range[0]) & (train['year'] <= year_range[1])]
filtered_test = test[(test['year'] >= year_range[0]) & (test['year'] <= year_range[1])]
filtered_forecast_test = forecast_test[(forecast_test['ds'].dt.year >= year_range[0]) & (forecast_test['ds'].dt.year <= year_range[1])]

# Gráfico 1: Média Anual do Preço do Petróleo
st.write('Média Anual do Preço do Petróleo (Últimos 10 Anos)')
fig1 = px.line(filtered_annual_mean, x='year', y='y', title='Média Anual do Preço do Petróleo (Últimos 10 Anos)')
fig1.update_layout(width=1000, height=500)
st.plotly_chart(fig1)

st.write(f'Tamanho do conjunto de treino: {len(train)}')
st.write(f'Tamanho do conjunto de teste: {len(test)}')

# Gráfico 2: Previsões do Preço do Petróleo
st.write('Previsões do Preço do Petróleo')
fig2 = model.plot(forecast)
st.pyplot(fig2)

st.write(f'MAE: {mae:.2f}')
st.write(f'MSE: {mse:.2f}')
st.write(f'RMSE: {rmse:.2f}')

# Gráfico 3: Previsão vs. Valores Reais
st.write('Previsão vs. Valores Reais')
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=filtered_train['ds'], y=filtered_train['y'], mode='lines', name='Treino'))
fig3.add_trace(go.Scatter(x=filtered_test['ds'], y=filtered_test['y'], mode='lines', name='Teste'))
fig3.add_trace(go.Scatter(x=filtered_test['ds'], y=filtered_forecast_test['yhat'], mode='lines', name='Previsão', line=dict(color='red')))
fig3.update_layout(title='Previsão do Preço Diário do Petróleo', xaxis_title='Data', yaxis_title='Preço', width=1000, height=500)
st.plotly_chart(fig3)
