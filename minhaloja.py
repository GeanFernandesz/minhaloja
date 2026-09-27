"""MinhaLoja v4 - Sistema Completo com PDV, OOP, Banco de Dados e Machine Learning"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import MinMaxScaler

sns.set(style="whitegrid")

# --- 1. ORIENTAÇÃO A OBJETOS ---
class Produto:
    def __init__(self, nome, preco, categoria, estoque):
        self.nome = nome
        self.preco = preco
        self.categoria = categoria
        self.estoque = estoque

    def __str__(self):
        return (f"{self.nome} | R$ {self.preco:.2f} | "
                f"Categoria: {self.categoria} | Estoque: {self.estoque}")

    def aplicar_desconto(self, percentual):
        assert 0 <= percentual <= 100, "❌ O desconto deve ser entre 0 e 100%!"
        self.preco = self.preco - (self.preco * (percentual / 100))

# --- 2. FUNÇÕES E PDV ---
def calcular_comissao(valor_venda, percentual):
    assert percentual is not None, "❌ O percentual não pode ser None!"
    assert 0 <= percentual <= 20, "❌ O percentual deve estar entre 0 e 20!"
    return valor_venda * (percentual / 100)

def calcular_desconto(valor, percentual):
    if percentual < 0 or percentual > 100:
        return None
    return valor - (valor * (percentual / 100))

def registrar_venda(produto, valor_final):
    print(f">> Venda registrada: {produto} por R$ {valor_final:.2f}")

arredondar = lambda v: round(v, 2)

# Catálogo inicial de produtos
catalogo = [
    Produto("Notebook", 3500.00, "Informática", 12),
    Produto("Smartphone", 2200.00, "Informática", 18),
    Produto("Cafeteira", 350.00, "Eletrodoméstico", 7),
    Produto("Geladeira", 2900.00, "Eletrodoméstico", 4),
    Produto("Fone Bluetooth", 180.00, "Acessório", 30),
]

# --- 3. BANCO DE DADOS E PANDAS ---
conn = sqlite3.connect(":memory:")
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS vendas(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data TEXT,
        produto TEXT,
        categoria TEXT,
        quantidade INTEGER,
        valor_unitario REAL
    )
""")

vendas_exemplo = [
    ('2024-01-15', 'Notebook',       'Informática',     1, 3500.00),
    ('2024-01-20', 'Cafeteira',      'Eletrodoméstico', 2,  350.00),
    ('2024-01-30', 'Fone Bluetooth', 'Acessório',       4,  180.00),
    ('2024-02-05', 'Smartphone',     'Informática',     1, 2200.00),
    ('2024-02-18', 'Fone Bluetooth', 'Acessório',       3,  180.00),
    ('2024-02-25', 'Notebook',       'Informática',     2, 3500.00),
    ('2024-03-10', 'Geladeira',      'Eletrodoméstico', 1, 2900.00),
    ('2024-03-22', 'Fone Bluetooth', 'Acessório',       5,  180.00),
    ('2024-03-28', 'Cafeteira',      'Eletrodoméstico', 1,  350.00),
    ('2024-04-02', 'Smartphone',     'Informática',     2, 2200.00),
    ('2024-04-19', 'Geladeira',      'Eletrodoméstico', 1, 2900.00),
    ('2024-04-25', 'Notebook',       'Informática',     1, 3500.00),
    ('2024-05-03', 'Cafeteira',      'Eletrodoméstico', 3,  350.00),
    ('2024-05-15', 'Smartphone',     'Informática',     2, 2200.00),
    ('2024-05-28', 'Fone Bluetooth', 'Acessório',       6,  180.00),
    ('2024-06-10', 'Geladeira',      'Eletrodoméstico', 2, 2900.00),
    ('2024-06-20', 'Notebook',       'Informática',     3, 3500.00),
    ('2024-06-30', 'Smartphone',     'Informática',     1, 2200.00),
]

cursor.executemany(
    "INSERT INTO vendas (data, produto, categoria, quantidade, valor_unitario) VALUES (?, ?, ?, ?, ?)",
    vendas_exemplo
)
conn.commit()

df = pd.read_sql("SELECT * FROM vendas", conn)
conn.close()

df['total'] = df['quantidade'] * df['valor_unitario']
df['data'] = pd.to_datetime(df['data'])
df['mes'] = df['data'].dt.month
df['mes_nome'] = df['data'].dt.strftime('%b')

# --- 4. MACHINE LEARNING ---
fat_mes = df.groupby('mes')['total'].sum().reset_index().sort_values('mes')
x = fat_mes[['mes']].values.astype(float)
y = fat_mes[['total']].values.astype(float)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_s = scaler_x.fit_transform(x)
y_s = scaler_y.fit_transform(y)

modelo = Sequential([
    Input(shape=(1,)),
    Dense(16, activation='relu'),
    Dense(1)
])
modelo.compile(optimizer='adam', loss='mean_squared_error')
modelo.fit(x_s, y_s, epochs=1000, verbose=0)

meses_futuros = np.array([[7], [8], [9]], dtype=float)
meses_futuros_s = scaler_x.transform(meses_futuros)
prev_s = modelo.predict(meses_futuros_s, verbose=0)
previsoes = scaler_y.inverse_transform(prev_s).ravel()

nomes_meses = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr', 5:'Mai', 6:'Jun',
               7:'Jul', 8:'Ago', 9:'Set'}

# --- 5. RELATÓRIO EXECUTIVO E DESAFIO POR CATEGORIA ---
def gerar_relatorio_executivo(df, catalogo, previsoes, meses_futuros):
    print("=" * 55)
    print("       🏪 MinhaLoja v4 — Relatório Executivo")
    print("=" * 55)

    print("\n📋 PDV — Venda e Comissão:")
    valor_final = arredondar(calcular_desconto(2200.00, 5))
    registrar_venda("Smartphone", valor_final)
    comissao = calcular_comissao(valor_final, 10)
    print(f"💰 Comissão gerada: R$ {comissao:.2f}")

    print("\n📦 Produto mais caro do catálogo:")
    mais_caro = max(catalogo, key=lambda p: p.preco)
    print(f"  {mais_caro}")

    print("\n🤖 Previsão de faturamento geral (TensorFlow):")
    for mes, prev in zip(meses_futuros, previsoes):
        print(f"  {nomes_meses[int(mes[0])]}: R$ {prev:,.2f}")

gerar_relatorio_executivo(df, catalogo, previsoes, meses_futuros)

print("\n" + "=" * 55)
print("🤖 Previsão de Julho (Mês 7) por Categoria:")
print("=" * 55)

categorias = ['Informática', 'Eletrodoméstico', 'Acessório']
for cat in categorias:
    df_filtrado = df[df['categoria'] == cat]
    fat_mes_cat = df_filtrado.groupby('mes')['total'].sum().reset_index()
    
    x_cat = fat_mes_cat[['mes']].values.astype(float)
    y_cat = fat_mes_cat[['total']].values.astype(float)
    
    x_cat_s = scaler_x.fit_transform(x_cat)
    y_cat_s = scaler_y.fit_transform(y_cat)
    
    modelo_cat = Sequential([
        Input(shape=(1,)),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    modelo_cat.compile(optimizer='adam', loss='mean_squared_error')
    modelo_cat.fit(x_cat_s, y_cat_s, epochs=500, verbose=0)
    
    mes_7 = np.array([[7]], dtype=float)
    mes_7_s = scaler_x.transform(mes_7)
    prev_s_cat = modelo_cat.predict(mes_7_s, verbose=0)
    prev_cat = scaler_y.inverse_transform(prev_s_cat).ravel()[0]
    
    print(f"  {cat}: R$ {prev_cat:,.2f}")
