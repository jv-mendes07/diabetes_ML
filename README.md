# Previsão de Diabetes (Projeto de Classificação)
## Projeto de Machine Learning

Neste projeto de dados, utilizei um conjunto de dados relativo ao diagnóstico de diabetes de várias pacientes consultadas, e com base em tal conjunto de dados, construi um modelo de classificação para prever e classificar se uma determinada paciente têm diabetes ou não.

Tal projeto é dividido entre às fases de **(1)** tratamento de dados, **(2)** análise exploratória de dados, **(3)** preparação e treino do modelo de machine learning e **(4)** ajuste de hiperparâmetros.

Utilizei Pandas e Numpy para manipulação de dados, Seaborn e Matplotlib para visualização de dados, e Sklearn para implementação, treinamento de modelos, avaliação de métricas e ajuste de hiperparâmetros.

Após a importação do dataset, verifiquei o formato de linhas e colunas do conjunto de dados:

* 768 linhas
* 9 colunas

Em seguida, visualizei às primeiras cinco linhas da tabela:

|   | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI  | DiabetesPedigreeFunction | Age | Outcome |
|---|-------------|---------|---------------|---------------|---------|------|--------------------------|-----|---------|
| 0 | 6           | 148     | 72            | 35            | 0       | 33.6 | 0.627                    | 50  | 1       |
| 1 | 1           | 85      | 66            | 29            | 0       | 26.6 | 0.351                    | 31  | 0       |
| 2 | 8           | 183     | 64            | 0             | 0       | 23.3 | 0.672                    | 32  | 1       |
| 3 | 1           | 89      | 66            | 23            | 94      | 28.1 | 0.167                    | 21  | 0       |
| 4 | 0           | 137     | 40            | 35            | 168     | 43.1 | 2.288                    | 33  | 1       |

Concluída a importação do dataset, comecei a limpar e tratar o conjunto de dados, como uma fase inicial para às próximas etapas do projeto:

## Tratamento de dados:

Na fase de tratamento de dados, realizei:

* **Reformatação textual do nome das colunas:**

Construi um list-compreehension para converter todos os nomes das colunas do dataset para minúsculo, desse modo quando fosse me referir ao nome das colunas, não teria que escrever com letra maiúscula a primeira letra de cada coluna:

```
# Formatação textual do nome das colunas, para que todas colunas estejam em minúsculo:

df.columns = [x.lower() for x in df.columns]
```
Saída com o nome das colunas transformado:

```
['pregnancies', 'glucose', 'bloodpressure', 'skinthickness', 'insulin',
       'bmi', 'diabetespedigreefunction', 'age', 'outcome'] 
 ```
* **Tratamento de dados nulos:**

Usei o método .isna().sum() para verificar a quantidade de dados nulos em cada coluna no conjunto de dados e obtive:

```
pregnancies                 0
glucose                     0
bloodpressure               0
skinthickness               0
insulin                     0
bmi                         0
diabetespedigreefunction    0
age                         0
outcome                     0
dtype: int64
```

Ou seja, não havia nenhum dado ausente registrado no conjunto de dados.

Após concluir os principais passos da limpeza de dados, decidi começar o processo de análise exploratória dos dados, para extrair informações e insights importantes sobre o conjunto de dados em questão:

## Análise Exploratória De Dados (EDA):

Antes de inicializar a etapa de análise exploratória, é imprescindível ter um dicionário de dados que explique rapidamente o quê cada coluna informa no conjunto de dados que será analisado:

#### Dicionário de Dados

* **pregnancies** - Número de gravidezes
* **glucose** - Nível de glicose no sangue
* **bloodpressure** - Medição da pressão sanguínea
* **skinthickness** - Espessura da pele
* **insulin** - Nível de insulina no sangue
* **bmi** - Indicador de massa corporal
* **diabetespedigreefunction** - Porcentagem de diabetes no sangue
* **age** - Idade
* **outcome** - Resultado de diabetes (1 - Sim; 0 = Não)

Tendo um dicionário de dados disponível, inicializei tal exploração analítica com uma pergunta básica:

#### (1) Qual foi a quantidade de pacientes diagnosticados com diabetes ou não?

Basicamente, após manipular os dados, obtive como resposta que 65 % (500 pacientes) das pacientes não foram diagnosticadas com diabetes, enquanto às demais 34 % (268) das pacientes receberam diagnóstico confirmativo de diabetes.

Plotei um gráfico de colunas para expor visualmente a quantidade de pacientes não diagnosticadas com diabetes em comparação com às pacientes diagnosticadas:

![](./img/graf_01.png)

Com tal informação inicial, comecei a analisar cada variável separadamente do dataset, para analisar como cada variável impacta, influencia ou é estatisticamente correlacionada com à variável target, isto é, com a variável que informa se a paciente têm diabetes ou não.

#### (2) A quantidade de gravidezes de uma paciente é correlacionada com o fato da paciente ter diabetes ou não?

Com o método .unique() é verificável que há pacientes sem nenhuma gravidez até pacientes com 17 gravidezes registradas, porém majoritariamente há mais pacientes que tiveram somente uma gravidez até então.

Ou com dados mais detalhados, 55 % das pacientes consultadas têm de nenhuma até 3 gravidezes registradas (0-3), enquanto às de mais 45 % das pacientes têm mais de 4 gravidezes.

Com esses dados, decidi ter em tabelas separadas, a quantidade de pacientes com ou sem diabetes agrupada pela quantidade de gravidezes tidas:

| outcome     | 0   |
|-------------|-----|
| pregnancies |     |
| 1           | 106 |
| 2           | 84  |
| 0           | 73  |
| 3           | 48  |
| 4           | 45  |
| 5           | 36  |
| 6           | 34  |
| 7           | 20  |
| 8           | 16  |
| 10          | 14  |
| 9           | 10  |
| 12          | 5   |
| 13          | 5   |
| 11          | 4   |
| 14          | 0   |
| 15          | 0   |
| 17          | 0   |

É vísivel que em suma maioria, às pacientes sem nenhuma gravidez e às que tiveram ao menos duas gravidezes são às pacientes que receberam menos diagnósticos confirmativos de diabetes.

| outcome     | 1  |
|-------------|----|
| pregnancies |    |
| 0           | 38 |
| 1           | 29 |
| 3           | 27 |
| 7           | 25 |
| 4           | 23 |
| 8           | 22 |
| 5           | 21 |
| 2           | 19 |
| 9           | 18 |
| 6           | 16 |
| 10          | 10 |
| 11          | 7  |
| 13          | 5  |
| 12          | 4  |
| 14          | 2  |
| 15          | 1  |
| 17          | 1  |

Interessantemente, às pacientes que mais receberam diagnósticos de diabetes foram às pacientes que não tiveram nenhuma gravidez até então.

E como é sabido que mulheres com diabetes tendem à ter mais dificuldade para engravidar, então é esperado ou suposto parcialmente que haja mais incidência de diabetes em mulheres sem nenhuma gravidez (constando que essas apresentam mais dificuldade para engravidarem).

