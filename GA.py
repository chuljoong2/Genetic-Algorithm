import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import numpy as np

# StudentsPerformance.csv 파일 불러오기
pd.set_option('display.max_columns', None)
df = pd.read_csv('./input/StudentsPerformance.csv', encoding='utf-8')
print(df)

# 회귀식 구하기
flt = ols('writing_score ~ reading_score', data=df).fit()
regression_gradient = flt.params.reading_score
regression_intercept = flt.params.Intercept
print(regression_gradient, regression_intercept)

# 산점도 출력하기
generation = 0


def print_scatterplot(population=None):
    if population is None:
        population = []
    fig = plt.figure(figsize=(8, 8))
    fig.set_facecolor('white')

    font_size = 15
    plt.title(f"Generation: {generation}", fontsize=font_size)
    plt.scatter(df['reading_score'], df['writing_score'])
    x = np.arange(0, 100)

    # 만약 입력값이 들어온다면 후보해들 그리기
    # 없다면 for문은 실행되지 않음
    for i in range(len(population)):
        plt.plot(x, population[i][0] * x + population[i][1], color="green")

    plt.plot(x, regression_gradient * x + regression_intercept, color="red", linewidth="2.5")
    plt.xlabel('Reading Score', fontsize=font_size)
    plt.ylabel('Writing Score', fontsize=font_size)

    plt.show()


print_scatterplot()


