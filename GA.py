import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import numpy as np
import random

# StudentsPerformance.csv 파일 불러 오기
pd.set_option('display.max_columns', None)
df = pd.read_csv('./input/StudentsPerformance.csv', encoding='utf-8')
print(df)

# 회귀식 구하기
flt = ols('writing_score ~ reading_score', data=df).fit()
regression_gradient = flt.params.reading_score
regression_intercept = flt.params.Intercept
print(regression_gradient, regression_intercept)

# 산점도 출력
generation = 0  # 현재 세대 번호


def print_scatterplot(p=None):
    if p is None:
        p = []

    fig = plt.figure(figsize=(8, 8))
    fig.set_facecolor('white')

    font_size = 15
    plt.title(f"Generation: {generation}", fontsize=font_size)
    plt.scatter(df['reading_score'], df['writing_score'])
    x = np.arange(0, 100)

    # 만약 입력 값이 들어 온다면 후보해 들 그리기
    # 없다면 for 문은 실행 되지 않음
    for i in range(len(p)):
        plt.plot(x, p[i][0] * x + p[i][1])

    plt.plot(x, regression_gradient * x + regression_intercept, color="red", linewidth="2.5")
    plt.xlabel('Reading Score', fontsize=font_size)
    plt.ylabel('Writing Score', fontsize=font_size)
    plt.xlim([0, 100])
    plt.ylim([0, 100])

    plt.show()


print_scatterplot()

#  점수 데이터를 리스트에 저장
scores = []
for row in df.index:
    reading_score = df['reading_score'][row]
    writing_score = df['writing_score'][row]
    scores.append([reading_score, writing_score])

total_scores_length = len(scores)
print(total_scores_length)
print(scores)


#  초기 세대 생성
population_length = 30  # 모집단의 수


def init_population():
    p = []
    while len(p) < population_length:
        # 임의의 점 두개 선택
        a = random.randint(0, total_scores_length - 1)
        b = random.randint(0, total_scores_length - 1)

        first_x, first_y = scores[a][0], scores[a][1]
        second_x, second_y = scores[b][0], scores[b][1]

        # 기울기가 0일 때 예외 처리와 ZeroDivisionError 에러 처리
        if second_y - first_y == 0 or second_x - first_x == 0:
            continue

        gradient = (second_y - first_y) / (second_x - first_x)  # 기울기
        intercept = first_y - gradient * first_x  # y 절편

        p.append([gradient, intercept])

    return p


population = init_population()
print_scatterplot(population)
print(population)
