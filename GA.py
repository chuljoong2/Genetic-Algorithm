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


#  평균 제곱 오차 구하기
def get_mean_squared_errores(p):
    mean_squared_errores = []
    for i in range(population_length):
        w = p[i][0]  # 기울기
        b = p[i][1]  # y 절편

        sum_squared_error = 0
        for score in scores:
            rs, ws = score[0], score[1]  # reading_score, writing_score

            predicted_ws = w * rs + b
            squared_error = (ws - predicted_ws) ** 2  # (실제값 - 예측값)의 제곱
            sum_squared_error += squared_error

        mean_squared_error = sum_squared_error / total_scores_length  # 평균 제곱 오차
        mean_squared_errores.append(mean_squared_error)

    return mean_squared_errores


# 적합도 구하기
def get_fittnesses(msnes):
    fittnesses = []
    sum_msnes = sum(msnes)

    # 평균 제곱 오차가 낮을 수록 적합도가 높기 때문에
    # (1 - 평균 제곱 오차 비율)을 통해서 평균 제곱 오차가 낮을 수록 적합도를 높게 설정
    for msn in msnes:
        fittnesses.append(1-msn/sum_msnes)

    return fittnesses


# 룰렛 휠에 사용될 누적 확률 리스트 만들기
def get_percentages(fits):
    percenatages = []
    sum_fits = sum(fits)
    for fit in fits:
        if len(percenatages) == 0:
            percenatages.append(fit / sum_fits)  # idx == 0, 누적할 확률이 없음
        else:
            percenatages.append(percenatages[-1] + fit / sum_fits)  # idx != 0, 이전 확률과 더해서 저장

    return percenatages


# 선택 연산
def selection(p):
    selected_p = []
    mean_squared_errores = get_mean_squared_errores(p)
    fittnesses = get_fittnesses(mean_squared_errores)
    percenatages = get_percentages(fittnesses)

    # 룰렛 휠 방식
    for _ in range(population_length):
        random_percentage = random.random()
        for i in range(population_length):
            if random_percentage < percenatages[i]:
                selected_p.append(p[i])
                break

    return selected_p


# 교차 연산
def crossover(p):
    crossover_p = []
    p.sort()  # 같은 직선끼리 교차할 확률을 줄이기 위함
    for i in range(population_length//2):
        crossover_p.append([(p[i][0] + p[i + population_length // 2][0])/2,
                            (p[i][1] + p[i + population_length // 2][1])/2])
        crossover_p.append([(p[i][0] + p[-1-i][0])/2, (p[i][1] + p[-1-i][1])/2])

    return crossover_p


# 돌연변이 연산
def mutation(p):
    mutated_p = []
    for i in range(population_length):
        random_percentage = random.random()
        # 돌연변이가 일어날 확률 1%
        if random_percentage > 0.975:
            mutated_p.append([p[i][0], p[i][1]-0.5])
        elif random_percentage > 0.95:
            mutated_p.append([p[i][0], p[i][1]+0.5])
        elif random_percentage > 0.925:
            mutated_p.append([p[i][0]-0.1, p[i][1]])
        elif random_percentage > 0.9:
            mutated_p.append([p[i][0]+0.1, p[i][1]])
        else:
            mutated_p.append([p[i][0], p[i][1]])

    return mutated_p


print_scatterplot()

#  점수 데이터를 리스트에 저장
scores = []
for row in df.index:
    reading_score = df['reading_score'][row]
    writing_score = df['writing_score'][row]
    scores.append([reading_score, writing_score])

total_scores_length = len(scores)


population = init_population()
print_scatterplot(population)

while generation < 100:
    generation += 1
    selected_population = selection(population)
    crossover_population = crossover(selected_population)
    mutated_population = mutation(crossover_population)
    population = mutated_population
    print_scatterplot(population)
