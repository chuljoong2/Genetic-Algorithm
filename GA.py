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
    for m in range(population_length):
        w = p[m][0]  # 기울기
        b = p[m][1]  # y 절편

        sse = 0
        for reading, writing in scores:
            p_ws = w * reading + b
            se = (writing - p_ws) ** 2  # (실제값 - 예측값)의 제곱
            sse += se

        mean_squared_error = sse / total_scores_length  # 평균 제곱 오차
        mean_squared_errores.append(mean_squared_error)

    return mean_squared_errores


# 적합도 구하기
def get_fittnesses(mean_squared_errores):
    fittnesses = []
    sum_mean_squared_errores = sum(mean_squared_errores)

    # 평균 제곱 오차가 낮을 수록 적합도가 높기 때문에
    # (1 - 평균 제곱 오차 비율)을 통해서 평균 제곱 오차가 낮을 수록 적합도를 높게 설정
    for mse in mean_squared_errores:
        fittnesses.append(sum_mean_squared_errores/mse)

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
        for k in range(population_length):
            if random_percentage < percenatages[k]:
                selected_p.append(p[k])
                break

    return selected_p


# 교차 연산
def crossover(p):
    crossover_p = []
    p.sort()  # 같은 직선끼리 교차할 확률을 줄이기 위함
    for i in range(population_length // 2):
        crossover_p.append([(p[i][0] + p[i + population_length // 2][0]) / 2,
                            (p[i][1] + p[i + population_length // 2][1]) / 2])
        crossover_p.append([(p[i][0] + p[-1 - i][0]) / 2, (p[i][1] + p[-1 - i][1]) / 2])

    return crossover_p


# 돌연변이 연산
def mutation(p):
    mutated_p = []
    for l in range(population_length):
        random_percentage = random.random()
        # 돌연변이가 일어날 확률 5%
        if random_percentage > 0.5:
            mutated_p.append([p[l][0]+0.1, p[l][1]-0.7])
        elif random_percentage > 0.0:
            mutated_p.append([p[l][0]-0.1, p[l][1]+0.7])
        else:
            mutated_p.append([p[l][0], p[l][1]])

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

result = []
est_gradient, est_intercept = 0, 0
while generation < 10:
    generation += 1
    est_mses = get_mean_squared_errores(population)

    # 평균 제곱 오차의 평균
    est_sum_mses = sum(est_mses)
    est_avg_mses = est_sum_mses / population_length
    result.append([generation, est_avg_mses])

    selected_population = selection(population)
    crossover_population = crossover(selected_population)
    mutated_population = mutation(crossover_population)
    population = mutated_population

    temp = []
    for t in range(population_length):
        temp.append([est_mses[t], t])

    # 마지막 세대의 최적해가 담김 (최종 추정된 회귀식)
    est_gradient = population[min(temp)[1]][0]
    est_intercept = population[min(temp)[1]][1]

    print_scatterplot(population)

#  라이브러리로 찿은 회귀식
opt_sum_sse, est_sum_sse = 0, 0
for score in scores:
    rs, ws = score[0], score[1]  # reading_score, writing_score

    predicted_ws = regression_gradient * rs + regression_intercept
    opt_se = (ws - predicted_ws) ** 2  # (실제값 - 예측값)의 제곱
    opt_sum_sse += opt_se
    predicted_ws = est_gradient * rs + est_intercept
    est_se = (ws - predicted_ws) ** 2  # (실제값 - 예측값)의 제곱
    est_sum_sse += est_se

# 평균 제곱 오차
opt_mse = opt_sum_sse / total_scores_length
est_mse = est_sum_sse / total_scores_length

fig = plt.figure(figsize=(8, 8))
fig.set_facecolor('white')

font_size = 15
plt.title("Mean MSE change", fontsize=font_size)
for i in range(len(result)):
    plt.scatter(result[i][0], result[i][1])

x = np.arange(0, generation)  # x의 범위 0 ~ generaion

plt.hlines(opt_mse, 0, 30, color='red', linestyle='solid', linewidth=2)
plt.xlabel('generaion', fontsize=font_size)
plt.ylabel('MSE', fontsize=font_size)
plt.xlim([0, 10])
plt.ylim([0, 100])

plt.show()


# 최종 추정회귀식과 회귀식 비교
fig = plt.figure(figsize=(8, 8))
fig.set_facecolor('white')

font_size = 15
plt.title("estimated regression comapre", fontsize=font_size)
plt.scatter(df['reading_score'], df['writing_score'])
x = np.arange(0, 100)

plt.plot(x, regression_gradient * x + regression_intercept, color="red", linewidth="2.5")
plt.plot(x, est_gradient * x + est_intercept, color="orange", linestyle="--", linewidth="2.5")
plt.xlabel('Reading Score', fontsize=font_size)
plt.ylabel('Writing Score', fontsize=font_size)
plt.xlim([0, 100])
plt.ylim([0, 100])

plt.show()


