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


def print_scatterplot(p=None, est=None):
    if p is None:
        p = []

    if est is None:
        est = []

    fig = plt.figure(figsize=(8, 8))
    fig.set_facecolor('white')

    fs = 15
    plt.title(f"Scatter Plot", fontsize=fs)
    plt.scatter(df['reading_score'], df['writing_score'])
    x = np.arange(0, 100)

    #  알고리즘 과정일 떼,
    if len(est) == 0:
        # 만약 입력 값이 들어 온다면 후보해 들 그리기
        # 없다면 for 문은 실행 되지 않음
        for i in range(len(p)):
            plt.plot(x, p[i][0] * x + p[i][1])

    #  마지막 추정된 회귀식과 비교할 때,
    else:
        plt.title("estimated regression comapre", fontsize=fs)
        plt.plot(x, est[0] * x + est[1], color="orange", linestyle="--", linewidth="2.5")

    plt.plot(x, regression_gradient * x + regression_intercept, color="red", linewidth="2.5")
    plt.xlabel('Reading Score', fontsize=fs)
    plt.ylabel('Writing Score', fontsize=fs)
    plt.xlim([0, 100])
    plt.ylim([0, 100])

    plt.show()


population_length = 30  # 모집단의 수


#  초기 세대 생성
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
    for i in range(len(p)):
        w = p[i][0]  # 기울기
        b = p[i][1]  # y절편

        sum_squared_error = 0
        for reading_score, writing_score in scores:
            predicted_ws = w * reading_score + b  # 예측값
            squared_error = (writing_score - predicted_ws) ** 2  # 오차의 제곱
            sum_squared_error += squared_error

        # 평균 제곱 오차
        mean_squared_error = sum_squared_error / total_scores_length
        mean_squared_errores.append(mean_squared_error)

    return mean_squared_errores


# 적합도 구하기
def get_fittnesses(mean_squared_errores):
    fittnesses = []
    sum_mean_squared_errores = sum(mean_squared_errores)

    # 평균 제곱 오차가 낮을수록 적합도가 높기 때문에
    # (전체 평균 제곱 오차의 합 / 평균 제곱 오차)를 통해서
    # 평균 제곱 오차가 낮을수록 적합도를 높게 설정
    for mse in mean_squared_errores:
        fittnesses.append(sum_mean_squared_errores/mse)

    return fittnesses


# 룰렛 휠에 사용될 누적 확률 리스트 만들기
def get_percentages(fittnesses):
    percenatages = []
    sum_fittnesses = sum(fittnesses)

    for fittness in fittnesses:
        # idx == 0, 누적할 확률이 없음
        if len(percenatages) == 0:
            percenatages.append(fittness / sum_fittnesses)
        # idx != 0, 이전 확률과 더해서 저장
        else:
            percenatages.append(percenatages[-1] + fittness / sum_fittnesses)

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
    half_population_length = population_length // 2

    p.sort()  # 같은 직선끼리 교차할 확률을 줄이기 위함
    for i in range(half_population_length):
        crossover_p.append([(p[i][0] + p[i + half_population_length][0]) / 2,
                            (p[i][1] + p[i + half_population_length][1]) / 2])
        crossover_p.append([(p[i][0] + p[-1 - i][0]) / 2,
                            (p[i][1] + p[-1 - i][1]) / 2])

    return crossover_p


# 돌연변이 연산
def mutation(p):
    mutated_p = []
    for i in range(population_length):
        random_percentage = random.random()
        # 돌연변이가 일어날 확률 1%
        if random_percentage > 0.997:
            mutated_p.append([p[i][0]+0.05, p[i][1]-0.])
        elif random_percentage > 0.990:
            mutated_p.append([p[i][0]-0.05, p[i][1]+0.1])
        else:
            mutated_p.append([p[i][0], p[i][1]])

    return mutated_p


print_scatterplot()

#  점수 데이터를 리스트에 저장
scores = []
for row in df.index:
    scores.append([df['reading_score'][row], df['writing_score'][row]])

total_scores_length = len(scores)


# 알고리즘 시작
population = init_population()
print_scatterplot(population)


result = []  # 평균 MSE 리스트
est_gradient, est_intercept = 0, 0  # 추정된 회귀식 기울기와 y절편

# 종료 조건 판별
while generation < 10:
    mses = get_mean_squared_errores(population)

    generation += 1
    # 선택 연산 -> 교차 연산 -> 돌연변이 연산
    selected_population = selection(population)
    crossover_population = crossover(selected_population)
    mutated_population = mutation(crossover_population)

    # 해당 세대의 오차크기 저장
    sum_est_mses = sum(mses)
    avg_est_mses = sum_est_mses/population_length
    result.append([generation, avg_est_mses])

    # 마지막 세대의 est_graident, est_intercept가 최동 추정된 회귀식
    mses_idx_pair = [[mses[i], i] for i in range(population_length)]  # 임시 리스트
    est_gradient = population[min(mses_idx_pair)[1]][0]
    est_intercept = population[min(mses_idx_pair)[1]][1]

    # 연산 반복
    population = mutated_population

    print_scatterplot(population)


# 라이브러리로 얻은 최적의 회귀식 MSE 구하기
opt_mse = get_mean_squared_errores([[regression_gradient,
                                     regression_intercept]]).pop()

# 추정된 회귀식과 최적의 회귀식 비교
figure = plt.figure(figsize=(8, 8))
figure.set_facecolor('white')

font_size = 15
plt.title("AVG MSE change compared OPT", fontsize=font_size)

for v in range(len(result)):
    if result[v][1] > 60:
        plt.scatter(result[v][0], 59)
        plt.text(result[v][0], 57, f"{int(result[v][1])}")
    else:
        plt.scatter(result[v][0], result[v][1])

plt.hlines(opt_mse, 0, generation, color='red', linestyle='solid', linewidth=2)
plt.xlabel('generaion', fontsize=font_size)
plt.ylabel('MSE', fontsize=font_size)
plt.xlim([0, generation])
plt.ylim([0, 60])

plt.show()

# 추정된 회귀식과 최적의 회귀식 그래프
print_scatterplot(est=[est_gradient, est_intercept])

# 추정된 회귀식과 최적의 회귀식
print("📌 추정된 회귀식")
print(f"y = {est_gradient} * x + {est_intercept}")
print("📌 최적의 회귀식")
print(f"y = {regression_gradient} * x + {regression_intercept}")
