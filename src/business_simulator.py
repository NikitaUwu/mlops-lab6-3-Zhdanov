import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
import time
import json
import requests
import random
from datetime import datetime, timedelta


def simulate_business_process(num_days=30, num_requests_per_day=1000):
    """
    Симулирует работу бизнес-процесса до и после внедрения ML-модели

    Args:
    num_days: Количество дней для симуляции
    num_requests_per_day: Количество запросов в день

    Returns:
    DataFrame с результатами симуляции
    """
    print(f"Simulating business process for {num_days} days with {num_requests_per_day} requests/day...")

    # Параметры до внедрения (ручная обработка)
    base_time_per_request = 24 * 60 # 24 часа в минутах
    base_satisfaction = 0.65 # 65% удовлетворенности
    base_conversion = 0.20 # 20% конверсия

    # Параметры после внедрения
    model_time_reduction = 0.7 # Снижение времени на 70%
    accuracy = 0.85 # Точность мsatisfaction_increase = 0.15 * accuracy # Удовлетворенность растет пропорционально точности
    conversion_increase = 0.05 * accuracy # Конверсия растет пропорционально точности

    # Симуляция
    results = []
    for day in range(num_days):
        for i in range(num_requests_per_day):
            # Генерация случайного запроса
            request_type = random.choice(['new', 'existing', 'problem'])
            customer_value = random.uniform(0, 1)
            is_critical = random.random() < 0.1 # 10% критических запросов

            # До внедрения
            base_time = base_time_per_request
            base_satisfaction_score = base_satisfaction
            base_conversion_rate = base_conversion

            # После внедрения
            if random.random() < accuracy:
                # Правильная классификация
                model_time = base_time * (1 - model_time_reduction)
                model_satisfaction = base_satisfaction + satisfaction_increase
                model_conversion = base_conversion + conversion_increase
            else:
                # Неправильная классификация
                model_time = base_time * 0.5 # Все равно быстрее ручной обработки
                model_satisfaction = base_satisfaction - 0.1 # Небольшое снижение из-за ошибки
                model_conversion = base_conversion - 0.05 # Небольшое снижение конверсии

            # Добавление в результаты
            timestamp = datetime.now() - timedelta(days=num_days-day, seconds=i*86400//num_requests_per_day)
            results.append({
                'timestamp': timestamp,
                'day': day,
                'request_id': i,
                'request_type': request_type,
                'customer_value': customer_value,
                'is_critical': int(is_critical),
                'base_time': base_time,
                'model_time': model_time,
                'base_satisfaction': base_satisfaction_score,
                'model_satisfaction': model_satisfaction,
                'base_conversion': base_conversion_rate,
                'model_conversion': model_conversion,
                'correct_classification': int(random.random() < accuracy)
            })

    return pd.DataFrame(results)


def calculate_roi(simulation_results, cost_per_request=10, num_requests_per_day=1000):
    """
    Рассчитывает ROI и срок окупаемости

    Args:
    simulation_results: результаты симуляции
    cost_per_request: стоимость обработки одного запроса вручную (USD)
    num_requests_per_day: количество запросов в день

    Returns:
    dict: результаты расчета ROI
    """
    # Расчет текущих операционных затрат
    daily_cost_base = num_requests_per_day * cost_per_request
    yearly_cost_base = daily_cost_base * 365

    # Расчет операционных затрат после внедрения
    time_reduction = 1 - simulation_results['model_time'].mean() / simulation_results['base_time'].mean()
    cost_reduction = time_reduction * 0.7 # Предполагаем, что 70% экономии времени = 70% экономии затрат
    daily_cost_model = daily_cost_base * (1 - cost_reduction)
    yearly_cost_model = daily_cost_model * 365

    # Расчет выгоды от повышения удовлетворенности
    satisfaction_increase = simulation_results['model_satisfaction'].mean() - simulation_results['base_satisfaction'].mean()
    yearly_retention_benefit = yearly_cost_base * satisfaction_increase * 0.5 # 50% от стоимости

    # Расчет выгоды от повышения конверсии
    conversion_increase = simulation_results['model_conversion'].mean() - simulation_results['base_conversion'].mean()
    average_order_value = 100 # USD
    yearly_conversion_benefit = num_requests_per_day * 365 * conversion_increase * average_order_value

    # Общая годовая выгода
    yearly_savings = (yearly_cost_base - yearly_cost_model) + yearly_retention_benefit + yearly_conversion_benefit

    # Оценка стоимости решения
    development_cost = 50000 # USD
    infrastructure_cost = 10000 # USD/год
    maintenance_cost = 20000 # USD/год

    total_cost = development_cost + infrastructure_cost + maintenance_cost

    # Расчет ROI и срока окупаемости
    roi = (yearly_savings - (infrastructure_cost + maintenance_cost)) / total_cost * 100
    payback_period = development_cost / (yearly_savings - infrastructure_cost - maintenance_cost)

    return {
        'yearly_savings': yearly_savings,
        'development_cost': development_cost,
        'infrastructure_cost': infrastructure_cost,
        'maintenance_cost': maintenance_cost,
        'total_cost': total_cost,
        'roi': roi,
        'payback_period_months': payback_period * 12,
        'yearly_net_benefit': yearly_savings - infrastructure_cost - maintenance_cost
    }


def visualize_results(simulation_results, roi_results):
    """Визуализирует результаты симуляции и ROI"""
    plt.figure(figsize=(15, 10))

    # Влияние на время обработки
    plt.subplot(2, 2, 1)
    plt.plot(simulation_results.groupby('day')['base_time'].mean(), label='Базовая')
    plt.plot(simulation_results.groupby('day')['model_time'].mean(), label='С моделью')
    plt.title('Среднее время обработки запроса')
    plt.xlabel('День')
    plt.ylabel('Минуты')
    plt.legend()
    plt.grid(True)
    # Влияние на удовлетворенность
    plt.subplot(2, 2, 2)
    plt.plot(simulation_results.groupby('day')['base_satisfaction'].mean(), label='Базовая')
    plt.plot(simulation_results.groupby('day')['model_satisfaction'].mean(), label='С моделью')
    plt.title('Удовлетворенность клиентов')
    plt.xlabel('День')
    plt.ylabel('Удовлетворенность')
    plt.legend()
    plt.grid(True)

    # Влияние на конверсию
    plt.subplot(2, 2, 3)
    plt.plot(simulation_results.groupby('day')['base_conversion'].mean(), label='Базовая')
    plt.plot(simulation_results.groupby('day')['model_conversion'].mean(), label='С моделью')
    plt.title('Конверсия')
    plt.xlabel('День')
    plt.ylabel('Конверсия')
    plt.legend()
    plt.grid(True)

    # Финансовый анализ
    plt.subplot(2, 2, 4)
    categories = ['Годовая выгода', 'Стоимость разработки', 'Годовые затраты']
    values = [
        roi_results['yearly_savings'],
        roi_results['development_cost'],
        roi_results['infrastructure_cost'] + roi_results['maintenance_cost']
    ]
    colors = ['g', 'r', 'orange']
    plt.bar(categories, values, color=colors)
    plt.title('Финансовый анализ')
    plt.ylabel('USD')
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.savefig('business_impact.png')
    plt.show()

    # Визуализация ROI
    plt.figure(figsize=(10, 5))
    metrics = ['ROI', 'Срок окупаемости (мес)']
    values = [roi_results['roi'], roi_results['payback_period_months']]
    plt.bar(metrics, values, color=['b', 'purple'])
    plt.title('Ключевые финансовые показатели')
    plt.ylabel('Значение')
    plt.savefig('roi_analysis.png')
    plt.show()


def main():
    # Симуляция бизнес-процесса
    simulation_results = simulate_business_process(num_days=30,
    num_requests_per_day=1000)

    # Расчет ROI
    roi_results = calculate_roi(simulation_results, cost_per_request=10,
    num_requests_per_day=1000)

    # Визуализация результатов
    visualize_results(simulation_results, roi_results)

    # Сохранение результатов
    simulation_results.to_csv('simulation_results.csv', index=False)

    with open('roi_results.json', 'w') as f:
        json.dump(roi_results, f, indent=2)

    # Вывод ключевых результатов
    print("\nКлючевые результаты симуляции:")
    print(f"Среднее время обработки до: {simulation_results['base_time'].mean():.2f} мин")
    print(f"Среднее время обработки после: {simulation_results['model_time'].mean():.2f} мин")
    print(f"Снижение времени: {(1 - simulation_results['model_time'].mean()/simulation_results['base_time'].mean())*100:.2f}%")

    print(f"\nУдовлетворенность до: {simulation_results['base_satisfaction'].mean():.4f}")
    print(f"Удовлетворенность после: {simulation_results['model_satisfaction'].mean():.4f}")
    print(f"Увеличение удовлетворенности: {(simulation_results['model_satisfaction'].mean() - simulation_results['base_satisfaction'].mean())*100:.2f} п.п.")

    print(f"\nКонверсия до: {simulation_results['base_conversion'].mean():.4f}")
    print(f"Конверсия после: {simulation_results['model_conversion'].mean():.4f}")
    print(f"Увеличение конверсии: {(simulation_results['model_conversion'].mean() - simulation_results['base_conversion'].mean())*100:.2f} п.п.")

    print("\nФинансовый анализ:")
    print(f"Годовая выгода: ${roi_results['yearly_savings']:,.2f}")
    print(f"Стоимость разработки: ${roi_results['development_cost']:,.2f}")
    print(f"Годовые затраты на инфраструктуру и поддержку: ${roi_results['infrastructure_cost'] + roi_results['maintenance_cost']:,.2f}")
    print(f"ROI: {roi_results['roi']:.2f}%")
    print(f"Срок окупаемости: {roi_results['payback_period_months']:.1f} месяцев")
    print(f"Годовая чистая выгода: ${roi_results['yearly_net_benefit']:,.2f}")


if __name__ == "__main__":
    main()