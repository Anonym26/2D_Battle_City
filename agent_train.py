"""
Модуль agent_train.py игры "2D Battle City".
Этот модуль предназначен для тренировки агента в среде "Tanks"
с использованием алгоритма Proximal Policy Optimization (PPO).

Основные функции:
1. Создание окружения и настройка параметров обучения.
2. Обучение модели с возможностью логирования через TensorBoard.
3. Периодическое сохранение модели на диск для последующего использования.
"""

# Импортируем необходимые библиотеки
import gym_tanks
import gymnasium as gym
import os
from stable_baselines3 import PPO
from multiprocessing import freeze_support

# Устанавливаем окружение для устранения возможных ошибок
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":
    """
    Основной модуль для тренировки агента в среде "Tanks" с использованием алгоритма PPO.
    """
    # Устраняем проблемы с многопоточностью
    freeze_support()

    # Задаем папки для моделей и логов
    test_name = "Test_Bot"

    # Указываем имя теста для данного запуска
    models_dir = f"models/PPO/{test_name}"
    logdir = f"logs/{test_name}"   # Для отображения графиков: выполните команду "tensorboard --logdir=logs"

    TIMESTEPS = 2048  # Количество шагов по умолчанию для PPO

    # Путь к файлу .zip с предварительно обученными весами
    start_steps = 2064384
    if start_steps > 0 and start_steps % TIMESTEPS == 0:
        weights_path = f"models/PPO/{test_name}/model_{start_steps}_steps.zip"
    else:
        # Начинаем обучение с нуля
        weights_path = None

    # Создаем папки, если они не существуют
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # Загружаем и сбрасываем окружение
    env = gym.make('gym_tanks/tanks-v0')
    env.reset()

    # Проверяем, существует ли файл с весами
    if weights_path is not None:
        # Загружаем модель с предварительно обученными весами
        model = PPO.load(weights_path, env=env, tensorboard_log=logdir, n_steps=TIMESTEPS, device="cuda")
    else:
        # Создаем новую модель с использованием алгоритма PPO
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir, n_steps=TIMESTEPS, device="cuda")

    # Параметры обучения
    SAVE_INTERVAL = 2048  # Интервал сохранения модели
    TOTAL_TRAINING_TIMESTEPS = 2066432  # Общее количество шагов для обучения

    total_timesteps = start_steps
    last_save = start_steps  # Отслеживаем последнюю точку сохранения

    while total_timesteps < TOTAL_TRAINING_TIMESTEPS:
        # Выполняем обучение модели
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        total_timesteps += TIMESTEPS

        # Проверяем, нужно ли сохранить модель
        if total_timesteps // SAVE_INTERVAL > last_save // SAVE_INTERVAL:
            print(f"Сохраняю модель в {models_dir}/model_{total_timesteps}_steps")
            model.save(f"{models_dir}/model_{total_timesteps}_steps")
            last_save = total_timesteps  # Обновляем последнюю точку сохранения

    env.close()
