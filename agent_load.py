"""
Модуль agent_load.py игры "2D Battle City".
Этот модуль предназначен для запуска предварительно обученной модели агента в среде "Tanks".

Основные функции:
1. Загрузка окружения и модели с предварительно обученными весами.
2. Воспроизведение действий модели в нескольких эпизодах с визуализацией процесса.
3. Вывод вознаграждений на каждом шаге симуляции.
"""

# Импорт необходимых библиотек
import gym_tanks
import gymnasium as gym
from stable_baselines3 import PPO
from multiprocessing import freeze_support

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == "__main__":
    """
    Модуль для запуска предварительно обученной модели агента в среде "Tanks".

    Скрипт загружает веса обученной модели PPO, запускает среду и выполняет несколько эпизодов симуляции,
    выводя вознаграждения за каждый шаг.
    """
    # Устраняем проблемы с многопоточностью
    freeze_support()

    # Загружаем и сбрасываем окружение
    env = gym.make('gym_tanks/tanks-v0')
    env.reset()

    TIMESTEPS = 10000

    # Указываем папку с моделями и путь к весам
    models_dir = "models/PPO/Test_Bot"
    model_path = f"{models_dir}/model_2066432_steps.zip"  # Укажите файл весов для загрузки

    # Загружаем модель
    model = PPO.load(model_path, env=env, n_steps=TIMESTEPS)

    # Векторизуем окружение
    env = model.get_env()

    episodes = 10  # Количество эпизодов для симуляции

    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            # Отображаем окружение
            env.render()
            # Предсказываем действие
            action, _state = model.predict(obs, deterministic=False)
            # Выполняем шаг в окружении
            obs, reward, done, info = env.step(action)
            print(reward)  # Выводим вознаграждение

    env.close()
