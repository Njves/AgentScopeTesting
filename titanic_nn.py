"""
Нейросеть на PyTorch для предсказания выживания пассажира Титаника.

Запуск:
    python titanic_nn.py                 # обучение + интерактивный ввод
    python titanic_nn.py --no-interactive # только обучение

Файл данных по умолчанию: titanic.csv (рядом со скриптом).
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ----------------------------------------------------------------------
# 1. Подготовка данных
# ----------------------------------------------------------------------
class Preprocessor:
    """Хранит параметры предобработки (медианы, средние, std), чтобы
    одинаково обрабатывать и обучающие данные, и ввод пользователя."""

    FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    def fit_transform(self, df: pd.DataFrame):
        # Заполняем пропуски значениями, рассчитанными по обучающей выборке
        self.age_median = df["Age"].median()
        self.fare_median = df["Fare"].median()
        self.embarked_mode = df["Embarked"].mode()[0]

        X = self._build_features(df)

        # Стандартизация числовых признаков (среднее 0, std 1)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0  # защита от деления на ноль
        return (X - self.mean) / self.std

    def transform(self, df: pd.DataFrame):
        X = self._build_features(df)
        return (X - self.mean) / self.std

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df["Age"] = df["Age"].fillna(self.age_median)
        df["Fare"] = df["Fare"].fillna(self.fare_median)
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode)

        # Кодируем категориальные признаки числами
        sex = df["Sex"].map({"male": 0, "female": 1}).fillna(0)
        embarked = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0)

        X = np.column_stack([
            df["Pclass"].astype(float),
            sex.astype(float),
            df["Age"].astype(float),
            df["SibSp"].astype(float),
            df["Parch"].astype(float),
            df["Fare"].astype(float),
            embarked.astype(float),
        ])
        return X.astype(np.float32)


# ----------------------------------------------------------------------
# 2. Архитектура сети
# ----------------------------------------------------------------------
class TitanicNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),  # один выход — логит выживания
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ----------------------------------------------------------------------
# 3. Обучение
# ----------------------------------------------------------------------
def train_model(csv_path: str):
    df = pd.read_csv(csv_path)

    pre = Preprocessor()
    X = pre.fit_transform(df)
    y = df["Survived"].values.astype(np.float32)

    # Разбиение на обучающую и проверочную выборки (80/20)
    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]

    Xtr = torch.tensor(X[train_idx])
    ytr = torch.tensor(y[train_idx])
    Xval = torch.tensor(X[val_idx])
    yval = torch.tensor(y[val_idx])

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr), batch_size=32, shuffle=True
    )

    model = TitanicNet(n_features=X.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_pred = (torch.sigmoid(model(Xval)) >= 0.5).float()
                acc = (val_pred == yval).float().mean().item()
            print(f"Эпоха {epoch:3d}/{epochs} | "
                  f"loss={loss.item():.4f} | точность на val={acc:.3f}")

    # Итоговая точность
    model.eval()
    with torch.no_grad():
        val_pred = (torch.sigmoid(model(Xval)) >= 0.5).float()
        final_acc = (val_pred == yval).float().mean().item()
    print(f"\nИтоговая точность на проверочной выборке: {final_acc:.1%}")

    return model, pre


# ----------------------------------------------------------------------
# 4. Предсказание для одного пассажира
# ----------------------------------------------------------------------
def predict_passenger(model, pre, passenger: dict):
    """passenger — словарь с ключами из Preprocessor.FEATURES."""
    df = pd.DataFrame([passenger])
    X = torch.tensor(pre.transform(df))
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(X)).item()
    survived = prob >= 0.5
    return survived, prob


def ask(prompt, cast, default=None):
    """Считывает значение с клавиатуры с приведением типа."""
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            return cast(raw)
        except ValueError:
            print("  Неверный ввод, попробуйте ещё раз.")


def interactive_loop(model, pre):
    print("\n=== Ввод данных пассажира (Enter — оставить пустым/по умолчанию) ===")
    while True:
        try:
            pclass = ask("Класс билета (1, 2 или 3): ", int)
            sex_raw = ask("Пол (м/ж): ", str).lower()
            sex = "female" if sex_raw in ("ж", "f", "female", "ж.") else "male"
            age = ask("Возраст (лет): ", float)
            sibsp = ask("Кол-во братьев/сестёр/супругов на борту (SibSp): ", int)
            parch = ask("Кол-во родителей/детей на борту (Parch): ", int)
            fare = ask("Стоимость билета (Fare): ", float)
            emb_raw = ask("Порт посадки (S=Southampton, C=Cherbourg, Q=Queenstown): ", str).upper()
            embarked = emb_raw if emb_raw in ("S", "C", "Q") else "S"

            passenger = {
                "Pclass": pclass, "Sex": sex, "Age": age,
                "SibSp": sibsp, "Parch": parch, "Fare": fare,
                "Embarked": embarked,
            }
            survived, prob = predict_passenger(model, pre, passenger)
            verdict = "ВЫЖИВЕТ" if survived else "НЕ выживет"
            print(f"\n>>> Прогноз: {verdict} "
                  f"(вероятность выживания ≈ {prob:.1%})\n")
        except (KeyboardInterrupt, EOFError):
            print("\nВыход.")
            break

        again = input("Ввести ещё одного пассажира? (д/н): ").strip().lower()
        if again not in ("д", "y", "yes", "да"):
            break


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="titanic.csv", help="путь к CSV-файлу")
    parser.add_argument("--no-interactive", action="store_true",
                        help="только обучить модель, без ручного ввода")
    args = parser.parse_args()

    model, pre = train_model(args.csv)

    if not args.no_interactive:
        interactive_loop(model, pre)


if __name__ == "__main__":
    main()
