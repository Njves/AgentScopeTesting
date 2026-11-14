from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import torch

from titanic_nn import Preprocessor, ROOT, load_model, predict, split_data, train_model


class TitanicTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv(ROOT / "titanic.csv")

    def test_features_missing_values_and_unknown_categories(self):
        training = self.data.iloc[:10].copy()
        training["Age"] = 25.0
        training["Fare"] = 10.0
        preprocessor = Preprocessor()
        features = preprocessor.fit_transform(training)
        state = deepcopy(vars(preprocessor))
        passengers = training.iloc[:2].copy()
        passengers.loc[0, ["Age", "Fare", "Cabin", "Embarked"]] = np.nan
        passengers.loc[1, "Ticket"] = "NEWPREFIX 42"
        transformed = preprocessor.transform(passengers)
        self.assertEqual(features.shape, (10, 11))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transformed[0, 2], 0)
        self.assertEqual(transformed[0, 8], 0)
        self.assertEqual(transformed[1, 7], -1)
        np.testing.assert_array_equal(features[:, 5], training.SibSp + training.Parch + 1)
        np.testing.assert_array_equal(features[:, 6], features[:, 5] == 1)
        pd.testing.assert_series_equal(preprocessor.means, state["means"])
        self.assertEqual(preprocessor.categories, state["categories"])

    def test_invalid_inputs(self):
        for column, value in [("Pclass", 4), ("Age", -1), ("Fare", np.inf), ("SibSp", 0.5)]:
            with self.subTest(column=column):
                invalid = self.data.iloc[:10].copy()
                invalid[column] = value
                with self.assertRaises(ValueError):
                    Preprocessor().fit_transform(invalid)
        with self.assertRaises(ValueError):
            Preprocessor().fit_transform(self.data.drop(columns="Sex"))
        with self.assertRaises(ValueError):
            split_data(self.data.assign(Survived=2))

    def test_splits_and_training_round_trip(self):
        training, validation, test = split_data(self.data)
        self.assertEqual([len(training), len(validation), len(test)], [623, 178, 90])
        self.assertFalse(set(training.index) & set(validation.index))
        self.assertFalse(set(training.index) & set(test.index))
        self.assertFalse(set(validation.index) & set(test.index))
        with TemporaryDirectory() as directory:
            model, preprocessor, metrics = train_model(ROOT / "titanic.csv", directory, epochs=2)
            expected_means = training[["Age", "Fare"]].mean()
            pd.testing.assert_series_equal(preprocessor.means, expected_means)
            restored_model, restored_preprocessor = load_model(directory)
            np.testing.assert_array_equal(
                predict(model, preprocessor, test), predict(restored_model, restored_preprocessor, test),
            )
            self.assertEqual(metrics["epochs_run"], 2)
            history = pd.read_csv(Path(directory) / "history.csv")
            best = history.loc[metrics["best_epoch"] - 1]
            self.assertAlmostEqual(best.val_loss, metrics["best_val_loss"])
            x = torch.from_numpy(preprocessor.transform(validation))
            y = torch.tensor(validation.Survived.to_numpy(dtype=np.float32))
            with torch.no_grad():
                loss = torch.nn.BCEWithLogitsLoss()(restored_model(x), y).item()
            self.assertAlmostEqual(loss, metrics["best_val_loss"])
            self.assertEqual(len(pd.read_csv(Path(directory) / "test_predictions.csv")), 90)


if __name__ == "__main__":
    unittest.main()
