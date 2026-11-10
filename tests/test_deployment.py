import unittest

import numpy as np
from PIL import Image

from deployment import best_frames, load_config, preprocess_image, sweep_summary


class DeploymentHelpersTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_config_contains_six_classes(self):
        self.assertEqual(len(self.config["classes"]), 6)
        self.assertEqual(self.config["classes"][-1], "no_plane")

    def test_preprocess_returns_model_tensor(self):
        image = Image.fromarray(np.full((500, 700), 128, dtype=np.uint8), mode="L")
        tensor = preprocess_image(image, self.config)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)

    def test_sweep_summary_uses_max_non_plane_score(self):
        results = [
            {"frame": 1, "predicted_class": "no_plane", "confidence": 0.8,
             "probabilities": {"biparietal": 0.1, "abdominal": 0.1, "heart": 0.1,
                                "spine": 0.1, "femur": 0.1, "no_plane": 0.5}},
            {"frame": 2, "predicted_class": "biparietal", "confidence": 0.8,
             "probabilities": {"biparietal": 0.8, "abdominal": 0.02, "heart": 0.02,
                                "spine": 0.02, "femur": 0.02, "no_plane": 0.12}},
        ]
        summary = sweep_summary(results, self.config)
        self.assertAlmostEqual(summary["best_score"], 0.88)
        self.assertTrue(summary["usable"])

    def test_best_frames_selects_highest_probability(self):
        row_a = {"frame": 1, "probabilities": {name: 0.0 for name in self.config["classes"]}}
        row_b = {"frame": 2, "probabilities": {name: 0.0 for name in self.config["classes"]}}
        row_a["probabilities"]["heart"] = 0.3
        row_b["probabilities"]["heart"] = 0.7
        selected = best_frames([row_a, row_b], self.config)
        self.assertEqual(selected["heart"]["frame"], 2)


if __name__ == "__main__":
    unittest.main()
