"""Comparison planning: models x steps x seeds from the gateway's catalog, bounded, every model given
the same prompt and seed; steps only where a model can change them."""
import unittest

from spielwiese.compare import MAX_CELLS, CompareError, plan

CATALOG = {
    "z-image-turbo": {"id": "z-image-turbo", "label": "Z-Image Turbo", "available": True, "provider": "comfyui",
                      "aspect_ratios": ["1:1", "4:3", "3:4", "16:9", "9:16"],
                      "steps": {"default": 8, "adjustable": True, "max": 60}},
    "qwen-image-2.1": {"id": "qwen-image-2.1", "label": "Qwen-Image 2.1", "available": True, "aspect_ratios": ["1:1"],
                       "provider": "comfyui",
                       "steps": {"default": 20, "adjustable": True, "max": 60}},
    "iris-turbo": {"id": "iris-turbo", "label": "Iris-3B Turbo", "available": True, "provider": "iris",
                   "aspect_ratios": ["1:1", "4:3", "3:4", "16:9", "9:16"],
                   "steps": {"default": 4, "adjustable": False, "max": 4}},
    "ideogram-4": {"id": "ideogram-4", "label": "Ideogram 4", "available": False, "aspect_ratios": ["1:1"],
                   "steps": {"default": 20, "adjustable": True, "max": 60}},
}


class TestPlan(unittest.TestCase):
    def test_same_prompt_and_seed_for_every_model(self):
        p = plan({"prompt": " a fox ", "models": ["z-image-turbo", "iris-turbo"], "seeds": [7]}, CATALOG)
        self.assertEqual(p["prompt"], "a fox")
        self.assertEqual(p["aspect_ratio"], "1:1")
        self.assertEqual([(c["model"], c["steps"], c["seed"]) for c in p["cells"]],
                         [("z-image-turbo", 8, 7), ("iris-turbo", 4, 7)])
        self.assertEqual(p["cells"][0]["label"], "Z-Image Turbo")
        self.assertEqual([c["lane"] for c in p["cells"]], ["comfyui", "iris"])   # one lane per provider

    def test_no_seed_means_one_shared_random_seed(self):
        p = plan({"prompt": "x", "models": ["z-image-turbo", "iris-turbo"]}, CATALOG)
        self.assertEqual(len({c["seed"] for c in p["cells"]}), 1)

    def test_steps_grid_only_where_adjustable(self):
        p = plan({"prompt": "x", "models": ["z-image-turbo", "iris-turbo"], "steps": [16, 4, 8], "seeds": [1, 2]},
                 CATALOG)
        self.assertEqual([(c["model"], c["steps"], c["seed"]) for c in p["cells"]],
                         [("z-image-turbo", 4, 1), ("z-image-turbo", 4, 2), ("z-image-turbo", 8, 1),
                          ("z-image-turbo", 8, 2), ("z-image-turbo", 16, 1), ("z-image-turbo", 16, 2),
                          ("iris-turbo", 4, 1), ("iris-turbo", 4, 2)])
        self.assertEqual([c["send_steps"] for c in p["cells"]], [True] * 6 + [False] * 2)

    def test_refusals(self):
        for body in ({"prompt": "", "models": ["z-image-turbo"]},
                     {"prompt": "x" * 4001, "models": ["z-image-turbo"]},
                     {"prompt": "x", "models": []},
                     {"prompt": "x", "models": ["nope"]},
                     {"prompt": "x", "models": ["ideogram-4"]},                               # not installed
                     {"prompt": "x", "models": ["z-image-turbo"], "aspect_ratio": "21:9"},
                     {"prompt": "x", "models": ["z-image-turbo"], "steps": [0]},
                     {"prompt": "x", "models": ["z-image-turbo"], "steps": [61]},
                     {"prompt": "x", "models": ["z-image-turbo"], "steps": ["8"]},
                     {"prompt": "x", "models": ["z-image-turbo"], "seeds": [-1]},
                     {"prompt": "x", "models": ["z-image-turbo"], "seeds": [1, 2, 3, 4, 5]},
                     {"prompt": "x", "models": ["z-image-turbo"], "evil": 1},
                     {"prompt": "x", "models": "z-image-turbo"}):
            with self.assertRaises(CompareError, msg=body):
                plan(body, CATALOG)

    def test_a_model_without_the_aspect_gets_its_closest_one(self):
        """Refusing the whole comparison over one model made comparing impossible (user, 2026-10-18)."""
        p = plan({"prompt": "x", "models": ["z-image-turbo", "qwen-image-2.1"], "aspect_ratio": "16:9", "seeds": [1]},
                 CATALOG)
        self.assertEqual([(c["model"], c["aspect_ratio"]) for c in p["cells"]],
                         [("z-image-turbo", "16:9"), ("qwen-image-2.1", "1:1")])
        self.assertEqual(p["aspect_ratio"], "16:9")
        wide_only = {**CATALOG["z-image-turbo"], "id": "w", "aspect_ratios": ["4:3", "16:9"]}
        p = plan({"prompt": "x", "models": ["w"], "aspect_ratio": "9:16", "seeds": [1]}, {"w": wide_only})
        self.assertEqual(p["cells"][0]["aspect_ratio"], "4:3")   # closest to 9:16 among 4:3 and 16:9

    def test_cell_cap(self):
        many = {f"m{i}": {**CATALOG["z-image-turbo"], "id": f"m{i}"} for i in range(8)}
        with self.assertRaises(CompareError):
            plan({"prompt": "x", "models": list(many), "steps": [4, 8, 12, 16, 20, 24], "seeds": [1]}, many)
        self.assertLessEqual(len(plan({"prompt": "x", "models": list(many)[:4], "steps": [4, 8, 12, 16],
                                       "seeds": [1, 2]}, many)["cells"]), MAX_CELLS)
