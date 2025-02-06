import unittest
from yacs.config import CfgNode
from scripts.f_environment import *
config_path = 'config/SwinCVS_config.yaml'

class TestEnv(unittest.TestCase):
    def test_get_config(self):
        config, experiment_name = get_config(config_path)
        self.assertIsInstance(config, CfgNode)
        self.assertIsInstance(experiment_name, str)

class TestValidateConfig(unittest.TestCase):
    def setUp(self):
        # Create a base config
        self.config_dict = read_config(config_path)
        self.config = config_to_yacs(self.config_dict)
        self.config.defrost()
        self.config.SEED = 42
        self.config.BACKBONE.PRETRAINED = 'swinv2_base_patch4'

    def test_swinv2_backbone(self):
        self.config.MODEL.LSTM = False
        experiment_name = validate_config(self.config)
        self.assertEqual(experiment_name, "SwinV2_backbone_IMNP_sd42")

    def test_frozen_swinvcs(self):
        self.config.MODEL.LSTM = True
        self.config.MODEL.E2E = False
        experiment_name = validate_config(self.config)
        self.assertEqual(experiment_name, "SwinCVS_frozen_IMNP_sd42")

    def test_e2e_without_multiclassifier(self):
        self.config.MODEL.LSTM = True
        self.config.MODEL.E2E = True
        self.config.MODEL.MULTICLASSIFIER = False
        experiment_name = validate_config(self.config)
        self.assertEqual(experiment_name, "SwinCVS_E2E_IMNP_sd42")

    def test_e2e_with_multiclassifier(self):
        self.config.MODEL.LSTM = True
        self.config.MODEL.E2E = True
        self.config.MODEL.MULTICLASSIFIER = True
        experiment_name = validate_config(self.config)
        self.assertEqual(experiment_name, "SwinCVS_E2E_MC_IMNP_sd42")

    def test_frozen_swinvcs_endoscapes_weights(self):
        self.config.MODEL.LSTM = True
        self.config.MODEL.E2E = False
        self.config.BACKBONE.PRETRAINED = 'endoscapes'
        experiment_name = validate_config(self.config)
        self.assertEqual(experiment_name, "SwinCVS_frozen_ENDP_sd42")

    def test_inference_with_weights(self):
        self.config.MODEL.LSTM = False
        self.config.MODEL.INFERENCE = True
        self.config.MODEL.INFERENCE_WEIGHTS = "swinv2_model_weights_sd3"
        experiment_name = validate_config(self.config)
        self.assertEqual(experiment_name, "SwinV2_backbone_sd3_INFERENCE")


if __name__ == "__main__":
    unittest.main()