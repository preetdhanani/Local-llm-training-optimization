import os
import pytest
from src.config import TrainConfig
from src.pipelines.run_full_pipeline import run_full_pipeline

def test_full_pipeline_invalid_columns(invalid_csv_data):
    """Test that pipeline fails gracefully with clear error on missing columns."""
    cfg = TrainConfig()
    cfg.dataset_type = "local"
    cfg.dataset_path = invalid_csv_data
    cfg.dataset_format = "custom_columns"
    cfg.prompt_column = "my_prompt"
    cfg.chosen_column = "my_chosen" # This column is missing in invalid_csv_data
    cfg.model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    cfg.wandb_mode = "disabled"
    
    with pytest.raises(ValueError) as excinfo:
        run_full_pipeline(cfg)
    
    assert "Required columns" in str(excinfo.value)
    assert "not found in dataset" in str(excinfo.value)

def test_full_pipeline_success(valid_csv_data, tmp_path):
    """
    Test a full success run (Phase 0 -> Phase 2).
    Uses a tiny random model on CPU to ensure speed and portability.
    """
    # Override output dirs for test isolation
    test_outputs = tmp_path / "outputs"
    test_logs = tmp_path / "logs"
    os.makedirs(test_outputs, exist_ok=True)
    os.makedirs(test_logs, exist_ok=True)

    cfg = TrainConfig()
    cfg.dataset_type = "local"
    cfg.dataset_path = valid_csv_data
    cfg.dataset_format = "custom_columns"
    cfg.prompt_column = "my_prompt"
    cfg.chosen_column = "my_chosen"
    cfg.rejected_column = "my_rejected"
    cfg.dataset_len = 3
    
    # Use a tiny model and CPU settings
    cfg.model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    cfg.use_4bit = False
    cfg.batch_size = 1
    cfg.epochs = 1
    cfg.wandb_mode = "disabled"
    cfg.sft_eval_steps = 1
    cfg.sft_save_steps = 1
    cfg.dpo_eval_steps = 1
    cfg.dpo_save_steps = 1
    
    # Run the full pipeline
    # Note: We expect this to take ~30-60s on CPU
    run_full_pipeline(cfg)
    
    # Verify outputs
    assert os.path.exists(cfg.sft_output_dir)
    assert os.path.exists(cfg.dpo_output_dir)
    
    # Check for SFT adapter
    sft_adapter = os.path.join(cfg.sft_output_dir, "adapter_config.json")
    assert os.path.exists(sft_adapter), "SFT adapter config not found"
    
    # Check for DPO adapter
    dpo_adapter = os.path.join(cfg.dpo_output_dir, "adapter_config.json")
    assert os.path.exists(dpo_adapter), "DPO adapter config not found"
