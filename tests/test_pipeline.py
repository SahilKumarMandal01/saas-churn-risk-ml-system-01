from src.pipeline.training_pipeline import TrainingPipeline

def test_training_pipeline_initialization():
    pipeline = TrainingPipeline()
    assert pipeline is not None