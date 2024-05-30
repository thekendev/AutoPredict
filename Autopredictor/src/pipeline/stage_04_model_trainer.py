from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer
from src.logging import logger

STAGE_NAME = "model trainer stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
            config_manager = ConfigurationManager()
            model_trainer_config = config_manager.get_model_trainer_config()

            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train()
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    obj = ModelTrainerTrainingPipeline()
    obj.main()