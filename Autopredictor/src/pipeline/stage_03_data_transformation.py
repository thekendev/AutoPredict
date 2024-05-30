from Autopredictor.src.config.configuration import ConfigurationManager
from Autopredictor.src.components.data_transformation import DataTransformation
from Autopredictor.src.logging import logger
from pathlib import Path 




STAGE_NAME = "Data transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config_manager = ConfigurationManager()
                data_transformation_config = config_manager.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.create_pipeline()
                data_transformation.transform_data()
            else:
                raise Exception("Your data schema is not valid")
        except Exception as e:
            logger.exception("Exception occurred during data transformation:", exc_info=True)
            print(e)
