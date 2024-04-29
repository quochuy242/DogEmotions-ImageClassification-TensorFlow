import os
import src
import src.pipeline


def main() -> None:
    try:
        src.pipeline.ModelTrainingPipeline(model="CNN").run_pipeline
        src.pipeline.ModelEvaluationPipeline(model="CNN").run_pipeline
    except Exception as e:
        src.logging.exception(e)
    return None


if __name__ == "__main__":
    main()
