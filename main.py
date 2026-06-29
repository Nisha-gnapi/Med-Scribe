from pipeline import MedicalTranscriptPipeline
from Optional6_testQuality import run_validation_pipeline

def main():

    pipeline = MedicalTranscriptPipeline()

    result=pipeline.run(transcripts="record1.json",
        verbose=True
    )
    validation_score=run_validation_pipeline(result["transcript"])
    return result,validation_score
if __name__ == "__main__":

    main()