from fastrag.generators.openvino import OpenVINOGenerator
from haystack import Pipeline

# Create OpenVINO generator
generator = OpenVINOGenerator(
    model="microsoft/Phi-3.5-mini-instruct",
    compressed_model_dir="lokinfey/Phi-3.5-mini-instruct-ov-int4",
    device_openvino="AUTO:GPU.1,CPU.0",
    task="text-generation",
    generation_kwargs={
        "max_new_tokens": 500,
    }
)

# Create a pipeline
pipeline = Pipeline()

# Add the generator to the pipeline
pipeline.add_component("generator", generator)

# Run the pipeline
results = pipeline.run({"generator": {"prompt": "Who is the best American actor?"}})

# Print the results
print(results)