import pandas as pd
import logging
from typing import List
import subprocess
import threading
import json
import math

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class CustomThread(threading.Thread):
    """
    Custom thread class to retrieve return values from threaded functions.
    """
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        if kwargs is None:
            kwargs = {}
        super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return

def load_data(labeltype:str):
    """
    Load labeled and unlabeled data from CSV files.
    """
    try:
        if labeltype == "behavior":
            data = pd.read_csv("data/text/behavior_labels.csv")
        elif labeltype == "stance":
            data = pd.read_csv("data/text/stance_labels.csv")
        return data
            
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def generate_prompt(data: pd.DataFrame, persona, datatype) -> List[str]:
    unique_labels = None  # Initialize unique_labels

    logger.info("Starting to generate few-shot prompts.")
    try:
        # Use only the 'text' and 'labels' columns from labeled data
        text_with_labels = data[['text', 'label']].copy()

      
       
        # Exclude rows with the label "Not about vaccines" if needed (not shown)

        if datatype == "stance":
            unique_labels = ['Vaccinated/planning to vaccinate', 'No information about the author’s vaccine intentions', 'Not vaccinated/not planning to vaccinate']
        elif datatype == "behavior":
            unique_labels = ['Vaccination schedule, spacing, or timing', 'Vaccines are mentioned positively' ,'Protection from disease' ,'About vaccines but does not fit any of the previous categories' ,'Vaccines are mentioned negatively' ,'Seeking information about vaccines', 'Adverse effects']

        # Group labeled data by label for sampling examples
        grouped = text_with_labels.groupby('label')

        prompts = []

        for text in data['text']:
            # Clean the unlabeled text by removing newlines and quotes
            clean_text = text.replace("\n", " ").replace('"', '')

            # For each label group, randomly sample up to 3 examples (or fewer if not available)
            sample_docs = grouped.apply(lambda x: x.sample(min(len(x), 3))).reset_index(drop=True)

            # make sure samples do not equal text
            sample_docs = sample_docs[sample_docs['text'] != text]

            # Construct example string: each example is delineated by markers
            examples = " ".join(
                f'<ExampleStart> {row["clean_text"]} : {unique_labels[row["label"]]} <ExampleEnd>'
                for _, row in sample_docs.assign(
                    clean_text=sample_docs["text"]
                    .str.replace("\n", " ", regex=False)
                    .str.replace("\\", "", regex=False)
                ).iterrows()
            )

            # Build the prompt with clear instructions and boundaries.
            prompt = (
                f"You are a {persona}. Your job is to look at the following labeled examples and then label an unlabeled document.\n"
                f"Available categories: {', '.join(unique_labels)}.\n\n"
                f"Here are some examples for guidance:\n{examples}\n\n"
                f"Now, classify the following text between the tokens <<<START>>> and <<<END>>>.\n"
                f"<<<START>>>\n{clean_text}\n<<<END>>>\n\n"
                f"Return only a single label (from the available categories) without any additional text or formatting."
            )
            prompts.append(prompt)

        logger.info(f"Generated {len(prompts)} prompts.")
        return prompts
    except Exception as e:
        logger.error(f"Error during prompt generation: {e}")
        raise

def generate_ollama_requests(input_prompts: List[str]) -> List[str]:
    """
    Generate JSON requests for the Ollama model.
    Each request is a JSON object specifying the model, user message content, and streaming setting.
    """
    logger.info("Generating Ollama requests.")
    requests = [
        {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "seed": 101,
                "temperature": 0
            }
        } for prompt in input_prompts
    ]
    return requests

def write_requests_to_file(requests: List[dict], filename: str = "requests.json"):
    """
    Write each JSON request to the specified file, one per line.
    This function also cleans up escape sequences for clarity.
    """
    logger.info(f"Writing {len(requests)} requests to {filename}.")
    with open(filename, "a") as f:
        for request in requests:
            # Dump JSON as a single line and perform cleanup on escape sequences.
            request_str = json.dumps(request)
            # Clean up escape sequences and unwanted characters
            request_str = (
                request_str.replace("\\n", " ")
                           .replace("\\u2019", "`")
                           .replace('"false"', 'false')
                           .replace("\\\\", " ")
                           .replace("\\u201c", "")
                           .replace("\\u201d", "")
                           .replace("\\u2026", "")
                           .replace("\\", "")
                           .replace("(", "")
                           .replace(")", "")
            )
            f.write(request_str + "\n")

def main():
    """
    Main pipeline:
      1. Load labeled and unlabeled data.
      2. Process the unlabeled data in batches.
      3. For each batch, generate few-shot prompts using a separate thread.
      4. Convert these prompts into Ollama JSON requests.
      5. Write each JSON request to a file (one per line).
      
    Notes:
      - Future improvements include containerizing Ollama, applying Bennett's patches to the shell script,
        and running multiple Ollama instances in parallel on specified ports.
    """
    for datatype in ["behavior", "stance"]:
        for persona in ["mom", "immunologist", "doctor", "teacher", "robot"]:
            try:
                logger.info("Starting main pipeline.")
                data = load_data("stance")

                batch_size = 100
                num_batches = math.ceil(len(data) / batch_size)

                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size
                    unlabeled_batch = data.iloc[start_idx:end_idx]

                    # Run few-shot prompt generation in a separate thread
                    thread = CustomThread(target=generate_prompt, args=(data, persona,datatype))
                    thread.start()
                    few_shot_prompts = thread.join()

                    # Generate Ollama requests from the prompts
                    ollama_requests = generate_ollama_requests(few_shot_prompts)

                    import os
                    if not os.path.exists("data/requests/stance"):
                        os.makedirs("data/requests/stance")
                    if not os.path.exists("data/requests/behavior"):
                        os.makedirs("data/requests/behavior")
                    # Write the requests to file, one per line in JSON format
                    write_requests_to_file(ollama_requests, f"data/requests/{datatype}/requests_{persona}.json")

                logger.info("Pipeline completed successfully.")
            except Exception as e:
                logger.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()