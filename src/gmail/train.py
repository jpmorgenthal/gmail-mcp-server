import argparse
import json
import requests

def read_training_data(file_path: str) -> list[dict]:
    """
    Reads training data from a JSON file.

    Args:
        file_path (str): Path to the training data file.

    Returns:
        list[dict]: A list of training data entries.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data.get("training_data", [])
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON in file '{file_path}'.")
        return []

def post_to_url(url: str, entry: dict) -> dict:
    """
    Sends a POST request to the specified URL with the given content.

    Args:
        url (str): The URL to send the POST request to.
        content (dict): The content to include in the POST request.

    Returns:
        dict: The response from the server.
    """


    prompt = "The following is to be used for training the model. " \
            "You were asked to provide an analysis of email content and provide a one word classification " \
            "The content is a JSON object with the one or more of the following elements: " \
            "input - provides the data you were asked to evaluate "\
            "label - the response you provided "\
            "evaluation - what you should have provided or letting you know your evaluation was correct " \
            
    content = {
            "model": "ai/deepseek-r1-distill-llama:8B-Q4_K_M",
            "messages": [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": "{'input':" + f"{entry.get('input')}" + "}"
                },
                {
                    "role": "user",
                    "content": "{'evaluation':"+ f"{entry.get('evaluation')}" + "}"
                }
            ],
            "stream": False,
        }
    print(f"Posting content: {content}")
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=content, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error posting to {url}: {e}")
        return {}

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process training data from a file.")
    parser.add_argument(
        "--training-file",
        required=True,
        help="Path to the JSON file containing training data."
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Read training data from the file
    training_data = read_training_data(args.training_file)

    # Define the URL for the POST request
    url = "http://localhost:9001/engines/llama.cpp/v1/chat/completions"

    # Iterate over the training data and send POST requests
    for entry in training_data:
#        print(f"Posting content: {entry}")
        response = post_to_url(url, entry)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()