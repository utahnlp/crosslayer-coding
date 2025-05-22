import requests
import json

url = "http://34.41.125.189:8000/datasets/EleutherAI%2Fpythia-70m%2Fpile-uncopyrighted_train/norm_stats"
output_filename = "norm_stats_downloaded.json"

try:
    print(f"Attempting to download from {url}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
    data = response.json()
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Successfully downloaded and saved to {output_filename}")
except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
except requests.exceptions.ConnectionError as conn_err:
    print(f"Connection error occurred: {conn_err}")
except requests.exceptions.Timeout as timeout_err:
    print(f"Timeout error occurred: {timeout_err}")
except requests.exceptions.RequestException as req_err:
    print(f"An error occurred during the request: {req_err}")
except json.JSONDecodeError:
    print("Failed to decode JSON from the response. The content might not be valid JSON.")
    # Optionally, save the raw content for inspection
    with open("norm_stats_raw_content.txt", "w") as f:
        f.write(response.text)
    print("Raw response content saved to norm_stats_raw_content.txt")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
