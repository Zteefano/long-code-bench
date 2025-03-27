import os
import requests
import time

# Optionally set your GitHub personal access token as an environment variable GITHUB_TOKEN.
# This helps avoid rate limits for many API calls.
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
headers = {"Accept": "application/vnd.github.v3+json"}
if GITHUB_TOKEN:
    headers["Authorization"] = f"token {GITHUB_TOKEN}"

# Read the list of repositories from the file "repo_32K"
# Each line in the file should be in the format "owner/repo"
with open("repo_128K.txt", "r") as file:
    repos = [line.strip() for line in file if line.strip()]

num_files = 0

for repo in repos:
    try:
        owner, repo_name = repo.split("/")
    except ValueError:
        print(f"Skipping invalid repository entry: {repo}")
        continue

    # Use the GitHub API to get the full tree of the repository (recursive)
    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/HEAD?recursive=1"


    # Check if there is timeout error, if yes, wait a minute and try again until it works
    while True:
        try:
            response = requests.get(api_url, headers=headers, timeout=5)
            break
        except requests.exceptions.Timeout:
            print(f"Timeout error for {repo}, waiting 60 seconds...")
            time.sleep(60)


    if response.status_code != 200:
        print(f"Error fetching {repo}: {response.status_code} - {response.json().get('message')}")
        continue

    data = response.json()
    tree = data.get("tree", [])
    file_count = 0

    # Iterate through the tree and count files (blobs)
    for item in tree:
        if item["type"] == "blob":  # file
            # Exclude files in a top-level "tests" directory
            if item["path"].startswith("tests/"):
                continue
            # Optionally exclude any file with '.git' in its path (usually unnecessary)
            if ".git" in item["path"]:
                continue
            file_count += 1
    
    num_files += file_count

    print(f"{repo}: {file_count} files")

print(f"Total files in all repositories: {num_files}")
print(f"Total repositories: {len(repos)}")
print(f"Average files per repository: {num_files / len(repos)}")


