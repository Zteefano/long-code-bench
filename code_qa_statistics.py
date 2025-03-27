import os
import requests
import time

# Optional: set your GitHub personal access token to avoid rate limits.
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
headers = {"Accept": "application/vnd.github.v3+json"}
if GITHUB_TOKEN:
    headers["Authorization"] = f"token {GITHUB_TOKEN}"

# List of bucket files.
bucket_files = [
    "repo_32K.txt",
    "repo_64K.txt",
    "repo_128K.txt",
    "repo_256K.txt",
    "repo_512K.txt",
    "repo_1M.txt"
]

# Global dictionary to store file counts for each repo.
# Format: repo (str) -> file_count (int)
global_repo_file_counts = {}

# Dictionary to store stats per bucket.
# Format: bucket filename -> list of file counts for that bucket.
bucket_file_counts = {}

def count_files_for_repo(repo_full_name):
    """
    Given a repository full name (owner/repo),
    use the GitHub API to get the recursive file tree from HEAD
    and count the number of files (blobs), excluding:
       - files in a top-level "tests" directory (path starts with "tests/")
       - files with ".git" in the path.
    Returns the file count (int) or None on error.
    """
    owner_repo = repo_full_name.strip()
    try:
        owner, repo_name = owner_repo.split("/")
    except ValueError:
        print(f"Invalid repo format: {repo_full_name}")
        return None

    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/HEAD?recursive=1"
    try:
        response = requests.get(api_url, headers=headers)
    except Exception as e:
        print(f"Error calling API for {repo_full_name}: {e}")
        return None

    if response.status_code != 200:
        try:
            error_message = response.json().get("message", "")
        except Exception:
            error_message = response.text
        print(f"Error fetching {repo_full_name}: {response.status_code} - {error_message}")
        return None

    data = response.json()
    tree = data.get("tree", [])
    file_count = 0
    for item in tree:
        if item.get("type") == "blob":  # count only files
            path = item.get("path", "")
            # Exclude files that are in a top-level "tests" directory.
            if path.startswith("tests/"):
                continue
            # Exclude any file with ".git" in its path.
            if ".git" in path:
                continue
            file_count += 1

    return file_count

# Process each bucket file.
for bucket in bucket_files:
    if not os.path.exists(bucket):
        print(f"Bucket file {bucket} does not exist. Skipping.")
        continue

    bucket_counts = []  # list to store file counts for repos in this bucket

    with open(bucket, "r") as f:
        repos = [line.strip() for line in f if line.strip()]

    print(f"Processing bucket file {bucket} with {len(repos)} repositories...")
    for repo in repos:
        # If already processed, reuse the file count.
        if repo in global_repo_file_counts:
            file_count = global_repo_file_counts[repo]
        else:
            file_count = count_files_for_repo(repo)
            # To avoid hitting rate limits, add a slight delay (adjust as needed)
            time.sleep(0.5)
            if file_count is None:
                print(f"Could not count files for {repo}, skipping.")
                continue
            global_repo_file_counts[repo] = file_count

        bucket_counts.append(file_count)
        print(f"{repo}: {file_count} files")
    bucket_file_counts[bucket] = bucket_counts

print("\nAverage number of files per bucket:")
for bucket, counts in bucket_file_counts.items():
    if counts:
        avg = sum(counts) / len(counts)
        print(f"{bucket}: {avg:.2f} files on average over {len(counts)} repos")
    else:
        print(f"{bucket}: No valid repositories processed.")

# Now, process the repo_tokens.txt file.
repo_tokens = {}
tokens_file = "repo_tokens.txt"
if os.path.exists(tokens_file):
    with open(tokens_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                repo, token_str = line.split(",")
                repo = repo.strip()
                tokens = int(token_str.strip())
                repo_tokens[repo] = tokens
            except Exception as e:
                print(f"Error parsing line in {tokens_file}: '{line}'. Error: {e}")
else:
    print(f"{tokens_file} not found.")

# Calculate overall average tokens per file for all repos present in the buckets.
total_tokens = 0
total_files = 0
for repo, file_count in global_repo_file_counts.items():
    if repo in repo_tokens:
        total_tokens += repo_tokens[repo]
        total_files += file_count
    else:
        print(f"Repo {repo} not found in {tokens_file}; skipping tokens for this repo.")

if total_files > 0:
    avg_tokens_per_file = total_tokens / total_files
    print(f"\nOverall average tokens per file (from all bucket repos): {avg_tokens_per_file:.2f}")
else:
    print("No files counted; cannot compute tokens per file.")
