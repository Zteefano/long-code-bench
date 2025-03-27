#!/usr/bin/env python3
import csv
import re
import time
import requests
from bs4 import BeautifulSoup
from src.long_code_bench.data.count_tokens import count_repo_tokens

def get_github_repo_from_pypi(pypi_url: str) -> str | None:
    """
    Given a PyPI package URL, fetch the page and extract the GitHub repository ID
    (in the format 'user/reponame') from any link that points to github.com.
    """
    try:
        response = requests.get(pypi_url)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {pypi_url}: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    for a in soup.find_all('a', href=True):
        href = a['href']
        if "github.com" in href:
            match = re.search(r"github\.com/([^/]+/[^/]+)", href)
            if match:
                return match.group(1)
    return None

def main():
    csv_file = "/Users/romanil/Work/top-pypi-packages.csv"
    output_file = "repo_tokens_2.txt"
    start_offset = 0  # Skip the first 200 processed repos.
    max_repos = 100     # Process the next 500 repositories.

    # Do not clear the output file (since you already have the first 200)
    with open(output_file, "a") as f:
        pass

    processed = 0
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx < start_offset:
                continue

            if processed >= max_repos:
                print(f"Reached the limit of {max_repos} new repositories. Stopping further processing.")
                break

            project_name = row.get("project") or row.get("project name")
            if not project_name:
                print("Skipping row without project name:", row)
                continue

            pypi_url = f"https://pypi.org/project/{project_name}/"
            print(f"Processing {pypi_url} ...")

            repo_id = get_github_repo_from_pypi(pypi_url)
            if repo_id:
                print(f"  Found GitHub repo: {repo_id}")
                try:
                    token_count = count_repo_tokens(repo_id)
                    line = f"{repo_id}, {token_count}"
                    print(f"  Token count: {token_count}")
                    # Append the result after processing each package.
                    with open(output_file, "a") as out:
                        out.write(line + "\n")
                    processed += 1
                except Exception as e:
                    print(f"  Error processing repo {repo_id}: {e}")
            else:
                print("  No GitHub repository found for this package.")

            # Pause briefly between requests.
            time.sleep(1)

if __name__ == "__main__":
    main()
