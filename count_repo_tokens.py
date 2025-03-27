import os
import argparse
import tiktoken
from src.long_code_bench.data.count_tokens import count_repo_tokens

def main():
    
    reponame = "psf/requests"

    token_count = count_repo_tokens(reponame)
    print(f"The repository '{reponame}' has {token_count} tokens.")

if __name__ == "__main__":
    main()
