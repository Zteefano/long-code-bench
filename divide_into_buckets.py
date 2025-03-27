#!/usr/bin/env python3
def main():
    input_file = "repo_tokens.txt"
    
    # Define the output files and corresponding token count ranges.
    # The ranges are inclusive.
    output_configs = {
        "repo_32K.txt": (0, 32767),
        "repo_64K.txt": (32768, 65535),
        "repo_128K.txt": (65536, 131071),
        "repo_256K.txt": (131072, 262143),
        "repo_512K.txt": (262144, 524287),
        "repo_1M.txt": (524288, 1048575),
    }
    
    # Clear output files (create empty files)
    for filename in output_configs:
        with open(filename, "w") as f:
            pass  # just to clear the file

    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Each line should be in the format: "user/reponame, token_count"
            parts = line.split(",")
            if len(parts) < 2:
                print(f"Skipping invalid line: {line}")
                continue
            
            repo_id = parts[0].strip()
            try:
                token_count = int(parts[1].strip())
            except ValueError:
                print(f"Invalid token count for line: {line}")
                continue

            # Check which range the token_count falls into.
            for filename, (low, high) in output_configs.items():
                if low <= token_count <= high:
                    with open(filename, "a") as out:
                        out.write(repo_id + "\n")
                    # If a repo should only appear in one file, break after writing.
                    break

if __name__ == "__main__":
    main()
