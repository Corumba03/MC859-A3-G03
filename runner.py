import subprocess
import time
import os
import glob

def main():
    input_dir = "instances"   # Directory where input .txt files are stored
    log_dir = "logs"          # Directory where log files will be written

    # Create logs directory if it doesn't exist, then clean it
    os.makedirs(log_dir, exist_ok=True)
    for f in glob.glob(os.path.join(log_dir, "*")):
        try:
            os.remove(f)
        except Exception as e:
            print(f"Warning: Could not remove {f}: {e}")

    t_out = 60*30 # Time limit for each instance

    # Collect all .txt files from input_dir, sorted alphabetically
    files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))

    # For each input file, run main.py and capture the output in a log file
    for file in files:
        # Extract filename without extension to use in log naming
        base = os.path.basename(file).replace(".txt", "")
        log_file = os.path.join(log_dir, f"{base}.log")
        filename = os.path.splitext(os.path.basename(file))[0]

        print(f"[{time.strftime('%H:%M')}] Running main.py with input {filename}...")

        start = time.time()
        try:
            # Run main.py with the input file piped to stdin
            with open(file, "r") as fin, open(log_file, "w") as fout:
                subprocess.run(
                    ["python", "main.py"],  # Command being executed
                    stdin=fin,              # Input redirected from current .txt file
                    stdout=fout,            # Output redirected to the log file
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    timeout=t_out              # Timeout in seconds (10s here)
                )
            end = time.time()

            # Append execution time to the log
            with open(log_file, "a") as fout:
                fout.write(f"\n--- Finished in {end - start:.2f} seconds ---\n")

            print(f"Finished {filename} in {end - start:.2f} seconds\n")

        except subprocess.TimeoutExpired:
            # Handle timeout case
            with open(log_file, "a") as fout:
                fout.write(f"\n--- Execution timed out after {t_out} seconds ---\n")
            print(f"Timeout: {file}")

        except Exception as e:
            # Handle any other errors
            with open(log_file, "a") as fout:
                fout.write(f"\n--- Error: {e} ---\n")
            print(f"❌ Error running {file}: {e}")

if __name__ == "__main__":
    main()
    print("All files processed. Check the logs directory for outputs.")
