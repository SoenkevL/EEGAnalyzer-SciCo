import subprocess
import multiprocessing
import mne
import argparse

def plot_eeg(eeg_file):
    # Read EEG data
    try:
        raw = mne.io.read_raw(eeg_file)
    except FileNotFoundError:
        print(f"Error: File '{eeg_file}' not found")
        exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)
    # Plot EEG data
    raw.plot(block=True)
# Set up argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot EEG data')
    parser.add_argument('eeg_file', help='Path to EEG file')
    args = parser.parse_args()
    eeg_file = args.eeg_file
    # eeg_file = '/home/soenkevanloh/Documents/EEGAnalyzer/example/eeg/PN001-preprocessed-raw.fif'
    # subprocess.run(["python3", "display_eeg.py", f"{eeg_file}"])
    p1 = multiprocessing.Process(target=plot_eeg, args=(eeg_file,))
    p1.start()
    print('This is still running despite the eeg')
    p1.join()