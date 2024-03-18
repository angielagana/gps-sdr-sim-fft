import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", type=str, help="Path to the first signal file")
    parser.add_argument("--file2", type=str, help="Path to the second signal file")
    return parser.parse_args()

# Load signal data from gpssim.bin file
def load_signal_data(file_path, bits=16):
    try:
        if bits == 16:
            # Load data from the binary file as shorts
            data = np.fromfile(file_path, dtype=np.int16)
        else:
            # Load data from the binary file as bytes
            data = np.fromfile(file_path, dtype=np.uint8)

        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit()

# Convert interleaved shorts to complex values
def interleaved_shorts_to_complex(data):
    data_complex = data.astype(np.float32) / 32768.0
    data_complex = data_complex.view(np.complex64)

    return data_complex

# Perform FFT analysis on the signal
def perform_fft(signal, sampling_rate):
    signal_fft = np.fft.fft(signal)
    
    # Calculate the frequency axis
    n = len(signal)
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    
    return freq, signal_fft

def main():
    args = parse_args()

    file_path1 = args.file1
    file_path2 = args.file2
    
    # Load signal data
    bits = 16  # Assuming 16-bit data, change if necessary
    signal_data1 = load_signal_data(file_path1, bits)
    signal_data2 = load_signal_data(file_path2, bits)
    
    # Convert interleaved shorts to complex values
    signal_complex1 = interleaved_shorts_to_complex(signal_data1)
    signal_complex2 = interleaved_shorts_to_complex(signal_data2)
    
    # Sampling rate (replace with the actual sampling rate used)
    sampling_rate = 2.5e6  
    
    # Perform FFT analysis
    freq1, signal_fft1 = perform_fft(signal_complex1, sampling_rate)
    freq2, signal_fft2 = perform_fft(signal_complex2, sampling_rate)
    
    plt.figure(figsize=(10, 6))
    plt.plot(freq1, np.abs(signal_fft1), label='File 1')
    plt.plot(freq2, np.abs(signal_fft2), label='File 2')
    plt.title('Signal Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

