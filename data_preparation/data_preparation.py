import librosa
import librosa.display
import os
import soundfile as sf
import noisereduce as nr

def load_audio_file(file_path):
    audio_data, sampling_rate = librosa.load(file_path, sr=None)
    return audio_data, sampling_rate

def process_audio_file(input_path, output_path, target_sr=16000):
    # Wczytanie pliku audio
    audio_data, sampling_rate = load_audio_file(input_path)
    
    # Przpróbkowanie
    resampled_audio = librosa.resample(
        audio_data, 
        orig_sr=sampling_rate,
        target_sr=target_sr)
    
    # Odszumienie
    denoised_audio = nr.reduce_noise(y=resampled_audio, sr=target_sr)
    
    # Stworzenie folderu docelowego, jeśli jeszcze nie istnieje
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Zapisz przetworzone audio do pliku
    sf.write(output_path, denoised_audio, target_sr)
    
    return denoised_audio, target_sr

def process_all_audio_files(input_root_dir, output_root_dir):
    # Stworzenie folderu docelowego, jeśli jeszcze nie istnieje
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
    
    # Walk through the input directory structure
    for root, dirs, files in os.walk(input_root_dir):
        relative_path = os.path.relpath(root, input_root_dir)
        
        # pomijamy roota
        if relative_path == '.':
            continue
        
        # utorzenie analogicznego folderu
        output_dir = os.path.join(output_root_dir, relative_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # przechodzimy przez pliki audio
        for file in files:
            if file.endswith('.wav'):
                input_file_path = os.path.join(root, file)
                
                # plik wyjściowy
                base_name = os.path.splitext(file)[0]
                output_file = f"{base_name}_processed.wav"
                output_file_path = os.path.join(output_dir, output_file)
                
                # przetwarzanie pliku audio
                process_audio_file(input_file_path, output_file_path)

if __name__ == "__main__":
    main_directory = os.path.dirname(os.getcwd())
    
    # folder z danymi wejściowymi
    input_data_directory = os.path.join(main_directory, "data", "AudioMNIST", "data")
    # folder na dany wejściowe
    output_data_directory = os.path.join(main_directory, "data", "AudioMNIST", "processed")
    
    # Przetwarzanie wszystkich plików audio
    print("Processing...")
    process_all_audio_files(input_data_directory, output_data_directory)
    print("Complete!")