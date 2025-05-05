import marimo

__generated_with = "0.11.17"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Folder/file structure
        `data/[speaker]/[number]_[speaker]_[repeats].wav`
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Play random file""")
    return


@app.cell
def _():
    import librosa
    import IPython.display as ipd
    import os
    import random
    import matplotlib.pyplot as plt
    import soundfile as sf

    mainpath = 'D:\\codethings\\GADy\\AudioMINST\\data'
    testdir = 'D:\\codethings\\GADy\\AudioMINST\\Test'

    # We got 60 speakers here
    randomized_speaker = str(random.randint(1,60))
    if int(randomized_speaker) < 10:
        randomized_speaker = '0' + str(randomized_speaker)

    # Digits
    random_digit = random.randint(0,9)

    # Each number is repeated 50 times
    repeat_num = random.randint(0,49)

    # Concat, get random audio file
    random_file = mainpath + '\\' + str(randomized_speaker) + '\\' + str(random_digit) + '_' + str(randomized_speaker) + '_' + str(repeat_num) + '.wav'

    print(random_file)
    print(f'Path exists? : {os.path.exists(random_file)}' + '\n')

    audio_data, sr = librosa.load(random_file, sr = 16000)

    print(sf.info(random_file))
    print('\n')

    ipd.Audio(random_file)
    return (
        audio_data,
        ipd,
        librosa,
        mainpath,
        os,
        plt,
        random,
        random_digit,
        random_file,
        randomized_speaker,
        repeat_num,
        sf,
        sr,
        testdir,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Plot comparison between 16kHz vs default librosa 22050 Hz""")
    return


@app.cell
def _(audio_data, librosa, plt):
    plt.figure(figsize=(14, 5))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio_data, sr=22050, color='blue')
    plt.title('22050 Hz audio')

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(audio_data, sr=16000, color='green')
    plt.title('16000 Hz audio')

    plt.tight_layout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Różnice w długościach próbek - to nie jest resample, to plt tak interpretuje "sr = ..." 
        #### Aby sprawdzić jaki sr ma plik, używamy soundfile.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Pora na spectrogramy""")
    return


@app.cell
def _(audio_data, librosa, plt, sr):
    amplitude = librosa.stft(audio_data)
    decibels = librosa.amplitude_to_db(abs(amplitude))

    librosa.display.specshow(
        decibels, 
        sr=sr, 
        x_axis='time',
        y_axis='hz'
    )
    plt.colorbar()
    return amplitude, decibels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Testowanie noisereduce (default)""")
    return


@app.cell
def _(audio_data, librosa, plt, sf, sr, testdir):
    import noisereduce as nr
    _reduced_noise = nr.reduce_noise(audio_data, sr)
    dir_var = testdir + '\\' + 'test.wav'
    print(dir_var)
    sf.write(dir_var, _reduced_noise, sr)
    reduced_amp = librosa.stft(_reduced_noise)
    reduced_db = librosa.amplitude_to_db(abs(reduced_amp))
    librosa.display.specshow(reduced_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    return dir_var, nr, reduced_amp, reduced_db


@app.cell
def _(dir_var, ipd):
    ipd.Audio(dir_var)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Testowanie noisereduce (z samplem szumu)""")
    return


@app.cell
def _(audio_data, ipd, nr, sf, sr, testdir):
    sample_qty = int(sr * 60 / 1000)
    noise = audio_data[:sample_qty]
    _reduced_noise = nr.reduce_noise(audio_data, sr=sr, y_noise=noise)
    noisereduce_wsample = testdir + '\\' + 'noisereduce_wsample.wav'
    sf.write(noisereduce_wsample, _reduced_noise, sr)
    ipd.Audio(noisereduce_wsample)
    return noise, noisereduce_wsample, sample_qty


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Testowanie scipy.signal.wiener()""")
    return


@app.cell
def _(audio_data, librosa, plt, sf, sr, testdir):
    import scipy
    filtered = scipy.signal.wiener(audio_data)
    wiener_testing = dir_var_1 = testdir + '\\' + 'wiener.wav'
    sf.write(wiener_testing, filtered, sr)
    amp_wiener = librosa.stft(filtered)
    db_wiener = librosa.amplitude_to_db(abs(amp_wiener))
    librosa.display.specshow(db_wiener, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    return amp_wiener, db_wiener, dir_var_1, filtered, scipy, wiener_testing


@app.cell
def _(ipd, wiener_testing):
    ipd.Audio(wiener_testing)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Sprawdzanie spektrogramu
        ### resample->wiener
        """
    )
    return


@app.cell
def _(audio_data, ipd, librosa, random_file, scipy, sf, sr, testdir):
    # print(random_file.split('\\')[-1])
    resampled_denoised_filename = 'wiener_rd_' + random_file.split('\\')[-1]
    print(resampled_denoised_filename)
    wiener_test= testdir + '\\' + resampled_denoised_filename

    resample_wiener = librosa.resample(y=audio_data, target_sr=16000, orig_sr=sr)
    test_wiener = scipy.signal.wiener(resample_wiener)
    sf.write(wiener_test, test_wiener, 16000)
    ipd.Audio(wiener_test)
    return (
        resample_wiener,
        resampled_denoised_filename,
        test_wiener,
        wiener_test,
    )


@app.cell
def _(librosa, plt, sr, test_wiener):
    amp_test = librosa.stft(test_wiener)
    db_test = librosa.amplitude_to_db(abs(amp_test))
    librosa.display.specshow(
        db_test,
        sr=sr,
        x_axis='time',
        y_axis='hz'
    )
    plt.colorbar()
    return amp_test, db_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Sprawdzanie spektrogramu
        ### resample->noisereduce
        """
    )
    return


@app.cell
def _(audio_data, ipd, librosa, nr, random_file, sf, sr, testdir):
    test_noisereduce = 'nr_rd_' + random_file.split('\\')[-1]
    noisereduce_path = testdir + '\\' + test_noisereduce


    resample_noisereduce = librosa.resample(y=audio_data, target_sr=16000, orig_sr=sr)
    clean_noisereduce = nr.reduce_noise(resample_noisereduce, sr=16000)
    sf.write(noisereduce_path, clean_noisereduce, 16000)
    print(noisereduce_path)
    ipd.Audio(noisereduce_path)
    return (
        clean_noisereduce,
        noisereduce_path,
        resample_noisereduce,
        test_noisereduce,
    )


@app.cell
def _(clean_noisereduce, db_test, librosa, plt, sr):
    amp_nr = librosa.stft(clean_noisereduce)
    db_nr = librosa.amplitude_to_db(abs(clean_noisereduce))
    librosa.display.specshow(
        db_test,
        sr=sr,
        x_axis='time',
        y_axis='hz'
    )
    plt.colorbar()
    return amp_nr, db_nr


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
