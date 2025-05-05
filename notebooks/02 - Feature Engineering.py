import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Imports""")
    return


@app.cell
def _():
    import marimo as mo
    import librosa
    import IPython.display as ipd
    import os
    import random
    import matplotlib.pyplot as plt
    import soundfile as sf
    import ipywidgets as widgets
    import numpy as np
    import warnings

    path_dn = 'D:\\codethings\\GADy\\AudioMINST\\processed_data'
    path_pre = 'D:\\codethings\\GADy\\AudioMINST\\data'
    return (
        ipd,
        librosa,
        mo,
        np,
        path_dn,
        path_pre,
        plt,
        random,
        warnings,
        widgets,
    )


@app.cell
def _(path_dn, path_pre, path_randomizer):
    Current = path_randomizer()
    random_pre = path_pre + Current
    random_dn = path_dn + Current
    return random_dn, random_pre


@app.cell
def _(random):
    def path_randomizer():
        # We got 60 speakers here
        randomized_speaker = str(random.randint(1,60))
        if int(randomized_speaker) < 10:
            randomized_speaker = '0' + str(randomized_speaker)

        # Digits
        random_digit = random.randint(0,9)

        # Each number is repeated 50 times
        repeat_num = random.randint(0,49)

        # Concat, get random audio file
        random_file = '\\' + str(randomized_speaker) + '\\' + str(random_digit) + '_' + str(randomized_speaker) + '_' + str(repeat_num) + '.wav'

        random_file_path ='\\' + str(randomized_speaker) + '\\' + str(random_digit) + '_' + str(randomized_speaker) + '_' + str(repeat_num) + '.wav'
        return random_file_path

    print(path_randomizer())
    return (path_randomizer,)


@app.cell
def _(ipd):
    def button_display(output, file_path):
        with output:
            ipd.clear_output(wait=True)
            ipd.display(ipd.Audio(file_path))
    return (button_display,)


@app.cell
def _(button_display, ipd, random_dn, random_pre, widgets):
    # Functions to call ipd audio players

    def pre_player():
        # "Output buffer" widget
        o_audio_pre = widgets.Output()
        # ipd.display(o_audio_pre)
        ipd.display(ipd.Markdown("Audio Before"))
        button_display(o_audio_pre, random_pre)

    def dn_player():
        # "Output buffer" widget
        o_audio_dn = widgets.Output()
        # ipd.display(o_audio_dn)
        ipd.display(ipd.Markdown("Audio After"))
        button_display(o_audio_dn, random_dn)
    return dn_player, pre_player


@app.cell(hide_code=True)
def _(pre_player):
    pre_player()
    return


@app.cell(hide_code=True)
def _(dn_player):
    dn_player()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# De-noised audio experiments""")
    return


@app.cell
def _(librosa, random_dn, random_pre):
    pre_audio_data, sr_p = librosa.load(random_pre, sr=None)
    dn_audio_data, sr_dn = librosa.load(random_dn, sr=None)
    return dn_audio_data, pre_audio_data, sr_dn, sr_p


@app.cell
def _(dn_audio_data, librosa, pre_audio_data):
    pre_stft = librosa.stft(pre_audio_data)
    pre_db = librosa.amplitude_to_db(abs(pre_stft))

    dn_stft = librosa.stft(dn_audio_data)
    dn_db = librosa.amplitude_to_db(abs(dn_stft))
    return dn_db, pre_db


@app.cell(hide_code=True)
def _(dn_db, librosa, plt, pre_db):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    librosa.display.specshow(
        pre_db,
        x_axis='time',
        y_axis='hz',
    )
    plt.colorbar()
    plt.title("Before")

    plt.subplot(1, 2, 2)
    librosa.display.specshow(
        dn_db,
        x_axis='time',
        y_axis='hz',
    )
    plt.colorbar()
    plt.title("After")


    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Theory
    ## Key feature extraction methods for spoken digits:
    -  Mel-Spectrogram
    -  MFCCs (Mel-Frequency Cepstral Coefficients)
    -  Delta and Delta-Delta Features
    ## Data Augmentation:
    - Time stretching
    - Pitch shifting
    - Background noise injection
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Mel-Spectrogram""")
    return


@app.cell(hide_code=True)
def _(mo, slider_fmax, slider_hop_length, slider_mels, slider_n_fft):
    mo.md(f"""n_mels value: {slider_mels.value} <br> fmax value: {slider_fmax.value} <br> n_fft value: {slider_n_fft.value} <br> hop_length value: {slider_hop_length.value}""")
    return


@app.cell(hide_code=True)
def _(mo):
    slider_mels = mo.ui.slider(start=0, stop=512, step=16,
                               value=128, label="n_mels (128)")
    slider_fmax = mo.ui.slider(start=0, stop=16000, step=100,
                               value=10000, label="fmax (10k)")
    slider_n_fft = mo.ui.slider(start=0, stop=2048, step=128,
                               value=2048, label="n_fft (2048)")
    slider_hop_length = mo.ui.slider(start=0, stop=512, step=32,
                               value=512, label="hop_length (512)")
    slider_mels, slider_fmax, slider_n_fft, slider_hop_length
    return slider_fmax, slider_hop_length, slider_mels, slider_n_fft


@app.cell(hide_code=True)
def _(slider_fmax, slider_hop_length, slider_mels, slider_n_fft):
    n_mels = slider_mels.value    # number of mel bands (frequency bins) to use in mel-spectrogram. In short, this is done to divide frequency range into bands that better match human auditory perception.
    fmax = slider_fmax.value    # Frequency cutoff. For speech, most important information is below 8000 Hz, thus 10kHz is reasonable.
    n_fft = slider_n_fft.value # How many samples are analyzed in each frame when converting from time domain to frequency domain
    hop_length = slider_hop_length.value # Number of samples between successive frames. Controls how much the window "slides" between fft calculations. Smaller hop lengths is more frames and smoother spectrograms, but increase computational cost.
    return fmax, hop_length, n_fft, n_mels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Comparison of Mel-Spectrograms - before and after denoising""")
    return


@app.cell(hide_code=True)
def _(
    dn_audio_data,
    fmax,
    hop_length,
    librosa,
    n_fft,
    n_mels,
    np,
    plt,
    pre_audio_data,
    sr_dn,
    sr_p,
    warnings,
):
    ### Before vs After ###
    warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

    ## Before ##
    melspec_pre = librosa.feature.melspectrogram(
        y=pre_audio_data,
        sr=sr_p,
        n_mels=n_mels,
        fmax=fmax,
        n_fft=n_fft,
        hop_length=hop_length
    )

    ms_db_pre = librosa.power_to_db(melspec_pre, ref=np.max)

    ## After ##
    melspec_dn = librosa.feature.melspectrogram(
        y=dn_audio_data,
        sr=sr_dn,
        n_mels=n_mels,
        fmax=fmax,
        n_fft=n_fft,
        hop_length=hop_length
    )

    ms_db_dn = librosa.power_to_db(melspec_dn, ref=np.max)

    ## Plot it
    plt.figure(figsize=(14, 5))
    # Before #
    plt.subplot(1, 2, 1)
    librosa.display.specshow(
        ms_db_pre,
        sr=sr_p,
        x_axis="time",
        y_axis="mel",
        fmax=fmax
    )
    plt.colorbar()
    plt.title("Before")
    # After #
    plt.subplot(1, 2, 2)
    librosa.display.specshow(
        ms_db_dn,
        sr=sr_dn,
        x_axis="time",
        y_axis="mel",
        fmax=fmax
    )
    plt.colorbar()
    plt.title("After")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Again, for comparison, normal spectrograms:""")
    return


@app.cell(hide_code=True)
def _(dn_db, librosa, plt, pre_db):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    librosa.display.specshow(
        pre_db,
        x_axis='time',
        y_axis='hz',
    )
    plt.colorbar()
    plt.title("Before")

    plt.subplot(1, 2, 2)
    librosa.display.specshow(
        dn_db,
        x_axis='time',
        y_axis='hz',
    )
    plt.colorbar()
    plt.title("After")


    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
