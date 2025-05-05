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
    mo.md(r"""# Mel-Spectrogram""")
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
def _(mo):
    mo.md(r"""## Comparison of Mel-Spectrograms - before and after denoising""")
    return


@app.cell(hide_code=True)
def _(slider_fmax, slider_hop_length, slider_mels, slider_n_fft):
    n_mels = slider_mels.value    # number of mel bands (frequency bins) to use in mel-spectrogram. In short, this is done to divide frequency range into bands that better match human auditory perception.
    fmax = slider_fmax.value    # Frequency cutoff. For speech, most important information is below 8000 Hz, thus 10kHz is reasonable.
    n_fft = slider_n_fft.value # How many samples are analyzed in each frame when converting from time domain to frequency domain
    hop_length = slider_hop_length.value # Number of samples between successive frames. Controls how much the window "slides" between fft calculations. Smaller hop lengths is more frames and smoother spectrograms, but increase computational cost.
    return fmax, hop_length, n_fft, n_mels


@app.cell
def _(
    dn_audio_data,
    fmax,
    hop_length,
    librosa,
    n_fft,
    n_mels,
    np,
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
    return ms_db_dn, ms_db_pre


@app.cell(hide_code=True)
def _(fmax, librosa, ms_db_dn, ms_db_pre, plt, sr_dn, sr_p):
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
    plt.colorbar(format='%+2.0f dB')
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
    plt.colorbar(format='%+2.0f dB')
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
    plt.colorbar(format='%+2.0f dB')
    plt.title("Before")

    plt.subplot(1, 2, 2)
    librosa.display.specshow(
        dn_db,
        x_axis='time',
        y_axis='hz',
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title("After")


    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# MFCCs""")
    return


@app.cell(hide_code=True)
def _(mo):
    slider_nmfccs = mo.ui.slider(start=0, stop=24, step=1,
                               value=13, label="n_mfccs (13)")

    slider_nmfccs
    return (slider_nmfccs,)


@app.cell
def _(slider_nmfccs):
    n_mfcc=slider_nmfccs.value
    return (n_mfcc,)


@app.cell
def _():
    # n_mels = slider_mels.value  
    # fmax = slider_fmax.value    
    # n_fft = slider_n_fft.value 
    # hop_length = slider_hop_length.value
    return


@app.cell(hide_code=True)
def _(dn_audio_data, hop_length, librosa, n_fft, n_mfcc, pre_audio_data, sr_p):
    mfccs_pre = librosa.feature.mfcc(
        y = pre_audio_data,
        sr = sr_p,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    mfccs_dn = librosa.feature.mfcc(
        y = dn_audio_data,
        sr = sr_p,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return mfccs_dn, mfccs_pre


@app.cell(hide_code=True)
def _(mfccs_dn, mfccs_pre, plt):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(mfccs_pre,aspect='auto', origin='lower', cmap='RdBu_r')
    plt.grid(True, linestyle='--', color='black', alpha=0.3)
    plt.title('MFCC - Before denoising')
    plt.xlabel('Time (frame)')
    plt.ylabel('MFCCs')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(mfccs_dn,aspect='auto', origin='lower', cmap='RdBu_r')
    plt.grid(True, linestyle='--', color='black', alpha=0.3)
    plt.title('MFCC - After denoising')
    plt.xlabel('Time (frame)')
    plt.ylabel('MFCCs')
    plt.colorbar()

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Δ & Δ² features
    #### https://librosa.org/doc/main/generated/librosa.feature.delta.html
    """
    )
    return


@app.cell
def _(librosa, mfccs_dn, mfccs_pre):
    delta_pre = librosa.feature.delta(mfccs_pre)
    delta2_pre = librosa.feature.delta(mfccs_pre, order=2)

    delta_dn = librosa.feature.delta(mfccs_dn)
    delta2_dn = librosa.feature.delta(mfccs_dn, order=2)
    return delta2_dn, delta2_pre, delta_dn, delta_pre


@app.cell(hide_code=True)
def _(delta_dn, delta_pre, plt):
    fig_delta, axs_delta = plt.subplots(1, 2, figsize=(14, 5))

    # Delta - Before Denoising
    im_delta_pre = axs_delta[0].imshow(delta_pre, aspect='auto', origin='lower', cmap='RdBu_r')
    axs_delta[0].grid(True, linestyle='--', color='black', alpha=0.3)
    axs_delta[0].set_title('Delta MFCC - Before Denoising')
    axs_delta[0].set_xlabel('Time (frame)')
    axs_delta[0].set_ylabel('MFCC Index')
    fig_delta.colorbar(im_delta_pre, ax=axs_delta[0])

    # Delta - After Denoising
    im_delta_dn = axs_delta[1].imshow(delta_dn, aspect='auto', origin='lower', cmap='RdBu_r')
    axs_delta[1].grid(True, linestyle='--', color='black', alpha=0.3)
    axs_delta[1].set_title('Delta MFCC - After Denoising')
    axs_delta[1].set_xlabel('Time (frame)')
    axs_delta[1].set_ylabel('MFCC Index')
    fig_delta.colorbar(im_delta_dn, ax=axs_delta[1])

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(delta2_dn, delta2_pre, plt):
    fig_delta2, axs_delta2 = plt.subplots(1, 2, figsize=(14, 5))

    # Delta-Delta - Before Denoising
    im_delta2_pre = axs_delta2[0].imshow(delta2_pre, aspect='auto', origin='lower', cmap='RdBu_r')
    axs_delta2[0].grid(True, linestyle='--', color='black', alpha=0.3)
    axs_delta2[0].set_title('Delta-Delta MFCC - Before Denoising')
    axs_delta2[0].set_xlabel('Time (frame)')
    axs_delta2[0].set_ylabel('MFCC Index')
    fig_delta2.colorbar(im_delta2_pre, ax=axs_delta2[0])

    # Delta-Delta - After Denoising
    im_delta2_dn = axs_delta2[1].imshow(delta2_dn, aspect='auto', origin='lower', cmap='RdBu_r')
    axs_delta2[1].grid(True, linestyle='--', color='black', alpha=0.3)
    axs_delta2[1].set_title('Delta-Delta MFCC - After Denoising')
    axs_delta2[1].set_xlabel('Time (frame)')
    axs_delta2[1].set_ylabel('MFCC Index')
    fig_delta2.colorbar(im_delta2_dn, ax=axs_delta2[1])

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""# Spectral features: Centroid & Bandwidth""")
    return


@app.cell
def _(dn_audio_data, librosa, pre_audio_data, sr_dn, sr_p):
    # Spectral Centroid
    spectral_centroid_pre = librosa.feature.spectral_centroid(y=pre_audio_data, sr=sr_p)[0]
    spectral_centroid_dn = librosa.feature.spectral_centroid(y=dn_audio_data, sr=sr_dn)[0]

    # Spectral Bandwidth
    spectral_bandwidth_pre = librosa.feature.spectral_bandwidth(y=pre_audio_data, sr=sr_p)[0]
    spectral_bandwidth_dn = librosa.feature.spectral_bandwidth(y=dn_audio_data, sr=sr_dn)[0]
    return


@app.cell
def _(mo):
    mo.md(r"""# Zero Crossing Rate""")
    return


@app.cell
def _(dn_audio_data, librosa, pre_audio_data):
    # Zero Crossing Rate
    zcr_pre = librosa.feature.zero_crossing_rate(pre_audio_data)[0]
    zcr_dn = librosa.feature.zero_crossing_rate(dn_audio_data)[0]
    return


if __name__ == "__main__":
    app.run()
