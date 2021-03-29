import os
import numpy as np
import matplotlib.pyplot as plt
import shutil


def analyze_feature_distribution():
    full_tempogram = np.load('full_tempogram.npz.npy', allow_pickle=True)

    plt.hist(full_tempogram, 100, facecolor='blue', alpha=0.5)
    plt.show()

    full_mfcc = np.load('full_mfcc.npz.npy', allow_pickle=True)
    plt.hist(full_mfcc, 100, facecolor='blue', alpha=0.5)
    plt.show()

    full_mel_spectrogram = np.load('full_mel_spectrogram.npz.npy', allow_pickle=True)
    plt.hist(full_mel_spectrogram, 100, facecolor='blue', alpha=0.5)
    plt.show()

    full_cqt = np.load('full_cqt.npz.npy', allow_pickle=True)
    plt.hist(full_cqt, 100, facecolor='blue', alpha=0.5)
    plt.show()

    feature_dict = {
        'spectral_centroid': [0, 1],
        'spectral_bandwidth': [1, 2],
        'onset_envelope': [3, 4],
        'pulse': [3, 4],
        'pulse_lognorm': [4, 5],
        'dtempo': [5, 6],
        'tempogram': [6, 390],
        'mfcc': [390, 410],
        'mel_spectrogram': [410, 538],
        'cqt': [538, 622]
    }

    dataset_dir = 'D:\conductor-dataset\dataset\\'
    name_list = os.listdir(dataset_dir)

    full_tempogram = []
    full_mfcc = []
    full_mel_spectrogram = []
    full_cqt = []
    for i, name in enumerate(name_list):

        feature_npy = dataset_dir + name + '\\' + 'audiofeature-30fps.npy'
        feature = np.load(feature_npy, allow_pickle=True)
        print(i, name, np.shape(feature))

        tempogram = feature[feature_dict['tempogram'][0]:feature_dict['tempogram'][1], :]
        full_tempogram.extend(np.reshape(tempogram, [-1]).tolist())

        mfcc = feature[feature_dict['mfcc'][0]:feature_dict['mfcc'][1], :]
        full_mfcc.extend(np.reshape(mfcc, [-1]).tolist())

        mel_spectrogram = feature[feature_dict['mel_spectrogram'][0]:feature_dict['mel_spectrogram'][1], :]
        full_mel_spectrogram.extend(np.reshape(mel_spectrogram, [-1]).tolist())

        cqt = feature[feature_dict['cqt'][0]:feature_dict['cqt'][1], :]
        full_cqt.extend(np.reshape(cqt, [-1]).tolist())

        if i % 10 == 0:
            np.save('full_tempogram.npz', full_tempogram)
            np.save('full_mfcc.npz', full_mfcc)
            np.save('full_mel_spectrogram.npz', full_mel_spectrogram)
            np.save('full_cqt.npz', full_cqt)

    np.save('full_tempogram.npz', full_tempogram)
    np.save('full_mfcc.npz', full_mfcc)
    np.save('full_mel_spectrogram.npz', full_mel_spectrogram)
    np.save('full_cqt.npz', full_cqt)


source_dir = 'D:\\conductor-dataset\\dataset\\'
dst_dir = 'small_dataset\\'

namelist = os.listdir(source_dir)
copy_list = ['high-pass-results.npy', 'low-pass-results.npy', 'audiofeature-30fps.npy']
for i in range(len(namelist)):

    name = namelist[i]
    source_floder = source_dir + name + '\\'
    dst_floder = dst_dir + name + '\\'

    for item in copy_list:
        print(i, '【from】', source_floder + item, '【to】', dst_floder + item)
        shutil.copyfile(source_floder + item, dst_floder + item)
