import os
import csv

import mir_eval as me
import numpy as np
import matplotlib.pyplot as plt
import medleydb as mdb

import predict.predict_on_audio as pred

class Predict_args:
    def __init__(self, audio_fpath, task, save_dir, output_format, threshold, use_neg):
        self.audio_fpath = audio_fpath
        self.task = task
        self.save_dir = save_dir
        self.output_format = output_format
        self.threshold = threshold
        self.use_neg = use_neg


def predict_mcgill(mcgill_file='data/mcGillData.csv'):
    filenames, inst_tokens, inst_descs, notenames, notenames2, true_f0s, midis, paths, file_tokens = load_mcgill_data(
        mcgill_file)
    for path in paths:
        p_args = Predict_args(path, 'melody1', 'data/dsmcgillpredictions', 'singlef0', 0.3, True)
        pred.main(p_args)


def load_mcgill_data(mcgill_file):
    filenames = []
    inst_tokens = []
    inst_descs = []
    notenames = []
    notenames2 = []
    true_f0s = []
    midis = []
    paths = []
    file_tokens = []
    with open(mcgill_file, newline='') as csvfile:
        print('reading', mcgill_file)
        csvreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in csvreader:
            filename = row['filename']
            instrument_token = row['instrumentToken']
            instrument_desc = row['instrumentDesc']
            notename = row['notename']
            notename2 = row['notename2']
            true_f0 = row['trueF0']
            midi = row['midi']
            path = row['path']
            file_token = row['fileToken']

            filenames.append(filename)
            inst_tokens.append(instrument_token)
            inst_descs.append(instrument_desc)
            notenames.append(notename)
            notenames2.append(notename2)
            true_f0s.append(true_f0)
            midis.append(midi)
            paths.append(path)
            file_tokens.append(file_token)

    return filenames, inst_tokens, inst_descs, notenames, notenames2, true_f0s, midis, paths, file_tokens


def predict_mdb():
    tracklist = mdb.load_all_multitracks()
    for mtrack in tracklist:
        title = mtrack.track_id
        if mtrack.has_melody:
            p_args = Predict_args(mtrack.mix_path, 'melody1', 'data/predictions2', 'singlef0', 0.3, True)
            pred.main(p_args)
        else:
            print('No melody for', title)


def predict_file(audio_fpath, output_dir):
    p_args = Predict_args(audio_fpath, 'melody1', output_dir, 'singlef0', 0.3, True)
    pred.main(p_args)


if __name__ == "__main__":
    # predict_mcgill()
    # predict_mcgill('data/mcGillDataLocal.csv')
    # predict_mdb()
    predict_file('/home/simon/PycharmProjects/me-pytorch-sf/data/03-AClassicEducation_NightOwl-SingerSongwriter-0220_0232-reference-m27LUFS.flac', '/home/simon/PycharmProjects/me-pytorch-sf/data/hv_test/NightOwlNet3/deepsalience')


def old():
    predict = True
    # tasks = ['melody1', 'bass', 'melody2', 'melody3', 'multif0', 'pitch', 'vocal']
    tasks = ['pitch']
    for task in tasks:
        print(task)

        #mtrack1 = mdb.MultiTrack('LizNelson_Rainfall')
        #mtrack2 = mdb.MultiTrack('Phoenix_ScotchMorris')




        for filename in os.listdir('data/audio'):
            songname = filename[:len(filename) - 4]
            print(songname)
            if predict:
                print('In predict:')
                p_args = Predict_args('data/audio/' + filename, task, 'data/predictions', 'singlef0', 0.3, True)
                pred.main(p_args)
            ref_filename = 'MedleyDB_sample/Annotations/Pitch_Annotations/' + songname + '.csv'
            est_filename = 'data/predictions/' + songname + '_' + task + '_singlef0.csv'

            #ref_time, ref_freq = me.io.load_ragged_time_series(ref_filename, delimiter=',')
            est_time, est_freq = me.io.load_time_series(est_filename)

            time = []
            audio = []
            other = []
            with open(ref_filename, newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in csvreader:
                    time.append(float(row[0]))
                    audio.append(float(row[1]))

            ref_time_a = np.array(time)
            ref_freq_a = np.array(audio)
            ref_length = int(np.ceil(est_time[-1]*175.78125))
            ref_time = np.zeros(ref_length)
            ref_freq = np.zeros(ref_length)

            for i in range(ref_length):
                ref_time[i] = i/175.78125
                val = (np.abs(ref_time_a - ref_time[i])).min()
                if val < 0.01:
                    idx = (np.abs(ref_time_a - ref_time[i])).argmin()
                    ref_freq[i] = ref_freq_a[idx]


            print(ref_time)
            print(est_time)

            ref_time_p = ref_time
            ref_freq_p = ref_freq

            est_time_p = est_time
            est_freq_p = est_freq

            ref_time_p[ref_freq_p <= 0] = None
            ref_freq_p[ref_freq_p <= 0] = None

            est_time_p[est_freq_p <= 0] = None
            est_freq_p[est_freq_p <= 0] = None

            #f = interpolate.interp1d(est_time_p, est_freq_p, kind='nearest')
            #est_freq_i = f(ref_time_p)

            (ref_v, ref_c, est_v, est_c) = me.melody.to_cent_voicing(ref_time, ref_freq, est_time, est_freq)
            raw_pitch = me.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
            raw_chroma = me.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)

            print(raw_pitch)
            print(raw_chroma)

            plt.title(songname + task)
            plt.xlabel('time in s')
            plt.ylabel('freq in hz')
            plt.plot(est_time_p, est_freq_p, c='r', linewidth=2, label='prediction (rpa: ' + '%.2f' % raw_pitch + ' rca: ' + '%.2f' % raw_chroma + ')')
            plt.plot(ref_time_p, ref_freq_p, c='b', label='truth')
            #plt.plot(ref_time_p, est_freq_i-ref_freq_p, c='g', label='difference')
            plt.legend()
            #plt.savefig('data/rparca_singlef0_' + task + '_' + songname)
            plt.show()

