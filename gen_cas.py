import pickle
import random
import time
from absl import app, flags

# flags
FLAGS = flags.FLAGS
# observation and prediction time settings:
# for weibo   dataset, we use 1800 (0.5 hour) or 3600 (1 hour) or 7200 (2 hour) as observation time
#                      we use 3600*24 (86400, 1 day) as prediction time
# for aps     dataset, we use 365*3 (1095, 3 years) or 365*5+1 (1826, 5 years)
#                       or 365*7+1 (2556, 7 years) as observation time*
#                      we use 365*20+5 (7305, 20 years) as prediction time
flags.DEFINE_integer('observation_time', 1095, 'Observation time.')
flags.DEFINE_integer('prediction_time', 7305, 'Prediction time.')
# path

flags.DEFINE_string('data', 'data/aps/', 'Dataset path.')


def generate_cascades(ob_time, pred_time, filename, file_train, file_val, file_test, seed):

    # a list to save the cascades
    filtered_data = list()
    cascades_type = dict()  # 0 for train, 1 for val, 2 for test
    cascades_time_dict = dict()
    cascades_total = 0
    cascades_valid_total = 0
    label_zero = 0  # number of incremental popularity(zero)

    # Important node: for weibo dataset, if you want to compare CasFlow with baselines such as DeepHawkes and CasCN,
    # make sure the ob_time is set consistently.
    if ob_time in [3600, 3600*2, 3600*3]:  # end_hour is set to 19 in DeepHawkes and CasCN, but it should be 18
        end_hour = 19
    else:
        end_hour = 18

    with open(filename) as file:
        for line in file:
            # split the cascades into 5 parts
            # 1: cascade id
            # 2: user/item id
            # 3: publish date/time
            # 4: number of adoptions
            # 5: a list of adoptions
            cascades_total += 1
            parts = line.split('\t')
            cascade_id = parts[0]
            if cascade_id == '2742':
                a = 1
            # filter cascades by their publish date/time
            if 'weibo' in FLAGS.data:
                # timezone invariant
                hour = int(time.strftime('%H', time.gmtime(float(parts[2])))) + 8
                if hour < 8 or hour >= end_hour:
                    continue
            elif 'twitter' in FLAGS.data:
                month = int(time.strftime('%m', time.localtime(float(parts[2]))))
                day = int(time.strftime('%d', time.localtime(float(parts[2]))))
                if month == 4 and day > 10:
                    continue
            elif 'aps' in FLAGS.data:
                publish_time = parts[2]
                if publish_time > '1997':
                    continue
            else:
                pass

            paths = parts[4].strip().split(' ')

            observation_path = list()
            # number of observed popularity
            p_o = 0
            for p in paths:
                # observed adoption/participant
                nodes = p.split(':')[0].split('/')
                time_now = int(p.split(':')[1])
                if time_now < ob_time:
                    p_o += 1
                # save observed adoption/participant into 'observation_path'
                observation_path.append((nodes, time_now))

            # filter cascades which observed popularity less than 10
            if p_o < 10:
                continue

            # sort list by their publish time/date
            observation_path.sort(key=lambda tup: tup[1])

            # for each cascade, save its publish time into a dict
            if 'aps' in FLAGS.data:
                cascades_time_dict[cascade_id] = int(0)
            else:
                cascades_time_dict[cascade_id] = int(parts[2])

            o_path = list()

            for i in range(len(observation_path)):
                nodes = observation_path[i][0]
                t = observation_path[i][1]
                o_path.append('/'.join(nodes) + ':' + str(t))

            # write data into the targeted file, if they are not excluded
            line = parts[0] + '\t' + parts[1] + '\t' + parts[2] + '\t' \
                   + parts[3] + '\t' + ' '.join(o_path) + '\n'
            filtered_data.append(line)
            cascades_valid_total += 1

    # open three files to save train, val, and test set, respectively
    with open(file_train, 'w') as data_train, \
            open(file_val, 'w') as data_val, \
            open(file_test, 'w') as data_test:

        def shuffle_cascades():
            # shuffle all cascades
            shuffle_time = list(cascades_time_dict.keys())
            random.seed(seed)
            random.shuffle(shuffle_time)

            count = 0
            # split dataset
            for key in shuffle_time:
                if count < cascades_valid_total * .7:
                    cascades_type[key] = 0  # training set, 70%
                elif count < cascades_valid_total * .85:
                    cascades_type[key] = 1  # validation set, 15%
                else:
                    cascades_type[key] = 2  # test set, 15%
                count += 1

        shuffle_cascades()

        # number of valid cascades
        print("Number of valid cascades: {}/{}".format(cascades_valid_total, cascades_total))

        # 3 lists to save the filtered sets
        filtered_data_train = list()
        filtered_data_val = list()
        filtered_data_test = list()
        for line in filtered_data:
            cascade_id = line.split('\t')[0]
            if cascades_type[cascade_id] == 0:
                filtered_data_train.append(line)
            elif cascades_type[cascade_id] == 1:
                filtered_data_val.append(line)
            elif cascades_type[cascade_id] == 2:
                filtered_data_test.append(line)
            else:
                print('What happened?')

        print("Number of valid train cascades: {}".format(len(filtered_data_train)))
        print("Number of valid   val cascades: {}".format(len(filtered_data_val)))
        print("Number of valid  test cascades: {}".format(len(filtered_data_test)))

        # shuffle the train set again
        random.seed(seed)
        random.shuffle(filtered_data_train)

        def file_write(file_name):
            # write file, note that compared to the original 'dataset.txt', only cascade_id and each of the
            # observed adoptions are saved, plus label information at last
            file_name.write(cascade_id + '\t' + '\t'.join(observation_path) + '\t' + label + '\n')

        # write cascades into files
        for line in filtered_data_train + filtered_data_val + filtered_data_test:
            # split the cascades into 5 parts
            parts = line.split('\t')
            cascade_id = parts[0]
            observation_path = list()
            label = int()
            edges = set()
            paths = parts[4].split(' ')

            for p in paths:
                nodes = p.split(':')[0].split('/')

                time_now = int(p.split(':')[1])
                if time_now < ob_time:
                    observation_path.append(','.join(nodes) + ':' + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ':' + nodes[i] + ':1')
                # add label information depends on prediction_time, e.g., 24 hours for weibo dataset
                if time_now < pred_time:
                    label += 1

            if label == len(observation_path):
                label_zero += 1

            # calculate the incremental popularity
            label = str(label - len(observation_path))

            # write files by cascade type
            # 0 to train, 1 to val, 2 to test
            if cascade_id in cascades_type and cascades_type[cascade_id] == 0:
                file_write(data_train)
            elif cascade_id in cascades_type and cascades_type[cascade_id] == 1:
                file_write(data_val)
            elif cascade_id in cascades_type and cascades_type[cascade_id] == 2:
                file_write(data_test)

        print('number of incremental popularity(zero): ', label_zero)


def main(argv):
    time_start = time.time()

    print('Dataset path: {}\n'.format(FLAGS.data))

    generate_cascades(FLAGS.observation_time, FLAGS.prediction_time,
                      FLAGS.data + 'dataset.txt',
                      FLAGS.data + f'train_{FLAGS.observation_time}.txt',
                      FLAGS.data + f'val_{FLAGS.observation_time}.txt',
                      FLAGS.data + f'test_{FLAGS.observation_time}.txt',
                      seed=0)

    print('Processing time: {:.2f}s'.format(time.time()-time_start))


if __name__ == '__main__':
    app.run(main)
