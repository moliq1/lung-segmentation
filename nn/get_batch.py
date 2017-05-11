import numpy as np
import os
import glob
from random import shuffle
import tensorflow as tf

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class DataSet(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.nodule_dict = self.get_nodule_dict()
        self.nonnodules = self.get_non_nodule()
        self.data_dir = '/mnt/sdk/lung_segmentation_2d_training_data/'
        self.label_dir = '/mnt/sdk/mask/'

    def get_nodule_dict(self):
        from collections import defaultdict
        nodule_dict = defaultdict(list)
        with open('/home/molingqiang/molingqiang/VOCdevkit/VOC2007/ImageSets/Main/trainval_all.txt', 'r') as f:
            for line in f:
                nodule_dict[line.strip().split('-')[0]].append(int(line.strip().split('-')[1]) + 10001)
        return nodule_dict


    def get_non_nodule(self):
        total_cases = os.listdir('/home/molingqiang/data/lung_segmentation_2d_training_data')
        nodule_case = self.nodule_dict.keys()
        nonnodule_cases = [i for i in total_cases if i not in nodule_case]
        # print 'number of non nodule cases : ',len(nonnodule_cases)
        if len(nonnodule_cases) > 80:
            shuffle(nonnodule_cases)
            nonnodule_cases = nonnodule_cases[:80]
        return nonnodule_cases

    def get_training_data(self):

        training_data = []
        training_label = []
        have_keys = os.listdir('/mnt/sdk/lung_segmentation_2d_training_data')
        for case in self.nodule_dict.keys():
            if case in have_keys:
                inds = self.nodule_dict[case]
                data_path = [os.path.join(self.data_dir, case, str(ind)+'.png') for ind in inds]
                label_path = [os.path.join(self.label_dir, case, str(ind)+'.png') for ind in inds]
                training_data.extend(data_path)
                training_label.extend(label_path)
        print len(training_data), len(training_label)

        for case in self.nonnodules:
            data_path = glob.glob(self.data_dir + case + '/*')
            label_path = glob.glob(self.label_dir + case + '/*')
            # print case
            if len(data_path) != len(label_path):
                continue
            training_data.extend(data_path)
            training_label.extend(label_path)

        assert len(training_data) == len(training_label)

        perm = np.random.permutation(np.arange(len(training_label)))
        data_path = [training_data[i] for i in perm]
        label_path = [training_label[i] for i in perm]

        return data_path, label_path

    def write_tfrecords(self, data_path, label_path):

        from tqdm import tqdm
        image_queue = tf.train.string_input_producer(data_path, shuffle=False)
        label_queue = tf.train.string_input_producer(label_path, shuffle=False)
        reader = tf.WholeFileReader()
        key, ivalue = reader.read(image_queue)
        _, lvalue = reader.read(label_queue)
        Img = tf.image.decode_png(ivalue)
        Label = tf.image.decode_png(lvalue)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        filename = 'lung.tfrecords'
        writer = tf.python_io.TFRecordWriter(filename)
        for ind in tqdm(range(len(data_path))):
            Image = sess.run(Img)
            mask = sess.run(Label)
            imageRaw = Image.tostring()
            maskRaw = mask.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(imageRaw),
                'mask_raw' : _bytes_feature(maskRaw)}))
            writer.write(example.SerializeToString())

        coord.request_stop()
        coord.join(threads)


    def read_and_decode(self):
        filename_queue = tf.train.string_input_producer(['/home/molingqiang/molingqiang/lung_segement/nn/lung_seg.tfrecords'])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               'mask_raw': tf.FixedLenFeature([], tf.string),
                                           })

        img = tf.decode_raw(features['image_raw'], tf.uint8)
        lab = tf.decode_raw(features['mask_raw'], tf.uint8)

        image = tf.reshape(img, [512, 512, 1])
        label = tf.reshape(lab, [512, 512])

        image = tf.cast(image, tf.float32) * (1. / 255)
        label = tf.cast(label, tf.float32)


        return image, label

    def get_batch(self):
        img, label = self.read_and_decode()
        img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                        batch_size=self.batch_size, capacity=2000,
                                                        min_after_dequeue=100)
        return img_batch, label_batch

    #
    # def load_training_files(self):
    #     self.train_neg_files = glob.glob(self.train_data_folder_neg+'*')
    #     self.train_neg_files.remove('/mnt/1T/data/nodule_2d_patch_neg/label_dict.npy')
    #     self.train_pos_files = glob.glob(self.train_data_folder_pos+'*')
    #     # shuffle(self.train_pos_files)
    #     # shuffle(self.train_neg_files)
    #
    # def data_path_to_array(self, paths):
    #
    #     train = [np.load(i) for i in paths]
    #     train = np.array(train, dtype=np.float32)
    #     train = np.expand_dims(train, 3)
    #     return train
    #
    # # def label_path_to_array(self, path):
    #
    # def data_update(self):
    #     self.ind = self.ind % len(self.train_neg_files)
    #     epoch_data = self.train_neg_files[self.ind:self.ind + 25000] + self.train_pos_files
    #     label_data = np.hstack([np.zeros(25000, dtype=np.int32), np.ones(len(self.train_pos_files), dtype=np.int32)])
    #     # shuffle
    #     self.epoch_data_shuf = []
    #     self.label_data_shuf = []
    #     index_shuf = range(len(epoch_data))
    #     shuffle(index_shuf)
    #     for i in index_shuf:
    #         self.epoch_data_shuf.append(epoch_data[i])
    #         self.label_data_shuf.append(label_data[i])
    #     self.ind += 25000
    #
    # def get_batch(self):
    #
    #     if self.index+self.batch_size > len(self.epoch_data_shuf):
    #         self.index = 0
    #     x_batch = self.data_path_to_array(self.epoch_data_shuf[self.index:self.index+self.batch_size])
    #     y_batch = self.label_data_shuf[self.index:self.index+self.batch_size]
    #     self.index += self.batch_size
    #     return x_batch, y_batch


if __name__ == '__main__':

    dataset = DataSet(batch_size=5)
    training_data, training_label = dataset.get_training_data()
    print len(training_data), len(training_label)
    print training_data[:5]
    dataset.write_tfrecords(training_data, training_label)
    # dataset.get_batch()



# number of non nodule cases :  285
# 7030 7030
# 28209 28209