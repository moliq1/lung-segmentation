from FCN import fcn
from get_batch import DataSet
import numpy as np
from draw_mask import plot_boundaries
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class test(object):

    def __init__(self):
        # self.tensorboard_dir = '/home/molingqiang/molingqiang/lung_segement/tf_log'
        self.pretrain_model = '/home/molingqiang/molingqiang/lung_segement/checkpoint/epoch_112640'
        self.batch_size = 32
        self.sess = tf.Session()
        self.nn = fcn()
        self.dataset = DataSet(batch_size=self.batch_size)
        self.saver = tf.train.Saver()


    def restore_session(self, checkpoint_dir):
        self.saver.restore(self.sess, checkpoint_dir)
        print 'model restore from'+checkpoint_dir

    def run(self):
        import scipy.misc

        self.restore_session(self.pretrain_model)

        img_batch, label_batch = self.dataset.get_batch()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        for i in xrange(1):

            image, label= self.sess.run([img_batch, label_batch])
            label[label > 1] = 1
            feed_dict = {self.nn.input: image, self.nn.label: label}
            pred = self.sess.run(self.nn.pred, feed_dict=feed_dict)

            # save bimary mask
            for k in range(self.batch_size):
                scipy.misc.imsave(os.path.join('/home/molingqiang/scrips/paper/result', str(k)+'_im.png'), image[k, :, :, 0]*255)
                scipy.misc.imsave(os.path.join('/home/molingqiang/scrips/paper/result', str(k)+'_mask.png'), pred[k]*255)

            # for k in range(self.batch_size):
            #     im = plot_boundaries(image[k, :, :, 0]*255, pred[k])
            #     scipy.misc.imsave(os.path.join('/home/molingqiang/molingqiang/lung_segement/example_pred', str(k) + '.png'), im)

        coord.request_stop()
        coord.join(threads)

    def pred_dcm(self, dcm_dir, save_path=None):
        from dicom_utils import read_dcm_to_array
        import time
        self.restore_session(self.pretrain_model)
        for i in range(1):
            start = time.time()
            images = read_dcm_to_array(dcm_dir)
            images = images.astype(np.uint8) / 255.0
            images = images.astype(np.float32)
            pred_mask = np.zeros(shape=images.shape)
            # save_path = os.path.join(save_folder, dcm_dir.split('/')[-1])
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            test_size_batch = 32
            images = np.expand_dims(images, 3)
            for i in range(images.shape[0] / test_size_batch):
                train = images[i * test_size_batch:(i * test_size_batch + test_size_batch)]
                # seg = seg_image[i * 10:(i * 10 + 10)]
                feed_dict = {self.nn.input: train}
                pred = self.sess.run(self.nn.pred, feed_dict=feed_dict)
                pred_mask[i * test_size_batch:(i * test_size_batch + test_size_batch),:,:] = pred

            if not images.shape[0] % test_size_batch == 0:
                idx = images.shape[0] % test_size_batch
                train = images[-idx:]
                feed_dict = {self.nn.input: train}
                pred = self.sess.run(self.nn.pred, feed_dict=feed_dict)
                pred_mask[-idx:, :, :] = pred

            print 'predication complete, time comsuming: %.2f'%(time.time() - start)

            if save_path:
                np.save(os.path.join(save_path, 'fcn_pred.npy'), pred_mask.astype(np.int8))
                print 'save prediction to', save_path

if __name__ == '__main__':
    tester = test()
    tester.run()
    # tester.pred_dcm('/mnt/1T/LIDC-IDRI/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192', save_path='/home/molingqiang/scrips')



