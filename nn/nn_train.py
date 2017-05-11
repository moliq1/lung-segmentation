from FCN import fcn
from get_batch import DataSet
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class train_model(object):

    def __init__(self):
        # self.tensorboard_dir = '/home/molingqiang/molingqiang/lung_segement/tf_log'
        self.checkpoint_dir = '/home/molingqiang/molingqiang/lung_segement/checkpoint'
        self.batch_size = 5
        self.nepoch = 20
        self.training_size = 28209
        self.display_step = 1000
        self.sess = tf.Session()
        self.nn = fcn()
        self.dataset = DataSet(batch_size=self.batch_size)
        self.saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=1)


    def restore_session(self, checkpoint_dir):
        self.saver.restore(self.sess, checkpoint_dir)
        print 'model restore from'+checkpoint_dir
    #
    # def accuracy(self, predictions, labels):
    #     return (100.0 * np.sum(np.argmax(predictions, 1) == labels)
    #             / predictions.shape[0])

    def train(self, restore_session=None):
        print '-' * 30
        print 'Initilize variables'
        print '-' * 30
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if restore_session is not None:
            self.restore_session(restore_session)

        print 'Start training'
        print '-' * 30

        # checkpoint dir create
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # tensorboard summary
        # if not os.path.exists(self.tensorboard_dir):
        #     os.makedirs(self.tensorboard_dir)
        # train_writer = tf.summary.FileWriter(self.tensorboard_dir, self.sess.graph)

        # training
        img_batch, label_batch = self.dataset.get_batch()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        for i in xrange(self.nepoch*self.training_size/self.batch_size):

            if i * self.batch_size % 28160 == 0:
                print 'epoch %d finished'%(i*self.batch_size/28160)
                self.saver.save(self.sess, save_path=os.path.join(self.checkpoint_dir, 'epoch_%d'%i))

            image, label= self.sess.run([img_batch, label_batch])
            label[label > 1] = 1
            feed_dict = {self.nn.input: image, self.nn.label: label}
            self.sess.run(self.nn.optimizer, feed_dict=feed_dict)

            if i % 10 == 0 :
                msg = "Iteration: {0:>6}, Training Loss: {1:>6.6f}"
                train_loss = self.sess.run(self.nn.loss,feed_dict=feed_dict)
                print msg.format(i, train_loss)

        coord.request_stop()
        coord.join(threads)




        # epoch = 1
        # self.dataset.data_update()
        #
        # for i in range(self.train_iters):
        #     x_batch, y_batch = self.dataset.get_batch()
        #     feed_dict = {self.nn.data: x_batch, self.nn.label: y_batch, self.nn.dropout: 0.5}
        #     self.sess.run(self.nn.optimizer, feed_dict=feed_dict)
        #
        #     # if (i *self.batch_size) % 20000 == 0 and i > 0:
        #     #     save_path = os.path.join(self.checkpoint_dir, str(i)+ '.ckpt')
        #     #     self.saver.save(sess=self.sess, save_path=save_path)
        #     #     print 'model %d saved' % (i)
        #
        #     if (i * self.batch_size) % 42500 == 0 and i > 0:
        #         self.dataset.data_update()
        #         print '-' * 30
        #         print 'Epoch: %d finished!!! ' % (epoch)
        #         print '-' * 30
        #         if epoch >= 2:
        #             save_path = os.path.join(self.checkpoint_dir, str(i)+ '.ckpt')
        #             self.saver.save(sess=self.sess, save_path=save_path)
        #             print 'model %d saved'%(i)
        #         epoch += 1
        #
        #     # show training loss
        #     if i % 50 == 0 or i + 1 == self.train_iters:
        #         msg = "Iteration: {0:>6}, Training Loss: {1:>6.6f}, train accuracy: {2:>6.4f}"
        #         train_loss, summary, prediction = self.sess.run([self.nn.loss,self.nn.merged, self.nn.prediction], feed_dict=feed_dict)
        #         # train_loss = self.sess.run(self.nn.loss, feed_dict=feed_dict)
        #         # summary = self.sess.run(self.nn.merged, feed_dict=feed_dict)
        #         train_writer.add_summary(summary, i)
        #         print msg.format(i + 1, train_loss, self.accuracy(prediction, y_batch))


if __name__ == '__main__':
    trainer = train_model()
    trainer.train('/home/molingqiang/molingqiang/lung_segement/checkpoint/epoch_11264')



