import datapipeline
import model
import tensorflow as tf
from Utils.dshandlehook import DSHandleHook

def main():
    #Use tf dataset api to create multi-threaded input queue with buffer to scale up training
    data = datapipeline.Dataset()
    state, action, next_state, _, _, handle, iter_test_handle = data()
    # state: current stack of 4 image frames, action: corresponding 4 actions, next_state: next generated game image
    gameenginemodel = model.Model(state, action, next_state)
    # testiing tensors: mean squared error loss
    testvar = [gameenginemodel.loss]
    # variable initializer
    scf = tf.train.Scaffold(init_op=tf.global_variables_initializer(), local_init_op=tf.local_variables_initializer())
    #fetches only the test queue
    ds_handle_hook = DSHandleHook(iter_test_handle, _)
    hooks = [ds_handle_hook]
    #monitored training session restores optimized weights from the directory
    with tf.train.MonitoredTrainingSession(scaffold=scf,
                                           checkpoint_dir='./ModelCheckpoints',
                                           summary_dir='./TFBoard',
                                           save_checkpoint_steps=500, hooks=hooks) as sess:
        n = 100
        sumloss = 0
        for i in range(n):
            loss = sess.run(testvar, feed_dict={handle: ds_handle_hook.train_handle})
            sumloss += loss[0]
    testmse = sumloss / n
    # mean squared loss calculated
    print("Test mean squared error: " + str(testmse))

if __name__ == "__main__":
    main()