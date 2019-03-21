import datapipeline
import model
import tensorflow as tf
from Utils.dshandlehook import DSHandleHook

def main():
    #Use tf dataset api to create multi-threaded input queue pipelines with buffer to scale up training
    data = datapipeline.Dataset()
    state,action,next_state, iter_train_handle, iter_val_handle,handle,_ = data()
    # state: current stack of 4 image frames, action: corresponding 4 actions, next_state: next generated game image
    gameenginemodel = model.Model(state, action, next_state)
    # training tensors: step and loss
    trainvar = [gameenginemodel.step, gameenginemodel.loss]
    # variable initializer
    scf = tf.train.Scaffold(init_op= tf.global_variables_initializer(), local_init_op= tf.local_variables_initializer())
    #using to switch between train, val data queues dynamically during training using feedable iterators that maintain state
    ds_handle_hook = DSHandleHook(iter_train_handle, iter_val_handle)
    #used to generate initial string handlers to switch train and val as handle missing during first run
    hooks = [ds_handle_hook]
    #monitored training session takes care of model checkpointing, restoring and tensorboard
    with tf.train.MonitoredTrainingSession(scaffold=scf,
                                           checkpoint_dir='./ModelCheckpoints',
                                           summary_dir='./TFBoard',
                                           save_checkpoint_steps=500, hooks= hooks) as sess:
        for i in range(10000):
            #feed_dict used to switch multi-threaded input queues between train and val
            step, loss = sess.run(trainvar, feed_dict={handle: ds_handle_hook.train_handle})
            print("Training loss: "+str(loss))
            if (i%500 == 0):
                #validation
                loss = sess.run(gameenginemodel.loss, feed_dict={handle: ds_handle_hook.valid_handle})
                print("Validation loss: " + str(loss))

if __name__ == "__main__":
    main()