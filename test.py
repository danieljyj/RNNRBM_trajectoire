
import numpy as np 
import tensorflow as tf

elems = np.array([0, 0, 0, 0, 0, 0])
initializer = (np.array(0), np.array(1))
fibonaccis = tf.scan(lambda a, _: (a[1], a[0] + a[1]), elems, initializer)
# fibonaccis == ([1, 1, 2, 3, 5, 8], [1, 2, 3, 5, 8, 13])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    fibo=sess.run(fibonaccis)
    print(fibo)
    print(type(fibo))
    print(initializer)
    print(elems)
print("finished")