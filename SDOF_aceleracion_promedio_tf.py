import tensorflow as tf
@tf.function
def SDOF_aceleracion_promedio_tf(m, kt, bt, p, Fs, x0=0.0, v0=0.0):
    """Solve SDOF system using Newmark's method in TensorFlow."""
    dt_local = 1.0 / Fs
    N = tf.shape(p)[0]
    x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
    v0 = tf.convert_to_tensor(v0, dtype=tf.float32)
    kt = tf.convert_to_tensor(kt, dtype=tf.float32)
    bt = tf.convert_to_tensor(bt, dtype=tf.float32)
    p = tf.convert_to_tensor(p, dtype=tf.float32)
    def body(t, x_prev, v_prev, a_prev, x_ta, v_ta, a_ta):
        term1 = (4 / (dt_local**2)) * x_prev
        term2 = (4 / dt_local) * v_prev
        term3 = a_prev
        term4 = (2 / dt_local) * x_prev + v_prev
        c_i = 2 * tf.sqrt(kt[t]) * bt[t]
        ptongo = p[t] + term1 + term2 + term3 + c_i * term4
        ktongo = (4 / (dt_local**2)) + (2 / dt_local) * c_i + kt[t]
        new_x = ptongo / ktongo
        new_v = (2 / dt_local) * (new_x - x_prev) - v_prev
        new_a = p[t] - c_i * new_v - kt[t] * new_x
        x_ta = x_ta.write(t, new_x)
        v_ta = v_ta.write(t, new_v)
        a_ta = a_ta.write(t, new_a)
        return t+1, new_x, new_v, new_a, x_ta, v_ta, a_ta

    x_ta = tf.TensorArray(tf.float32, size=N)
    v_ta = tf.TensorArray(tf.float32, size=N)
    a_ta = tf.TensorArray(tf.float32, size=N)
    x_prev = x0
    v_prev = v0
    a_prev = p[0] - 2 * tf.sqrt(kt[0]) * bt[0] * v_prev - kt[0] * x_prev
    x_ta = x_ta.write(0, x_prev)
    v_ta = v_ta.write(0, v_prev)
    a_ta = a_ta.write(0, a_prev)
    t = tf.constant(1)
    t, x_prev, v_prev, a_prev, x_ta, v_ta, a_ta = tf.while_loop(
        lambda t, *_: t < N,
        body,
        [t, x_prev, v_prev, a_prev, x_ta, v_ta, a_ta],
        parallel_iterations=10
    )
    return x_ta.stack(), v_ta.stack(), a_ta.stack()
