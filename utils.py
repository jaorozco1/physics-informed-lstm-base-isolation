import tensorflow as tf
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from scipy import signal


def integrate_displacement(acc, dt, initial_displ=0.0, initial_vel=0.0):
    """Double integrate acceleration to displacement with initial conditions."""
    initial_displ_tiled = tf.repeat(tf.expand_dims(initial_displ, axis=-1), acc.shape[1], axis=-1)
    initial_vel_tiled = tf.repeat(tf.expand_dims(initial_vel, axis=-1), acc.shape[1], axis=-1)
    vel = tf.cumsum(acc, axis=1) * dt + initial_vel_tiled
    disp = tf.cumsum(vel, axis=1) * dt + initial_displ_tiled
    return disp, vel

def normalize(seq):
    mean = tf.reduce_mean(seq)
    std = tf.math.reduce_std(seq) + 1e-8
    return (seq - mean) / std

def calculate_nrmse(pred, true):
    """Calculate NRMSE with error-proof handling for zero or NaN cases."""
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    if true.size == 0 or pred.size == 0 or np.all(np.isnan(true)) or np.all(np.isnan(pred)):
        return np.nan
    mse = np.nanmean((pred - true) ** 2)
    if mse == 0:
        return 0.0
    variance = np.nanvar(true)
    if variance == 0:
        return np.nan
    nrmse = np.sqrt(mse) / np.sqrt(variance)
    return nrmse if not np.isnan(nrmse) else np.nan

def butter_bandpass(lowcut, highcut, fs, order=40, btype='band'):
    """Design a Butterworth filter with specified type and cutoff frequencies."""
    nyquist = 0.5 * fs
    if btype == 'low':
        cutoff = highcut / nyquist
        b, a = butter(order, cutoff, btype='low')
    elif btype == 'high':
        cutoff = lowcut / nyquist
        b, a = butter(order, cutoff, btype='high')
    else:
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
    return b, a

def chanchullo_barrido(acc, fs, graf=0, ffiltro=[0.03, 20], fs_new=50, nalto=4, N_DOF=None):
    """Post-process acceleration to derive displacement using the chanchullo_barrido approach."""
    if N_DOF is None:
        raise ValueError("N_DOF must be provided")
    t = np.arange(acc.shape[1]) / fs
    vel = np.cumsum(acc, axis=1) * (1.0 / fs)
    b_low, a_low = butter_bandpass(0.01, ffiltro[1], fs, order=8, btype='low')
    vel = filtfilt(b_low, a_low, vel, axis=1)
    t_new = np.arange(0, t[-1], 1.0 / fs_new)
    vel_resampled = np.zeros((N_DOF, len(t_new)))
    for i in range(N_DOF):
        interp_func = interp1d(t, vel[i, :], kind='cubic', fill_value="extrapolate")
        vel_resampled[i, :] = interp_func(t_new)
    b_high, a_high = butter_bandpass(ffiltro[0], 20.0, fs_new, order=nalto, btype='high')
    vel_resampled = filtfilt(b_high, a_high, vel_resampled, axis=1)
    vel_final = np.zeros((N_DOF, len(t)))
    for i in range(N_DOF):
        interp_func = interp1d(t_new, vel_resampled[i, :], kind='cubic', fill_value="extrapolate")
        vel_final[i, :] = interp_func(t)
    vel_final = signal.detrend(vel_final, axis=1, type='linear')
    disp = np.cumsum(vel_final, axis=1) * (1.0 / fs)
    disp = filtfilt(b_high, a_high, disp, axis=1)
    disp = signal.detrend(disp, axis=1, type='linear')
    if graf:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(t, vel_final.T)
        plt.grid()
        plt.title('Velocity')
        plt.figure()
        plt.plot(t, disp.T)
        plt.grid()
        plt.title('Displacement')
        plt.figure()
        plt.plot(t, acc.T, label='Original Acceleration')
        derived_acc = np.gradient(np.gradient(disp, t, axis=1), t, axis=1)
        for i in range(N_DOF):
            plt.plot(t, derived_acc[i, :], '--', label=f'Derived Acc CH {i+1}' if i == 0 else "")
        plt.grid()
        plt.legend()
        plt.title('Acceleration Comparison')
        plt.show()
    return disp

def chanchullo_barrido_tf(displ, fs, ffiltro=[0.03, 20], fs_new=50, nalto=4):
    """TensorFlow-based displacement post-processing."""
    t = tf.cast(tf.range(displ.shape[1]) / fs, dtype=tf.float32)
    print(f"chanchullo_barrido_tf input shape: {displ.shape}, t shape: {t.shape}")
    disp_processed = displ
    window_size = tf.cast(tf.math.round(fs / ffiltro[1]), tf.int32)
    window = tf.ones([window_size, 1, 1], dtype=tf.float32) / tf.cast(window_size, tf.float32)
    disp_processed = tf.nn.conv1d(tf.expand_dims(disp_processed, axis=-1), window, stride=1, padding="SAME")
    disp_processed = tf.squeeze(disp_processed, axis=-1)
    t_new = tf.cast(tf.range(0, t[-1], 1.0 / fs_new), dtype=tf.float32)
    num_points_new = tf.cast(tf.shape(t_new)[0], tf.int32)
    disp_resampled = tf.image.resize(tf.expand_dims(disp_processed, axis=-1), [num_points_new, N_DOF], method='bilinear')
    disp_resampled = tf.squeeze(disp_resampled, axis=-1)
    window_size_high = tf.cast(tf.math.round(fs_new / ffiltro[0]), tf.int32)
    window_high = tf.ones([window_size_high, 1, 1], dtype=tf.float32) / tf.cast(window_size_high, tf.float32)
    disp_mean = tf.nn.conv1d(tf.expand_dims(disp_resampled, axis=-1), window_high, stride=1, padding="SAME")
    disp_processed = disp_resampled - tf.squeeze(disp_mean, axis=-1)
    num_points_orig = tf.cast(displ.shape[1], tf.int32)
    disp_resampled_back = tf.image.resize(tf.expand_dims(disp_processed, axis=-1), [num_points_orig, N_DOF], method='bilinear')
    disp_processed = tf.squeeze(disp_resampled_back, axis=-1)
    time_indices = tf.cast(tf.range(displ.shape[1], dtype=tf.float32), tf.float32)
    ones = tf.ones([displ.shape[1]], dtype=tf.float32)
    X = tf.stack([ones, time_indices], axis=1)
    def detrend_dof(dof_data):
        coeffs = tf.linalg.lstsq(X, tf.expand_dims(dof_data, axis=-1))
        trend = tf.matmul(X, coeffs)
        return dof_data - tf.squeeze(trend, axis=-1)
    disp_processed = tf.transpose(tf.map_fn(detrend_dof, tf.transpose(disp_processed), fn_output_signature=tf.float32))
    disp_mean_high = tf.nn.conv1d(tf.expand_dims(disp_processed, axis=-1), window_high, stride=1, padding="SAME")
    disp_processed = disp_processed - tf.squeeze(disp_mean_high, axis=-1)
    disp_processed = tf.transpose(tf.map_fn(detrend_dof, tf.transpose(disp_processed), fn_output_signature=tf.float32))
    return disp_processed

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

@tf.function(jit_compile=True)
def SDOF_aceleracion_promedio_tfNOTF(m, kt, bt, p, Fs, x0=0.0, v0=0.0):
    """Solve SDOF system using Newmark's method in TensorFlow."""
    dt_local = 1.0 / Fs
    N = p.shape[0]
    x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
    v0 = tf.convert_to_tensor(v0, dtype=tf.float32)
    kt = tf.convert_to_tensor(kt, dtype=tf.float32)
    bt = tf.convert_to_tensor(bt, dtype=tf.float32)
    p = tf.convert_to_tensor(p, dtype=tf.float32)
    x_list, v_list, a_list = [], [], []
    x_prev, v_prev = x0, v0
    a_prev = p[0] - 2 * tf.sqrt(kt[0]) * bt[0] * v_prev - kt[0] * x_prev
    x_list.append(x_prev)
    v_list.append(v_prev)
    a_list.append(a_prev)
    for i in range(1, N):
        term1 = (4/(dt_local**2)) * x_prev
        term2 = (4/dt_local) * v_prev
        term3 = a_prev
        term4 = 2/dt_local * x_prev + v_prev
        c_i = 2 * tf.sqrt(kt[i]) * bt[i]
        ptongo = p[i] + term1 + term2 + term3 + c_i * term4
        ktongo = (4/(dt_local**2)) + (2/dt_local)*c_i + kt[i]
        new_x = ptongo / ktongo
        new_v = (2/dt_local) * (new_x - x_prev) - v_prev
        new_a = p[i] - c_i * new_v - kt[i] * new_x
        x_list.append(new_x)
        v_list.append(new_v)
        a_list.append(new_a)
        x_prev, v_prev, a_prev = new_x, new_v, new_a
    return tf.stack(x_list), tf.stack(v_list), tf.stack(a_list)

@tf.function
def SDOF_aceleracion_promedio_tf_new(m, kt, bt, p, Fs, x0=0.0, v0=0.0, a0=0.0):
    """Solve SDOF system using Newmark's average acceleration method in TensorFlow."""
    dt_local = tf.cast(1.0 / Fs, dtype=tf.float32)
    N = tf.shape(p)[0]
    x_prev = tf.cast(x0, dtype=tf.float32)
    v_prev = tf.cast(v0, dtype=tf.float32)
    a_prev = tf.cast(a0, dtype=tf.float32)
    m = tf.cast(m, dtype=tf.float32)
    kt = tf.cast(kt, dtype=tf.float32)
    bt = tf.cast(bt, dtype=tf.float32)
    p = tf.cast(p, dtype=tf.float32)
    epsilon = 1e-10
    def adjust_initials():
        a_prev_adj = a_prev
        bt_0_safe = tf.math.maximum(tf.abs(bt[0]), epsilon) * tf.sign(bt[0])
        kt_0_safe = tf.math.maximum(tf.abs(kt[0]), epsilon) * tf.sign(kt[0])
        v_prev_adj = (p[0] - m * a_prev_adj - kt[0] * x_prev) / bt_0_safe
        x_prev_adj = (p[0] - bt[0] * v_prev_adj - m * a_prev_adj) / kt_0_safe
        return x_prev_adj, v_prev_adj, a_prev_adj
    x_prev, v_prev, a_prev = tf.cond(
        tf.not_equal(a_prev, 0.0),
        adjust_initials,
        lambda: (x_prev, v_prev, a_prev)
    )
    def cond(i, x_prev, v_prev, a_prev, x_ta, v_ta, a_ta):
        return i < N
    def body(i, x_prev, v_prev, a_prev, x_ta, v_ta, a_ta):
        term1 = (4.0 / (dt_local**2)) * x_prev
        term2 = (4.0 / dt_local) * v_prev
        term3 = a_prev
        term4 = (2.0 / dt_local) * x_prev + v_prev
        c_i = bt[i]
        ptongo = p[i] + m * (term1 + term2 + term3) + c_i * term4
        ktongo = tf.math.maximum(m * (4.0 / (dt_local**2)) + (2.0 / dt_local) * c_i + kt[i], epsilon)
        new_x = tf.math.divide_no_nan(ptongo, ktongo)
        new_v = (2.0 / dt_local) * (new_x - x_prev) - v_prev
        new_a = tf.math.divide_no_nan(p[i] - c_i * new_v - kt[i] * new_x, m)
        x_ta = x_ta.write(i, new_x)
        v_ta = v_ta.write(i, new_v)
        a_ta = a_ta.write(i, new_a)
        return i + 1, new_x, new_v, new_a, x_ta, v_ta, a_ta
    x_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    v_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    a_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    x_ta = x_ta.write(0, x_prev)
    v_ta = v_ta.write(0, v_prev)
    a_ta = a_ta.write(0, a_prev)
    i = tf.constant(1, dtype=tf.int32)
    i, x_prev, v_prev, a_prev, x_ta, v_ta, a_ta = tf.while_loop(
        cond, body,
        [i, x_prev, v_prev, a_prev, x_ta, v_ta, a_ta],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([]),
            tf.TensorShape([]),
            tf.TensorShape([]),
            tf.TensorShape(None),
            tf.TensorShape(None),
            tf.TensorShape(None)
        ]
    )
    return x_ta.stack(), v_ta.stack(), a_ta.stack()

@tf.function
def SDOF_aceleracion_promedio_tf2(m, kt, bt, p, Fs, x0=0.0, v0=0.0, a0=0.0):
    """Solve SDOF system with clipping and safeguards."""
    dt_local = tf.cast(1.0 / Fs, dtype=tf.float32)
    N = tf.shape(p)[0]
    x0 = tf.cast(x0, dtype=tf.float32)
    v0 = tf.cast(v0, dtype=tf.float32)
    a0 = tf.cast(a0, dtype=tf.float32)
    m = tf.cast(m, dtype=tf.float32)
    kt = tf.cast(kt, dtype=tf.float32)
    bt = tf.cast(bt, dtype=tf.float32)
    p = tf.cast(p, dtype=tf.float32)
    x_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    v_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    a_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    x_array = x_array.write(0, x0)
    v_array = v_array.write(0, v0)
    a_array = a_array.write(0, a0)
    x_prev = x0
    v_prev = v0
    a_prev = a0
    epsilon = 1e-10
    def adjust_initials():
        nonlocal x_prev, v_prev, a_prev
        a_prev_adj = a0
        bt_0_safe = tf.math.maximum(tf.abs(bt[0]), epsilon) * tf.sign(bt[0])
        kt_0_safe = tf.math.maximum(tf.abs(kt[0]), epsilon) * tf.sign(kt[0])
        v_prev_adj = (p[0] - m * a_prev_adj - kt[0] * x_prev) / bt_0_safe
        x_prev_adj = (p[0] - bt[0] * v_prev_adj - m * a_prev_adj) / kt_0_safe
        return x_prev_adj, v_prev_adj, a_prev_adj
    x_prev, v_prev, a_prev = tf.cond(
        tf.not_equal(a0, 0.0),
        adjust_initials,
        lambda: (x_prev, v_prev, a_prev)
    )
    def body(i, x_prev, v_prev, a_prev, x_array, v_array, a_array):
        term1 = (4.0 / (dt_local**2)) * x_prev
        term2 = (4.0 / dt_local) * v_prev
        term3 = a_prev
        term4 = (2.0 / dt_local) * x_prev + v_prev
        c_i = bt[i]
        ptongo = p[i] + m * (term1 + term2 + term3) + c_i * term4
        ktongo = m * (4.0 / (dt_local**2)) + c_i * (2.0 / dt_local) + kt[i]
        ktongo_safe = tf.math.maximum(tf.abs(ktongo), epsilon) * tf.sign(ktongo)
        new_x = tf.cond(
            tf.greater(tf.abs(ktongo), epsilon),
            lambda: ptongo / ktongo_safe,
            lambda: x_prev
        )
        new_v = (2.0 / dt_local) * (new_x - x_prev) - v_prev
        m_safe = tf.math.maximum(tf.abs(m), epsilon) * tf.sign(m)
        new_a = tf.cond(
            tf.greater(tf.abs(m), epsilon),
            lambda: (p[i] - c_i * new_v - kt[i] * new_x) / m_safe,
            lambda: a_prev
        )
        x_array = x_array.write(i, new_x)
        v_array = v_array.write(i, new_v)
        a_array = a_array.write(i, new_a)
        return i + 1, new_x, new_v, new_a, x_array, v_array, a_array
    _, _, _, _, x_array, v_array, a_array = tf.while_loop(
        lambda i, *args: i < N,
        body,
        (tf.constant(1), x_prev, v_prev, a_prev, x_array, v_array, a_array),
        shape_invariants=(tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]),
                          tf.TensorShape(None), tf.TensorShape(None), tf.TensorShape(None))
    )
    x_list = x_array.stack()
    v_list = v_array.stack()
    a_list = a_array.stack()
    displ_min, displ_max = -0.8, 0.8
    vel_min, vel_max = -2.0, 2.0
    acc_min, acc_max = -5.0, 5.0
    x_list = tf.clip_by_value(x_list, displ_min, displ_max)
    v_list = tf.clip_by_value(v_list, vel_min, vel_max)
    a_list = tf.clip_by_value(a_list, acc_min, acc_max)
    return x_list, v_list, a_list

def compute_freq_loss(pred_freq_mode1, pred_freq_mode2, target_freq_mode1, target_freq_mode2):
    """Compute frequency loss with tolerance-based penalty."""
    tolerance = 0.1
    dev_mode1 = tf.abs(pred_freq_mode1 - target_freq_mode1)
    dev_mode2 = tf.abs(pred_freq_mode2 - target_freq_mode2)
    loss_mode1 = tf.where(dev_mode1 > tolerance, (dev_mode1 - tolerance) ** 2, 0.0)
    loss_mode2 = tf.where(dev_mode2 > tolerance, (dev_mode2 - tolerance) ** 2, 0.0)
    return tf.reduce_mean(loss_mode1 + loss_mode2)

def adaptive_balance_losses(tape, losses, params):
    """Dynamically balance loss terms based on gradient norms."""
    dynamic_weights = []
    for loss_term in losses:
        grad_norm = tf.linalg.global_norm(tape.gradient(loss_term, params))
        dynamic_weights.append(1.0 / (grad_norm + 1e-6))
    total_weight = tf.reduce_sum(dynamic_weights)
    dynamic_weights = [w / total_weight for w in dynamic_weights]
    return dynamic_weights

