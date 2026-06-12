import tensorflow as tf
from utils import normalize
import numpy as np
from SDOF_aceleracion_promedio_tf import SDOF_aceleracion_promedio_tf

@tf.function
def process_window(
    w_tf,
    modal_y_prev_mode1, modal_ydot_prev_mode1, prev_damp_seq_mode1, prev_freq_seq_mode1,
    modal_y_prev_mode2, modal_ydot_prev_mode2, prev_damp_seq_mode2, prev_freq_seq_mode2,
    modal_y_prev_mode3, modal_ydot_prev_mode3, prev_damp_seq_mode3, prev_freq_seq_mode3,
    hidden_state_mode1, hidden_state_mode2, hidden_state_mode3, hidden_state_contrib,
    current_seismic_norm, current_cnn_feature_norm, next_seismic_norm, next_cnn_feature_norm,
    measured_win_n1, true_displ_n1, p_n1_mode1, p_n1_mode2, p_n1_mode3,
    model_mode1, model_mode2, model_mode3, model_contrib,
    optimizer, max_acc, max_displ, Fs,
    ACC_LOSS_WEIGHT, DISPL_LOSS_WEIGHT, SMOOTH_WEIGHT_DAMP, SMOOTH_WEIGHT_FREQ,
    PHI_REG_LOSS_WEIGHT, MODE_SUM_LOSS_WEIGHT, DAMP_VARIATION_LIMIT, FREQ_VARIATION_LIMIT,
    PHI, R, ACTIVE_MODES, WINDOW_SIZE, N_DOF, use_clip=True
):
    # Initialize outputs
    phi_mode1_t_trans = tf.zeros([N_DOF, WINDOW_SIZE], dtype=tf.float32)
    phi_mode2_t_trans = tf.zeros([N_DOF, WINDOW_SIZE], dtype=tf.float32)
    phi_mode3_t_trans = tf.zeros([N_DOF, WINDOW_SIZE], dtype=tf.float32)
    
    # Normalize previous sequences
    prev_damp_seq_mode1_norm = normalize(prev_damp_seq_mode1)
    prev_freq_seq_mode1_norm = normalize(prev_freq_seq_mode1)
    prev_damp_seq_mode2_norm = tf.zeros_like(prev_damp_seq_mode1) if ACTIVE_MODES < 2 else normalize(prev_damp_seq_mode2)
    prev_freq_seq_mode2_norm = tf.zeros_like(prev_freq_seq_mode1) if ACTIVE_MODES < 2 else normalize(prev_freq_seq_mode2)
    prev_damp_seq_mode3_norm = tf.zeros_like(prev_damp_seq_mode1) if ACTIVE_MODES < 3 else normalize(prev_damp_seq_mode3)
    prev_freq_seq_mode3_norm = tf.zeros_like(prev_freq_seq_mode1) if ACTIVE_MODES < 3 else normalize(prev_freq_seq_mode3)

    # === STAGE 1: PRELIMINARY SDOF SOLVE ===
    # Mode 1
    interp_freq_mode1_prev = tf.cond(tf.greater(w_tf, 0), lambda: prev_freq_seq_mode1, lambda: tf.repeat(1.2, WINDOW_SIZE))
    interp_damp_mode1_prev = tf.cond(tf.greater(w_tf, 0), lambda: prev_damp_seq_mode1, lambda: tf.repeat(0.05, WINDOW_SIZE))
    kt_mode1_prev = (2.0 * tf.cast(tf.constant(np.pi), tf.float32) * interp_freq_mode1_prev)**2
    x_mode1_prev, _, a_int_mode1_prev = SDOF_aceleracion_promedio_tf(1.0, kt_mode1_prev, interp_damp_mode1_prev, p_n1_mode1, Fs, x0=modal_y_prev_mode1, v0=modal_ydot_prev_mode1)
    
    # Mode 2
    x_mode2_prev = tf.zeros_like(x_mode1_prev)
    a_int_mode2_prev = tf.zeros_like(a_int_mode1_prev)
    if ACTIVE_MODES >= 2:
        interp_freq_mode2_prev = tf.cond(tf.greater(w_tf, 0), lambda: prev_freq_seq_mode2, lambda: tf.repeat(3.25, WINDOW_SIZE))
        interp_damp_mode2_prev = tf.cond(tf.greater(w_tf, 0), lambda: prev_damp_seq_mode2, lambda: tf.repeat(0.025, WINDOW_SIZE))
        kt_mode2_prev = (2.0 * tf.cast(tf.constant(np.pi), tf.float32) * interp_freq_mode2_prev)**2
        x_mode2_prev, _, a_int_mode2_prev = SDOF_aceleracion_promedio_tf(1.0, kt_mode2_prev, interp_damp_mode2_prev, p_n1_mode2, Fs, x0=modal_y_prev_mode2, v0=modal_ydot_prev_mode2)

    # Mode 3
    x_mode3_prev = tf.zeros_like(x_mode1_prev)
    a_int_mode3_prev = tf.zeros_like(a_int_mode1_prev)
    if ACTIVE_MODES >= 3:
        interp_freq_mode3_prev = tf.cond(tf.greater(w_tf, 0), lambda: prev_freq_seq_mode3, lambda: tf.repeat(5.0, WINDOW_SIZE))
        interp_damp_mode3_prev = tf.cond(tf.greater(w_tf, 0), lambda: prev_damp_seq_mode3, lambda: tf.repeat(0.02, WINDOW_SIZE))
        kt_mode3_prev = (2.0 * tf.cast(tf.constant(np.pi), tf.float32) * interp_freq_mode3_prev)**2
        x_mode3_prev, _, a_int_mode3_prev = SDOF_aceleracion_promedio_tf(1.0, kt_mode3_prev, interp_damp_mode3_prev, p_n1_mode3, Fs, x0=modal_y_prev_mode3, v0=modal_ydot_prev_mode3)

    # === STAGE 2: CONSTRUCT FEATURES ===
    window_feat_mode1 = tf.stack([current_seismic_norm, a_int_mode1_prev / max_acc, x_mode1_prev / max_displ, current_cnn_feature_norm, next_seismic_norm, next_cnn_feature_norm, prev_damp_seq_mode1_norm, prev_freq_seq_mode1_norm], axis=1)[tf.newaxis, :, :]
    window_feat_mode2 = tf.zeros_like(window_feat_mode1) if ACTIVE_MODES < 2 else tf.stack([current_seismic_norm, a_int_mode2_prev / max_acc, x_mode2_prev / max_displ, current_cnn_feature_norm, next_seismic_norm, next_cnn_feature_norm, prev_damp_seq_mode2_norm, prev_freq_seq_mode2_norm], axis=1)[tf.newaxis, :, :]
    window_feat_mode3 = tf.zeros_like(window_feat_mode1) if ACTIVE_MODES < 3 else tf.stack([current_seismic_norm, a_int_mode3_prev / max_acc, x_mode3_prev / max_displ, current_cnn_feature_norm, next_seismic_norm, next_cnn_feature_norm, prev_damp_seq_mode3_norm, prev_freq_seq_mode3_norm], axis=1)[tf.newaxis, :, :]
    
    # Contributions model features
    features = [current_seismic_norm, a_int_mode1_prev / max_acc, x_mode1_prev / max_displ, prev_damp_seq_mode1_norm, prev_freq_seq_mode1_norm]
    if ACTIVE_MODES >= 2:
        features.extend([a_int_mode2_prev / max_acc, x_mode2_prev / max_displ, prev_damp_seq_mode2_norm, prev_freq_seq_mode2_norm])
    if ACTIVE_MODES >= 3:
        features.extend([a_int_mode3_prev / max_acc, x_mode3_prev / max_displ, prev_damp_seq_mode3_norm, prev_freq_seq_mode3_norm])
    features.extend([current_cnn_feature_norm, next_seismic_norm, next_cnn_feature_norm])
    window_feat_contrib = tf.stack(features, axis=1)[tf.newaxis, :, :] if N_DOF > 1 else tf.zeros([1, WINDOW_SIZE, 1], dtype=tf.float32)

    with tf.GradientTape() as tape:
        # === STAGE 3: GET MODEL PREDICTIONS ===
        # Mode 1
        if ACTIVE_MODES >= 1:
            pred_out_mode1, hidden_state_mode1 = model_mode1(window_feat_mode1, initial_state=hidden_state_mode1, training=True)
            new_damp_seq_mode1 = pred_out_mode1[0][:, 0]
            new_freq_seq_mode1 = pred_out_mode1[0][:, 1]
        else:
            new_damp_seq_mode1 = prev_damp_seq_mode1
            new_freq_seq_mode1 = prev_freq_seq_mode1

        # Mode 2
        new_damp_seq_mode2 = prev_damp_seq_mode2
        new_freq_seq_mode2 = prev_freq_seq_mode2
        if ACTIVE_MODES >= 2 and model_mode2 is not None:
            pred_out_mode2, hidden_state_mode2 = model_mode2(window_feat_mode2, initial_state=hidden_state_mode2, training=True)
            new_damp_seq_mode2 = pred_out_mode2[0][:, 0]
            new_freq_seq_mode2 = pred_out_mode2[0][:, 1]

        # Mode 3
        new_damp_seq_mode3 = prev_damp_seq_mode3
        new_freq_seq_mode3 = prev_freq_seq_mode3
        if ACTIVE_MODES >= 3 and model_mode3 is not None:
            pred_out_mode3, hidden_state_mode3 = model_mode3(window_feat_mode3, initial_state=hidden_state_mode3, training=True)
            new_damp_seq_mode3 = pred_out_mode3[0][:, 0]
            new_freq_seq_mode3 = pred_out_mode3[0][:, 1]

        # Contributions LSTM
        if N_DOF > 1 and model_contrib is not None:
            pred_phi, hidden_state_contrib = model_contrib(window_feat_contrib, initial_state=hidden_state_contrib, training=True)
            phi_mode1_t = pred_phi[:, :, 0:N_DOF]
            phi_mode2_t = pred_phi[:, :, N_DOF:2*N_DOF] if ACTIVE_MODES >= 2 else tf.zeros([1, WINDOW_SIZE, N_DOF], dtype=tf.float32)
            phi_mode3_t = pred_phi[:, :, 2*N_DOF:3*N_DOF] if ACTIVE_MODES >= 3 else tf.zeros([1, WINDOW_SIZE, N_DOF], dtype=tf.float32)
            phi_mode1_t_trans = tf.transpose(phi_mode1_t[0])
            phi_mode2_t_trans = tf.transpose(phi_mode2_t[0]) if ACTIVE_MODES >= 2 else tf.zeros([N_DOF, WINDOW_SIZE], dtype=tf.float32)
            phi_mode3_t_trans = tf.transpose(phi_mode3_t[0]) if ACTIVE_MODES >= 3 else tf.zeros([N_DOF, WINDOW_SIZE], dtype=tf.float32)
        else:
            phi_mode1_t_trans = tf.ones([N_DOF, WINDOW_SIZE], dtype=tf.float32)  # For N_DOF=1, scale is 1
            phi_mode2_t_trans = tf.zeros([N_DOF, WINDOW_SIZE], dtype=tf.float32)
            phi_mode3_t_trans = tf.zeros([N_DOF, WINDOW_SIZE], dtype=tf.float32)

        # === STAGE 4: FINAL SDOF SOLVE ===
        kt_mode1 = (2.0 * tf.cast(tf.constant(np.pi), tf.float32) * new_freq_seq_mode1)**2
        x_mode1, v_mode1, a_int_mode1 = SDOF_aceleracion_promedio_tf(1.0, kt_mode1, new_damp_seq_mode1, p_n1_mode1, Fs, x0=modal_y_prev_mode1, v0=modal_ydot_prev_mode1)
        
        x_mode2, v_mode2, a_int_mode2 = tf.zeros_like(x_mode1), tf.zeros_like(v_mode1), tf.zeros_like(a_int_mode1)
        if ACTIVE_MODES >= 2:
            kt_mode2 = (2.0 * tf.cast(tf.constant(np.pi), tf.float32) * new_freq_seq_mode2)**2
            x_mode2, v_mode2, a_int_mode2 = SDOF_aceleracion_promedio_tf(1.0, kt_mode2, new_damp_seq_mode2, p_n1_mode2, Fs, x0=modal_y_prev_mode2, v0=modal_ydot_prev_mode2)
        
        x_mode3, v_mode3, a_int_mode3 = tf.zeros_like(x_mode1), tf.zeros_like(v_mode1), tf.zeros_like(a_int_mode1)
        if ACTIVE_MODES >= 3:
            kt_mode3 = (2.0 * tf.cast(tf.constant(np.pi), tf.float32) * new_freq_seq_mode3)**2
            x_mode3, v_mode3, a_int_mode3 = SDOF_aceleracion_promedio_tf(1.0, kt_mode3, new_damp_seq_mode3, p_n1_mode3, Fs, x0=modal_y_prev_mode3, v0=modal_ydot_prev_mode3)

        # === STAGE 5: COMPUTE TOTAL RESPONSE AND LOSSES ===
        mode1_contrib_acc = phi_mode1_t_trans * tf.expand_dims(a_int_mode1, axis=0)
        mode2_contrib_acc = phi_mode2_t_trans * tf.expand_dims(a_int_mode2, axis=0)
        mode3_contrib_acc = phi_mode3_t_trans * tf.expand_dims(a_int_mode3, axis=0)
        a_total_n1 = mode1_contrib_acc + mode2_contrib_acc + mode3_contrib_acc + tf.expand_dims(-p_n1_mode1, axis=0) * R

        mode1_displ_contrib = phi_mode1_t_trans * tf.expand_dims(x_mode1, axis=0)
        mode2_displ_contrib = phi_mode2_t_trans * tf.expand_dims(x_mode2, axis=0)
        mode3_displ_contrib = phi_mode3_t_trans * tf.expand_dims(x_mode3, axis=0)
        displ_total_n1 = mode1_displ_contrib + mode2_displ_contrib + mode3_displ_contrib

        loss_acc = tf.reduce_mean(tf.reduce_mean((a_total_n1 - measured_win_n1)**2, axis=1))
        var_acc = tf.reduce_mean(tf.math.reduce_variance(measured_win_n1, axis=1)) + 1e-6
        loss_acc_norm = loss_acc / var_acc
        loss_acc_weighted = ACC_LOSS_WEIGHT * loss_acc_norm

        loss_displ = tf.reduce_mean(tf.reduce_mean((displ_total_n1 - true_displ_n1)**2, axis=1))
        var_displ = tf.reduce_mean(tf.math.reduce_variance(true_displ_n1, axis=1)) + 1e-6
        loss_displ_norm = loss_displ / var_displ
        loss_displ_weighted = DISPL_LOSS_WEIGHT * loss_displ_norm

        # Smoothness losses
        smooth_loss_damp_mode1 = tf.reduce_mean(tf.maximum(tf.abs((new_damp_seq_mode1[1:] - new_damp_seq_mode1[:-1]) / (new_damp_seq_mode1[:-1] + 1e-8)) - DAMP_VARIATION_LIMIT, 0.0))
        smooth_loss_damp_mode2 = tf.reduce_mean(tf.maximum(tf.abs((new_damp_seq_mode2[1:] - new_damp_seq_mode2[:-1]) / (new_damp_seq_mode2[:-1] + 1e-6)) - DAMP_VARIATION_LIMIT, 0.0)) if ACTIVE_MODES >= 2 else tf.constant(0.0, dtype=tf.float32)
        smooth_loss_damp_mode3 = tf.reduce_mean(tf.maximum(tf.abs((new_damp_seq_mode3[1:] - new_damp_seq_mode3[:-1]) / (new_damp_seq_mode3[:-1] + 1e-6)) - DAMP_VARIATION_LIMIT, 0.0)) if ACTIVE_MODES >= 3 else tf.constant(0.0, dtype=tf.float32)
        smooth_loss_freq_mode1 = tf.reduce_mean(tf.maximum(tf.abs(new_freq_seq_mode1[1:] - new_freq_seq_mode1[:-1]) - FREQ_VARIATION_LIMIT, 0.0))
        smooth_loss_freq_mode2 = tf.reduce_mean(tf.maximum(tf.abs(new_freq_seq_mode2[1:] - new_freq_seq_mode2[:-1]) - FREQ_VARIATION_LIMIT, 0.0)) if ACTIVE_MODES >= 2 else tf.constant(0.0, dtype=tf.float32)
        smooth_loss_freq_mode3 = tf.reduce_mean(tf.maximum(tf.abs(new_freq_seq_mode3[1:] - new_freq_seq_mode3[:-1]) - FREQ_VARIATION_LIMIT, 0.0)) if ACTIVE_MODES >= 3 else tf.constant(0.0, dtype=tf.float32)
        
        smooth_loss_damp = smooth_loss_damp_mode1 + smooth_loss_damp_mode2 + smooth_loss_damp_mode3
        smooth_loss_freq = smooth_loss_freq_mode1 + smooth_loss_freq_mode2 + smooth_loss_freq_mode3

        # Regularization for mode shapes (skip for N_DOF=1)
        loss_phi_reg = tf.constant(0.0, dtype=tf.float32)
        if N_DOF > 1:
            phi_mode1_baseline = tf.repeat(tf.expand_dims(PHI[:, 0], axis=1), WINDOW_SIZE, axis=1)
            phi_mode2_baseline = tf.repeat(tf.expand_dims(PHI[:, 1], axis=1), WINDOW_SIZE, axis=1) if ACTIVE_MODES >= 2 else tf.zeros_like(phi_mode1_baseline)
            phi_mode3_baseline = tf.repeat(tf.expand_dims(PHI[:, 2], axis=1), WINDOW_SIZE, axis=1) if ACTIVE_MODES >= 3 else tf.zeros_like(phi_mode1_baseline)
            loss_phi_reg = tf.reduce_mean((phi_mode1_t_trans - phi_mode1_baseline)**2 +
                                          (phi_mode2_t_trans - phi_mode2_baseline)**2 +
                                          (phi_mode3_t_trans - phi_mode3_baseline)**2)
        loss_phi_reg_weighted = PHI_REG_LOSS_WEIGHT * loss_phi_reg

        # Mode sum constraint (skip for N_DOF=1)
        loss_mode_sum = tf.constant(0.0, dtype=tf.float32)
        if N_DOF > 1:
            mode_sum = phi_mode1_t_trans + phi_mode2_t_trans + phi_mode3_t_trans
            r_j = tf.ones([N_DOF, WINDOW_SIZE], dtype=tf.float32)
            loss_mode_sum = tf.reduce_mean((mode_sum - r_j)**2)
        loss_mode_sum_weighted = MODE_SUM_LOSS_WEIGHT * loss_mode_sum

        loss_win = (loss_acc_weighted + loss_displ_weighted +
                    SMOOTH_WEIGHT_DAMP * smooth_loss_damp +
                    SMOOTH_WEIGHT_FREQ * smooth_loss_freq +
                    loss_phi_reg_weighted + loss_mode_sum_weighted)

    # === STAGE 6: COMPUTE AND APPLY GRADIENTS ===
    trainable_vars = []
    if N_DOF > 1 and model_contrib is not None:
        trainable_vars += model_contrib.trainable_variables
    if ACTIVE_MODES >= 1:
        trainable_vars += model_mode1.trainable_variables
    if ACTIVE_MODES >= 2 and model_mode2 is not None:
        trainable_vars += model_mode2.trainable_variables
    if ACTIVE_MODES >= 3 and model_mode3 is not None:
        trainable_vars += model_mode3.trainable_variables

    grads = tape.gradient(loss_win, trainable_vars)
    grad_norm_tf = tf.linalg.global_norm(grads) if any(g is not None for g in grads) else tf.constant(0.0)
    if use_clip:
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=500.0)
    optimizer.apply_gradients(zip(grads, trainable_vars))

    # === STAGE 7: UPDATE AND RETURN STATES ===
    new_modal_y_prev_mode1, new_modal_ydot_prev_mode1 = x_mode1[-1], v_mode1[-1]
    new_modal_y_prev_mode2, new_modal_ydot_prev_mode2 = x_mode2[-1], v_mode2[-1]
    new_modal_y_prev_mode3, new_modal_ydot_prev_mode3 = x_mode3[-1], v_mode3[-1]

    return (
        new_modal_y_prev_mode1, new_modal_ydot_prev_mode1, new_damp_seq_mode1, new_freq_seq_mode1,
        new_modal_y_prev_mode2, new_modal_ydot_prev_mode2, new_damp_seq_mode2, new_freq_seq_mode2,
        new_modal_y_prev_mode3, new_modal_ydot_prev_mode3, new_damp_seq_mode3, new_freq_seq_mode3,
        hidden_state_mode1, hidden_state_mode2, hidden_state_mode3, hidden_state_contrib,
        loss_win, loss_acc_weighted, loss_displ_weighted,
        SMOOTH_WEIGHT_DAMP * smooth_loss_damp, SMOOTH_WEIGHT_FREQ * smooth_loss_freq,
        loss_phi_reg_weighted, loss_mode_sum_weighted, grad_norm_tf,
        a_total_n1, displ_total_n1, phi_mode1_t_trans, phi_mode2_t_trans, phi_mode3_t_trans
    )