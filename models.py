import tensorflow as tf
from tensorflow.keras import layers

class PhysicsInformedLSTM(tf.keras.Model):
    def __init__(self, input_dim=8, hidden_dim=64, mode=1):
        super().__init__()
        self.mode = mode
        self.project = layers.TimeDistributed(layers.Dense(hidden_dim))
        self.lstm1 = layers.LSTM(hidden_dim, return_sequences=True, return_state=True)
        self.norm = layers.LayerNormalization()
        self.drop = layers.Dropout(0.05)
        self.dense1 = layers.TimeDistributed(layers.Dense(hidden_dim, activation='relu'))
        self.dense2 = layers.TimeDistributed(layers.Dense(2))  # Damping, frequency

    def call(self, x, initial_state=None, training=False):
        shortcut = self.project(x)
        if initial_state is None:
            lstm1_out, h1, c1 = self.lstm1(x, training=training)
        else:
            lstm1_out, h1, c1 = self.lstm1(x, initial_state=initial_state[:2], training=training)
        
        res = self.norm(lstm1_out + shortcut)
        res = self.drop(res, training=training)
        dense_out = self.dense1(res)
        raw = self.dense2(dense_out)
        
        rd = raw[..., 0:1]
        rf = raw[..., 1:2]
        
        # --- Damping Calculation (Unchanged) ---
        # The tanh function scales the output to the range [0.01, 0.4] for all modes.
        damp_seq = (tf.tanh(rd) + 1) / 2 * (0.4 - 0.01) + 0.01
        
        # --- Frequency Calculation (Modified for 3 Modes) ---
        # The sigmoid function scales the output based on the specified mode.
        if self.mode == 1:
            # Scales to the range [0.1, 1.7] Hz
            freq_seq = tf.sigmoid(rf) * (1.7 - 0.1) + 0.1
        elif self.mode == 2:
            # Scales to the range [2.0, 5.0] Hz
            freq_seq = tf.sigmoid(rf) * (5.0 - 2.0) + 2.0
        elif self.mode == 3:
            # NEW: Scales to the range [5.0, 10.0] Hz
            freq_seq = tf.sigmoid(rf) * (10.0 - 5.0) + 5.0
        else:
            # Default or error case if an unsupported mode is given
            freq_seq = tf.zeros_like(rf) 

        return tf.concat([damp_seq, freq_seq], axis=-1), [h1, c1]
    
class PhysicsInformedLSTM2(tf.keras.Model):
    def __init__(self, input_dim=8, hidden_dim=64, mode=1):
        super().__init__()
        self.mode = mode
        self.project = layers.TimeDistributed(layers.Dense(hidden_dim))
        self.lstm1 = layers.LSTM(hidden_dim, return_sequences=True, return_state=True)
        self.norm = layers.LayerNormalization()
        self.drop = layers.Dropout(0.05)
        self.dense1 = layers.TimeDistributed(layers.Dense(hidden_dim, activation='relu'))
        self.dense2 = layers.TimeDistributed(layers.Dense(2))  # Damping, frequency

    def call(self, x, initial_state=None, training=False):
        shortcut = self.project(x)
        if initial_state is None:
            lstm1_out, h1, c1 = self.lstm1(x, training=training)
        else:
            lstm1_out, h1, c1 = self.lstm1(x, initial_state=initial_state[:2], training=training)
        
        res = self.norm(lstm1_out + shortcut)
        res = self.drop(res, training=training)
        dense_out = self.dense1(res)
        raw = self.dense2(dense_out)
        
        rd = raw[..., 0:1]
        rf = raw[..., 1:2]
        
        # --- Damping Calculation (Unchanged) ---
        # The tanh function scales the output to the range [0.01, 0.4] for all modes.
        damp_seq = (tf.tanh(rd) + 1) / 2 * (0.4 - 0.01) + 0.01
        
        # --- Frequency Calculation (Modified for 3 Modes) ---
        # The sigmoid function scales the output based on the specified mode.
        if self.mode == 1:
            # Escala al rango [0.1, 1.5] Hz
            freq_seq = tf.sigmoid(rf) * (1.5 - 0.1) + 0.1
        elif self.mode == 2:
            # Escala al rango [1.5, 4.0] Hz
            freq_seq = tf.sigmoid(rf) * (4.0 - 1.5) + 1.5
        elif self.mode == 3:
            # Escala al rango [4.0, 10.0] Hz
            freq_seq = tf.sigmoid(rf) * (10.0 - 4.0) + 4.0
        else:
            # Caso por defecto o de error si se da un modo no soportado
            freq_seq = tf.zeros_like(rf)

        return tf.concat([damp_seq, freq_seq], axis=-1), [h1, c1]

class PhysicsInformedLSTM_Modoriginal(tf.keras.Model):
    def __init__(self, input_dim=8, hidden_dim=32, output_timesteps=100, mode=1):
        super().__init__()
        self.output_timesteps = output_timesteps
        self.mode = mode
        l2_reg = tf.keras.regularizers.l2(1e-5)
        self.project = layers.TimeDistributed(layers.Dense(hidden_dim, kernel_regularizer=l2_reg))
        self.lstm1 = layers.LSTM(
            hidden_dim,
            return_sequences=True,
            return_state=True,
            kernel_regularizer=l2_reg,
            recurrent_regularizer=l2_reg
        )
        self.lstm2 = layers.LSTM(
            hidden_dim,
            return_sequences=True,
            return_state=True,
            kernel_regularizer=l2_reg,
            recurrent_regularizer=l2_reg
        )
        self.norm = layers.LayerNormalization()
        self.drop = layers.Dropout(0.1)
        self.dense1 = layers.TimeDistributed(layers.Dense(hidden_dim, activation='relu', kernel_regularizer=l2_reg))
        self.dense3 = layers.TimeDistributed(layers.Dense(3, kernel_regularizer=l2_reg))
        self.res_proj1 = layers.TimeDistributed(layers.Dense(hidden_dim)) if input_dim != hidden_dim else None
        self.res_proj2 = layers.TimeDistributed(layers.Dense(hidden_dim))

    def call(self, x, initial_state=None, training=False):
        shortcut = self.project(x)
        if initial_state is None:
            lstm1_out, h1, c1 = self.lstm1(x, training=training)
        else:
            lstm1_out, h1, c1 = self.lstm1(x, initial_state=initial_state[:2], training=training)
        lstm2_out, h2, c2 = self.lstm2(lstm1_out, training=training)
        res = self.norm(lstm2_out + shortcut)
        res = self.drop(res, training=training)
        dense_out = self.dense1(res)
        raw = self.dense3(dense_out)
        rd = raw[..., 0:1]
        rf = raw[..., 1:2]
        rs = raw[..., 2:3]
        damp_seq = tf.sigmoid(rd) * (0.40 - 0.01) + 0.01
        if self.mode == 1:
            freq_seq = tf.sigmoid(rf) * 1.5 + 0.1
        else:
            freq_seq = tf.sigmoid(rf) * (4.1 - 2.1) + 2.1
        scale_factor_seq = tf.sigmoid(rs) * (1.01 - 0.99) + 0.99
        tf.print(f"Mode {self.mode} - Frequency seq mean/min/max:", tf.reduce_mean(freq_seq), tf.reduce_min(freq_seq), tf.reduce_max(freq_seq))
        tf.print(f"Mode {self.mode} - Damping seq mean/min/max:", tf.reduce_mean(damp_seq), tf.reduce_min(damp_seq), tf.reduce_max(damp_seq))
        tf.print(f"Mode {self.mode} - Scale factor seq mean/min/max:", tf.reduce_mean(scale_factor_seq), tf.reduce_min(scale_factor_seq), tf.reduce_max(scale_factor_seq))
        return tf.concat([damp_seq, freq_seq, scale_factor_seq], axis=-1), [h1, c1, h2, c2]

class PhysicsInformedLSTM_SDOF_Scalefactor(tf.keras.Model):
    def __init__(self, input_dim=8, hidden_dim=64, output_timesteps=100, mode=1):
        super().__init__()
        self.output_timesteps = output_timesteps
        self.mode = mode
        l2_reg = tf.keras.regularizers.l2(1e-5)
        self.project = layers.TimeDistributed(layers.Dense(hidden_dim, kernel_regularizer=l2_reg))
        self.lstm1 = layers.LSTM(
            hidden_dim,
            return_sequences=True,
            return_state=True,
            kernel_regularizer=l2_reg,
            recurrent_regularizer=l2_reg
        )
        self.norm = layers.LayerNormalization()
        self.drop = layers.Dropout(0.1)
        self.dense1 = layers.TimeDistributed(layers.Dense(hidden_dim, activation='relu', kernel_regularizer=l2_reg))
        self.dense3 = layers.TimeDistributed(layers.Dense(3, kernel_regularizer=l2_reg))
        self.res_proj1 = layers.TimeDistributed(layers.Dense(hidden_dim)) if input_dim != hidden_dim else None
        self.res_proj2 = layers.TimeDistributed(layers.Dense(hidden_dim))

    def call(self, x, initial_state=None, training=False):
        shortcut = self.project(x)
        if initial_state is None:
            lstm1_out, h1, c1 = self.lstm1(x, training=training)
        else:
            lstm1_out, h1, c1 = self.lstm1(x, initial_state=initial_state[:2], training=training)
        res = self.norm(lstm1_out + shortcut)
        res = self.drop(res, training=training)
        dense_out = self.dense1(res)
        raw = self.dense3(dense_out)
        rd = raw[..., 0:1]
        rf = raw[..., 1:2]
        rs = raw[..., 2:3]
        damp_seq = tf.sigmoid(rd) * (0.40 - 0.01) + 0.01
        if self.mode == 1:
            freq_seq = tf.sigmoid(rf) * 1.5 + 0.1
        else:
            freq_seq = tf.sigmoid(rf) * (4.1 - 2.1) + 2.1
        scale_factor_seq = tf.sigmoid(rs) * (1.01 - 0.99) + 0.99
        tf.print(f"Mode {self.mode} - Frequency seq mean/min/max:", tf.reduce_mean(freq_seq), tf.reduce_min(freq_seq), tf.reduce_max(freq_seq))
        tf.print(f"Mode {self.mode} - Damping seq mean/min/max:", tf.reduce_mean(damp_seq), tf.reduce_min(damp_seq), tf.reduce_max(damp_seq))
        tf.print(f"Mode {self.mode} - Scale factor seq mean/min/max:", tf.reduce_mean(scale_factor_seq), tf.reduce_min(scale_factor_seq), tf.reduce_max(scale_factor_seq))
        return tf.concat([damp_seq, freq_seq, scale_factor_seq], axis=-1), [h1, c1]

class CustomReduceLR:
    def __init__(self, optimizer, factor=0.5, patience=5, min_lr=1e-7, verbose=1):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_loss = float('inf')
        self.wait = 0
        self.initial_step = tf.cast(optimizer.iterations, tf.float32)

    def on_epoch_end(self, epoch, current_loss):
        current_step = tf.cast(self.optimizer.iterations, tf.float32)
        current_lr = self.optimizer.learning_rate(current_step)
        print(f"Epoch {epoch+1}, Current Loss: {current_loss:.4e}, Current LR: {current_lr.numpy():.2e}")
        if current_loss < self.best_loss - 1e-6:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(current_lr * self.factor, self.min_lr)
                self.optimizer.learning_rate.initial_lr.assign(new_lr)
                if self.verbose:
                    print(f"Epoch {epoch+1}: Reducing LR to {new_lr:.2e}")
                self.wait = 0

class AdaptiveGradientClipping:
    def __init__(self, initial_clip_norm=1.0, decay=0.9, threshold_factor=2.0):
        self.clip_norm = tf.Variable(initial_clip_norm, dtype=tf.float32, trainable=False)
        self.avg_norm = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.decay = decay
        self.threshold_factor = threshold_factor

    def update(self, grad_norm):
        self.avg_norm.assign(self.decay * self.avg_norm + (1 - self.decay) * grad_norm)
        new_clip_norm = self.threshold_factor * self.avg_norm
        self.clip_norm.assign(tf.maximum(new_clip_norm, 0.1))

    def clip_grads(self, grads):
        global_norm = tf.linalg.global_norm(grads)
        return tf.clip_by_global_norm(grads, self.clip_norm)[0], global_norm

class CustomWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, decay_steps, decay_rate=0.95):
        super().__init__()
        self.initial_lr = tf.Variable(initial_lr, dtype=tf.float32, trainable=False)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.decay_rate = decay_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.initial_lr * (step + 1.0) / self.warmup_steps
        decay_lr = self.initial_lr * tf.math.pow(self.decay_rate, tf.math.floor((step - self.warmup_steps) / self.decay_steps))
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: decay_lr
        )
    
class PhysicsInformedLSTM_Contributions(tf.keras.Model):
    def __init__(self, input_dim=12, hidden_dim=64, n_dof=6):
        super().__init__()
        self.n_dof = n_dof
        self.project = layers.TimeDistributed(layers.Dense(hidden_dim))
        self.lstm1 = layers.LSTM(hidden_dim, return_sequences=True, return_state=True)
        self.norm = layers.LayerNormalization()
        self.drop = layers.Dropout(0.2)
        self.dense1 = layers.TimeDistributed(layers.Dense(hidden_dim, activation='relu'))
        self.dense2 = layers.TimeDistributed(layers.Dense(n_dof * 2))  # phi_mode1_t and phi_mode2_t for each DOF

    def call(self, x, initial_state=None, training=False):
        shortcut = self.project(x)
        if initial_state is None:
            lstm1_out, h1, c1 = self.lstm1(x, training=training)
        else:
            lstm1_out, h1, c1 = self.lstm1(x, initial_state=initial_state[:2], training=training)
        res = self.norm(lstm1_out + shortcut)
        res = self.drop(res, training=training)
        dense_out = self.dense1(res)
        raw = self.dense2(dense_out)  # Shape: [batch, WINDOW_SIZE, n_dof * 2]
        # Apply sigmoid and scale to [-2, 2] for mode shapes
        phi = tf.sigmoid(raw) * 4.0 - 2.0  # [0,1] -> [0,4] -> [-2,2]
        return phi, [h1, c1]