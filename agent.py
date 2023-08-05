import tensorflow as tf

from collections import deque
from random import sample

MAX_MEMORY = 100_000

class Model(tf.keras.Model):
    def __init__(self, n_outputs, n_inputs, units=364, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.hidden1 = tf.keras.layers.Dense(units, activation='relu', input_shape=(n_inputs,),
                                             kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.output_ = tf.keras.layers.Dense(n_outputs,
                                             kernel_initializer=tf.keras.initializers.GlorotUniform())
    
    def call(self, input_):
        x = self.hidden1(input_)
        x = self.output_(x)
        
        return x

class Agent:
    
    def __init__(self, n_actions, n_inputs, batch_size, lr_schduler, lr=0.01, gamma=.99, epsilon_length=200) -> None:
        self.lr = tf.constant(lr, dtype=tf.float32)
        self.gamma = tf.constant(gamma, dtype=tf.float32)
        self.batch_size = tf.constant(batch_size, dtype=tf.int32)
        self.epsilonl = tf.constant(epsilon_length, dtype=tf.int32)
        self.epsilon = tf.Variable(0)
        self.n_games = tf.Variable(0)
        self.n_actions = tf.constant(n_actions, dtype=tf.int32)
        self.test_mode = tf.Variable(False)
        self.lr_schduler = tf.function(lr_schduler)
        
        self.model = Model(n_outputs=n_actions, n_inputs=n_inputs)
        self.optimizer = tf.optimizers.AdamW(lr)
        self.loss_fn = tf.losses.mean_squared_error
        
        self.memory = deque(maxlen=MAX_MEMORY)
    
    def predict(self, state) -> tf.Tensor:
        state = tf.cast(tf.constant([state]), tf.float32)
        
        proba = tf.random.uniform((1,), minval=0, maxval=self.epsilonl, dtype=tf.int32)
        
        self.epsilon.assign(tf.subtract(self.epsilonl, self.n_games))
        
        if proba < self.epsilon and not self.test_mode:
            return tf.random.uniform((1,), minval=0, maxval=self.n_actions, dtype=tf.int32)
        else:
            return tf.argmax(self.model(state), axis=1)
    
    def __call__(self, state):
        return self.predict(state)

    def remember(self, state, action, reward, next_state, done, truncate, preprocess=False):
        if preprocess:
            state = tf.cast(tf.constant([state]), dtype=tf.float32)
            action = tf.cast(tf.constant([[action]]), dtype=tf.float32)
            reward = tf.cast(tf.constant([[reward]]), dtype=tf.float32)
            next_state = tf.cast(tf.constant([next_state]), dtype=tf.float32)
            done = tf.cast(tf.constant([[done]]), dtype=tf.bool)
            truncate = tf.cast(tf.constant([[truncate]]), dtype=tf.bool)
        
        self.memory.append((state, action, reward, next_state, done, truncate))

    def train_step(self, states, actions, rewards, next_states, dones, truncates):
        next_q_values = self.model(next_states)
        
        runs = tf.subtract(tf.constant(1.0), tf.cast(tf.logical_or(dones, truncates), tf.float32))
        
        target = tf.add(rewards, tf.multiply(runs, tf.multiply(self.gamma, tf.reduce_max(next_q_values, axis=1))))
        
        target = tf.reshape(target, (-1, 1))
        
        mask = tf.one_hot(tf.cast(actions, tf.int32), self.n_actions)
        
        with tf.GradientTape() as tape:
            
            all_q_values = self.model(states)
            
            Q_values = tf.reduce_sum(tf.multiply(all_q_values, mask), axis=-1, keepdims=True)
            
            loss = tf.reduce_mean(self.loss_fn(target, Q_values))
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        
    def remember_and_train_short(self, state, action, reward, next_state, done, truncate):
        state = tf.cast(tf.constant([state]), dtype=tf.float32)
        action = tf.cast(tf.constant([[action]]), dtype=tf.float32)
        reward = tf.cast(tf.constant([[reward]]), dtype=tf.float32)
        next_state = tf.cast(tf.constant([next_state]), dtype=tf.float32)
        done = tf.cast(tf.constant([[done]]), dtype=tf.bool)
        truncate = tf.cast(tf.constant([[truncate]]), dtype=tf.bool)
        
        self.remember(state, action, reward, next_state, done, truncate, preprocess=False)
        self.train_short(state, action, reward, next_state, done, truncate)
    
    def train_short(self, state, action, reward, next_state, done, truncate):
        self.train_step(state, action, reward, next_state, done, truncate)
    
    def train_long(self):
        if len(self.memory) > self.batch_size:
            mini_sample = sample(self.memory, self.batch_size.numpy().item())
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones, truncates = zip(*mini_sample)
        
        states = tf.concat(states, axis=0)
        actions = tf.squeeze(tf.concat(actions, axis=0))
        rewards = tf.squeeze(tf.concat(rewards, axis=0))
        next_states = tf.concat(next_states, axis=0)
        dones = tf.squeeze(tf.concat(dones, axis=0))
        truncates = tf.squeeze(tf.concat(truncates, axis=0))
        
        self.train_step(states, actions, rewards, next_states, dones, truncates)
    
    def when_episode_done(self):
        self.n_games.assign_add(1)
        
        self.optimizer.learning_rate.assign(self.lr_schduler(self.n_games, self.optimizer.learning_rate))