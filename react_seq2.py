import nengo
import nengo.spa as spa
import numpy as np
import nengolib
import temporal

N_states = 4        
D = 48
seed = 2
np.random.seed(seed)
import random
random.seed(seed)
vocab = spa.Vocabulary(D, randomize=True)


class Environment(nengo.Node):
    def __init__(self, D, seed, interval=0.1):
        self.rng = np.random.RandomState(seed=seed)
        self.seq = np.arange(D)
        self.D = D
        self.rng.shuffle(self.seq)
        self.interval = interval
        self.loc = np.linspace(10, 90, D)
        super(Environment, self).__init__(self.update, size_in=1)

    def update(self, t, hide_last):
        step = int(t / self.interval) % (self.D + 2)
        v = np.zeros(self.D)
        cx = None
        if step > 0 and step <= self.D:
            if hide_last > 0.5 and step > 1:
                pass
            else:
                s = self.seq[step - 1] 
                v[s] = 1
                cx = self.loc[s]
        if cx is not None:
            self.update.im_func._nengo_html_ = '''
            <svg width="100%" height="100%" viewbox="0 0 100 100">
                <circle cx="{cx}" cy="50" r="10" style="stroke:red"/>
            </svg>
            '''.format(**locals())
        else:
            self.update.im_func._nengo_html_ = ''
            
        return v
        
        

model = spa.SPA()
with model:
    env = Environment(D=N_states, seed=2)
    env_trans = np.array([vocab.parse('A').v,
                          vocab.parse('B').v,
                          vocab.parse('C').v,
                          vocab.parse('D').v,]).T
    hide = nengo.Node(0)
    nengo.Connection(hide, env)
                          
                          
    model.vision = spa.State(D, vocab=vocab)
    
    model.motor = spa.AssociativeMemory(vocab, wta_output=False)
    
    model.mem = spa.State(D)
    
    bg_scale=0.5
    model.bg = spa.BasalGanglia(spa.Actions(
        'dot(vision, A) --> motor=mem*~A*%g'%bg_scale,
        'dot(vision, B) --> motor=mem*~B*%g'%bg_scale,
        'dot(vision, C) --> motor=mem*~C*%g'%bg_scale,
        'dot(vision, D) --> motor=mem*~D*%g'%bg_scale,
        '0.5 --> motor=0',
    ))
    model.thal = spa.Thalamus(model.bg)
    
    model.input = spa.Input(mem = 'A*A+B*B+C*C+D*D')
    
    nengo.Connection(env, model.vision.input, transform=env_trans)

    scale_direct = nengo.Node(lambda t, x: x[:-1]*x[-1], size_in=D+1)
    scale_chunk = nengo.Node(lambda t, x: x[:-1]*x[-1], size_in=D+1)
    
    nengo.Connection(model.vision.output, scale_direct[:-1])
    nengo.Connection(scale_direct, model.motor.input)
    ctrl_direct = nengo.Node(0)
    nengo.Connection(ctrl_direct, scale_direct[-1])
    
    
    delay = 0.8
    dt = 0.001
    chunk = nengolib.networks.LinearNetwork(
        nengolib.synapses.PureDelay(delay, order=8), n_neurons=500, synapse=0.04,
        radii=0.1, dt=dt, output_synapse=0.02)    
        
        
    chunk_scale = 1.0
    def stim_func(t):
        v = env.update(t)
        v = np.dot(env_trans, v)
        return np.dot(v, vocab.parse(vocab.keys[env.seq[0]]).v)
    def target_func(t):
        v = env.update(t)
        v = np.dot(env_trans, v)
        return v * chunk_scale 
        
        
    td = temporal.TemporalDecoder(chunk, seed=1)
    td.decode(scale_chunk[:-1], filename='t10', stim_time=5.0, stim_node=chunk.input, 
              stim_func=stim_func,
              target_func=target_func)
    nengo.Connection(model.vision.output, chunk.input, transform=[vocab.parse(vocab.keys[env.seq[0]]).v])



    nengo.Connection(scale_chunk, model.motor.input)
    ctrl_chunk = nengo.Node(0)
    nengo.Connection(ctrl_chunk, scale_chunk[-1])
