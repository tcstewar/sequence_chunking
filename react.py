import nengo
import nengo.spa as spa
import numpy as np

N_states = 4        
D = 48
vocab = spa.Vocabulary(D, randomize=True)


class Environment(nengo.Node):
    def __init__(self, D, seed, interval=0.1):
        self.rng = np.random.RandomState(seed=seed)
        self.seq = np.arange(D)
        self.D = D
        self.rng.shuffle(self.seq)
        self.interval = interval
        self.loc = np.linspace(10, 90, D)
        super(Environment, self).__init__(self.update, size_in=0)

    def update(self, t):
        step = int(t / self.interval) % (self.D + 2)
        v = np.zeros(self.D)
        cx = None
        if step > 0 and step <= self.D:
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
        
    model.vision = spa.State(D, vocab=vocab)
    
    model.motor = spa.AssociativeMemory(vocab, wta_output=False)
    
    model.mem = spa.State(D)
    
    bg_scale=0.5
    model.task = spa.State(D)
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
    
    direct_scale = 1.0
    nengo.Connection(model.vision.output, model.motor.input, transform=direct_scale)
    
    
