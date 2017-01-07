import nengo
import nengolib
import numpy as np

def target_func(t):
    t = (t+0.95) % 1.0
    
    if 0<t<0.05:
        return [1,0,0,0]
    elif 0.1<t<0.15:
        return [0,1,0,0]
    elif 0.2<t<0.25:
        return [0,0,1,0]
    elif 0.3<t<0.35:
        return [0,0,0,1]
    else:
        return [0,0,0,0]


model = nengo.Network()
#model.config[nengo.Ensemble].neuron_type=nengo.LIFRate()
with model:
    
    delay = 0.5
    dt = 0.001
    subnet = nengolib.networks.LinearNetwork(
        nengolib.synapses.PureDelay(delay, order=4), n_neurons=200, synapse=0.02,
        radii=0.1, dt=dt, output_synapse=0.02)    
        
        
    stim = nengo.Node(lambda t: 1 if t%1.0<0.05 else 0)
    
    nengo.Connection(stim, subnet.input)
    
    
    post = nengo.Ensemble(n_neurons=100, dimensions=4)
    error = nengo.Ensemble(n_neurons=100, dimensions=4)
    target = nengo.Node(target_func)
    nengo.Connection(target, error, transform=-1)
    nengo.Connection(post, error, transform=1)
    for ens in subnet.all_ensembles:
        c = nengo.Connection(ens, post, function=lambda x: [0,0,0,0],
                             learning_rule_type=nengo.PES(learning_rate=1e-4))
        nengo.Connection(error, c.learning_rule)
        
    
    stop_learn = nengo.Node(0)
    nengo.Connection(stop_learn, error.neurons,
                     transform=-5*np.ones((error.n_neurons, 1)))
    