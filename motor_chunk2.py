import nengo
import nengolib
import numpy as np
import temporal

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
        nengolib.synapses.PureDelay(delay, order=8), n_neurons=50, synapse=0.02,
        radii=0.1, dt=dt, output_synapse=0.02)    
        
        
    stim = nengo.Node(lambda t: 1 if t%1.0<0.05 else 0)
    
    nengo.Connection(stim, subnet.input)
    
    post = nengo.Ensemble(n_neurons=100, dimensions=4)
    
    td = temporal.TemporalDecoder(subnet, seed=1)
    td.decode(post, filename='t3', stim_time=5.0, stim_node=subnet.input, stim_func=lambda t: 1 if t%1.0<0.05 else 0,
                    target_func=target_func)
