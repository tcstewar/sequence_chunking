import nengo
import nengo.spa as spa
import numpy as np

n_lights = 4
t_between = 0.2
duty_cycle = 0.5

rng = np.random.RandomState(seed=1)
order = np.arange(n_lights)
rng.shuffle(order)



model = spa.SPA()
with model:
    def stimulus(t):
        if (t % t_between) > t_between * duty_cycle:
            return [0]*n_lights
        else:
            index = int(t / t_between) % n_lights
            return np.eye(n_lights)[index]
    stim = nengo.Node(stimulus)
    
    bg = nengo.networks.BasalGanglia(n_lights+1)
    thal = nengo.networks.Thalamus(n_lights+1)
    nengo.Connection(bg.output, thal.input)
    bias = nengo.Node(1)
    nengo.Connection(bias, bg.input[n_lights], transform=0.5)
    
    nengo.Connection(stim, bg.input[:n_lights])
    
    fingers = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=n_lights)
    
    nengo.Connection(thal.output[:n_lights], fingers.input)


    state = nengo.Ensemble(n_neurons=1000, dimensions=n_lights*2, radius=1.4)
    nengo.Connection(stim, state[:n_lights])
    nengo.Connection(state[:n_lights], state[n_lights:], synapse=0.1, transform=0.5)
    nengo.Connection(state[n_lights:], state[n_lights:], transform=0.9, synapse=0.1)
    
    predict = nengo.Ensemble(n_neurons=500, dimensions=n_lights)
    error = nengo.Ensemble(n_neurons=500, dimensions=n_lights)
    nengo.Connection(thal.output[:n_lights], error, transform=-1, synapse=0.05)
    nengo.Connection(predict, error, transform=1)
    
    c = nengo.Connection(state, predict, function=lambda x: [0]*n_lights, learning_rule_type=nengo.PES(pre_tau=0.05))
    nengo.Connection(error, c.learning_rule)
    
    nengo.Connection(predict, fingers.input)
    
    
    
    