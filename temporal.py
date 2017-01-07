import os

import numpy as np
import nengo


class TemporalDecoder(object):
    def __init__(self, network, seed=None):
        self.network = network
        self.set_seed(seed)

    def set_seed(self, seed):
        self.seed = seed
        rng = np.random.RandomState(seed=seed)
        for ens in self.network.all_ensembles:
            ens.seed = rng.randint(0x7FFFFFFF)

    def decode(self, post, stim_time, target_func,
               filename=None,
               stim_node=None, stim_func=None,
               synapse=nengo.Lowpass(0.005),
               solver=nengo.solvers.LstsqL2()):
        decoder = None
        fn = None
        if filename is not None and self.seed is not None:
            fn = '%s-%08x.npy' % (filename, self.seed)
            if os.path.exists(fn):
                decoder = np.load(fn)
        if decoder is None:
            decoder = self.compute_decoder(stim_time, target_func,
                                           synapse, solver,
                                           stim_node, stim_func)

        if fn is not None:
            np.save(fn, decoder)

        offset = 0
        for i, ens in enumerate(self.network.all_ensembles):
            n = ens.n_neurons
            nengo.Connection(ens.neurons, post,
                             transform=decoder[offset:offset+n,:].T)
            offset += n

    def compute_decoder(self, stim_time, target_func, synapse, solver,
                        stim_node=None, stim_func=None):
        model = nengo.Network(add_to_container=False)
        model.networks.append(self.network)
        with model:
            if stim_func is not None:
                if stim_node is None:
                    stim_node = getattr(self.network, 'input')
                stim = nengo.Node(stim_func)
                nengo.Connection(stim, stim_node, synapse=None)
            probes = []
            for ens in model.all_ensembles:
                probes.append(nengo.Probe(ens.neurons))
        sim = nengo.simulator.Simulator(model)
        sim.run(stim_time)
        data = [sim.data[p] for p in probes]

        data = np.hstack(data)
        target = np.array([target_func(t) for t in sim.trange()], dtype=float)

        data = synapse.filt(data, y0=0)
        target = synapse.filt(target, y0=0, dt=0.001)
        decoder, solver_info = solver(data, target)

        return decoder
