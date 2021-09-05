import time
import numpy as np
import markov.hmm_utility.hmm_utility as utility

from abc import abstractmethod, ABCMeta
from scipy.special import logsumexp

class HiddenMarkovMixin(metaclass=ABCMeta):
    def __init__(self, n_states: int, max_order: int = 1, init_logprobdist: np.ndarray = None,
        prior_transition_logprobdist: np.ndarray = None):

        assert max_order > 0, (
            "Exception: HiddenMarkov\n" + 
            "    cannot have zeroth-order markov process,\n" +
            "    [@param max_order] must be more than zero."
        )

        if init_logprobdist is None:
            init_logprobdist = np.log(np.ones(n_states) / n_states)

        assert init_logprobdist.size == n_states

        self.n_states  = n_states
        self.max_order = max_order

        self.init_logprobdist = init_logprobdist
        self.prior_transition_logprobdist = prior_transition_logprobdist

        self.temporary_buffers = []

    def init_k_observations(self, k_observations) -> tuple:
        if isinstance(k_observations, np.ndarray):
            if k_observations.ndim == 3:
                k_observations = [*k_observations]
            else:
                k_observations = [k_observations]
        
        observations = np.concatenate(k_observations, axis=0)
        
        if observations.ndim < 2:
            observations = observations.reshape((-1, 1))

        k_sequence_sizes = np.asarray([observation.shape[0] for observation
                in k_observations], dtype=int)

        k_sequence_indexes = np.append([0], k_sequence_sizes[1::])

        return observations, k_sequence_indexes, k_sequence_sizes

    def init_transition_logprobdist(self, rand_generator: np.random.Generator):
        if not self.prior_transition_logprobdist is None:
            self.transition_logprobdist =  self.prior_transition_logprobdist
            return

        distribution_shape = [self.n_states] * (self.max_order + 1)
        self.transition_logprobdist = rand_generator.random(distribution_shape)

        self.transition_logprobdist /= np.sum(self.transition_logprobdist,
                axis=-1)[...,np.newaxis]

        self.transition_logprobdist = np.log(self.transition_logprobdist)

    def set_temporary_buffer(self, **kwargs):
        for kwarg, arg in kwargs.items():
            self.temporary_buffers.append(kwarg)
            setattr(self, kwarg, arg)

    def construct_temporary_buffers(self, max_order:int, num_observations: int, k_sequences:int):
        graph_shape = np.ones(max_order, dtype=int) * self.n_states
        graph_size  = np.prod(graph_shape)

        self.set_temporary_buffer(
            init_logprobdist_vector=np.copy(self.init_logprobdist),
            init_logprobdist_shape=np.asarray(self.init_logprobdist.shape),

            transition_logprobdist_vector=self.transition_logprobdist.reshape(-1),
            transition_logprobdist_shape=np.asarray(self.transition_logprobdist.shape),

            emission_logprobgraph_shape=np.array([num_observations, self.n_states]),

            forward_logprobgraph_vector=np.zeros(num_observations * graph_size, dtype=float),
            forward_logprobgraph_shape=np.asarray((num_observations, *graph_shape)),

            backward_logprobgraph_vector=np.zeros(num_observations * graph_size, dtype=float),
            backward_logprobgraph_shape=np.asarray((num_observations, *graph_shape)),
            # padded to adopt same zeroth axis shape
            transition_logprobgraph_vector=np.zeros(num_observations * graph_size
                    * self.n_states, dtype=float),
            transition_logprobgraph_shape=np.asarray((num_observations, *graph_shape,
                    self.n_states)),

            state_logprobgraph_vector=np.zeros(num_observations * graph_size, dtype=float),
            state_logprobgraph_shape=np.asarray((num_observations, *graph_shape)),

            k_transition_tempbuffer_vector = np.zeros(k_sequences * graph_size
                    * self.n_states, dtype=float),
            k_transition_tempbuffer_shape  = np.asarray((k_sequences, *graph_shape,
                    self.n_states)),

            k_state_tempbuffer_vector=np.zeros(k_sequences * graph_size, dtype=float),
            k_state_tempbuffer_shape=np.asarray((k_sequences, *graph_shape)),

            observation_logprobgraph=np.zeros(k_sequences, dtype=float)
        )
        
        self.set_temporary_buffer(
            transition_tempbuffer_vector=np.copy(self.transition_logprobdist_vector),
            transition_tempbuffer_shape=np.copy(self.transition_logprobdist_shape)
        )

    def deconstruct_temporary_buffers(self):
        for temporary_buffer in self.temporary_buffers:
            delattr(self, temporary_buffer)
        
        self.temporary_buffers = []

    def train_forward_backward(self, k_observations: np.ndarray, max_iter: int = 1000,
        logtolerance: float = 0.00001, convergence_passes: int = 10, rand_seed: int = None,
        verbose: int = 1):
        
        rand_generator = np.random.default_rng(rand_seed)
        observations, k_sequence_indexes, k_sequence_sizes = self.init_k_observations(k_observations)

        k_sequences = k_sequence_indexes.size
        num_observations = observations.shape[0]
        max_order = min(min(k_sequence_sizes), self.max_order)
        graph_shape = np.ones(max_order, dtype=int) * self.n_states

        self.init_transition_logprobdist(rand_generator)
        self.init_emission_logprobdist(rand_generator, observations)
        self.construct_temporary_buffers(max_order, num_observations, k_sequences)

        printcount_padding = len(str(max_iter))
        training_begtime = time.time()
        convergence_count, observation_logprob = 0, None

        for emiter in range(max_iter):
            emission_logprobgraph_vector = self.compute_emission_logprobgraph(
                    observations).reshape(-1)

            utility.forward_backward_emiter(
                self.init_logprobdist_vector, self.init_logprobdist_shape,
                self.transition_logprobdist_vector, self.transition_logprobdist_shape,
                emission_logprobgraph_vector, self.emission_logprobgraph_shape,
                self.forward_logprobgraph_vector, self.forward_logprobgraph_shape,
                self.backward_logprobgraph_vector, self.backward_logprobgraph_shape,
                self.transition_logprobgraph_vector, self.transition_logprobgraph_shape,
                self.state_logprobgraph_vector, self.state_logprobgraph_shape,
                self.transition_tempbuffer_vector, self.transition_tempbuffer_shape,
                self.k_transition_tempbuffer_vector, self.k_transition_tempbuffer_shape,
                self.k_state_tempbuffer_vector, self.k_state_tempbuffer_shape,
                self.observation_logprobgraph, k_sequence_indexes, k_sequence_sizes,
                max_order, num_observations, self.n_states, k_sequences
            )

            self.maximization_emission_logprobdist(self.state_logprobgraph_vector,
                np.asarray((num_observations, *graph_shape)), self.observation_logprobgraph,
                observations, k_sequence_indexes, k_sequence_sizes, max_order, k_sequences)

            # reshaped to new transition dsitrbution shape
            self.transition_logprobdist_shape = self.transition_logprobdist_shape[:max_order + 1]

            if not observation_logprob is None and abs(logsumexp(self.observation_logprobgraph)
                - observation_logprob) < logtolerance:
                convergence_count += 1
            else:
                convergence_count = 0

            observation_logprob = logsumexp(self.observation_logprobgraph)

            if verbose:
                print(
                    f" ITER {emiter + 1:>{printcount_padding}}/{max_iter} || "
                    + f"ABS LOG Σ[P(O|λ)] {round(abs(observation_logprob), 5):<15} || "
                    + f"TE {round(time.time() - training_begtime)}s",
                    end=("\n" if emiter == max_iter - 1 or convergence_count
                            == convergence_passes else "\r")
                )

            if convergence_count == convergence_passes:
                break
        
        self.transition_logprobdist = np.reshape(self.transition_logprobdist_vector[:self.n_states
                ** (max_order + 1)], self.transition_logprobdist_shape)
        
        self.deconstruct_temporary_buffers()
        
        if convergence_count != convergence_passes:
            print(f"Warning: HiddenMarkov.train_forward_backward")
            print(f"    did not converge for {max_iter} iterations and logtolerance of {logtolerance}.")

    def predict_viterbi(self, predict_timesteps: int, prior_observations: np.ndarray = None
        ) -> np.ndarray:
        max_order = self.transition_logprobdist.ndim - 1
        
        # ndvector init and temporary buffer construction
        transition_logprobdist_vector = self.transition_logprobdist.reshape(-1)
        transition_logprobdist_shape  = np.asarray(self.transition_logprobdist.shape)

        transition_tempbuffer_vector = np.copy(transition_logprobdist_vector)
        transition_tempbuffer_shape  = np.copy(transition_logprobdist_shape)

        if not prior_observations is None:
            prior_observations, _, _ = self.init_k_observations(prior_observations)
            num_observations = prior_observations.shape[0]

            init_logprobdist_vector = self.init_logprobdist
            init_logprobdist_shape  = np.asarray(self.init_logprobdist.shape)

            emission_logprobgraph = self.compute_emission_logprobgraph(prior_observations)
            emission_logprobgraph_vector = emission_logprobgraph.reshape(-1)
            emission_logprobgraph_shape  = np.asarray(emission_logprobgraph.shape)

            maxprob_graphpath = np.ones(num_observations, dtype=int) * -1

            max_logprob_decoded = utility.viterbi_decode(
                init_logprobdist_vector, init_logprobdist_shape,
                transition_logprobdist_vector, transition_logprobdist_shape,
                emission_logprobgraph_vector, emission_logprobgraph_shape,
                transition_tempbuffer_vector, transition_tempbuffer_shape,
                maxprob_graphpath, max_order, num_observations, self.n_states
            )
        else:
            num_observations = 1
            maxprob_graphpath = np.array([np.argmax(self.init_logprobdist)], dtype=int)
        
        predicted_state_sequence = np.ones(predict_timesteps, dtype=int)
        utility.predict_states(
            transition_logprobdist_vector, transition_logprobdist_shape,
            transition_tempbuffer_vector, transition_tempbuffer_shape,
            maxprob_graphpath, predicted_state_sequence,
            predict_timesteps, max_order, num_observations, self.n_states
        )

        return self.sequence_expectation(predicted_state_sequence)

    @abstractmethod
    def init_emission_logprobdist(self, rand_generator: np.random.Generator, observations: np.ndarray):
        pass

    @abstractmethod
    def compute_emission_logprobgraph(self, observations: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def maximization_emission_logprobdist(self, state_logprobgraph_vector: np.ndarray,
        state_logprobgraph_shape: np.ndarray, observation_logprobgraph: np.ndarray,
        observations: np.ndarray, k_sequence_indexes: np.ndarray, k_sequence_sizes: np.ndarray,
        max_order: int, k_sequences: int):
        pass

    @abstractmethod
    def sequence_expectation(self, state_sequence: np.ndarray) -> np.ndarray:
        pass

class HiddenMarkovModel(HiddenMarkovMixin):
    """ Hidden Markov Model (HMM) with discrete distribution
        attributes:
            n_states: int
                the number of states in the modelled time-series.
            max_order: optional int
                the maximal order of the markov process which determines the number of prior
                timesteps that have influence on the current state.
                must be more than zero.
                defaults to one.
            init_logprobdist: optional np.ndarray, shape[n_states]
                the discrete probability distribution in Log-domain of the initial state of
                the time-series with respect to the [@param n_states] states
                defaults to uniform distribution.
            prior_transition_logprobdist: optional np.ndarray, shape[n_states ^ (max_order + 1)]
                the transition distribution in Log-domain to initialize training with.
                defaults to randomly generated transition matrix at training runtime.
    """
    def init_emission_logprobdist(self, rand_generator: np.random.Generator, observations: np.ndarray):
        self.observation_set = np.unique(observations, axis=0)
        self.emission_logprobdist = rand_generator.random((self.observation_set.shape[0],
                self.n_states))
        
        self.emission_logprobdist = np.log(self.emission_logprobdist / np.sum(
                self.emission_logprobdist, axis=0)[np.newaxis,...])

    def compute_emission_logprobgraph(self, observations: np.ndarray) -> np.ndarray:
        assert hasattr(self, "emission_logprobdist")
        assert hasattr(self, "observation_set")

        emission_logprob_mapper = {
            hash(observation.tostring()): emission_logprob for observation, emission_logprob
            in zip(self.observation_set, self.emission_logprobdist)
        }

        observation_set, indexes = np.unique(observations, return_inverse=True, axis=0)
        emission_logprobdist = np.full((observation_set.shape[0], self.n_states), np.nan)
        prior_observations, new_observations = self.observation_set.shape[0], 0
        
        for index, observation in enumerate(observation_set):
            try:
                emission_logprobdist[index] = emission_logprob_mapper[hash(observation.tostring())]
            except:
                new_observations += 1

        if new_observations:
            emission_logprobdist[~np.isnan(emission_logprobdist)] += np.log(prior_observations
                    / (new_observations + prior_observations))
            emission_logprobdist[np.isnan(emission_logprobdist)] = np.log(new_observations
                    / (new_observations + prior_observations))

        return emission_logprobdist[indexes]

    def maximization_emission_logprobdist(self, state_logprobgraph_vector: np.ndarray,
        state_logprobgraph_shape: np.ndarray, observation_logprobgraph: np.ndarray,
        observations: np.ndarray, k_sequence_indexes: np.ndarray, k_sequence_sizes: np.ndarray,
        max_order: int, k_sequences: int):

        state_logprobgraph = state_logprobgraph_vector.reshape(state_logprobgraph_shape)

        # k-observation log-probability weighted
        for kindex, ksize, observation_logprob in zip(k_sequence_indexes, k_sequence_sizes,
            observation_logprobgraph):

            state_logprobgraph[kindex:kindex + ksize] += observation_logprob
        
        # shape[O x N ^(max_order - 1) x N] -> shape[O x N]
        for _ in range(state_logprobgraph.ndim - 2):
            state_logprobgraph = logsumexp(state_logprobgraph, axis=1)

        observation_set, indexes = np.unique(observations, return_inverse=True, axis=0)

        self.emission_logprobdist = np.stack([logsumexp(state_logprobgraph[indexes == k], axis=0)
                for k in range(observation_set.shape[0])], axis=0).reshape(-1)
        
        prior_emission_logprob = logsumexp(state_logprobgraph, axis=0).reshape(-1)
        utility.logprodexp_broadcast(self.emission_logprobdist, -prior_emission_logprob, 0)

        self.observation_set = observation_set
        self.emission_logprobdist = np.reshape(self.emission_logprobdist,
                (observation_set.shape[0], self.n_states))

    def sequence_expectation(self, state_sequence):
        assert hasattr(self, "emission_logprobdist")
        assert hasattr(self, "observation_set")

        state_expectations = np.zeros(self.observation_set.shape)
        for state in range(self.observation_set.shape[0]):
            state_expectations[state] = self.observation_set[np.argmax(
                    self.emission_logprobdist[:,state])]

        return state_expectations[state_sequence]


if __name__ == "__main__":
    pass