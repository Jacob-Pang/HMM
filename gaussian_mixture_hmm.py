import numpy as np
import markov.hmm_utility.gmhmm_utility as utility

from scipy.cluster.vq import kmeans2
from scipy.special import logsumexp
from markov.hidden_markov_model import HiddenMarkovMixin
from markov.stats import mvgaussian_logprobdensity

class GaussianMixtureHMM(HiddenMarkovMixin):
    """ Gaussian Mixture Hidden Markov Model (GMHMM) with continuous distribution
        attributes:
            n_states: int
                the number of states in the modelled time-series.
                each state is modelled by an separate gaussian mixture.
            m_components: optional int
                the number of components in each gaussian mixture, note: where [@param m_components]
                is one, the gaussian mixture is a singular multivariate gaussian distribution.
                must be more than zero.
                defaults to one.
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
        notes:
            convergence may not be monotonic due to positive-definite corrections performed
            and floating-point errors in cholesky decomposition on the covariance matrices.
    """
    def __init__(self, n_states: int, m_components: int = 1, max_order: int = 1,
        init_logprobdist: np.ndarray = None, prior_transition_logprobdist: np.ndarray = None):
        super().__init__(n_states, max_order, init_logprobdist=init_logprobdist,
                prior_transition_logprobdist=prior_transition_logprobdist)

        assert m_components > 0, (
            "Exception: GaussianMixtureHMM\n" +
            "    undefined number of gaussian mixture components.\n" +
            "    [@param m_components] must be more than zero."
        )

        self.m_components = m_components

    def construct_temporary_buffers(self, max_order:int, num_observations:int, k_sequences:int):
        super().construct_temporary_buffers(max_order, num_observations, k_sequences)

        self.set_temporary_buffer(
            k_statecomponent_probgraph_vector=np.empty(num_observations * self.n_states
                    * self.m_components, dtype=float),
            k_statecomponent_probgraph_shape=np.asarray((num_observations, self.n_states,
                    self.m_components), dtype=int),

            k_state_weights_vector=np.empty(k_sequences * self.n_states, dtype=float),
            k_state_weights_shape=np.asarray((k_sequences, self.n_states), dtype=int),

            k_mixture_weights_vector=np.empty(k_sequences * self.mixture_weights.size,
                    dtype=float),
            k_mixture_weights_shape=np.asarray((k_sequences, *self.mixture_weights.shape),
                    dtype=int),
            k_mixture_means_vector=np.zeros(k_sequences * self.mixture_means.size,
                    dtype=float),
            k_mixture_means_shape=np.asarray((k_sequences, *self.mixture_means.shape),
                    dtype=int),
            k_mixture_covariance_vector=np.zeros(k_sequences * self.mixture_covariance.size,
                    dtype=float),
            k_mixture_covariance_shape=np.asarray((k_sequences, *self.mixture_covariance.shape),
                    dtype=int),
            
            means_product_tempbuffer_vector=np.empty(num_observations * self.n_states
                    * self.m_components * self.d_features, dtype=float),
            means_product_tempbuffer_shape=np.asarray((num_observations, self.n_states,
                    self.m_components, self.d_features), dtype=int),
            covariance_product_tempbuffer_vector=np.empty(num_observations * self.n_states
                    * self.m_components * self.d_features ** 2, dtype=float),
            covariance_product_tempbuffer_shape=np.asarray((num_observations, self.n_states,
                    self.m_components, self.d_features, self.d_features), dtype=int)
        )

    def init_emission_logprobdist(self, rand_generator: np.random.Generator, observations: np.ndarray):
        self.d_features = observations.shape[1]

        # mixture_means: shape[N x D]
        mixture_means, mixture_states = kmeans2(observations, rand_generator.choice(observations,
                size=self.n_states, axis=0))
        mixture_distortion = np.abs(observations - mixture_means[mixture_states])
    
        if np.sum(mixture_distortion) == 0:
            print("Warning: GaussianMixtureHMM.init_emission_logprobdist")
            print("    zero distortion found during state-clustering from observations.")
            print("    observations do not exhibit stochastic behaviour.")

        # mixture_means: shape[N x M x D]
        self.mixture_means = np.stack([
            component_means[np.newaxis,:] + (rand_generator.random((self.m_components,
                    self.d_features)) - 0.5) * np.average(mixture_distortion[mixture_states == state],
                    axis=0)[np.newaxis,:]
            for state, component_means in enumerate(mixture_means)
        ], axis=0)

        self.mixture_covariance = np.nan_to_num(np.stack([
            np.ones((self.m_components, self.d_features, self.d_features)) *
                np.cov(observations[mixture_states == state].transpose())[np.newaxis,...]
            for state in range(self.n_states)
        ], axis=0))

        self.mixture_weights = rand_generator.random((self.n_states, self.m_components))
        self.mixture_weights /= np.sum(self.mixture_weights, axis=1)[:,np.newaxis]

    def compute_emission_logprobgraph(self, observations: np.ndarray) -> np.ndarray:
        assert hasattr(self, "mixture_means")
        assert hasattr(self, "mixture_covariance")
        assert hasattr(self, "mixture_weights")

        self.mixture_emission_logprobgraph = np.stack([
            np.stack([
                mvgaussian_logprobdensity(observations, component_means, component_covariance)
                for component_means, component_covariance in zip(mixture_means, mixture_covariance)
            ], axis=1) for mixture_means, mixture_covariance in zip(self.mixture_means,
                    self.mixture_covariance)
        ], axis=1)

        with np.errstate(divide="ignore"):
            self.mixture_emission_logprobgraph += np.log(self.mixture_weights)[np.newaxis,...] 
  
        return logsumexp(self.mixture_emission_logprobgraph, axis=-1)

    def maximization_emission_logprobdist(self, state_logprobgraph_vector: np.ndarray,
        state_logprobgraph_shape: np.ndarray, observation_logprobgraph: np.ndarray,
        observations: np.ndarray, k_sequence_indexes: np.ndarray, k_sequence_sizes: np.ndarray,
        max_order: int, k_sequences: int):
        assert hasattr(self, "mixture_emission_logprobgraph")
        reducedstate_logprobgraph = state_logprobgraph_vector.reshape(state_logprobgraph_shape)

        # shape[O x N ^(max_order - 1) x N] -> shape[O x N]
        for _ in range(reducedstate_logprobgraph.ndim - 2):
            reducedstate_logprobgraph = logsumexp(reducedstate_logprobgraph, axis=1)

        mixture_weights_vector = self.mixture_weights.reshape(-1)
        mixture_means_vector = self.mixture_means.reshape(-1)
        mixture_covariance_vector = self.mixture_covariance.reshape(-1)

        utility.maximization_gaussian_mixture(
            reducedstate_logprobgraph.reshape(-1), np.asarray(reducedstate_logprobgraph.shape, dtype=int),
            self.mixture_emission_logprobgraph.reshape(-1), np.asarray(self.mixture_emission_logprobgraph
                    .shape, dtype=int),
            mixture_weights_vector, np.asarray(self.mixture_weights.shape, dtype=int),
            mixture_means_vector, np.asarray(self.mixture_means.shape, dtype=int),
            mixture_covariance_vector, np.asarray(self.mixture_covariance.shape, dtype=int),
            observations.reshape(-1), np.asarray(observations.shape, dtype=int),
            self.k_statecomponent_probgraph_vector, self.k_statecomponent_probgraph_shape,
            self.k_state_weights_vector, self.k_state_weights_shape,
            self.k_mixture_weights_vector, self.k_mixture_weights_shape,
            self.k_mixture_means_vector, self.k_mixture_means_shape,
            self.k_mixture_covariance_vector, self.k_mixture_covariance_shape,
            self.means_product_tempbuffer_vector, self.means_product_tempbuffer_shape,
            self.covariance_product_tempbuffer_vector, self.covariance_product_tempbuffer_shape,
            observation_logprobgraph, k_sequence_indexes, k_sequence_sizes, max_order,
            observations.shape[0], self.n_states, self.m_components, self.d_features, k_sequences
        )

        self.mixture_weights = mixture_weights_vector.reshape(self.mixture_weights.shape)
        self.mixture_means = mixture_means_vector.reshape(self.mixture_means.shape)
        self.mixture_covariance = mixture_covariance_vector.reshape(self.mixture_covariance.shape)

    def sequence_expectation(self, state_sequence):
        return np.sum(self.mixture_means * self.mixture_weights[...,np.newaxis],
                axis=1)[state_sequence]


if __name__ == "__main__":
    pass