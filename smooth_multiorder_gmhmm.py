import numpy as np
from scipy.special import logsumexp
from markov.gaussian_mixture_hmm import GaussianMixtureHMM
from markov.stats import savgolsmooth_derivatives, mvgaussian_logprobdensity

class SmoothMultiorderGMHMM(GaussianMixtureHMM):
    """ Smooth Multi-Order Gaussian Mixture Hidden Markov Model (GMHMM) with continuous distribution
        and smoothed discrete-time derivatives.
        maintains states with multiple derivative orders.

        attributes:
            n_states: int
                the number of zeroth-order states in the modelled time-series.
                the number of states in higher orders are reduced by one per differentiation order
                to ensure parsimony, the model should be explained by lower powers more than
                higher powers.
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
            derivative_max_order: optional int
                the maximal order of derivatives to compute for states and observations.
                must be non-negative.
                defaults to one.
            init_logprobdist: optional np.ndarray, shape[n_states]
                the discrete probability distribution in Log-domain of the initial state of
                the time-series with respect to the [@param n_states] states
                defaults to uniform distribution.
            prior_transition_logprobdist: optional np.ndarray, shape[n_states ^ (max_order + 1)]
                the transition distribution in Log-domain to initialize training with.
                defaults to randomly generated transition matrix at training runtime.
            smoothing_mode: optional str {"savitzky_golay"}
                defaults to "savitzky_golay".
            recursive_smoothing: optional int
                whether to apply smoothing on every order of differentiation (True) or only on
                the zeroth-order (False).
                defaults to False.
            savgol_window: optional int
                defaults to 5.
            savgol_polyorder: optional int
                defaults to 3.
        notes:
            convergence may not be monotonic due to positive-definite corrections performed
            and floating-point errors in cholesky decomposition on the covariance matrices.
    """
    def __init__(self, n_states: int, m_components: int = 1, max_order: int = 1,
        derivative_orders: int = 1, init_logprobdist: np.ndarray = None,
        prior_transition_logprobdist: np.ndarray = None, smoothing_mode: str = "savitzky_golay",
        recursive_smoothing: int = 0, savgol_window: int = 5, savgol_polyorder: int = 3):

        multiorder_n_states = n_states
        derivative_n_states = [n_states]

        for order in range(1, derivative_orders + 1):
            m_states = max(n_states - order, 0)
            if not m_states: break

            multiorder_n_states += m_states
            derivative_n_states.append(m_states)

        super().__init__(
            multiorder_n_states, m_components=m_components, max_order=max_order,
            init_logprobdist=init_logprobdist,
            prior_transition_logprobdist=prior_transition_logprobdist
        )

        self.derivative_n_states = np.asarray(derivative_n_states, dtype=int)
        self.derivative_indexes = np.append([0], np.cumsum(self.derivative_n_states)[:-1])
        self.derivative_orders = self.derivative_n_states.size - 1

        # smoothing parameters
        self.smoothing_mode = smoothing_mode
        self.recursive_smoothing = recursive_smoothing
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder

    # @override
    def train_forward_backward(self, k_observations: np.ndarray, max_iter: int = 1000,
        logtolerance: float = 0.00001, convergence_passes: int = 10, rand_seed: int = None,
        verbose: int = 1):
        # store temp variable overriding num_observations
        observations, k_sequence_indexes, k_sequence_sizes = self.init_k_observations(k_observations)

        self.derivative_observations = np.concatenate([
            savgolsmooth_derivatives(observations[index:index + size], self.derivative_orders,
                    self.savgol_window, self.savgol_polyorder, self.recursive_smoothing)
            for index, size in zip(k_sequence_indexes, k_sequence_sizes)
        ], axis=1)

        k_observations = [observations[index + self.derivative_orders:index + size]
                for index, size in zip(k_sequence_indexes, k_sequence_sizes)]

        super().train_forward_backward(
            k_observations, max_iter=max_iter, logtolerance=logtolerance,
            convergence_passes=convergence_passes, rand_seed=rand_seed,
            verbose=verbose
        )
        
        # remove temp variable
        delattr(self, "derivative_observations")

    def init_emission_logprobdist(self, rand_generator: np.random.Generator, observations: np.ndarray):
        assert hasattr(self, "derivative_observations")
        prior_n_states = self.n_states

        multiorder_mixture_weights = []
        multiorder_mixture_means = []
        multiorder_mixture_covariance = []

        for m_states, m_derivative in zip(self.derivative_n_states, self.derivative_observations):
            self.n_states = m_states
            super().init_emission_logprobdist(rand_generator, m_derivative)

            multiorder_mixture_weights.append(self.mixture_weights)
            multiorder_mixture_means.append(self.mixture_means)
            multiorder_mixture_covariance.append(self.mixture_covariance)

        self.n_states = prior_n_states

        self.mixture_weights = np.concatenate(multiorder_mixture_weights, axis=0)
        self.mixture_means = np.concatenate(multiorder_mixture_means, axis=0)
        self.mixture_covariance = np.concatenate(multiorder_mixture_covariance, axis=0)
    
    def compute_emission_logprobgraph(self, observations: np.ndarray) -> np.ndarray:
        assert hasattr(self, "mixture_means")
        assert hasattr(self, "mixture_covariance")
        assert hasattr(self, "mixture_weights")

        if hasattr(self, "derivative_observations"):
            multiorder_derivatives = self.derivative_observations
        else: # shape[K x O x D]
            multiorder_derivatives = savgolsmooth_derivatives(observations, self.derivative_orders,
                    self.savgol_window, self.savgol_polyorder, self.recursive_smoothing)

        self.mixture_emission_logprobgraph = np.concatenate([
            np.stack([
                np.stack([
                    mvgaussian_logprobdensity(m_derivative, component_means, component_covariance)
                    for component_means, component_covariance in zip(mixture_means, mixture_covariance)
                ], axis=1) for mixture_means, mixture_covariance in zip(self.mixture_means
                        [m_index:m_index + m_states], self.mixture_covariance
                        [m_index:m_index + m_states])
            ], axis=1) for m_states, m_index, m_derivative in zip(self.derivative_n_states,
                    self.derivative_indexes, multiorder_derivatives)
        ], axis=1)

        with np.errstate(divide="ignore"):
            self.mixture_emission_logprobgraph += np.log(self.mixture_weights)[np.newaxis,...] 

        return logsumexp(self.mixture_emission_logprobgraph, axis=-1)
    
    def maximization_emission_logprobdist(self, state_logprobgraph_vector: np.ndarray,
        state_logprobgraph_shape: np.ndarray, observation_logprobgraph: np.ndarray,
        observations: np.ndarray, k_sequence_indexes: np.ndarray, k_sequence_sizes: np.ndarray,
        max_order: int, k_sequences: int):
        assert hasattr(self, "mixture_emission_logprobgraph")
        assert hasattr(self, "derivative_observations")

        # shape[K x O x D] -> shape[O x N x D]
        observations = []

        for order, n_states in enumerate(self.derivative_n_states):
            for _ in range(n_states):
                observations.append(self.derivative_observations[order])

        observations = np.stack(observations, axis=1)

        super().maximization_emission_logprobdist(
            state_logprobgraph_vector, state_logprobgraph_shape,
            observation_logprobgraph, observations, k_sequence_indexes,
            k_sequence_sizes, max_order, k_sequences
        )

    def predict_viterbi(self, predict_timesteps: int, prior_observations: np.ndarray) -> np.ndarray:
        prior_observations, _, _ = self.init_k_observations(prior_observations)
        
        self.derivative_observations = savgolsmooth_derivatives(prior_observations,
                self.derivative_orders, self.savgol_window, self.savgol_polyorder,
                self.recursive_smoothing)

        states, expectations = super().predict_viterbi(predict_timesteps, prior_observations
                [self.derivative_orders:])

        stateorders, index = np.ones(self.n_states, dtype=int), 0
        for order, num_states in enumerate(self.derivative_n_states):
            for _ in range(num_states):
                stateorders[index] = order
                index += 1
        
        # shape[K x T x D]
        prior_timesteps = self.derivative_observations.shape[1]
        multiorder_observations = np.empty((self.derivative_orders + 1, prior_timesteps
                + predict_timesteps, self.d_features))

        multiorder_observations[:,:prior_timesteps] = self.derivative_observations
        delattr(self, "derivative_observations")

        for state, expectation in zip(states, expectations):
            order = stateorders[state]
            multiorder_observations[order,prior_timesteps] = expectation

            for cascade_order in range(order + 1, self.derivative_orders + 1):
                multiorder_observations[cascade_order,prior_timesteps] = (
                    multiorder_observations[cascade_order - 1,prior_timesteps] -
                    multiorder_observations[cascade_order - 1,prior_timesteps - 1]
                )

            for cascade_order in range(order - 1, -1, -1):
                multiorder_observations[cascade_order,prior_timesteps] = (
                    multiorder_observations[cascade_order + 1,prior_timesteps] +
                    multiorder_observations[cascade_order,prior_timesteps - 1]
                )

            prior_timesteps += 1
    
        return multiorder_observations[0,-predict_timesteps:]

    # @override
    def sequence_expectation(self, state_sequence):
        return state_sequence, super().sequence_expectation(state_sequence)


if __name__ == "__main__":
    pass