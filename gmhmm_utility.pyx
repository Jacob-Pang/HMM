import numpy as np
cimport ndvector
cimport hmm_utility_c
cimport gmhmm_utility_c

ctypedef double dtype
ctypedef ndvector.ndvector_dtype ndvector_dtype

def maximization_gaussian_mixture(
    dtype[:] reducedstate_logprobgraph_vector,      int[:] reducedstate_logprobgraph_shape,
    dtype[:] mixture_emission_logprobgraph_vector,  int[:] mixture_emission_logprobgraph_shape,
    dtype[:] mixture_weights_vector,                int[:] mixture_weights_shape,
    dtype[:] mixture_means_vector,                  int[:] mixture_means_shape,
    dtype[:] mixture_covariance_vector,             int[:] mixture_covariance_shape,
    dtype[:] k_observations_vector,                 int[:] k_observations_shape,
    dtype[:] k_statecomponent_logprobgraph_vector,  int[:] k_statecomponent_logprobgraph_shape,
    dtype[:] k_state_weights_vector,                int[:] k_state_weights_shape,
    dtype[:] k_mixture_weights_vector,              int[:] k_mixture_weights_shape,
    dtype[:] k_mixture_means_vector,                int[:] k_mixture_means_shape,
    dtype[:] k_mixture_covariance_vector,           int[:] k_mixture_covariance_shape,
    dtype[:] means_product_tempbuffer_vector,       int[:] means_product_tempbuffer_shape,
    dtype[:] covariance_product_tempbuffer_vector,  int[:] covariance_product_tempbuffer_shape,
    dtype[:] observation_logprobgraph, int[:] k_sequence_indexes, int[:] k_sequence_sizes,
    int max_order, int t_observations, int n_states, int m_components, int d_features,
    int k_sequences):

    cdef ndvector_dtype reducedstate_logprobgraph = hmm_utility_c.ndvector_memoryview_construct(
        &reducedstate_logprobgraph_vector[0], &reducedstate_logprobgraph_shape[0],
        reducedstate_logprobgraph_shape.shape[0]
    )
    cdef ndvector_dtype mixture_emission_logprobgraph = hmm_utility_c.ndvector_memoryview_construct(
        &mixture_emission_logprobgraph_vector[0], &mixture_emission_logprobgraph_shape[0],
        mixture_emission_logprobgraph_shape.shape[0]
    )
    cdef ndvector_dtype mixture_weights = hmm_utility_c.ndvector_memoryview_construct(
        &mixture_weights_vector[0], &mixture_weights_shape[0],
        mixture_weights_shape.shape[0]
    )
    cdef ndvector_dtype mixture_means = hmm_utility_c.ndvector_memoryview_construct(
        &mixture_means_vector[0], &mixture_means_shape[0],
        mixture_means_shape.shape[0]
    ) 
    cdef ndvector_dtype mixture_covariance = hmm_utility_c.ndvector_memoryview_construct(
        &mixture_covariance_vector[0], &mixture_covariance_shape[0],
        mixture_covariance_shape.shape[0]
    )
    cdef ndvector_dtype k_observations = hmm_utility_c.ndvector_memoryview_construct(
        &k_observations_vector[0], &k_observations_shape[0],
        k_observations_shape.shape[0]
    )
    cdef ndvector_dtype k_statecomponent_logprobgraph = hmm_utility_c.ndvector_memoryview_construct(
        &k_statecomponent_logprobgraph_vector[0], &k_statecomponent_logprobgraph_shape[0],
        k_statecomponent_logprobgraph_shape.shape[0]
    )
    cdef ndvector_dtype k_state_weights = hmm_utility_c.ndvector_memoryview_construct(
        &k_state_weights_vector[0], &k_state_weights_shape[0],
        k_state_weights_shape.shape[0]
    )
    cdef ndvector_dtype k_mixture_weights = hmm_utility_c.ndvector_memoryview_construct(
        &k_mixture_weights_vector[0], &k_mixture_weights_shape[0],
        k_mixture_weights_shape.shape[0]
    )
    cdef ndvector_dtype k_mixture_means = hmm_utility_c.ndvector_memoryview_construct(
        &k_mixture_means_vector[0], &k_mixture_means_shape[0],
        k_mixture_means_shape.shape[0]
    )
    cdef ndvector_dtype k_mixture_covariance = hmm_utility_c.ndvector_memoryview_construct(
        &k_mixture_covariance_vector[0], &  k_mixture_covariance_shape[0],
        k_mixture_covariance_shape.shape[0]
    )
    cdef ndvector_dtype means_product_tempbuffer = hmm_utility_c.ndvector_memoryview_construct(
        &means_product_tempbuffer_vector[0], &means_product_tempbuffer_shape[0],
        means_product_tempbuffer_shape.shape[0]
    )
    cdef ndvector_dtype covariance_product_tempbuffer = hmm_utility_c.ndvector_memoryview_construct(
        &covariance_product_tempbuffer_vector[0], &covariance_product_tempbuffer_shape[0],
        covariance_product_tempbuffer_shape.shape[0]
    )

    if k_observations.ndim < 3:
        gmhmm_utility_c.maximization_gaussian_mixture_params(&reducedstate_logprobgraph,
            &mixture_emission_logprobgraph, &mixture_weights, &mixture_means, &mixture_covariance,
            &k_observations, &k_observations_vector[0], &k_statecomponent_logprobgraph,
            &k_statecomponent_logprobgraph_vector[0], &k_state_weights, &k_mixture_weights,
            &k_mixture_means, &k_mixture_covariance, &means_product_tempbuffer, 
            &covariance_product_tempbuffer, &observation_logprobgraph[0], &k_sequence_indexes[0],
            &k_sequence_sizes[0], t_observations, n_states, m_components, d_features, k_sequences)
    else:
        gmhmm_utility_c.maximization_multiorder_gaussian_mixture_params(&reducedstate_logprobgraph,
            &mixture_emission_logprobgraph, &mixture_weights, &mixture_means, &mixture_covariance,
            &k_observations, &k_observations_vector[0], &k_statecomponent_logprobgraph,
            &k_statecomponent_logprobgraph_vector[0], &k_state_weights, &k_mixture_weights,
            &k_mixture_means, &k_mixture_covariance, &means_product_tempbuffer, 
            &covariance_product_tempbuffer, &observation_logprobgraph[0], &k_sequence_indexes[0],
            &k_sequence_sizes[0], t_observations, n_states, m_components, d_features, k_sequences)