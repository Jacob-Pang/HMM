cimport ndvector
cimport hmm_utility_c

from libc.stdlib cimport malloc, free
from libc.math cimport exp

ctypedef double dtype
ctypedef ndvector.ndvector_dtype ndvector_dtype

cdef inline void maximization_gaussian_mixture_params(ndvector_dtype* reducedstate_logprobgraph,
    ndvector_dtype* mixture_emission_logprobgraph, ndvector_dtype* mixture_weights,
    ndvector_dtype* mixture_means, ndvector_dtype* mixture_covariance,
    ndvector_dtype* k_observations, dtype* k_observations_vptr,
    ndvector_dtype* k_statecomponent_logprobgraph, dtype* k_statecomponent_logprobgraph_vptr,
    ndvector_dtype* k_state_weights, ndvector_dtype* k_mixture_weights,
    ndvector_dtype* k_mixture_means, ndvector_dtype* k_mixture_covariance,
    ndvector_dtype* means_product_tempbuffer,ndvector_dtype* covariance_product_tempbuffer,
    dtype* observation_logprobgraph, int* k_sequence_indexes, int* k_sequence_sizes,
    int t_observations, int n_states, int m_components, int d_features, int k_sequences):

    # γ(t,n,m) = γ(t,n)∙b(t,m)/b(t) : shape[O x N x M] --------------------------------------------------------
    cdef long statecomponent_tempbuffer_size = t_observations * n_states
    cdef dtype* k_statecomponent_tempbuffer = <dtype*>malloc(sizeof(dtype) * t_observations * n_states
            * m_components)
    
    ndvector.ndvector_init_copy(k_statecomponent_logprobgraph, mixture_emission_logprobgraph)

    ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_logsumexp, k_statecomponent_tempbuffer,
            mixture_emission_logprobgraph.vector_ptr, mixture_emission_logprobgraph.shape_ptr,
            mixture_emission_logprobgraph.dweight_ptr, mixture_emission_logprobgraph.size,
            mixture_emission_logprobgraph.ndim, -1)

    ndvector.ndvector_broadcast_vptr(&ndvector.logdivexp, k_statecomponent_logprobgraph.vector_ptr,
            k_statecomponent_tempbuffer, k_statecomponent_logprobgraph.size,
            statecomponent_tempbuffer_size, -1)
    
    ndvector.ndvector_broadcast_vptr(&ndvector.logprodexp, k_statecomponent_logprobgraph.vector_ptr,
            reducedstate_logprobgraph.vector_ptr, k_statecomponent_logprobgraph.size,
            reducedstate_logprobgraph.size, -1)

    # updating of Log weights -----------------------------------------------------------------------------
    cdef Py_ssize_t k, index
    cdef int kindex, ksize

    for k in range(k_sequences):
        kindex = k_sequence_indexes[k]
        ksize = k_sequence_sizes[k]

        hmm_utility_c.update_k_wrapper(k_statecomponent_logprobgraph, k_statecomponent_logprobgraph_vptr,
                kindex, ksize)
        
        # observation log-probability weighted
        for index in range(k_statecomponent_logprobgraph.size):
            k_statecomponent_logprobgraph.vector_ptr[index] = ndvector.logdivexp(
                k_statecomponent_logprobgraph.vector_ptr[index], observation_logprobgraph[k]
            )

        # Log Σ[γ(t,n,m)] : shape[N x M] -----------------------------------------------------------------
        ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_logsumexp, &k_mixture_weights.vector_ptr
                [k * k_mixture_weights.dweight_ptr[0]], k_statecomponent_logprobgraph.vector_ptr,
                k_statecomponent_logprobgraph.shape_ptr, k_statecomponent_logprobgraph.dweight_ptr,
                k_statecomponent_logprobgraph.size, k_statecomponent_logprobgraph.ndim, 0)
        
        # Log Σ[Σ[γ(t,n,m)]] : shape[N] ------------------------------------------------------------------
        ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_logsumexp, &k_state_weights.vector_ptr
                [k * k_state_weights.dweight_ptr[0]], &k_mixture_weights.vector_ptr
                [k * k_mixture_weights.dweight_ptr[0]], &k_mixture_weights.shape_ptr[1],
                &k_mixture_weights.dweight_ptr[1], k_mixture_weights.size // k_sequences,
                k_mixture_weights.ndim - 1, -1)
    
    # Log γ(n,m) temporary dividend base
    ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_logsumexp, mixture_weights.vector_ptr,
            k_mixture_weights.vector_ptr, k_mixture_weights.shape_ptr,
            k_mixture_weights.dweight_ptr, k_mixture_weights.size,
            k_mixture_weights.ndim, 0)

    # updating of means ----------------------------------------------------------------------------------
    for k in range(k_sequences):
        kindex = k_sequence_indexes[k]
        ksize = k_sequence_sizes[k]

        hmm_utility_c.update_k_wrapper(k_observations, k_observations_vptr, kindex, ksize)
        hmm_utility_c.update_k_wrapper(k_statecomponent_logprobgraph, k_statecomponent_logprobgraph_vptr,
                kindex, ksize)

        # γ(t,n,m)∙Ot : shape[K x N x M x D] ------------------------------------------------------------
        means_product_tempbuffer.size = ksize * n_states * m_components * d_features
        means_product_tempbuffer.shape_ptr[0] = ksize

        ndvector.ndvector_broadcast_vptr(&ndvector.vconstruct_dtype, means_product_tempbuffer.vector_ptr,
                k_statecomponent_logprobgraph.vector_ptr, means_product_tempbuffer.size,
                k_statecomponent_logprobgraph.size, -1)
        
        # shape[K x N x M]
        statecomponent_tempbuffer_size = ksize * n_states * m_components
        ndvector.ndvector_broadcast_vptr(&ndvector.vconstruct_dtype, k_statecomponent_tempbuffer,
                mixture_weights.vector_ptr, statecomponent_tempbuffer_size, mixture_weights.size, 0)
    
        ndvector.ndvector_broadcast_vptr(&ndvector.logdivexp, means_product_tempbuffer.vector_ptr,
                k_statecomponent_tempbuffer, means_product_tempbuffer.size,
                statecomponent_tempbuffer_size, -1)

        for index in range(means_product_tempbuffer.size):
            means_product_tempbuffer.vector_ptr[index] = exp(means_product_tempbuffer.vector_ptr[index])

        for index in range(ksize):
            ndvector.ndvector_broadcast_vptr(&ndvector.vproduct_dtype, &means_product_tempbuffer.vector_ptr
                    [index * means_product_tempbuffer.dweight_ptr[0]], &k_observations.vector_ptr
                    [index * d_features], means_product_tempbuffer.dweight_ptr[0], d_features, 0)
 
        # Σ[γ(t,n,m)∙Ot] : shape[N x M x D] -----------------------------------------------------------------
        ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_dtypesum, &k_mixture_means.vector_ptr
                [k * k_mixture_means.dweight_ptr[0]], means_product_tempbuffer.vector_ptr,
                means_product_tempbuffer.shape_ptr, means_product_tempbuffer.dweight_ptr,
                means_product_tempbuffer.size, means_product_tempbuffer.ndim, 0)

    ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_dtypesum, mixture_means.vector_ptr,
            k_mixture_means.vector_ptr, k_mixture_means.shape_ptr, k_mixture_means.dweight_ptr,
            k_mixture_means.size, k_mixture_means.ndim, 0)

    # ------------------------------------------------------------------------------------------------------
    for k in range(k_sequences):
        kindex = k_sequence_indexes[k]
        ksize = k_sequence_sizes[k]

        hmm_utility_c.update_k_wrapper(k_observations, k_observations_vptr, kindex, ksize)
        hmm_utility_c.update_k_wrapper(k_statecomponent_logprobgraph, k_statecomponent_logprobgraph_vptr,
                kindex, ksize)

        # shape[K x N x M x D x D] ------------------------------------------------------------------------
        covariance_product_tempbuffer.size = ksize * n_states * m_components * d_features ** 2
        covariance_product_tempbuffer.shape_ptr[0] = ksize

        ndvector.ndvector_broadcast_vptr(&ndvector.vconstruct_dtype, covariance_product_tempbuffer
                .vector_ptr, k_statecomponent_logprobgraph.vector_ptr, covariance_product_tempbuffer
                .size, k_statecomponent_logprobgraph.size, -1)

        # shape[K x N x M]
        statecomponent_tempbuffer_size = ksize * n_states * m_components
        ndvector.ndvector_broadcast_vptr(&ndvector.vconstruct_dtype, k_statecomponent_tempbuffer,
                mixture_weights.vector_ptr, statecomponent_tempbuffer_size, mixture_weights.size, 0)

        ndvector.ndvector_broadcast_vptr(&ndvector.logdivexp, covariance_product_tempbuffer
                .vector_ptr, k_statecomponent_tempbuffer, covariance_product_tempbuffer.size,
                statecomponent_tempbuffer_size, -1)

        for index in range(covariance_product_tempbuffer.size):
            covariance_product_tempbuffer.vector_ptr[index] = exp(covariance_product_tempbuffer
                    .vector_ptr[index])
        
        # shape[K x N x M x D] ----------------------------------------------------------------------------
        means_product_tempbuffer.size = ksize * n_states * m_components * d_features
        means_product_tempbuffer.shape_ptr[0] = ksize

        ndvector.ndvector_broadcast_vptr(&ndvector.vconstruct_dtype, means_product_tempbuffer
                .vector_ptr, mixture_means.vector_ptr, means_product_tempbuffer.size,
                mixture_means.size, 0)

        for index in range(ksize):
            ndvector.ndvector_broadcast_vptr(&ndvector.vrevsubtract_dtype, &means_product_tempbuffer
                    .vector_ptr[index * means_product_tempbuffer.dweight_ptr[0]], &k_observations
                    .vector_ptr[index * d_features], means_product_tempbuffer.size // ksize,
                    d_features, 0)

        # shape[K x N x M x D x D] ------------------------------------------------------------------------
        ndvector.ndvector_broadcast_vptr(&ndvector.vproduct_dtype, covariance_product_tempbuffer.vector_ptr,
                means_product_tempbuffer.vector_ptr, covariance_product_tempbuffer.size,
                means_product_tempbuffer.size, -1)

        for index in range(ksize * n_states * m_components):
            ndvector.ndvector_broadcast_vptr(&ndvector.vproduct_dtype, &covariance_product_tempbuffer
                    .vector_ptr[index * d_features ** 2], &means_product_tempbuffer.vector_ptr
                    [index * d_features], d_features ** 2, d_features, 0)
        
        # shape[N x M x D x D] ----------------------------------------------------------------------------
        ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_dtypesum, &k_mixture_covariance.vector_ptr
                [k * k_mixture_covariance.dweight_ptr[0]], covariance_product_tempbuffer.vector_ptr,
                covariance_product_tempbuffer.shape_ptr, covariance_product_tempbuffer.dweight_ptr,
                covariance_product_tempbuffer.size, covariance_product_tempbuffer.ndim, 0)

    free(k_statecomponent_tempbuffer)
    # -------------------------------------------------------------------------------------------------------
    ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_dtypesum, mixture_covariance.vector_ptr,
            k_mixture_covariance.vector_ptr, k_mixture_covariance.shape_ptr,
            k_mixture_covariance.dweight_ptr, k_mixture_covariance.size,
            k_mixture_covariance.ndim, 0)

    cdef dtype* mixture_weights_tempbuffer = <dtype*>malloc(sizeof(dtype) * n_states)

    ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_logsumexp, mixture_weights_tempbuffer,
            k_state_weights.vector_ptr, k_state_weights.shape_ptr, k_state_weights.dweight_ptr,
            k_state_weights.size, k_state_weights.ndim, 0)
    
    ndvector.ndvector_broadcast_vptr(&ndvector.logdivexp, mixture_weights.vector_ptr,
            mixture_weights_tempbuffer, mixture_weights.size, n_states, -1)

    for index in range(mixture_weights.size):
        mixture_weights.vector_ptr[index] = exp(mixture_weights.vector_ptr[index])

    free(mixture_weights_tempbuffer)

cdef inline void maximization_multiorder_gaussian_mixture_params(ndvector_dtype* reducedstate_logprobgraph,
    ndvector_dtype* mixture_emission_logprobgraph, ndvector_dtype* mixture_weights,
    ndvector_dtype* mixture_means, ndvector_dtype* mixture_covariance,
    ndvector_dtype* k_observations, dtype* k_observations_vptr,
    ndvector_dtype* k_statecomponent_logprobgraph, dtype* k_statecomponent_logprobgraph_vptr,
    ndvector_dtype* k_state_weights, ndvector_dtype* k_mixture_weights,
    ndvector_dtype* k_mixture_means, ndvector_dtype* k_mixture_covariance,
    ndvector_dtype* means_product_tempbuffer,ndvector_dtype* covariance_product_tempbuffer,
    dtype* observation_logprobgraph, int* k_sequence_indexes, int* k_sequence_sizes,
    int t_observations, int n_states, int m_components, int d_features, int k_sequences):
    """ modified for multiorder observations
    """
    # γ(t,n,m) = γ(t,n)∙b(t,m)/b(t) : shape[O x N x M] --------------------------------------------------------
    cdef long statecomponent_tempbuffer_size = t_observations * n_states
    cdef dtype* k_statecomponent_tempbuffer = <dtype*>malloc(sizeof(dtype) * t_observations * n_states
            * m_components)
    
    ndvector.ndvector_init_copy(k_statecomponent_logprobgraph, mixture_emission_logprobgraph)

    ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_logsumexp, k_statecomponent_tempbuffer,
            mixture_emission_logprobgraph.vector_ptr, mixture_emission_logprobgraph.shape_ptr,
            mixture_emission_logprobgraph.dweight_ptr, mixture_emission_logprobgraph.size,
            mixture_emission_logprobgraph.ndim, -1)

    ndvector.ndvector_broadcast_vptr(&ndvector.logdivexp, k_statecomponent_logprobgraph.vector_ptr,
            k_statecomponent_tempbuffer, k_statecomponent_logprobgraph.size,
            statecomponent_tempbuffer_size, -1)
    
    ndvector.ndvector_broadcast_vptr(&ndvector.logprodexp, k_statecomponent_logprobgraph.vector_ptr,
            reducedstate_logprobgraph.vector_ptr, k_statecomponent_logprobgraph.size,
            reducedstate_logprobgraph.size, -1)

    # updating of Log weights -----------------------------------------------------------------------------
    cdef Py_ssize_t k, index
    cdef int kindex, ksize

    for k in range(k_sequences):
        kindex = k_sequence_indexes[k]
        ksize = k_sequence_sizes[k]

        hmm_utility_c.update_k_wrapper(k_statecomponent_logprobgraph, k_statecomponent_logprobgraph_vptr,
                kindex, ksize)
        
        # observation log-probability weighted
        for index in range(k_statecomponent_logprobgraph.size):
            k_statecomponent_logprobgraph.vector_ptr[index] = ndvector.logdivexp(
                k_statecomponent_logprobgraph.vector_ptr[index], observation_logprobgraph[k]
            )

        # Log Σ[γ(t,n,m)] : shape[N x M] -----------------------------------------------------------------
        ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_logsumexp, &k_mixture_weights.vector_ptr
                [k * k_mixture_weights.dweight_ptr[0]], k_statecomponent_logprobgraph.vector_ptr,
                k_statecomponent_logprobgraph.shape_ptr, k_statecomponent_logprobgraph.dweight_ptr,
                k_statecomponent_logprobgraph.size, k_statecomponent_logprobgraph.ndim, 0)
        
        # Log Σ[Σ[γ(t,n,m)]] : shape[N] ------------------------------------------------------------------
        ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_logsumexp, &k_state_weights.vector_ptr
                [k * k_state_weights.dweight_ptr[0]], &k_mixture_weights.vector_ptr
                [k * k_mixture_weights.dweight_ptr[0]], &k_mixture_weights.shape_ptr[1],
                &k_mixture_weights.dweight_ptr[1], k_mixture_weights.size // k_sequences,
                k_mixture_weights.ndim - 1, -1)
    
    # Log γ(n,m) temporary dividend base
    ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_logsumexp, mixture_weights.vector_ptr,
            k_mixture_weights.vector_ptr, k_mixture_weights.shape_ptr,
            k_mixture_weights.dweight_ptr, k_mixture_weights.size,
            k_mixture_weights.ndim, 0)

    # updating of means ----------------------------------------------------------------------------------
    for k in range(k_sequences):
        kindex = k_sequence_indexes[k]
        ksize = k_sequence_sizes[k]

        hmm_utility_c.update_k_wrapper(k_observations, k_observations_vptr, kindex, ksize)
        hmm_utility_c.update_k_wrapper(k_statecomponent_logprobgraph, k_statecomponent_logprobgraph_vptr,
                kindex, ksize)

        # γ(t,n,m)∙Ot : shape[K x N x M x D] ------------------------------------------------------------
        means_product_tempbuffer.size = ksize * n_states * m_components * d_features
        means_product_tempbuffer.shape_ptr[0] = ksize

        ndvector.ndvector_broadcast_vptr(&ndvector.vconstruct_dtype, means_product_tempbuffer.vector_ptr,
                k_statecomponent_logprobgraph.vector_ptr, means_product_tempbuffer.size,
                k_statecomponent_logprobgraph.size, -1)
        
        # shape[K x N x M]
        statecomponent_tempbuffer_size = ksize * n_states * m_components
        ndvector.ndvector_broadcast_vptr(&ndvector.vconstruct_dtype, k_statecomponent_tempbuffer,
                mixture_weights.vector_ptr, statecomponent_tempbuffer_size, mixture_weights.size, 0)
    
        ndvector.ndvector_broadcast_vptr(&ndvector.logdivexp, means_product_tempbuffer.vector_ptr,
                k_statecomponent_tempbuffer, means_product_tempbuffer.size,
                statecomponent_tempbuffer_size, -1)

        for index in range(means_product_tempbuffer.size):
            means_product_tempbuffer.vector_ptr[index] = exp(means_product_tempbuffer.vector_ptr[index])

        for index in range(ksize * n_states):
            ndvector.ndvector_broadcast_vptr(&ndvector.vproduct_dtype, &means_product_tempbuffer.vector_ptr
                    [index * means_product_tempbuffer.dweight_ptr[1]], &k_observations.vector_ptr
                    [index * d_features], means_product_tempbuffer.dweight_ptr[1], d_features, 0)
 
        # Σ[γ(t,n,m)∙Ot] : shape[N x M x D] -----------------------------------------------------------------
        ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_dtypesum, &k_mixture_means.vector_ptr
                [k * k_mixture_means.dweight_ptr[0]], means_product_tempbuffer.vector_ptr,
                means_product_tempbuffer.shape_ptr, means_product_tempbuffer.dweight_ptr,
                means_product_tempbuffer.size, means_product_tempbuffer.ndim, 0)

    ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_dtypesum, mixture_means.vector_ptr,
            k_mixture_means.vector_ptr, k_mixture_means.shape_ptr, k_mixture_means.dweight_ptr,
            k_mixture_means.size, k_mixture_means.ndim, 0)

    # ------------------------------------------------------------------------------------------------------
    for k in range(k_sequences):
        kindex = k_sequence_indexes[k]
        ksize = k_sequence_sizes[k]

        hmm_utility_c.update_k_wrapper(k_observations, k_observations_vptr, kindex, ksize)
        hmm_utility_c.update_k_wrapper(k_statecomponent_logprobgraph, k_statecomponent_logprobgraph_vptr,
                kindex, ksize)

        # shape[K x N x M x D x D] ------------------------------------------------------------------------
        covariance_product_tempbuffer.size = ksize * n_states * m_components * d_features ** 2
        covariance_product_tempbuffer.shape_ptr[0] = ksize

        ndvector.ndvector_broadcast_vptr(&ndvector.vconstruct_dtype, covariance_product_tempbuffer
                .vector_ptr, k_statecomponent_logprobgraph.vector_ptr, covariance_product_tempbuffer
                .size, k_statecomponent_logprobgraph.size, -1)

        # shape[K x N x M]
        statecomponent_tempbuffer_size = ksize * n_states * m_components
        ndvector.ndvector_broadcast_vptr(&ndvector.vconstruct_dtype, k_statecomponent_tempbuffer,
                mixture_weights.vector_ptr, statecomponent_tempbuffer_size, mixture_weights.size, 0)

        ndvector.ndvector_broadcast_vptr(&ndvector.logdivexp, covariance_product_tempbuffer
                .vector_ptr, k_statecomponent_tempbuffer, covariance_product_tempbuffer.size,
                statecomponent_tempbuffer_size, -1)

        for index in range(covariance_product_tempbuffer.size):
            covariance_product_tempbuffer.vector_ptr[index] = exp(covariance_product_tempbuffer
                    .vector_ptr[index])
        
        # shape[K x N x M x D] ----------------------------------------------------------------------------
        means_product_tempbuffer.size = ksize * n_states * m_components * d_features
        means_product_tempbuffer.shape_ptr[0] = ksize

        ndvector.ndvector_broadcast_vptr(&ndvector.vconstruct_dtype, means_product_tempbuffer
                .vector_ptr, mixture_means.vector_ptr, means_product_tempbuffer.size,
                mixture_means.size, 0)

        for index in range(ksize * n_states):
            ndvector.ndvector_broadcast_vptr(&ndvector.vrevsubtract_dtype, &means_product_tempbuffer
                    .vector_ptr[index * means_product_tempbuffer.dweight_ptr[1]], &k_observations
                    .vector_ptr[index * d_features], means_product_tempbuffer.dweight_ptr[1],
                    d_features, 0)

        # shape[K x N x M x D x D] ------------------------------------------------------------------------
        ndvector.ndvector_broadcast_vptr(&ndvector.vproduct_dtype, covariance_product_tempbuffer.vector_ptr,
                means_product_tempbuffer.vector_ptr, covariance_product_tempbuffer.size,
                means_product_tempbuffer.size, -1)

        for index in range(ksize * n_states * m_components):
            ndvector.ndvector_broadcast_vptr(&ndvector.vproduct_dtype, &covariance_product_tempbuffer
                    .vector_ptr[index * d_features ** 2], &means_product_tempbuffer.vector_ptr
                    [index * d_features], d_features ** 2, d_features, 0)
        
        # shape[N x M x D x D] ----------------------------------------------------------------------------
        ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_dtypesum, &k_mixture_covariance.vector_ptr
                [k * k_mixture_covariance.dweight_ptr[0]], covariance_product_tempbuffer.vector_ptr,
                covariance_product_tempbuffer.shape_ptr, covariance_product_tempbuffer.dweight_ptr,
                covariance_product_tempbuffer.size, covariance_product_tempbuffer.ndim, 0)

    free(k_statecomponent_tempbuffer)
    # -------------------------------------------------------------------------------------------------------
    ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_dtypesum, mixture_covariance.vector_ptr,
            k_mixture_covariance.vector_ptr, k_mixture_covariance.shape_ptr,
            k_mixture_covariance.dweight_ptr, k_mixture_covariance.size,
            k_mixture_covariance.ndim, 0)

    cdef dtype* mixture_weights_tempbuffer = <dtype*>malloc(sizeof(dtype) * n_states)

    ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_logsumexp, mixture_weights_tempbuffer,
            k_state_weights.vector_ptr, k_state_weights.shape_ptr, k_state_weights.dweight_ptr,
            k_state_weights.size, k_state_weights.ndim, 0)
    
    ndvector.ndvector_broadcast_vptr(&ndvector.logdivexp, mixture_weights.vector_ptr,
            mixture_weights_tempbuffer, mixture_weights.size, n_states, -1)

    for index in range(mixture_weights.size):
        mixture_weights.vector_ptr[index] = exp(mixture_weights.vector_ptr[index])

    free(mixture_weights_tempbuffer)
