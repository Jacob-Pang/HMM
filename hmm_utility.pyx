import numpy as np
cimport ndvector
cimport hmm_utility_c
from libc.stdlib cimport malloc, free

ctypedef double dtype
ctypedef ndvector.ndvector_dtype ndvector_dtype

def logprodexp_broadcast(dtype[:] ndvector_vector, dtype[:] mdvector_vector, int axis):
    ndvector.ndvector_broadcast_vptr(&ndvector.logprodexp, &ndvector_vector[0], &mdvector_vector[0],
            ndvector_vector.shape[0], mdvector_vector.shape[0], axis)

def forward_backward_emiter(
    dtype[:] init_logprobdist_vector,        int[:] init_logprobdist_shape,
    dtype[:] transition_logprobdist_vector,  int[:] transition_logprobdist_shape,
    dtype[:] emission_logprobgraph_vector,   int[:] emission_logprobgraph_shape,
    dtype[:] forward_logprobgraph_vector,    int[:] forward_logprobgraph_shape,
    dtype[:] backward_logprobgraph_vector,   int[:] backward_logprobgraph_shape,
    dtype[:] transition_logprobgraph_vector, int[:] transition_logprobgraph_shape,
    dtype[:] state_logprobgraph_vector,      int[:] state_logprobgraph_shape,
    dtype[:] transition_tempbuffer_vector,   int[:] transition_tempbuffer_shape,
    dtype[:] k_transition_tempbuffer_vector, int[:] k_transition_tempbuffer_shape,
    dtype[:] k_state_tempbuffer_vector,      int[:] k_state_tempbuffer_shape,
    dtype[:] observation_logprobgraph, int[:] k_sequence_indexes, int[:] k_sequence_sizes,
    int max_order, int t_observations, int n_states, int k_sequences):
    """ assumptions
            [@param max_order] = min(*k_sequence_sizes, max_order)
    """
    assert init_logprobdist_vector.shape[0] == n_states
    assert transition_logprobdist_shape[0] == n_states
    assert emission_logprobgraph_shape[1] == n_states
    assert forward_logprobgraph_shape[1] == n_states
    assert transition_logprobgraph_shape[1] == n_states
    assert transition_logprobgraph_shape.shape[0] == max_order + 2
    assert k_transition_tempbuffer_shape[0] == k_sequences
    assert k_state_tempbuffer_shape[0] == k_sequences
    assert k_sequence_indexes.shape[0] == k_sequences
    assert k_sequence_sizes.shape[0] == k_sequences

    cdef ndvector_dtype init_logprobdist = hmm_utility_c.ndvector_memoryview_construct(
        &init_logprobdist_vector[0], &init_logprobdist_shape[0],
        init_logprobdist_shape.shape[0]
    )
    cdef ndvector_dtype transition_logprobdist = hmm_utility_c.ndvector_memoryview_construct(
        &transition_logprobdist_vector[0], &transition_logprobdist_shape[0],
        transition_logprobdist_shape.shape[0]
    )
    cdef ndvector_dtype k_emission_logprobgraph = hmm_utility_c.ndvector_memoryview_construct(
        &emission_logprobgraph_vector[0], &emission_logprobgraph_shape[0],
        emission_logprobgraph_shape.shape[0]
    )
    cdef ndvector_dtype k_forward_logprobgraph = hmm_utility_c.ndvector_memoryview_construct(
        &forward_logprobgraph_vector[0], &forward_logprobgraph_shape[0],
        forward_logprobgraph_shape.shape[0]
    )
    cdef ndvector_dtype k_backward_logprobgraph = hmm_utility_c.ndvector_memoryview_construct(
        &backward_logprobgraph_vector[0], &backward_logprobgraph_shape[0],
        backward_logprobgraph_shape.shape[0]
    )
    cdef ndvector_dtype k_transition_logprobgraph = hmm_utility_c.ndvector_memoryview_construct(
        &transition_logprobgraph_vector[0], &transition_logprobgraph_shape[0],
        transition_logprobgraph_shape.shape[0]
    )
    cdef ndvector_dtype k_state_logprobgraph = hmm_utility_c.ndvector_memoryview_construct(
        &state_logprobgraph_vector[0], &state_logprobgraph_shape[0],
        state_logprobgraph_shape.shape[0]
    )
    cdef ndvector_dtype logweightsum_tempbuffer = hmm_utility_c.ndvector_memoryview_construct(
        &transition_tempbuffer_vector[0], &transition_tempbuffer_shape[0],
        transition_tempbuffer_shape.shape[0]
    )
    cdef ndvector_dtype k_transition_tempbuffer = hmm_utility_c.ndvector_memoryview_construct(
        &k_transition_tempbuffer_vector[0], &k_transition_tempbuffer_shape[0],
        k_transition_tempbuffer_shape.shape[0]
    )
    cdef ndvector_dtype k_state_tempbuffer = hmm_utility_c.ndvector_memoryview_construct(
        &k_state_tempbuffer_vector[0], &k_state_tempbuffer_shape[0],
        k_state_tempbuffer_shape.shape[0]
    )

    hmm_utility_c.forward_backward_empass(&init_logprobdist, &transition_logprobdist, &k_emission_logprobgraph,
            &k_forward_logprobgraph, &k_backward_logprobgraph, &k_transition_logprobgraph,
            &k_state_logprobgraph, &logweightsum_tempbuffer, &k_transition_tempbuffer, 
            &k_state_tempbuffer, &emission_logprobgraph_vector[0], &forward_logprobgraph_vector[0],
            &backward_logprobgraph_vector[0], &transition_logprobgraph_vector[0],
            &state_logprobgraph_vector[0], &observation_logprobgraph[0], &k_sequence_indexes[0],
            &k_sequence_sizes[0], k_sequences, max_order, t_observations, n_states)

def viterbi_decode(
    dtype[:] init_logprobdist_vector,        int[:] init_logprobdist_shape,
    dtype[:] transition_logprobdist_vector,  int[:] transition_logprobdist_shape,
    dtype[:] emission_logprobgraph_vector,   int[:] emission_logprobgraph_shape,
    dtype[:] transition_tempbuffer_vector,   int[:] transition_tempbuffer_shape,
    int[:] maxprob_pathgraph, int max_order, int t_observations, int n_states):

    cdef ndvector_dtype init_logprobdist = hmm_utility_c.ndvector_memoryview_construct(
        &init_logprobdist_vector[0], &init_logprobdist_shape[0],
        init_logprobdist_shape.shape[0]
    )
    cdef ndvector_dtype transition_logprobdist = hmm_utility_c.ndvector_memoryview_construct(
        &transition_logprobdist_vector[0], &transition_logprobdist_shape[0],
        transition_logprobdist_shape.shape[0]
    )
    cdef ndvector_dtype emission_logprobgraph = hmm_utility_c.ndvector_memoryview_construct(
        &emission_logprobgraph_vector[0], &emission_logprobgraph_shape[0],
        emission_logprobgraph_shape.shape[0]
    )
    cdef ndvector_dtype logweightsum_tempbuffer = hmm_utility_c.ndvector_memoryview_construct(
        &transition_tempbuffer_vector[0], &transition_tempbuffer_shape[0],
        transition_tempbuffer_shape.shape[0]
    )

    return hmm_utility_c.viterbi(&init_logprobdist, &transition_logprobdist, &emission_logprobgraph,
            &logweightsum_tempbuffer, &maxprob_pathgraph[0], max_order, t_observations,
            n_states)

def predict_states(
    dtype[:] transition_logprobdist_vector,  int[:] transition_logprobdist_shape,
    dtype[:] transition_tempbuffer_vector,   int[:] transition_tempbuffer_shape,
    int[:] prior_observation_states, int[:] predicted_state_sequence, int num_predictions,
    int max_order, int t_observations, int n_states):
    
    cdef Py_ssize_t timestep, index, ndvector_index
    cdef int order
    cdef int* state_sequence = <int*>malloc(sizeof(int) * (t_observations + num_predictions))
    ndvector.vector_init_copy(state_sequence, &prior_observation_states[0], t_observations)
    
    cdef ndvector_dtype transition_logprobdist = hmm_utility_c.ndvector_memoryview_construct(
        &transition_logprobdist_vector[0], &transition_logprobdist_shape[0],
        transition_logprobdist_shape.shape[0]
    )
    cdef ndvector_dtype logweightsum_tempbuffer = hmm_utility_c.ndvector_memoryview_construct(
        &transition_tempbuffer_vector[0], &transition_tempbuffer_shape[0],
        transition_tempbuffer_shape.shape[0]
    )

    for timestep in range(t_observations, t_observations + num_predictions):
        order = min(timestep, max_order)
        hmm_utility_c.ndvector_logprob_mreducecast(&logweightsum_tempbuffer,
                &transition_logprobdist, order + 1)
        
        ndvector_index = 0
        for index in range(order):
            ndvector_index += (logweightsum_tempbuffer.dweight_ptr[index] * state_sequence
                    [timestep - max_order + index])
        
        state_sequence[timestep] = ndvector.vreduce_dtype_argmax(&logweightsum_tempbuffer
                .vector_ptr[ndvector_index], n_states)

    ndvector.vector_init_copy(&predicted_state_sequence[0], &state_sequence[t_observations],
            num_predictions)
    
    free(state_sequence)
