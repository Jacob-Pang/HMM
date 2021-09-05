cimport ndvector
from libc.stdlib cimport malloc, free
from libc.math cimport exp

ctypedef double dtype
ctypedef ndvector.ndvector_dtype ndvector_dtype

# supporting functions ---------------------------------------------------------------------------------------
cdef inline ndvector_dtype ndvector_memoryview_construct(dtype* vector_ptr, int* shape_ptr, int ndim):
    """ constructs ndvector_dtype from dtype memoryview
    """
    cdef ndvector_dtype ndvector_construct

    ndvector_construct.vector_ptr = vector_ptr
    ndvector_construct.shape_ptr = shape_ptr
    ndvector_construct.dweight_ptr = <long*>malloc(sizeof(long) * ndim)
    ndvector_construct.ndim = ndim

    ndvector.ndvector_init_size(&ndvector_construct)
    ndvector.ndvector_init_dweight(&ndvector_construct)

    return ndvector_construct

cdef inline void update_k_wrapper(ndvector_dtype* ndvector_ptr, dtype* vector_ptr, long k_sequence_index,
    long k_sequence_size):
    """ updates the vector, shape pointer to accomodate the Kth observation sequence.
    """
    ndvector_ptr.vector_ptr = &vector_ptr[ndvector_ptr.dweight_ptr[0] * k_sequence_index]
    ndvector_ptr.shape_ptr[0] = k_sequence_size

    ndvector.ndvector_init_size(ndvector_ptr)

cdef inline void remove_k_wrapper(ndvector_dtype* ndvector_ptr, dtype* vector_ptr, int t_observations):
    ndvector_ptr.vector_ptr = vector_ptr
    ndvector_ptr.shape_ptr[0] = t_observations

    ndvector.ndvector_init_size(ndvector_ptr)

cdef inline void ndvector_logprob_mreducecast(ndvector_dtype* ndvector_outptr, ndvector_dtype*
    ndvector_xptr, int order):
    """ reduces log-probability distributions to [@param order].
    """
    ndvector.ndvector_dtype_mreducecast(&ndvector.vreduce_logsumexp, ndvector_outptr, ndvector_xptr,
            order, 0)

    cdef long reduced_size = ndvector_outptr.size // ndvector_outptr.shape_ptr[ndvector_outptr.ndim - 1]
    cdef dtype* ndvector_priorptr = <dtype*>malloc(sizeof(dtype) * reduced_size)
    
    ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_logsumexp, ndvector_priorptr, ndvector_outptr
            .vector_ptr, ndvector_outptr.shape_ptr, ndvector_outptr.dweight_ptr, ndvector_outptr.size,
            ndvector_outptr.ndim, -1)

    ndvector.ndvector_broadcast_vptr(&ndvector.logdivexp, ndvector_outptr.vector_ptr, ndvector_priorptr,
            ndvector_outptr.size, reduced_size, -1)

    free(ndvector_priorptr)


cdef inline void forward(ndvector_dtype* init_logprobdist, ndvector_dtype* transition_logprobdist,
    ndvector_dtype* emission_logprobgraph, ndvector_dtype* forward_logprobgraph,
    ndvector_dtype* logweightsum_tempbuffer, int max_order, int t_observations,
    int n_states):
    """ parameters
            init_logprobdist: shape[N]
                the log-probability distribution for initializing in state N.
            transition_logprobdist: shape[N ^(max_order + 1)]
                the transition log-probability distribution encoding N states and order M,
                where the index [0, 1 ... q] corresponds to the sequence <0 1 ... q>
                and P(Qt = q|Q(t-m):Q(t-1) = 0...).
            emission_logprobgraph: shape[T x N]
                the emission log-probabilities of the observation sequence, P(Ot|Qt).
            forward_logprob_graph: shape[T x N ^max_order]
                return-pointer encoding the forward log-probability α where
                P(OT|λ) = Σ EXP(forward_logprob_graph[T])  
    """
    cdef Py_ssize_t timestep
    cdef int ndvector_order, mdvector_order
    cdef long ndvector_size, mdvector_size
    cdef long fgraph_tdweight = forward_logprobgraph.dweight_ptr[0]
    cdef long egraph_tdweight = emission_logprobgraph.dweight_ptr[0]
    cdef dtype* forward_logprob_tempbuffer = <dtype*>malloc(sizeof(dtype) * n_states ** max_order)

    ndvector.vector_init_copy(forward_logprob_tempbuffer, init_logprobdist.vector_ptr, n_states)
    ndvector.vector_broadcast(&ndvector.vsum_dtype, forward_logprob_tempbuffer,
            emission_logprobgraph.vector_ptr, n_states)

    ndvector.ndvector_logdtype_identitycast_vptr(forward_logprobgraph.vector_ptr,
            forward_logprob_tempbuffer, n_states, 1, n_states, max_order)

    ndvector_size = n_states
    ndvector_order = 1

    for timestep in range(1, t_observations):
        mdvector_order = min(timestep + 1, max_order)
        mdvector_size = n_states ** mdvector_order
        ndvector_logprob_mreducecast(logweightsum_tempbuffer, transition_logprobdist,
                ndvector_order + 1)

        ndvector.ndvector_broadcast_vptr(&ndvector.logprodexp, logweightsum_tempbuffer.vector_ptr,
                forward_logprob_tempbuffer, logweightsum_tempbuffer.size, ndvector_size, -1)

        if ndvector_order == max_order:
            ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_logsumexp, forward_logprob_tempbuffer,
                    logweightsum_tempbuffer.vector_ptr, logweightsum_tempbuffer.shape_ptr,
                    logweightsum_tempbuffer.dweight_ptr, logweightsum_tempbuffer.size,
                    logweightsum_tempbuffer.ndim, 0)
        else:
            ndvector.vector_init_copy(forward_logprob_tempbuffer, logweightsum_tempbuffer
                    .vector_ptr, mdvector_size)

        ndvector.ndvector_broadcast_vptr(&ndvector.logprodexp, forward_logprob_tempbuffer,
                &emission_logprobgraph.vector_ptr[egraph_tdweight * timestep], mdvector_size,
                n_states, 0)
        
        ndvector.ndvector_logdtype_identitycast_vptr(&forward_logprobgraph.vector_ptr[fgraph_tdweight
                * timestep], forward_logprob_tempbuffer, mdvector_size, mdvector_order, n_states,
                max_order)

        ndvector_size = mdvector_size
        ndvector_order = mdvector_order

    free(forward_logprob_tempbuffer)

cdef inline void backward(ndvector_dtype* transition_logprobdist, ndvector_dtype* emission_logprobgraph,
    ndvector_dtype* backward_logprobgraph, ndvector_dtype* logweightsum_tempbuffer,
    int max_order, int t_observations, int n_states):
    """ parameters
            transition_logprobdist: shape[N ^(max_order + 1)]
                the transition log-probability distribution encoding N states and order M,
                where the index [0, 1 ... q] corresponds to the sequence <0 1 ... q>
                and P(Qt = q|Q(t-m):Q(t-1) = 0...).
            emission_logprobgraph: shape[T x N]
                the emission log-probabilities of the observation sequence, P(Ot|Qt).
            backward_logprob_graph: shape[T x N ^max_order]
                return-pointer encoding the backward log-probability β where
                P(O|λ) = Σ[P(O1|S) P(Q1=S) EXP(backward_logprobgraph[1]])         
    """
    cdef Py_ssize_t timestep
    cdef int ndvector_order, mdvector_order
    cdef long ndvector_size, mdvector_size
    cdef long bgraph_tdweight = backward_logprobgraph.dweight_ptr[0]
    cdef long egraph_tdweight = emission_logprobgraph.dweight_ptr[0]
    cdef dtype* backward_logprob_tempbuffer = <dtype*>malloc(sizeof(dtype) * n_states ** max_order)

    mdvector_size = n_states ** max_order
    mdvector_order = max_order

    ndvector.vector_init_zeros(backward_logprob_tempbuffer, mdvector_size)
    ndvector.vector_init_zeros(&backward_logprobgraph.vector_ptr[bgraph_tdweight * (t_observations - 1)],
            mdvector_size)

    for timestep in range(t_observations - 2, -1, -1):
        ndvector_order = min(timestep + 1, max_order)
        ndvector_size = n_states ** ndvector_order
        ndvector_logprob_mreducecast(logweightsum_tempbuffer, transition_logprobdist,
                ndvector_order + 1)

        ndvector.ndvector_broadcast_vptr(&ndvector.logprodexp, logweightsum_tempbuffer.vector_ptr,
                backward_logprob_tempbuffer, logweightsum_tempbuffer.size, mdvector_size, 0)

        ndvector.ndvector_broadcast_vptr(&ndvector.logprodexp, logweightsum_tempbuffer.vector_ptr,
                &emission_logprobgraph.vector_ptr[egraph_tdweight * (timestep + 1)],
                logweightsum_tempbuffer.size, n_states, 0)

        ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_logsumexp, backward_logprob_tempbuffer,
                logweightsum_tempbuffer.vector_ptr, logweightsum_tempbuffer.shape_ptr,
                logweightsum_tempbuffer.dweight_ptr, logweightsum_tempbuffer.size,
                logweightsum_tempbuffer.ndim, -1)

        ndvector.ndvector_logdtype_identitycast_vptr(&backward_logprobgraph.vector_ptr[bgraph_tdweight
                * timestep], backward_logprob_tempbuffer, ndvector_size, ndvector_order, n_states,
                max_order)

        mdvector_size = ndvector_size
        mdvector_order = ndvector_order

    free(backward_logprob_tempbuffer)

cdef inline dtype expectation_logprob(ndvector_dtype* transition_logprobdist, ndvector_dtype*
    emission_logprobgraph, ndvector_dtype* transition_logprobgraph, ndvector_dtype* state_logprobgraph,
    ndvector_dtype* forward_logprobgraph, ndvector_dtype* backward_logprobgraph, ndvector_dtype*
    logweightsum_tempbuffer, int max_order, int t_observations, int n_states):
    """ computes the transition sequence expected probability ξ = P(Q(t-m+1):Qt,Q(T+1)|O,λ),
        state sequence expected probability γ = P(Q(t-m+1):Qt|O,λ), and the observation
        sequence probability P(O|λ), where
                P(O|λ) = Σ[αT•βT]
                ξt = [αt•A(t:t+1)•B(t+1)•β(t+1)]/P(O|λ)
                γt = [αt•βt]/P(O|λ)

        parameters
            transition_logprobdist: shape[N ^(max_order + 1)]
                the transition log-probability distribution encoding N states and order M,
                where the index [0, 1 ... q] corresponds to the sequence <0 1 ... q>
                and P(Qt = q|Q(t-m):Q(t-1) = 0...).
            emission_logprobgraph: shape[T x N]
                the emission log-probabilities of the observation sequence, P(Ot|Qt).
            forward_logprobgraph: shape[T x N ^max_order]
                the forward log-probabilities α computed in the forward algorithm.
            backward_logprob_graph: shape[T x N ^max_order]
                the backward log-probabilities β computed in the backward algorithm.
            transition_logprobgraph: shape[(T - max_order) x N^(max_order + 1)]
                return-pointer for log-probability ξ.
            state_logprobgraph: [T x N ^max_order]
                return-pointer for log-probability γ.
        returns
            observation_logprob: dtype
                P(O|λ)
    """
    cdef Py_ssize_t timestep
    cdef dtype observation_logprob = ndvector.vreduce_logsumexp(&forward_logprobgraph.vector_ptr
            [forward_logprobgraph.dweight_ptr[0] * (t_observations - 1)],
            n_states ** max_order)

    ndvector.ndvector_init_copy(state_logprobgraph, forward_logprobgraph)
    
    ndvector.vector_broadcast(&ndvector.logprodexp, state_logprobgraph.vector_ptr,
            backward_logprobgraph.vector_ptr, state_logprobgraph.size)
    
    for timestep in range(state_logprobgraph.size):
        state_logprobgraph.vector_ptr[timestep] = ndvector.logdivexp(state_logprobgraph.vector_ptr
                [timestep], observation_logprob)
    
    ndvector_logprob_mreducecast(logweightsum_tempbuffer, transition_logprobdist, max_order + 1)
    
    ndvector.ndvector_broadcast_vptr(&ndvector.vconstruct_dtype, transition_logprobgraph.vector_ptr,
            logweightsum_tempbuffer.vector_ptr, transition_logprobgraph.size,
            logweightsum_tempbuffer.size, 0)

    cdef long transition_tstepsize = transition_logprobgraph.size // transition_logprobgraph.shape_ptr[0]

    for timestep in range(max_order, t_observations):
        ndvector.ndvector_broadcast_vptr(&ndvector.logprodexp, &transition_logprobgraph.vector_ptr
                [transition_logprobgraph.dweight_ptr[0] * (timestep - max_order)],
                &forward_logprobgraph.vector_ptr[forward_logprobgraph.dweight_ptr[0] * (timestep - 1)],
                transition_tstepsize, forward_logprobgraph.size // forward_logprobgraph.shape_ptr[0], -1)
        ndvector.ndvector_broadcast_vptr(&ndvector.logprodexp, &transition_logprobgraph.vector_ptr
                [transition_logprobgraph.dweight_ptr[0] * (timestep - max_order)], 
                &emission_logprobgraph.vector_ptr[emission_logprobgraph.dweight_ptr[0] * timestep],
                transition_tstepsize, emission_logprobgraph.size // emission_logprobgraph.shape_ptr[0], 0)
        ndvector.ndvector_broadcast_vptr(&ndvector.logprodexp, &transition_logprobgraph.vector_ptr
                [transition_logprobgraph.dweight_ptr[0] * (timestep - max_order)], 
                &backward_logprobgraph.vector_ptr[backward_logprobgraph.dweight_ptr[0] * timestep],
                transition_tstepsize, backward_logprobgraph.size // backward_logprobgraph.shape_ptr[0], 0)
    
    for timestep in range(transition_logprobgraph.size):
        transition_logprobgraph.vector_ptr[timestep] = ndvector.logdivexp(transition_logprobgraph
                .vector_ptr[timestep], observation_logprob)
    
    return observation_logprob

cdef inline void maximization_transition_logprobdist(ndvector_dtype* transition_logprobdist,
    ndvector_dtype* k_transition_logprobgraph, ndvector_dtype* k_state_logprobgraph,
    dtype* transition_logprobgraph, dtype* state_logprobgraph, ndvector_dtype* k_transition_tempbuffer,
    ndvector_dtype* k_state_tempbuffer, dtype* observation_logprobgraph, int* k_sequence_indexes,
    int* k_sequence_sizes, int k_sequences, int max_order, int n_states):
    """ updates transition log-probability distribution A = P(Qt|Q(t-m):Q(t-1))
        for K observation sequences where
                A = Σk[Σ(ξk)/P(Ok|λ)]/Σk[Σ(γk)/P(Ok|λ)]

        parameters:
            transition_logprobgraph: shape[K x (Tk - max_order) x N^(max_order + 1)]
                the transition sequence expected log-probability LN(ξ) where
                ξ = P(Q(t-m+1):Qt,Q(T+1)|O,λ) for K observation sequences and the
                shape K parameter padded to |O|.
            state_logprobgraph: shape[K x Tk x N ^max_order] = [O x N ^max_order]
                the state sequence expected probability LN(γ) where γ = P(Q(t-m+1):Qt|O,λ)
                for K observation sequences.
            observation_logprobgraph: shape[K]
                the observation sequence log-probability LN(P(Ok|λ)) for K observation
                sequences.
            transition_logprobdist: shape[N ^(max_order + 1)]
                return-pointer for the transition log-probability distribution A, encoding
                N states and order M, where the index [0, 1 ... q] corresponds to the
                sequence <0 1 ... q> and P(Qt = q|Q(t-m):Q(t-1) = 0...).
    """
    cdef Py_ssize_t k
    cdef dtype* state_tempbuffer = <dtype*>malloc(sizeof(dtype) * n_states ** max_order)

    for k in range(k_sequences):
        update_k_wrapper(k_transition_logprobgraph, transition_logprobgraph, k_sequence_indexes[k],
                k_sequence_sizes[k] - max_order)

        update_k_wrapper(k_state_logprobgraph, state_logprobgraph, k_sequence_indexes[k],
                k_sequence_sizes[k] - max_order)

        ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_logsumexp,
                &k_transition_tempbuffer.vector_ptr[k * k_transition_tempbuffer.dweight_ptr[0]],
                k_transition_logprobgraph.vector_ptr, k_transition_logprobgraph.shape_ptr,
                k_transition_logprobgraph.dweight_ptr, k_transition_logprobgraph.size,
                k_state_logprobgraph.ndim, 0)
        
        ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_logsumexp,
                &k_state_tempbuffer.vector_ptr[k * k_state_tempbuffer.dweight_ptr[0]],
                &k_state_logprobgraph.vector_ptr[(max_order - 1) * k_state_logprobgraph.dweight_ptr[0]],
                k_state_logprobgraph.shape_ptr, k_state_logprobgraph.dweight_ptr, k_state_logprobgraph.size,
                k_state_logprobgraph.ndim, 0)

    ndvector.ndvector_broadcast_vptr(&ndvector.logdivexp, k_transition_tempbuffer.vector_ptr,
            observation_logprobgraph, k_sequences * k_transition_tempbuffer.dweight_ptr[0],
            k_sequences, -1)

    ndvector.ndvector_broadcast_vptr(&ndvector.logdivexp, k_state_tempbuffer.vector_ptr,
            observation_logprobgraph, k_sequences * k_state_tempbuffer.dweight_ptr[0],
            k_sequences, -1)

    ndvector.ndvector_dtype_mreducecast(&ndvector.vreduce_logsumexp, transition_logprobdist,
            k_transition_tempbuffer, max_order + 1, 0)

    ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_logsumexp, state_tempbuffer, k_state_tempbuffer
            .vector_ptr, k_state_tempbuffer.shape_ptr, k_state_tempbuffer.dweight_ptr,
            k_state_tempbuffer.size, k_state_tempbuffer.ndim, 0)

    ndvector.ndvector_broadcast_vptr(&ndvector.logdivexp, transition_logprobdist.vector_ptr,
            state_tempbuffer, transition_logprobdist.size, n_states ** max_order, -1)

    free(state_tempbuffer)

cdef inline void forward_backward_empass(ndvector_dtype* init_logprobdist, ndvector_dtype* transition_logprobdist,
    ndvector_dtype* k_emission_logprobgraph, ndvector_dtype* k_forward_logprobgraph,
    ndvector_dtype* k_backward_logprobgraph, ndvector_dtype* k_transition_logprobgraph,
    ndvector_dtype* k_state_logprobgraph, ndvector_dtype* logweightsum_tempbuffer,
    ndvector_dtype* k_transition_tempbuffer, ndvector_dtype* k_state_tempbuffer,
    dtype* emission_logprobgraph, dtype* forward_logprobgraph, dtype* backward_logprobgraph,
    dtype* transition_logprobgraph, dtype* state_logprobgraph, dtype* observation_logprobgraph,
    int* k_sequence_indexes, int* k_sequence_sizes, int k_sequences, int max_order, int t_observations,
    int n_states):

    cdef Py_ssize_t k
    cdef int kindex, ksize

    for k in range(k_sequences):
        kindex = k_sequence_indexes[k]
        ksize = k_sequence_sizes[k]

        update_k_wrapper(k_emission_logprobgraph,   emission_logprobgraph,   kindex, ksize)
        update_k_wrapper(k_forward_logprobgraph,    forward_logprobgraph,    kindex, ksize)
        update_k_wrapper(k_backward_logprobgraph,   backward_logprobgraph,   kindex, ksize)
        update_k_wrapper(k_transition_logprobgraph, transition_logprobgraph, kindex, ksize - max_order)
        update_k_wrapper(k_state_logprobgraph,      state_logprobgraph,      kindex, ksize)

        forward(init_logprobdist, transition_logprobdist, k_emission_logprobgraph,
                k_forward_logprobgraph, logweightsum_tempbuffer, max_order,
                t_observations, n_states)

        backward(transition_logprobdist, k_emission_logprobgraph, k_backward_logprobgraph,
                logweightsum_tempbuffer, max_order, t_observations, n_states)

        observation_logprobgraph[k] = expectation_logprob(transition_logprobdist,
                k_emission_logprobgraph, k_transition_logprobgraph, k_state_logprobgraph,
                k_forward_logprobgraph, k_backward_logprobgraph, logweightsum_tempbuffer,
                max_order, t_observations, n_states)

    maximization_transition_logprobdist(transition_logprobdist, k_transition_logprobgraph,
            k_state_logprobgraph, transition_logprobgraph, state_logprobgraph, k_transition_tempbuffer,
            k_state_tempbuffer, observation_logprobgraph, k_sequence_indexes, k_sequence_sizes,
            k_sequences, max_order, n_states)

cdef inline dtype viterbi(ndvector_dtype* init_logprobdist, ndvector_dtype* transition_logprobdist,
    ndvector_dtype* emission_logprobgraph, ndvector_dtype* logweightsum_tempbuffer,
    int* maxprob_pathgraph, int max_order, int t_observations, int n_states):

    cdef ndvector_dtype viterbi_logprobgraph
    viterbi_logprobgraph.size = t_observations * n_states ** max_order
    viterbi_logprobgraph.ndim = max_order + 1
    viterbi_logprobgraph.vector_ptr = <dtype*>malloc(sizeof(dtype) * viterbi_logprobgraph.size)
    viterbi_logprobgraph.shape_ptr = <int*>malloc(sizeof(dtype) * viterbi_logprobgraph.ndim)
    viterbi_logprobgraph.dweight_ptr = <long*>malloc(sizeof(long) * viterbi_logprobgraph.ndim)

    viterbi_logprobgraph.shape_ptr[0] = t_observations
    ndvector.vector_init_repeat(&viterbi_logprobgraph.shape_ptr[1], n_states, max_order)
    ndvector.ndvector_init_dweight(&viterbi_logprobgraph)

    cdef int* backpointer_pathgraph_vptr = <int*>malloc(sizeof(int) * viterbi_logprobgraph.size)

    cdef Py_ssize_t timestep
    cdef int ndvector_order, mdvector_order
    cdef long ndvector_size, mdvector_size
    cdef long vgraph_tdweight = viterbi_logprobgraph.dweight_ptr[0]

    ndvector.vector_init_copy(viterbi_logprobgraph.vector_ptr, init_logprobdist.vector_ptr,
            n_states)
    ndvector.vector_broadcast(&ndvector.vsum_dtype, viterbi_logprobgraph.vector_ptr,
            emission_logprobgraph.vector_ptr, n_states)

    ndvector_size = n_states
    ndvector_order = 1

    for timestep in range(1, t_observations):
        mdvector_order = min(timestep + 1, max_order)
        mdvector_size = n_states ** mdvector_order
        ndvector_logprob_mreducecast(logweightsum_tempbuffer, transition_logprobdist,
                ndvector_order + 1)

        ndvector.ndvector_broadcast_vptr(&ndvector.logprodexp, logweightsum_tempbuffer.vector_ptr,
                &viterbi_logprobgraph.vector_ptr[(timestep - 1) * vgraph_tdweight],
                logweightsum_tempbuffer.size, ndvector_size, -1)

        ndvector.ndvector_broadcast_vptr(&ndvector.logprodexp, logweightsum_tempbuffer.vector_ptr,
                &emission_logprobgraph.vector_ptr[timestep * emission_logprobgraph.dweight_ptr[0]],
                logweightsum_tempbuffer.size, n_states, 0)

        if ndvector_order == max_order:
            ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_dtypemax,
                    &viterbi_logprobgraph.vector_ptr[timestep * vgraph_tdweight],
                    logweightsum_tempbuffer.vector_ptr, logweightsum_tempbuffer.shape_ptr,
                    logweightsum_tempbuffer.dweight_ptr, logweightsum_tempbuffer.size,
                    logweightsum_tempbuffer.ndim, 0)

            ndvector.ndvector_reducecast_vptr(&ndvector.vreduce_dtype_argmax,
                    &backpointer_pathgraph_vptr[timestep * vgraph_tdweight],
                    logweightsum_tempbuffer.vector_ptr, logweightsum_tempbuffer.shape_ptr,
                    logweightsum_tempbuffer.dweight_ptr, logweightsum_tempbuffer.size,
                    logweightsum_tempbuffer.ndim, 0)
        else:
            ndvector.vector_init_copy(&viterbi_logprobgraph.vector_ptr[timestep * vgraph_tdweight],
                    logweightsum_tempbuffer.vector_ptr, mdvector_size)

        ndvector_size = mdvector_size
        ndvector_order = mdvector_order

    cdef dtype max_logprob = ndvector.vector_reduce(&ndvector.vmax_dtype, &viterbi_logprobgraph
            .vector_ptr[(t_observations - 1) * vgraph_tdweight], ndvector_size)
    
    ndvector.ndvector_mdargmax(
        &maxprob_pathgraph[t_observations - max_order],
        &viterbi_logprobgraph.vector_ptr[(t_observations - 1) * vgraph_tdweight],
        &viterbi_logprobgraph.shape_ptr[1],
        &viterbi_logprobgraph.dweight_ptr[1],
        viterbi_logprobgraph.size // viterbi_logprobgraph.shape_ptr[0],
        max_order
    )

    for timestep in range(t_observations - 1, max_order - 1, -1):
        reversed_timestep = timestep - max_order

        maxprob_pathgraph[reversed_timestep] = ndvector.ndvector_indexing_vptr(
            &backpointer_pathgraph_vptr[timestep * vgraph_tdweight],
            &maxprob_pathgraph[reversed_timestep + 1],
            &viterbi_logprobgraph.dweight_ptr[1], max_order
        )

    ndvector.ndvector_deconstruct(&viterbi_logprobgraph)
    free(backpointer_pathgraph_vptr)
    
    return max_logprob
