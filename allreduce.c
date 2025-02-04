#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

typedef struct all_reduce_t_ *all_reduce_t;
typedef void (*all_reduce_impl_t)(const void *, void *, all_reduce_t);

struct all_reduce_t_ {
  MPI_Comm          comm, inter_node_comm, intra_node_comm;
  int               size, rank;
  int               inter_node_rank, inter_node_size;
  int               node_leader;
  all_reduce_impl_t reduce;
};

static all_reduce_t reduce_ = 0;

static inline void ar_malloc_(void **ptr, size_t size, const char *file,
                              unsigned line) {
  *ptr = malloc(size);
  if (ptr) return;
  fprintf(stderr, "ar_malloc failed at %s:%d\n", file, line);
  exit(1);
}

#define ar_malloc(ptr, size)                                           \
  ar_malloc_((void **)ptr, sizeof(**(ptr)) * (size), __FILE__, __LINE__)

static inline void ar_free_(void **ptr) {
  if (*ptr) free(*ptr);
  *ptr = NULL;
}

#define ar_free(ptr) ar_free_((void **)(ptr))

static inline void binary_fifo_impl(const void *sendbuf, void *recvbuf,
                                    int rank, int size, MPI_Comm comm) {
  MPI_Request req;
  MPI_Status  status;
  int         offset   = size / 2;
  *((double *)recvbuf) = *((double *)sendbuf);

  while (offset > 0) {
    if (rank < offset) {
      // Receive from rank + offset
      double data;
      MPI_Irecv(&data, 1, MPI_DOUBLE, rank + offset, 0, comm, &req);
      MPI_Wait(&req, &status);
      *((double *)recvbuf) += data;
    } else if (rank < 2 * offset) {
      // Send to rank - offset
      MPI_Send(recvbuf, 1, MPI_DOUBLE, rank - offset, 0, comm);
    }
    offset /= 2;
  }

  offset = 1;
  while (offset <= size / 2) {
    if (rank < offset) {
      // Send to rank + offset
      MPI_Send(recvbuf, 1, MPI_DOUBLE, rank + offset, 0, comm);
    } else if (rank < 2 * offset) {
      // receive from rank - offset
      MPI_Irecv(recvbuf, 1, MPI_DOUBLE, rank - offset, 0, comm, &req);
      MPI_Wait(&req, &status);
    }
    offset *= 2;
  }
}

static inline void binary_fifo(const void *sendbuf, void *recvbuf,
                               all_reduce_t reduce) {
  binary_fifo_impl(sendbuf, recvbuf, reduce->rank, reduce->size,
                   reduce->comm);
}

static inline void binary_fifo_v2(const void *sendbuf, void *recvbuf,
                                  all_reduce_t reduce) {
  // Intra-node reduction:
  MPI_Reduce(sendbuf, recvbuf, 1, MPI_DOUBLE, MPI_SUM, 0,
             reduce->intra_node_comm);

  // Intra-node reduction:
  double partial = *((double *)recvbuf);
  binary_fifo_impl(&partial, recvbuf, reduce->inter_node_rank,
                   reduce->inter_node_size, reduce->inter_node_comm);

  // Broadcast within the node:
  MPI_Bcast(recvbuf, 1, MPI_DOUBLE, 0, reduce->intra_node_comm);
}

static inline void mpi_allreduce(const void *sendbuf, void *recvbuf,
                                 all_reduce_t reduce) {
  MPI_Allreduce(sendbuf, recvbuf, 1, MPI_DOUBLE, MPI_SUM, reduce->comm);
}

static inline void ar_setup(all_reduce_t *reduce_, MPI_Comm comm) {
  ar_malloc(reduce_, 1);
  all_reduce_t reduce = *reduce_;

  MPI_Comm_dup(comm, &reduce->comm);
  MPI_Comm_size(reduce->comm, &reduce->size);
  MPI_Comm_rank(reduce->comm, &reduce->rank);

  // Create the intra_node_comm:
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, reduce->rank,
                      MPI_INFO_NULL, &reduce->intra_node_comm);
  int node_rank;
  MPI_Comm_rank(reduce->intra_node_comm, &node_rank);
  reduce->node_leader = (node_rank == 0);

  // Create the inter_node_comm:
  MPI_Comm_split(comm, reduce->node_leader == 1, reduce->rank,
                 &reduce->inter_node_comm);
  MPI_Comm_size(reduce->inter_node_comm, &reduce->inter_node_size);
  MPI_Comm_rank(reduce->inter_node_comm, &reduce->inter_node_rank);

  // Add all the reduction implementations:
  all_reduce_impl_t reductions[] = {&mpi_allreduce, &binary_fifo_v2};
  const size_t      num_reductions =
      sizeof(reductions) / sizeof(reductions[0]);

  const size_t num_trials = 10000;
  double       tmin       = DBL_MAX;
  for (size_t j = 0; j < num_reductions; j++) {
    double sum = 0, input = 1;

    // Benchmark the reduction.
    MPI_Barrier(reduce->comm);
    double ts = MPI_Wtime();
    for (int i = 0; i < num_trials; i++)
      (*reductions[j])(&input, &sum, reduce);
    MPI_Barrier(reduce->comm);
    double te = MPI_Wtime();

    // Print timing and accuracy informations.
    if (sum != reduce->size) {
      fprintf(stderr, "Error: %lf != %d\n", sum, reduce->size);
      exit(1);
    }
    if (reduce->rank == 0)
      printf("Time for %zu: %lf\n", j, (te - ts) / num_trials);
    fflush(stdout), fflush(stderr);

    // Update the minimum time and method
    if ((te - ts) < tmin) {
      tmin           = te - ts;
      reduce->reduce = reductions[j];
    }
  }

  return;
}

static inline void ar_finalize(all_reduce_t *reduce) {
  MPI_Barrier((*reduce)->comm);
  MPI_Comm_free(&(*reduce)->comm);
  MPI_Comm_free(&(*reduce)->intra_node_comm);
  MPI_Comm_free(&(*reduce)->inter_node_comm);
  ar_free(reduce);
}

// ======================================================================
// Fortran interface
void ar_setup_(MPI_Fint *fcomm, int *err) {
  *err          = 1;
  MPI_Comm comm = MPI_Comm_f2c(*fcomm);
  ar_setup(&reduce_, comm);
  *err = 0;
}

void ar_gop_(const void *sendbuf, void *recvbuf) {
  reduce_->reduce(sendbuf, recvbuf, reduce_);
}

void ar_finalize_(int *err) {
  *err = 1;
  ar_finalize(&reduce_);
  *err = 0;
}

#undef ar_malloc
#undef ar_free
