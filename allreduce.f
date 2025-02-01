      program allreduce
        implicit none

        include 'mpif.h'

        integer ierr,rank_,size_
        real*8 input,output

        call mpi_init(ierr)
        call mpi_comm_size(MPI_COMM_WORLD,size_,ierr)
        call mpi_comm_rank(MPI_COMM_WORLD,rank_,ierr)

        call ar_setup(MPI_COMM_WORLD,ierr)

        input=1
        output=0
        call ar_gop(input,output)

        if (rank_.eq.0) then
          write(6,*) 'sum=',output
        endif

        call ar_finalize(ierr)

        call mpi_finalize()
      endprogram
