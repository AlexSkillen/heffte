#include "heffte.h"

/*!
 * \brief HeFFTe example 5, computing the Cosine Transform DCT using an arbitrary number of MPI ranks.
 *
 * Performing the discrete Cosine Transform (DCT) on three dimensional data in a box of 64 by 64 by 64
 * split across an arbitrary number of MPI ranks.
 */
void compute_dct(MPI_Comm comm){

    int me; // this process rank within the comm
    MPI_Comm_rank(comm, &me);

    int num_ranks; // total number of ranks in the comm
    MPI_Comm_size(comm, &num_ranks);

    // Using input configuration with pencil data format in X direction
    // and output configuration with pencil data in the Z direction.
    // This format uses only two internal reshape operation.
    std::array<int, 2> proc_grid = heffte::make_procgrid(num_ranks);
    std::array<int, 3> input_grid = {1, proc_grid[0], proc_grid[1]};
    std::array<int, 3> output_grid = {proc_grid[0], proc_grid[1], 1};

    // Describe all the indexes across all ranks
    heffte::box3d<> const world = {{0, 0, 0}, {63, 63, 127}};

    // Split the world box into a 2D grid of boxes
    std::vector<heffte::box3d<>> inboxes  = heffte::split_world(world, input_grid);
    std::vector<heffte::box3d<>> outboxes = heffte::split_world(world, output_grid);

    // Select the backend to use, prefer FFTW and fallback to the stock backend
    // The real-to-real transforms have _cos and _sin appended
    #ifdef Heffte_ENABLE_FFTW
    using backend_tag = heffte::backend::fftw_cos;
    #else
    using backend_tag = heffte::backend::stock_cos;
    #endif

    // define the heffte class and the input and output geometry
    // note that rtransform is just an alias to fft3d
    heffte::rtransform<backend_tag> tcos(inboxes[me], outboxes[me], comm);
    heffte::rtransform<heffte::backend::stock> fft(inboxes[me], outboxes[me], comm);


    tcos.nullify_executor(1);
    tcos.nullify_executor(2);

    fft.nullify_executor(2);


    double dx=0.1; 
    int world_plane = 64 * 128;
    int world_stride = 64;
    std::vector<double> world_input(64 * 64 * 128);
    for( int i=0; i<128; i++ )
        for( int j=0; j<64; j++ )
            for( int k=0; k<64; k++ )
                world_input[i* world_plane + j * world_stride + k] = std::sin(dx*i) + std::cos(dx*(i-j+k));


    std::vector<double> input(tcos.size_inbox());
   
    // set the strides for the triple indexes
    int local_plane = inboxes[me].size[0] * inboxes[me].size[1];
    int local_stride = inboxes[me].size[0];
    // note the order of the loops corresponding to the default order (0, 1, 2)
    // order (0, 1, 2) means that the data in dimension 0 is contiguous
    for(int i=inboxes[me].low[2]; i <= inboxes[me].high[2]; i++)
        for(int j=inboxes[me].low[1]; j <= inboxes[me].high[1]; j++)
            for(int k=inboxes[me].low[0]; k <= inboxes[me].high[0]; k++)
                input[(i - inboxes[me].low[2]) * local_plane
                      + (j - inboxes[me].low[1]) * local_stride + k - inboxes[me].low[0]]
                    = world_input[i * world_plane + j * world_stride + k];


    std::cout<<me<<" ";
    for(size_t i=0; i<10; i++)
        std::cout<<input[i]<<" ";
    std::cout<<std::endl;



    // vectors with the correct sizes to store the input and output data
    // taking the size of the input and output boxes
    std::vector<double> output(tcos.size_outbox());
    std::vector<std::complex<double>> output_complex(tcos.size_outbox());

    // the workspace vector is of a real type too
    std::vector<double> workspace(tcos.size_workspace());


    tcos.forward(input.data(), output.data());
    fft.forward(output.data(), output_complex.data());


    // compute the inverse or backward transform
    std::vector<std::complex<double>> inverse_complex(tcos.size_inbox());
    std::vector<double> inverse(tcos.size_inbox());

    fft.backward(output_complex.data(), inverse_complex.data());
    
    for( int i=0; i<inverse_complex.size(); i++) 
        inverse[i] = std::real(inverse_complex[i]);

    tcos.backward(inverse.data(), inverse.data(), workspace.data());

    for(size_t i=0; i<inverse.size(); i++) {
        inverse[i] /= 64*64*128*2;
    }


    std::cout<<me<<" ";
    for(size_t i=0; i<10; i++)
        std::cout<<inverse[i]<<" ";
    std::cout<<std::endl;


    double err = 0.0;
    for(size_t i=0; i<inverse.size(); i++)
        err = std::max(err, std::abs(inverse[i] - input[i]));

    // print the error for each MPI rank
    std::cout << std::scientific;
    for(int i=0; i<num_ranks; i++){
        if (me == i) std::cout << "rank " << i << " error: " << err << std::endl;
        MPI_Barrier(comm);
    }
}

int main(int argc, char** argv){

    MPI_Init(&argc, &argv);

    compute_dct(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
