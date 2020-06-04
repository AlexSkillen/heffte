/*
    -- HEFFTE (version 0.2) --
       Univ. of Tennessee, Knoxville
       @date
*/

#ifndef HEFFTE_PLAN_LOGIC_H
#define HEFFTE_PLAN_LOGIC_H

#include "heffte_geometry.h"
#include "heffte_common.h"

namespace heffte {

/*!
 * \brief Defines a set of tweaks and options to use in the plan generation.
 */
struct plan_options{
    template<typename backend_tag> plan_options(backend_tag const)
        : use_reorder(default_plan_options<backend_tag>::use_reorder),
          use_alltoall(true)
    {}
    //! \brief Defines whether to transpose the data on reshape or to use strided 1-D ffts.
    bool use_reorder;
    //! \brief Defines whether to use point to point or all to all communications.
    bool use_alltoall;
};

/*!
 * \brief Returns the default backend options associated with the given backend.
 */
template<typename backend_tag>
plan_options default_options(){
    return plan_options(backend_tag());
}

/*!
 * \brief The logic plan incorporates the order and types of operations in a transform.
 *
 * The logic_plan is used to separate the logic of the order of basic operations (reshape or fft execute)
 * from the constructor of the fft3d and fft3d_r2c classes.
 * In this manner, detection of pencils vs. brick distribution of the data and/or making decisions regarding
 * the transposition of indexing can be done in sufficiently robust and complex logic without
 * clutter of the main classes or unnecessary repetition of code.
 *
 * Node that reshape operation \b i will be performed only if in_shape[i] and out_shape[i] are different.
 */
struct logic_plan3d{
    //! \brief Holds the input shapes for the 4 forward reshapes (backwards reverses in and out).
    std::vector<box3d> in_shape[4];
    //! \brief Holds the output shapes for the 4 forward reshapes (backwards reverses in and out).
    std::vector<box3d> out_shape[4];
    //! \brief Direction of the 1-D fft.
    std::array<int, 3> fft_direction;
    //! \brief The total number of indexes in all directions.
    long long index_count;
    //! \brief Extra options used in the plan creation.
    plan_options const options;
};

/*!
 * \brief Returns true for each direction where the boxes form pencils (size matches the world size).
 */
inline std::array<bool, 3> pencil_directions(box3d const world, std::vector<box3d> const &boxes){
    std::array<bool, 3> is_pencil = {true, true, true};
    for(auto const b : boxes){
        for(int i=0; i<3; i++)
            is_pencil[i] = is_pencil[i] and (world.size[i] == b.size[i]);
    }
    return is_pencil;
}

/*!
 * \brief Creates the logic plan with the provided user input.
 *
 * \param boxes is the current distribution of the data across the MPI comm
 * \param r2c_direction is the direction is the direction of shrinking of the data for an r2c transform
 *              the c2c case should use -1
 * \param opts is a set of plan_options to use
 *
 * \returns the plan for reshape and 1-D fft transformations
 */
logic_plan3d plan_operations(ioboxes const &boxes, int r2c_direction, plan_options const opts);

}

#endif