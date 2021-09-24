//
// Created by lucka on 2021-10-11.
//

#ifndef LETHE_IB_PARTICLES_DEM_H
#define LETHE_IB_PARTICLES_DEM_H

#include <core/ib_particle.h>
#include <core/ib_stencil.h>
#include <core/lethegridtools.h>


#include <solvers/navier_stokes_base.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>

using namespace dealii;
using namespace std;


/**
 * A solver class for the DEM used in conjunction with IB particles and gls_sharp_navier_stokes.
 * This class defines and use some function of the DEM class that has been modified and simplified to use IB_particles.
 *
 * @tparam dim An integer that denotes the dimension of the space in which
 * the flow is solved
 *
 * @ingroup solvers
 * @author Lucka Barbeau, Bruno Blais, Shahab Golshan 2021
 */

template <int dim>
class IBParticlesDEM
{
public:



  void
  initialize(SimulationParameters<dim> p_nsparam, MPI_Comm&     mpi_communicator_input);


  void
  update_particles(std::vector<IBParticle<dim>> particles);


  /**
   * @brief
   * Function that allow subtime stepping to allow contact between particle.
   * upcomming PR.
   */
  void
  particles_dem(double dt, bool is_at_start);

  /**
   * @brief Calculate non-linear (Hertzian) particle-particle contact force
   */
  void
  calculate_pp_contact_force(const double &               dt_dem,
                             std::vector<Tensor<1, dim>> &contact_force,
                             std::vector<Tensor<1, 3>> &contact_torque);


  /**
   * @brief Calculate non-linear (Hertzian) particle-wall contact force
   */
  void
  calculate_pw_contact_force(const double &               dt_dem,
                             std::vector<Tensor<1, dim>> &contact_force,
                             std::vector<Tensor<1, 3>> &contact_torque);



  void
  update_particles_boundary_contact(std::vector<IBParticle<dim>>& particles, DoFHandler<dim> & dof_handler);


  std::vector<IBParticle<dim>> dem_particles;

private:
  // A struct to store contact tangential history
  struct ContactTangentialHistory
  {
    Tensor<1, dim> tangential_relative_velocity;
    Tensor<1, dim> tangential_overlap;
  };
  // Particles contact history

  // A struct to store boundary cells' information
  struct BoundaryCellsInfo
  {
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
      for(unsigned int i=0; i<dim; ++i){
          ar & normal_vector;
          ar &  point_on_boundary;
        }
    }

    Tensor<1, dim> normal_vector;
    Point<dim>     point_on_boundary;

  };

  /** This function is used to find the projection of vector_a on
   * vector_b
   * @param vector_a A vector which is going to be projected on vector_b
   * @param vector_b The projection vector of vector_a
   * @return The projection of vector_a on vector_b
   */
  inline Tensor<1, dim>
  find_projection(const Tensor<1, dim> &vector_a,
                  const Tensor<1, dim> &vector_b)
  {
    Tensor<1, dim> vector_c;
    vector_c = ((vector_a * vector_b) / (vector_b.norm_square())) * vector_b;

    return vector_c;
  }

  SimulationParameters<dim> parameters;

  MPI_Comm          mpi_communicator;

  // Particles contact history
  std::map<unsigned int, std::map<unsigned int, ContactTangentialHistory>>
    pp_contact_map;
  std::map<unsigned int, std::map<unsigned int, ContactTangentialHistory>>
    pw_contact_map;

  std::vector<std::vector<BoundaryCellsInfo>> boundary_cells;

};


#endif // LETHE_IB_PARTICLES_DEM_H