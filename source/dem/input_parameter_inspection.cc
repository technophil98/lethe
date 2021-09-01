#include <dem/find_maximum_particle_size.h>
#include <dem/input_parameter_inspection.h>

using namespace dealii;

template <int dim>
void
input_parameter_inspection(const DEMSolverParameters<dim> &dem_parameters,
                           const ConditionalOStream &      pcout,
                           const double &standard_deviation_multiplier)
{
  // Getting the input parameters as local variable
  auto   parameters          = dem_parameters;
  auto   physical_properties = dem_parameters.lagrangian_physical_properties;
  double rayleigh_time_step  = 0;

  for (unsigned int i = 0; i < physical_properties.particle_type_number; ++i)
    rayleigh_time_step =
      std::max(M_PI_2 * physical_properties.particle_average_diameter[i] *
                 sqrt(2 * physical_properties.density_particle[i] *
                      (2 + physical_properties.poisson_ratio_particle[i]) *
                      (1 - physical_properties.poisson_ratio_particle[i]) /
                      physical_properties.youngs_modulus_particle[i]) /
                 (0.1631 * physical_properties.poisson_ratio_particle[i] +
                  0.8766),
               rayleigh_time_step);

  const double time_step_rayleigh_ratio =
    parameters.simulation_control.dt / rayleigh_time_step;
  pcout << "DEM time-step is " << time_step_rayleigh_ratio * 100
        << "% of Rayleigh time step" << std::endl;

  if (time_step_rayleigh_ratio > 0.15)
    {
      pcout << "Warning: It is recommended to decrease the time-step"
            << std::endl;
    }
  else if (time_step_rayleigh_ratio < 0.01)
    {
      pcout << "Warning: It is recommended to increase the time-step"
            << std::endl;
    }

  // Checking particle size range
  for (unsigned int i = 0; i < physical_properties.particle_type_number; ++i)
    {
      if (physical_properties.particle_average_diameter.at(i) -
            standard_deviation_multiplier *
              physical_properties.particle_size_std.at(i) <
          0)
        {
          pcout
            << "Warning: Requested particle size distribution for type: " << i
            << " is not well-defined. Using requested distribution may lead to "
               "changes in sampled particle sizes to avoid negative particle "
               "diameters. You can consider decreasing the standard deviation of "
               "size distribution or increasing the particle diameter instead."
            << std::endl;
        }
    }

  // Insertion parameters check
  const double insertion_distance_per_particle =
    0.5 * (parameters.insertion_info.distance_threshold - 1);

  if (parameters.insertion_info.insertion_method ==
        Parameters::Lagrangian::InsertionInfo::InsertionMethod::non_uniform &&
      parameters.insertion_info.random_number_range >=
        insertion_distance_per_particle)
    pcout
      << "Warning: Particles may have collision at the insertion step. This can lead"
         " to high initial velocities (due to initial overlap) or errors when using "
         "less stable integration schemes. It is recommended to decrease the random "
         "number range or to increase the insertion distance threshold."
      << std::endl;

  // Check to see if the cylinder motion is applied to cylinder triangulation
  if (parameters.grid_motion.motion_type ==
        Parameters::Lagrangian::GridMotion<dim>::MotionType::cylinder_motion &&
      parameters.mesh.grid_type != "cylinder")
    pcout
      << "Warning: cylinder_motion should only be used for cylindrical geometries."
      << std::endl;

  // Check to see if the dem_3d solver is used for cylinder motion
  if (parameters.grid_motion.motion_type ==
        Parameters::Lagrangian::GridMotion<dim>::MotionType::cylinder_motion &&
      dim == 2)
    throw std::runtime_error(
      "Grid motion of type 'cylinder_motion' can only be solved with three dimensional"
      "(dem_3d) solver.");

  // Parallel simulations with load-balancing lead to deformed manifolds in
  // simulations with grid motion
  if (parameters.grid_motion.motion_type !=
        Parameters::Lagrangian::GridMotion<dim>::MotionType::none &&
      parameters.model_parameters.load_balance_method !=
        Parameters::Lagrangian::ModelParameters::LoadBalanceMethod::none)
    pcout
      << "Warning: Parallel simulations with load-balancing may lead to deformed manifolds in simulations with grid motion."
      << std::endl;
}

template void
input_parameter_inspection(const DEMSolverParameters<2> &dem_parameters,
                           const ConditionalOStream &    pcout,
                           const double &standard_deviation_multiplier);

template void
input_parameter_inspection(const DEMSolverParameters<3> &dem_parameters,
                           const ConditionalOStream &    pcout,
                           const double &standard_deviation_multiplier);
