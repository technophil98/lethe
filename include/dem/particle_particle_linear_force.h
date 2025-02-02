/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 - 2019 by the Lethe authors
 *
 * This file is part of the Lethe library
 *
 * The Lethe library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the Lethe distribution.
 *
 * ---------------------------------------------------------------------
 */

#include <dem/dem_solver_parameters.h>
#include <dem/particle_particle_contact_force.h>
#include <dem/particle_particle_contact_info_struct.h>

#include <deal.II/particles/particle.h>
#include <deal.II/particles/particle_iterator.h>

#include <math.h>

#include <iostream>
#include <vector>

using namespace dealii;

#ifndef particle_particle_linear_force_h
#  define particle_particle_linear_force_h

/**
 * Calculation of the linear particle-particle contact force using the
 * information obtained from the fine search and physical properties of
 * particles
 *
 * @author Shahab Golshan, Bruno Blais, Polytechnique Montreal 2019-
 */

template <int dim>
class ParticleParticleLinearForce : public ParticleParticleContactForce<dim>
{
public:
  ParticleParticleLinearForce<dim>(
    const DEMSolverParameters<dim> &dem_parameters);

  /**
   * Carries out the calculation of the particle-particle contact force
   * using linear (Hookean) model
   *
   * @param local_adjacent_particles Required information for the calculation
   * of the local-local particle-particle contact force. This information
   * was obtained in the fine search
   * @param ghost_adjacent_particles Required information for the calculation
   * of the local-ghost particle-particle contact force. This information
   * was obtained in the fine search
   * @param dt DEM time-step
   * @param momentum An unordered_map of momentum of particles
   * @param force Force acting on particles
   */
  virtual void
  calculate_particle_particle_contact_force(
    std::unordered_map<
      types::particle_index,
      std::unordered_map<types::particle_index,
                         particle_particle_contact_info_struct<dim>>>
      &local_adjacent_particles,
    std::unordered_map<
      types::particle_index,
      std::unordered_map<types::particle_index,
                         particle_particle_contact_info_struct<dim>>>
      &                          ghost_adjacent_particles,
    const double &               dt,
    std::vector<Tensor<1, dim>> &momentum,
    std::vector<Tensor<1, dim>> &force) override;

private:
  /**
   * Carries out the calculation of the particle-particle linear contact
   * force and torques based on the updated values in contact_info
   *
   * @param contact_info A container that contains the required information for
   * calculation of the contact force for a particle pair in contact
   * @param normal_relative_velocity_value Normal relative contact velocity
   * @param normal_unit_vector Contact normal unit vector
   * @param normal_overlap Contact normal overlap
   * @param particle_one_properties Properties of particle one in contact
   * @param particle_two_properties Properties of particle two in contact
   * @param normal_force Contact normal force
   * @param tangential_force Contact tangential force
   * @param particle_one_tangential_torque Contact tangential torque on particle one
   * @param particle_two_tangential_torque Contact tangential torque on particle two
   * @param rolling_friction_torque Contact rolling resistance torque
   */
  void
  calculate_linear_contact_force_and_torque(
    particle_particle_contact_info_struct<dim> &contact_info,
    const double &                              normal_relative_velocity_value,
    const Tensor<1, dim> &                      normal_unit_vector,
    const double &                              normal_overlap,
    const ArrayView<const double> &             particle_one_properties,
    const ArrayView<const double> &             particle_two_properties,
    Tensor<1, dim> &                            normal_force,
    Tensor<1, dim> &                            tangential_force,
    Tensor<1, dim> &                            particle_one_tangential_torque,
    Tensor<1, dim> &                            particle_two_tangential_torque,
    Tensor<1, dim> &                            rolling_resistance_torque);

  // Normal and tangential contact forces, tangential and rolling torques,
  // normal unit vector of the contact and contact relative velocity in the
  // normal direction
  Tensor<1, dim>               normal_unit_vector;
  Tensor<1, dim>               normal_force;
  Tensor<1, dim>               tangential_force;
  Tensor<1, dim>               particle_one_tangential_torque;
  Tensor<1, dim>               particle_two_tangential_torque;
  Tensor<1, dim>               rolling_resistance_torque;
  double                       normal_relative_velocity_value;
  RollingResistanceTorqueModel rolling_reistance_model;
};

#endif
