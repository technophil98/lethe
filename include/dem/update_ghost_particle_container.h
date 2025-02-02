/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 - 2020 by the Lethe authors
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

 *
 * Author: Shahab Golshan, Polytechnique Montreal, 2019
 */
#include <deal.II/particles/particle_handler.h>

using namespace dealii;

#ifndef update_ghost_particle_container_h
#  define update_ghost_particle_container_h

/**
 * Updates the iterators to local particles in a map of particles
 * (particle_container)
 *
 * @param particle_container A map of particles which is used to update
 * the iterators to particles in particle-particle and particle-wall fine search
 * outputs after calling sort particles into cells function
 * @param particle_handler Particle handler to access all the particles in the
 * system
 */

template <int dim>
void
update_ghost_particle_container(
  std::unordered_map<types::particle_index, Particles::ParticleIterator<dim>>
    &                                    ghost_particle_container,
  const Particles::ParticleHandler<dim> *particle_handler);

#endif /* update_ghost_particle_container_h */
