#include <dem/locate_ghost_particles.h>

using namespace dealii;

template <int dim>
void
locate_ghost_particles_in_cells(
  const Particles::ParticleHandler<dim> &particle_handler,
  std::unordered_map<types::particle_index, Particles::ParticleIterator<dim>>
    &ghost_particle_container,
  std::unordered_map<
    types::particle_index,
    std::unordered_map<types::particle_index,
                       particle_particle_contact_info_struct<dim>>>
    &ghost_adjacent_particles)
{
  update_ghost_particle_container<dim>(ghost_particle_container,
                                       &particle_handler);

  update_ghost_iterator_particle_particle_contact_container<dim>(
    ghost_adjacent_particles, ghost_particle_container);
}

template void
locate_ghost_particles_in_cells(
  const Particles::ParticleHandler<2> &particle_handler,
  std::unordered_map<types::particle_index, Particles::ParticleIterator<2>>
    &ghost_particle_container,
  std::unordered_map<
    types::particle_index,
    std::unordered_map<types::particle_index,
                       particle_particle_contact_info_struct<2>>>
    &ghost_adjacent_particles);

template void
locate_ghost_particles_in_cells(
  const Particles::ParticleHandler<3> &particle_handler,
  std::unordered_map<types::particle_index, Particles::ParticleIterator<3>>
    &ghost_particle_container,
  std::unordered_map<
    types::particle_index,
    std::unordered_map<types::particle_index,
                       particle_particle_contact_info_struct<3>>>
    &ghost_adjacent_particles);
