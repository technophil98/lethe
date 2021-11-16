#include <core/bdf.h>
#include <core/sdirk.h>

#include <solvers/tracer_scratch_data.h>

template <int dim>
void
TracerScratchData<dim>::allocate()
{
  // Initialize size of arrays
  this->n_q_points = fe_values_tracer.get_quadrature().size();
  this->n_dofs     = fe_values_tracer.get_fe().n_dofs_per_cell();

  // Initialize arrays related to quadrature
  this->JxW = std::vector<double>(n_q_points);

  // Forcing term array
  this->source = std::vector<double>(n_q_points);

  // Initialize arrays related to velocity and pressure
  this->velocities.first_vector_component = 0;
  // Velocity
  this->velocity_values = std::vector<Tensor<1, dim>>(n_q_points);
  // Tracer
  this->tracer_values     = std::vector<double>(n_q_points);
  this->tracer_gradients  = std::vector<Tensor<1, dim>>(n_q_points);
  this->tracer_laplacians = std::vector<double>(n_q_points);

  // Velocity for BDF schemes
  this->previous_tracer_values =
    std::vector<std::vector<double>>(maximum_number_of_previous_solutions(),
                                     std::vector<double>(n_q_points));

  // Velocity for SDIRK schemes
  this->stages_tracer_values =
    std::vector<std::vector<double>>(max_number_of_intermediary_stages(),
                                     std::vector<double>(n_q_points));


  // Initialize arrays related to shape functions
  // Velocity shape functions
  this->phi =
    std::vector<std::vector<double>>(n_q_points, std::vector<double>(n_dofs));
  this->grad_phi = std::vector<std::vector<Tensor<1, dim>>>(
    n_q_points, std::vector<Tensor<1, dim>>(n_dofs));
  this->hess_phi = std::vector<std::vector<Tensor<2, dim>>>(
    n_q_points, std::vector<Tensor<2, dim>>(n_dofs));
  this->laplacian_phi =
    std::vector<std::vector<double>>(n_q_points, std::vector<double>(n_dofs));
}

template <int dim>
void
DGTracerScratchData<dim>::allocate_cell()
{
  // Cell allocations
  // Initialize size of arrays
  this->cell_n_q_points = fe_values_tracer.get_quadrature().size();
  this->cell_n_dofs     = fe_values_tracer.get_fe().n_dofs_per_cell();

  // Initialize arrays related to quadrature
  this->cell_JxW = std::vector<double>(cell_n_q_points);

  // Forcing term array
  this->cell_source = std::vector<double>(cell_n_q_points);

  // Initialize arrays related to velocity and pressure
  this->velocities.first_vector_component = 0;
  // Velocity
  this->cell_velocity_values = std::vector<Tensor<1, dim>>(cell_n_q_points);
  // Tracer
  this->cell_tracer_values     = std::vector<double>(cell_n_q_points);
  this->cell_tracer_gradients  = std::vector<Tensor<1, dim>>(cell_n_q_points);
  this->cell_tracer_laplacians = std::vector<double>(cell_n_q_points);

  // Velocity for BDF schemes
  this->cell_previous_tracer_values =
    std::vector<std::vector<double>>(maximum_number_of_previous_solutions(),
                                     std::vector<double>(cell_n_q_points));

  // Velocity for SDIRK schemes
  this->cell_stages_tracer_values =
    std::vector<std::vector<double>>(max_number_of_intermediary_stages(),
                                     std::vector<double>(cell_n_q_points));

  // Initialize arrays related to shape functions
  // Velocity shape functions
  this->cell_phi =
    std::vector<std::vector<double>>(cell_n_q_points,
                                     std::vector<double>(cell_n_dofs));
  this->cell_grad_phi = std::vector<std::vector<Tensor<1, dim>>>(
    cell_n_q_points, std::vector<Tensor<1, dim>>(cell_n_dofs));
  this->cell_hess_phi = std::vector<std::vector<Tensor<2, dim>>>(
    cell_n_q_points, std::vector<Tensor<2, dim>>(cell_n_dofs));
  this->cell_laplacian_phi =
    std::vector<std::vector<double>>(cell_n_q_points,
                                     std::vector<double>(cell_n_dofs));
}

template <int dim>
void
DGTracerScratchData<dim>::allocate_face(const unsigned int face_n_dofs,
                                        const unsigned int face_n_q_points)
{
  // Face allocations
  // Initialize size of arrays
  this->face_n_dofs     = face_n_dofs;
  this->face_n_q_points = face_n_q_points;

  // Initialize arrays related to quadrature
  this->face_JxW = std::vector<double>(face_n_q_points);

  // Forcing term array
  this->face_source = std::vector<double>(face_n_q_points);

  // Initialize arrays related to velocity and pressure
  this->velocities.first_vector_component = 0;

  // Velocity
  this->face_velocity_values = std::vector<Tensor<1, dim>>(face_n_q_points);

  // Tracer
  this->face_tracer_jump =
    std::vector<std::vector<double>>(face_n_q_points,
                                     std::vector<double>(face_n_dofs));
  this->face_phi_inflow =
    std::vector<std::vector<double>>(face_n_q_points,
                                     std::vector<double>(face_n_dofs));
  this->face_phi_outflow =
    std::vector<std::vector<double>>(face_n_q_points,
                                     std::vector<double>(face_n_dofs));
  this->face_tracer_average_gradients =
    std::vector<std::vector<Tensor<1, dim>>>(
      face_n_q_points, std::vector<Tensor<1, dim>>(face_n_dofs));
}

template <int dim>
void
DGTracerScratchData<dim>::allocate_boundary(
  const unsigned int boundary_n_dofs,
  const unsigned int boundary_n_q_points)
{
  // Boundary allocations
  // Initialize size of arrays
  this->boundary_n_dofs     = boundary_n_dofs;
  this->boundary_n_q_points = boundary_n_q_points;

  // Initialize arrays related to quadrature
  this->boundary_JxW = std::vector<double>(face_n_q_points);

  // Forcing term array
  this->boundary_source = std::vector<double>(boundary_n_q_points);

  // Initialize arrays related to velocity and pressure
  this->velocities.first_vector_component = 0;

  // Velocity
  this->face_velocity_values = std::vector<Tensor<1, dim>>(boundary_n_q_points);
  // Tracer
  this->boundary_tracer_values = std::vector<double>(boundary_n_q_points);
  this->boundary_tracer_gradients =
    std::vector<Tensor<1, dim>>(boundary_n_q_points);
  this->boundary_tracer_laplacians = std::vector<double>(boundary_n_q_points);
}


template class TracerScratchData<2>;
template class TracerScratchData<3>;
template class DGTracerScratchData<2>;
template class DGTracerScratchData<3>;
