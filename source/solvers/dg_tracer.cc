#include <core/bdf.h>
#include <core/sdirk.h>
#include <core/time_integration_utilities.h>
#include <core/utilities.h>

#include <solvers/dg_tracer.h>
#include <solvers/tracer_assemblers.h>
#include <solvers/tracer_scratch_data.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/vector_tools.h>

template <int dim>
void
DGTracer<dim>::setup_assemblers()
{
  this->assemblers.clear();

  // Time-stepping schemes
  if (is_bdf(this->simulation_control->get_assembly_method()))
    {
      this->assemblers.push_back(
        std::make_shared<TracerAssemblerBDF<dim>>(this->simulation_control));
    }
  // Core assembler
  /* No assembler per se for the core: this assembly will be inside this .cc
   file
   * this->assemblers.push_back(std::make_shared<DGTracerAssemblerCore<dim>>(
    this->simulation_control, this->simulation_parameters.physical_properties));
    */
}

template <int dim>
void
DGTracer<dim>::assemble_system_matrix()
{
  this->system_matrix = 0;
  setup_assemblers();

  const DoFHandler<dim> *dof_handler_fluid =
    multiphysics->get_dof_handler(PhysicsID::fluid_dynamics);
  auto scratch_data = DGTracerScratchData<dim>(*this->fe,
                                               *this->cell_quadrature,
                                               *this->face_quadrature,
                                               *this->mapping,
                                               dof_handler_fluid->get_fe());

  using Iterator = typename DoFHandler<dim>::active_cell_iterator;

  const auto cell_worker = [&](const Iterator &          cell,
                               DGTracerScratchData<dim> &scratch_data,
                               DGMethodsCopyData &       copy_data) {
    copy_data.cell_is_local = cell->is_locally_owned();
    if (!copy_data.cell_is_local)
      return;

    auto &source_term = simulation_parameters.source_term->tracer_source;
    source_term.set_time(simulation_control->get_current_time());

    scratch_data.reinit(cell,
                        this->evaluation_point,
                        this->previous_solutions,
                        this->solution_stages,
                        &source_term);

    const DoFHandler<dim> *dof_handler_fluid =
      multiphysics->get_dof_handler(PhysicsID::fluid_dynamics);

    typename DoFHandler<dim>::active_cell_iterator velocity_cell(
      &(*triangulation), cell->level(), cell->index(), dof_handler_fluid);

    if (multiphysics->fluid_dynamics_is_block())
      {
        scratch_data.reinit_velocity(velocity_cell,
                                     *multiphysics->get_block_solution(
                                       PhysicsID::fluid_dynamics));
      }
    else
      {
        scratch_data.reinit_velocity(velocity_cell,
                                     *multiphysics->get_solution(
                                       PhysicsID::fluid_dynamics));
      }



    // Scheme and physical properties
    const double diffusivity =
      this->simulation_parameters.physical_properties.tracer_diffusivity;
    const auto method = this->simulation_control->get_assembly_method();

    // Loop and quadrature informations
    const auto &       JxW_vec    = scratch_data.cell_JxW;
    const unsigned int n_q_points = scratch_data.cell_n_q_points;
    const unsigned int n_dofs     = scratch_data.cell_n_dofs;

    // Copy data elements
    copy_data.reset(cell, scratch_data.cell_n_dofs);
    auto &local_matrix = copy_data.local_matrix;

    // assembling local matrix and right hand side
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        // Gather into local variables the relevant fields
        const Tensor<1, dim> tracer_gradient =
          scratch_data.cell_tracer_gradients[q];
        const Tensor<1, dim> velocity = scratch_data.cell_velocity_values[q];

        // Store JxW in local variable for faster access;
        const double JxW = JxW_vec[q];

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
              const Tensor<1, dim> grad_phi_T_i = scratch_data.cell_grad_phi[q][i];

            for (unsigned int j = 0; j < n_dofs; ++j)
              {
                const Tensor<1, dim> grad_phi_T_j =
                  scratch_data.cell_grad_phi[q][j];
                const auto phi_T_j = scratch_data.cell_phi[q][j];

                // Weak form : - D * laplacian T +  u * gradT - f=0
                local_matrix(i, j) += diffusivity *
                                      grad_phi_T_i   // \nabla \phi_i
                                      * grad_phi_T_j // \nabla \phi_j
                                      * JxW;         // dx

                local_matrix(i, j) += -velocity * grad_phi_T_i * phi_T_j * JxW;
              }
          }
      } // end loop on quadrature points



    for (auto &assembler : this->assemblers)
      {
        // TODO S'ASSURER QUE L'ASSEMBLAGE EN SS EST POSSIBLE
        // assembler->assemble_matrix(scratch_data, copy_data);
      }
  };

  const auto face_worker = [&](const Iterator &          cell,
                               const unsigned int &      f,
                               const unsigned int &      sf,
                               const Iterator &          ncell,
                               const unsigned int &      nf,
                               const unsigned int &      nsf,
                               DGTracerScratchData<dim> &scratch_data,
                               DGMethodsCopyData &       copy_data) {
    copy_data.cell_is_local = cell->is_locally_owned();
    if (!copy_data.cell_is_local)
      return;

    auto &source_term = simulation_parameters.source_term->tracer_source;
    source_term.set_time(simulation_control->get_current_time());

    scratch_data.reinit(cell,
                        f,
                        sf,
                        ncell,
                        nf,
                        nsf,
                        this->evaluation_point,
                        this->previous_solutions,
                        this->solution_stages,
                        &source_term);

    const DoFHandler<dim> *dof_handler_fluid =
      multiphysics->get_dof_handler(PhysicsID::fluid_dynamics);

    typename DoFHandler<dim>::active_cell_iterator velocity_cell(
      &(*triangulation), cell->level(), cell->index(), dof_handler_fluid);

    if (multiphysics->fluid_dynamics_is_block())
      {
        scratch_data.reinit_velocity(velocity_cell,
                                     f,
                                     *multiphysics->get_block_solution(
                                       PhysicsID::fluid_dynamics));
      }
    else
      {
        scratch_data.reinit_velocity(velocity_cell,
                                     f,
                                     *multiphysics->get_solution(
                                       PhysicsID::fluid_dynamics));
      }



    FEInterfaceValues<dim> &fe_iv = scratch_data.fe_interface_values_tracer;

    const auto &       JxW_vec    = scratch_data.face_JxW;
    const unsigned int n_q_points = scratch_data.face_n_q_points;
    const unsigned int n_dofs     = scratch_data.face_n_dofs;


      // Copy data elements
      copy_data.face_data.emplace_back(n_dofs);
    DGMethodsCopyDataFace &copy_data_face = copy_data.face_data.back();
    copy_data_face.joint_dof_indices      = fe_iv.get_interface_dof_indices();
    copy_data_face.local_matrix.reinit(n_dofs, n_dofs);
      auto &local_matrix = copy_data_face.local_matrix;

      const std::vector<Tensor<1, dim>> &normals = scratch_data.face_normals;

    // Scheme and physical properties
    const double diffusivity =
      this->simulation_parameters.physical_properties.tracer_diffusivity;
    const auto method = this->simulation_control->get_assembly_method();

    // Loop and quadrature informations
    const double extent1 = scratch_data.face_size;
    const double extent2 = scratch_data.face_size_neighbor;
    const double penalty =
      get_penalty_factor(scratch_data.fe_tracer_degree, extent1, extent2);


    // assembling local matrix and right hand side
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        // Gather into local variables the relevant fields
        const Tensor<1, dim> velocity = scratch_data.face_velocity_values[q];
          const double velocity_dot_n = velocity * normals[q];
        // Store JxW in local variable for faster access;
        const double JxW = JxW_vec[q];

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            for (unsigned int j = 0; j < n_dofs; ++j)
              {
                // Weak form : - D * laplacian T +  u * gradT - f=0

                local_matrix(i, j) += -diffusivity * normals[q] *
                                      fe_iv.average_gradient(i, q) *
                                      fe_iv.jump(j, q) * JxW;

                local_matrix(i, j) += -diffusivity * fe_iv.jump(i, q) // \phi_i
                                      * fe_iv.average_gradient(j, q) *
                                      normals[q] // n*\nabla \phi_j
                                      * JxW;     // dx

                local_matrix(i, j) += penalty * diffusivity * fe_iv.jump(i, q) *
                                      fe_iv.jump(j, q) * JxW;

                local_matrix(i, j) +=
                  fe_iv.jump(i, q) // [\phi_i]
                  * fe_iv.shape_value((velocity_dot_n > 0), j, q) *
                  velocity_dot_n * JxW;
              }
          }
      } // end loop on quadrature points



    for (auto &assembler : this->assemblers)
      {
        // TODO S'ASSURER QUE L'ASSEMBLAGE EN SS EST POSSIBLE
        // assembler->assemble_matrix(scratch_data, copy_data);
      }
  };

  const auto boundary_worker = [&](const Iterator &          cell,
                                   const unsigned int &      face_no,
                                   DGTracerScratchData<dim> &scratch_data,
                                   DGMethodsCopyData &       copy_data) {
    copy_data.cell_is_local = cell->is_locally_owned();
    if (!copy_data.cell_is_local)
      return;

    auto &source_term = simulation_parameters.source_term->tracer_source;
    source_term.set_time(simulation_control->get_current_time());
    scratch_data.reinit(cell,
                        face_no,
                        this->evaluation_point,
                        this->previous_solutions,
                        this->solution_stages,
                        &source_term);

    const DoFHandler<dim> *dof_handler_fluid =
      multiphysics->get_dof_handler(PhysicsID::fluid_dynamics);

    typename DoFHandler<dim>::active_cell_iterator velocity_cell(
      &(*triangulation), cell->level(), cell->index(), dof_handler_fluid);

    if (multiphysics->fluid_dynamics_is_block())
      {
        scratch_data.reinit_velocity(velocity_cell,
                                     face_no,
                                     *multiphysics->get_block_solution(
                                       PhysicsID::fluid_dynamics));
      }
    else
      {
        scratch_data.reinit_velocity(velocity_cell,
                                     face_no,
                                     *multiphysics->get_solution(
                                       PhysicsID::fluid_dynamics));
      }



    // Scheme and physical properties
    const double diffusivity =
      this->simulation_parameters.physical_properties.tracer_diffusivity;
    const auto method = this->simulation_control->get_assembly_method();

    // Loop and quadrature informations
    const FEFaceValuesBase<dim> &fe_face =
      scratch_data.fe_interface_values_tracer.get_fe_face_values(0);
    const auto &       JxW_vec    = scratch_data.boundary_JxW;
    const unsigned int n_q_points = scratch_data.boundary_n_q_points;
    const unsigned int n_dofs     = scratch_data.boundary_n_dofs;
    const double       extent1    = scratch_data.boundary_size;
    const double       penalty =
      get_penalty_factor(scratch_data.fe_tracer_degree, extent1, extent1);

    copy_data.face_data.emplace_back(n_dofs);
    DGMethodsCopyDataFace &copy_data_face = copy_data.face_data.back();
    copy_data_face.local_matrix.reinit(n_dofs, n_dofs);
    // Copy data elements
    auto &local_matrix = copy_data_face.local_matrix;
    copy_data_face.joint_dof_indices =
      scratch_data.fe_interface_values_tracer.get_interface_dof_indices();


    const std::vector<Tensor<1, dim>> &normals = scratch_data.boundary_normals;


    // assembling local matrix and right hand side
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        // Gather into local variables the relevant fields
        const Tensor<1, dim> tracer_gradient =
          scratch_data.boundary_tracer_gradients[q];
        const Tensor<1, dim> velocity = scratch_data.face_velocity_values[q];

        // Store JxW in local variable for faster access;
        const double JxW = JxW_vec[q];

        const double velocity_dot_n = velocity * normals[q];
        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            for (unsigned int j = 0; j < n_dofs; ++j)
              {
                // Weak form : - D * laplacian T +  u * gradT - f=0
                // if (cell->face(face_no)->boundary_id() == 0) //TODO CHECK
                // IF BOUNDARY
                //{
                // Copy data elements
                for (unsigned int i_bc = 0;
                     i_bc < this->simulation_parameters
                              .boundary_conditions_tracer.size;
                     ++i_bc)
                  {
                    if (cell->face(face_no)->boundary_id() == i_bc)
                      {
                        // Dirichlet condition : imposed temperature at i_bc
                        if (this->simulation_parameters
                              .boundary_conditions_tracer.type[i_bc] ==
                            BoundaryConditions::BoundaryType::tracer_dirichlet)
                          {
                            local_matrix(i, j) +=
                              -diffusivity * normals[q] *
                              fe_face.shape_grad(i, q)    // n*\nabla \phi_i
                              * fe_face.shape_value(j, q) // \phi_j
                              * JxW;                      // dx

                            local_matrix(i, j) +=
                              -diffusivity * fe_face.shape_value(i, q) // \phi_i
                              * normals[q] *
                              fe_face.shape_grad(j, q)
                              // n*\nabla \phi_j
                              * JxW; // dx

                            local_matrix(i, j) +=
                              diffusivity * penalty *
                              fe_face.shape_value(i, q)          // \phi_i
                              * fe_face.shape_value(j, q) * JxW; // dx
                          }
                      }
                  }
                if (velocity_dot_n > 0)
                  {
                    local_matrix(i, j) += fe_face.shape_value(i, q)   // \phi_i
                                          * fe_face.shape_value(j, q) // \phi_j
                                          * velocity_dot_n // \beta . n
                                          * JxW;           // dx
                  }
              }
          }
      } // end loop on quadrature points



    for (auto &assembler : this->assemblers)
      {
        // TODO S'ASSURER QUE L'ASSEMBLAGE EN SS EST POSSIBLE
        // assembler->assemble_matrix(scratch_data, copy_data);
      }
  };

  AffineConstraints<double> constraints_used;
  const auto                copier = [&](const DGMethodsCopyData &copy_data) {
    if (!copy_data.cell_is_local)
      return;
    constraints_used.distribute_local_to_global(copy_data.local_matrix,
                                                copy_data.local_dof_indices,
                                                system_matrix);
    // std::cout<<"Start copier matrix"<<std::endl;
    // copy_data.local_matrix.print(std::cout);
    for (auto &cdf : copy_data.face_data)
      {
        constraints_used.distribute_local_to_global(cdf.local_matrix,
                                                    cdf.joint_dof_indices,
                                                    system_matrix);
        // cdf.local_matrix.print(std::cout);
      }
    // system_matrix.print(std::cout);
  };

  MeshWorker::mesh_loop(this->dof_handler.begin_active(),
                        this->dof_handler.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        DGMethodsCopyData(this->fe->n_dofs_per_cell()),
                        MeshWorker::assemble_own_cells |
                          MeshWorker::assemble_boundary_faces |
                          MeshWorker::assemble_own_interior_faces_once,
                        boundary_worker,
                        face_worker);

  system_matrix.compress(VectorOperation::add);
}

template <int dim>
void
DGTracer<dim>::assemble_system_rhs()
{
  // TimerOutput::Scope t(this->computing_timer, "Assemble RHS");
  this->system_rhs = 0;
  setup_assemblers();

  const DoFHandler<dim> *dof_handler_fluid =
    multiphysics->get_dof_handler(PhysicsID::fluid_dynamics);

  auto scratch_data = DGTracerScratchData<dim>(*this->fe,
                                               *this->cell_quadrature,
                                               *this->face_quadrature,
                                               *this->mapping,
                                               dof_handler_fluid->get_fe());

  using Iterator = typename DoFHandler<dim>::active_cell_iterator;

  const auto cell_worker = [&](const Iterator &          cell,
                               DGTracerScratchData<dim> &scratch_data,
                               DGMethodsCopyData &       copy_data) {
    copy_data.cell_is_local = cell->is_locally_owned();
    if (!copy_data.cell_is_local)
      return;

    auto &source_term = simulation_parameters.source_term->tracer_source;
    source_term.set_time(simulation_control->get_current_time());

    scratch_data.reinit(cell,
                        this->evaluation_point,
                        this->previous_solutions,
                        this->solution_stages,
                        &source_term);

    const DoFHandler<dim> *dof_handler_fluid =
      multiphysics->get_dof_handler(PhysicsID::fluid_dynamics);

    typename DoFHandler<dim>::active_cell_iterator velocity_cell(
      &(*triangulation), cell->level(), cell->index(), dof_handler_fluid);

    if (multiphysics->fluid_dynamics_is_block())
      {
        scratch_data.reinit_velocity(velocity_cell,
                                     *multiphysics->get_block_solution(
                                       PhysicsID::fluid_dynamics));
      }
    else
      {
        scratch_data.reinit_velocity(velocity_cell,
                                     *multiphysics->get_solution(
                                       PhysicsID::fluid_dynamics));
      }



    // Scheme and physical properties
    const double diffusivity =
      this->simulation_parameters.physical_properties.tracer_diffusivity;
    const auto method = this->simulation_control->get_assembly_method();

    // Loop and quadrature informations
    const auto &       JxW_vec    = scratch_data.cell_JxW;
    const unsigned int n_q_points = scratch_data.cell_n_q_points;
    const unsigned int n_dofs     = scratch_data.cell_n_dofs;

      // Copy data elements
      copy_data.reset(cell, scratch_data.cell_n_dofs);
auto &local_rhs = copy_data.local_rhs;

    // assembling local matrix and right hand side
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        // Gather into local variables the relevant fields
        const Tensor<1, dim> tracer_gradient =
          scratch_data.cell_tracer_gradients[q];
        const Tensor<1, dim> velocity = scratch_data.cell_velocity_values[q];

        // Store JxW in local variable for faster access;
        const double JxW = JxW_vec[q];

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            const double phi_T_i      = scratch_data.cell_phi[q][i];
            const Tensor<1, dim> grad_phi_T_i = scratch_data.cell_grad_phi[q][i];

            // rhs for : - D * laplacian T +  u * grad T - f=0
            local_rhs(i) += (phi_T_i * scratch_data.cell_source[q] * JxW);

            // minus Ax
            local_rhs(i) -= (diffusivity * grad_phi_T_i * tracer_gradient -
                             velocity * grad_phi_T_i * scratch_data.cell_tracer_values[q]) *
                            JxW;
          }
      } // end loop on quadrature points



    for (auto &assembler : this->assemblers)
      {
        // TODO S'ASSURER QUE L'ASSEMBLAGE EN SS EST POSSIBLE
        // assembler->assemble_rhs(scratch_data, copy_data);
      }
  };

  const auto face_worker = [&](const Iterator &          cell,
                               const unsigned int &      f,
                               const unsigned int &      sf,
                               const Iterator &          ncell,
                               const unsigned int &      nf,
                               const unsigned int &      nsf,
                               DGTracerScratchData<dim> &scratch_data,
                               DGMethodsCopyData &       copy_data) {
    copy_data.cell_is_local = cell->is_locally_owned();
    if (!copy_data.cell_is_local)
      return;

    auto &source_term = simulation_parameters.source_term->tracer_source;
    source_term.set_time(simulation_control->get_current_time());

    scratch_data.reinit(cell,
                        f,
                        sf,
                        ncell,
                        nf,
                        nsf,
                        this->evaluation_point,
                        this->previous_solutions,
                        this->solution_stages,
                        &source_term);

    const DoFHandler<dim> *dof_handler_fluid =
      multiphysics->get_dof_handler(PhysicsID::fluid_dynamics);

    typename DoFHandler<dim>::active_cell_iterator velocity_cell(
      &(*triangulation), cell->level(), cell->index(), dof_handler_fluid);

    if (multiphysics->fluid_dynamics_is_block())
      {
        scratch_data.reinit_velocity(velocity_cell,
                                     f,
                                     *multiphysics->get_block_solution(
                                       PhysicsID::fluid_dynamics));
      }
    else
      {
        scratch_data.reinit_velocity(velocity_cell,
                                     f,
                                     *multiphysics->get_solution(
                                       PhysicsID::fluid_dynamics));
      }


    FEInterfaceValues<dim> &fe_iv = scratch_data.fe_interface_values_tracer;

    // Loop and quadrature informations
    const auto &       JxW_vec    = scratch_data.face_JxW;
    const unsigned int n_q_points = scratch_data.face_n_q_points;
    const unsigned int n_dofs     = scratch_data.face_n_dofs;

      // Copy data elements
      copy_data.face_data.emplace_back(n_dofs);
    DGMethodsCopyDataFace &copy_data_face = copy_data.face_data.back();
      copy_data_face.local_rhs.reinit(n_dofs);
    copy_data_face.joint_dof_indices      = fe_iv.get_interface_dof_indices();
      auto &local_rhs = copy_data_face.local_rhs;

    const std::vector<Tensor<1, dim>> &normals = scratch_data.face_normals;
    // Scheme and physical properties
    const double diffusivity =
      this->simulation_parameters.physical_properties.tracer_diffusivity;
    const auto method = this->simulation_control->get_assembly_method();

    std::vector<double>         tracer_jump;
    std::vector<Tensor<1, dim>> tracer_gradient_average;
    get_function_jump(fe_iv, this->evaluation_point, tracer_jump);
    get_function_gradient_average(fe_iv,
                                  this->evaluation_point,
                                  tracer_gradient_average);

    // Loop and quadrature informations
    const double extent1 = scratch_data.face_size;
    const double extent2 = scratch_data.face_size_neighbor;
    const double penalty =
      get_penalty_factor(scratch_data.fe_tracer_degree, extent1, extent2);

    // assembling local matrix and right hand side
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        // Gather into local variables the relevant fields
        const Tensor<1, dim> tracer_gradient =
          scratch_data.cell_tracer_gradients[q];
        const Tensor<1, dim> velocity = scratch_data.cell_velocity_values[q];
        const double         velocity_dot_n = velocity * normals[q];

        // Store JxW in local variable for faster access;
        const double JxW = JxW_vec[q];

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            // Gather into local variables the relevant fields
            const auto phi_T_i      = scratch_data.cell_phi[q][i];
            const auto grad_phi_T_i = scratch_data.cell_grad_phi[q][i];

            // rhs for : - D * laplacian T +  u * grad T - f=0

            // minus Ax
            local_rhs(i) -= -diffusivity * normals[q] *
                            fe_iv.average_gradient(i, q) * tracer_jump[q] * JxW;
            local_rhs(i) -= -diffusivity * fe_iv.jump(i, q) // \phi_i
                            * tracer_gradient_average[q] *
                            normals[q] // n*\nabla \phi_j
                            * JxW;     // dx

            local_rhs(i) -=
              penalty * diffusivity * fe_iv.jump(i, q) * tracer_jump[q] * JxW;

            local_rhs(i) -= fe_iv.jump(i, q) // [\phi_i]
                            * scratch_data.face_tracer_values[q] *
                            velocity_dot_n * JxW;
          //TODO INCERTAIN DE COMMENT TRADUIRE LE UPWIND TRACER VALUES
          }
      } // end loop on quadrature points

    for (auto &assembler : this->assemblers)
      {
        // TODO S'ASSURER QUE L'ASSEMBLAGE EN SS EST POSSIBLE
        // assembler->assemble_rhs(scratch_data, copy_data);
      }
  };

  const auto boundary_worker = [&](const Iterator &          cell,
                                   const unsigned int &      face_no,
                                   DGTracerScratchData<dim> &scratch_data,
                                   DGMethodsCopyData &       copy_data) {
    copy_data.cell_is_local = cell->is_locally_owned();
    if (!copy_data.cell_is_local)
      return;

    auto &source_term = simulation_parameters.source_term->tracer_source;
    source_term.set_time(simulation_control->get_current_time());

    scratch_data.reinit(cell,
                        face_no,
                        this->evaluation_point,
                        this->previous_solutions,
                        this->solution_stages,
                        &source_term);

    const DoFHandler<dim> *dof_handler_fluid =
      multiphysics->get_dof_handler(PhysicsID::fluid_dynamics);

    typename DoFHandler<dim>::active_cell_iterator velocity_cell(
      &(*triangulation), cell->level(), cell->index(), dof_handler_fluid);

    if (multiphysics->fluid_dynamics_is_block())
      {
        scratch_data.reinit_velocity(velocity_cell,
                                     face_no,
                                     *multiphysics->get_block_solution(
                                       PhysicsID::fluid_dynamics));
      }
    else
      {
        scratch_data.reinit_velocity(velocity_cell,
                                     face_no,
                                     *multiphysics->get_solution(
                                       PhysicsID::fluid_dynamics));
      }



    // Scheme and physical properties
    const double diffusivity =
      this->simulation_parameters.physical_properties.tracer_diffusivity;
    const auto method = this->simulation_control->get_assembly_method();
    FEInterfaceValues<dim> &fe_iv = scratch_data.fe_interface_values_tracer;
    const FEFaceValuesBase<dim> &fe_face = fe_iv.get_fe_face_values(0);

    // Loop and quadrature informations
    const auto &       JxW_vec    = scratch_data.boundary_JxW;
    const unsigned int n_q_points = scratch_data.boundary_n_q_points;
    const unsigned int n_dofs     = scratch_data.boundary_n_dofs;
    const double       extent1    = scratch_data.boundary_size;
    const double       penalty =
      get_penalty_factor(scratch_data.fe_tracer_degree, extent1, extent1);

    copy_data.face_data.emplace_back(n_dofs);
    DGMethodsCopyDataFace &copy_data_face = copy_data.face_data.back();
    copy_data_face.local_rhs.reinit(n_dofs);
    // Copy data elements
    auto &local_rhs                  = copy_data_face.local_rhs;
    copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();

    const std::vector<Tensor<1, dim>> &normals = scratch_data.boundary_normals;


    unsigned int actual_i_bc;
    bool         is_dirichlet_bc = false;
    for (unsigned int i_bc = 0;
         i_bc < this->simulation_parameters.boundary_conditions_tracer.size;
         ++i_bc)
      {
        if (cell->face(face_no)->boundary_id() == i_bc)
          {
            actual_i_bc     = i_bc;
            is_dirichlet_bc = true;
          }
      }

    // assembling local matrix and right hand side
    std::vector<double> g = scratch_data.boundary_dirichlet;
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        // Dirichlet condition : imposed temperature at i_bc
        if (is_dirichlet_bc)
          {
            scratch_data.compute_dirichlet_values(
              actual_i_bc,
              *this->simulation_parameters.boundary_conditions_tracer
                 .tracer[actual_i_bc]);
            g = scratch_data.boundary_dirichlet;
          }

        // Gather into local variables the relevant fields
        const Tensor<1, dim> tracer_gradient =
          scratch_data.boundary_tracer_gradients[q];
        const Tensor<1, dim> velocity = scratch_data.face_velocity_values[q];

        // Store JxW in local variable for faster access;
        const double JxW            = JxW_vec[q];
        const double velocity_dot_n = velocity * normals[q];

        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            if (cell->face(face_no)->boundary_id() == actual_i_bc &&
                this->simulation_parameters.boundary_conditions_tracer
                    .type[actual_i_bc] ==
                  BoundaryConditions::BoundaryType::tracer_dirichlet)
              {
                // rhs for : - D * laplacian T +  u * grad T - f=0

                local_rhs(i) += penalty * diffusivity *
                                fe_face.shape_value(i, q) // \phi_i
                                * g[q]                    // g
                                * JxW;                    // dx
                local_rhs(i) += -diffusivity * normals[q] *
                                fe_face.shape_grad(i, q) // n*\nabla \phi_i
                                * g[q]                   // g
                                * JxW;                   // dx
                if (velocity_dot_n < 0)
                  local_rhs(i) +=
                    -fe_face.shape_value(i, q) * g[q] * velocity_dot_n * JxW;

                // minus Ax
                local_rhs(i) -= -diffusivity * normals[q] *
                                fe_face.shape_grad(i, q) // n*\nabla \phi_i
                                *
                                scratch_data.boundary_tracer_values[q] // \phi_j
                                * JxW;                                 // dx

                local_rhs(i) -= -diffusivity *
                                fe_face.shape_value(i, q) // \phi_i
                                * normals[q] *
                                scratch_data.boundary_tracer_gradients[q]
                                // n*\nabla \phi_j
                                * JxW; // dx

                local_rhs(i) -=
                  diffusivity * penalty * fe_face.shape_value(i, q) // \phi_i
                  * scratch_data.boundary_tracer_values[q] * JxW;   // dx

                if (velocity_dot_n > 0)
                  {
                    local_rhs(i) -=
                      fe_face.shape_value(i, q)                // \phi_i
                      * scratch_data.boundary_tracer_values[q] // \phi_j
                      * velocity_dot_n                         // \beta . n
                      * JxW;                                   // dx
                  }
              }
          }
      } // end loop on quadrature points



    for (auto &assembler : this->assemblers)
      {
        // TODO S'ASSURER QUE L'ASSEMBLAGE EN SS EST POSSIBLE
        // assembler->assemble_rhs(scratch_data, copy_data);
      }
  };

  AffineConstraints<double> constraints_used;
  const auto                copier = [&](const DGMethodsCopyData &copy_data) {
    if (!copy_data.cell_is_local)
      return;

    constraints_used.distribute_local_to_global(copy_data.local_rhs,
                                                copy_data.local_dof_indices,
                                                system_rhs);
    // std::cout<<"Start copier rhs"<<std::endl;
    // copy_data.local_rhs.print(std::cout);
    for (auto &cdf : copy_data.face_data)
      {
        constraints_used.distribute_local_to_global(cdf.local_rhs,
                                                    cdf.joint_dof_indices,
                                                    system_rhs);
        // cdf.local_rhs.print(std::cout);
      }
    // system_rhs.print(std::cout);
  };

  MeshWorker::mesh_loop(this->dof_handler.begin_active(),
                        this->dof_handler.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        DGMethodsCopyData(this->fe->n_dofs_per_cell()),
                        MeshWorker::assemble_own_cells |
                          MeshWorker::assemble_boundary_faces |
                          MeshWorker::assemble_own_interior_faces_once,
                        boundary_worker,
                        face_worker);

  this->system_rhs.compress(VectorOperation::add);
}


template <int dim>
void
DGTracer<dim>::attach_solution_to_output(DataOut<dim> &data_out)
{
  data_out.add_data_vector(dof_handler, present_solution, "dg_tracer");
}

template <int dim>
double
DGTracer<dim>::calculate_L2_error()
{
  auto mpi_communicator = triangulation->get_communicator();

  FEValues<dim> fe_values(*mapping,
                          *fe,
                          *cell_quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell =
    fe->dofs_per_cell; // This gives you dofs per cell

  std::vector<types::global_dof_index> local_dof_indices(
    dofs_per_cell); //  Local connectivity

  const unsigned int n_q_points = cell_quadrature->size();

  std::vector<double> q_exact_solution(n_q_points);
  std::vector<double> q_scalar_values(n_q_points);

  auto &exact_solution = simulation_parameters.analytical_solution->tracer;
  exact_solution.set_time(simulation_control->get_current_time());

  double l2error = 0.;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values.get_function_values(present_solution, q_scalar_values);

          // Retrieve the effective "connectivity matrix" for this element
          cell->get_dof_indices(local_dof_indices);

          // Get the exact solution at all gauss points
          exact_solution.value_list(fe_values.get_quadrature_points(),
                                    q_exact_solution);

          for (unsigned int q = 0; q < n_q_points; q++)
            {
              double sim   = q_scalar_values[q];
              double exact = q_exact_solution[q];
              l2error += (sim - exact) * (sim - exact) * fe_values.JxW(q);
            }
        }
    }
  l2error = Utilities::MPI::sum(l2error, mpi_communicator);
  return l2error;
}

template <int dim>
void
DGTracer<dim>::finish_simulation()
{
  auto         mpi_communicator = triangulation->get_communicator();
  unsigned int this_mpi_process(
    Utilities::MPI::this_mpi_process(mpi_communicator));

  if (this_mpi_process == 0 &&
      simulation_parameters.analytical_solution->verbosity ==
        Parameters::Verbosity::verbose)
    {
      error_table.omit_column_from_convergence_rate_evaluation("cells");

      if (simulation_parameters.simulation_control.method ==
          Parameters::SimulationControl::TimeSteppingMethod::steady)
        {
          error_table.evaluate_all_convergence_rates(
            ConvergenceTable::reduction_rate_log2);
        }
      error_table.set_scientific("error_tracer", true);
      error_table.set_precision("error_tracer",
                                simulation_control->get_log_precision());
      error_table.write_text(std::cout);
    }
}

template <int dim>
void
DGTracer<dim>::percolate_time_vectors()
{
  for (unsigned int i = previous_solutions.size() - 1; i > 0; --i)
    {
      previous_solutions[i] = previous_solutions[i - 1];
    }
  previous_solutions[0] = this->present_solution;
}

template <int dim>
void
DGTracer<dim>::finish_time_step()
{
  percolate_time_vectors();
}

template <int dim>
void
DGTracer<dim>::postprocess(bool first_iteration)
{
  if (simulation_parameters.analytical_solution->calculate_error() == true &&
      !first_iteration)
    {
      double tracer_error = calculate_L2_error();

      error_table.add_value("cells",
                            this->triangulation->n_global_active_cells());
      error_table.add_value("error_tracer", tracer_error);

      if (simulation_parameters.analytical_solution->verbosity ==
          Parameters::Verbosity::verbose)
        {
          this->pcout << "L2 error tracer : " << tracer_error << std::endl;
        }
    }

  if (simulation_parameters.post_processing.calculate_tracer_statistics)
    {
      calculate_tracer_statistics();
      if (simulation_control->get_step_number() %
            this->simulation_parameters.post_processing.output_frequency ==
          0)
        this->write_tracer_statistics();
    }
}

template <int dim>
void
DGTracer<dim>::calculate_tracer_statistics()
{
  auto mpi_communicator = triangulation->get_communicator();

  FEValues<dim> fe_values(*mapping,
                          *fe,
                          *cell_quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell =
    fe->dofs_per_cell; // This gives you dofs per cell

  std::vector<types::global_dof_index> local_dof_indices(
    dofs_per_cell); //  Local connectivity

  const unsigned int  n_q_points = cell_quadrature->size();
  std::vector<double> q_tracer_values(n_q_points);

  double volume_integral  = 0;
  double max_tracer_value = DBL_MIN;
  double min_tracer_value = DBL_MAX;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values.get_function_values(present_solution, q_tracer_values);

          // Retrieve the effective "connectivity matrix" for this element
          cell->get_dof_indices(local_dof_indices);

          for (unsigned int q = 0; q < n_q_points; q++)
            {
              volume_integral += q_tracer_values[q] * fe_values.JxW(q);
              max_tracer_value = std::max(q_tracer_values[q], max_tracer_value);
              min_tracer_value = std::min(q_tracer_values[q], min_tracer_value);
            }
        }
    }
  volume_integral      = Utilities::MPI::sum(volume_integral, mpi_communicator);
  double global_volume = GridTools::volume(*triangulation, *mapping);
  double tracer_average = volume_integral / global_volume;

  double variance_integral = 0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values.get_function_values(present_solution, q_tracer_values);

          // Retrieve the effective "connectivity matrix" for this element
          cell->get_dof_indices(local_dof_indices);

          for (unsigned int q = 0; q < n_q_points; q++)
            {
              variance_integral += (q_tracer_values[q] - tracer_average) *
                                   (q_tracer_values[q] - tracer_average) *
                                   fe_values.JxW(q);
            }
        }
    }

  variance_integral = Utilities::MPI::sum(variance_integral, mpi_communicator);
  double tracer_variance      = variance_integral / global_volume;
  double tracer_std_deviation = std::sqrt(tracer_variance);

  this->pcout << "Tracer statistics : " << std::endl;
  this->pcout << "\t     Min : " << min_tracer_value << std::endl;
  this->pcout << "\t     Max : " << max_tracer_value << std::endl;
  this->pcout << "\t Average : " << tracer_average << std::endl;
  this->pcout << "\t Std-Dev : " << tracer_std_deviation << std::endl;

  statistics_table.add_value("time", simulation_control->get_current_time());
  statistics_table.add_value("min", min_tracer_value);
  statistics_table.add_value("max", max_tracer_value);
  statistics_table.add_value("average", tracer_average);
  statistics_table.add_value("std-dev", tracer_std_deviation);
}

template <int dim>
void
DGTracer<dim>::write_tracer_statistics()
{
  auto mpi_communicator = triangulation->get_communicator();

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::string filename =
        simulation_parameters.post_processing.tracer_output_name + ".dat";
      std::ofstream output(filename.c_str());

      statistics_table.write_text(output);
    }
}

template <int dim>
void
DGTracer<dim>::pre_mesh_adaptation()
{
  solution_transfer.prepare_for_coarsening_and_refinement(present_solution);

  for (unsigned int i = 0; i < previous_solutions.size(); ++i)
    {
      previous_solutions_transfer[i].prepare_for_coarsening_and_refinement(
        previous_solutions[i]);
    }
}

template <int dim>
void
DGTracer<dim>::post_mesh_adaptation()
{
  auto mpi_communicator = triangulation->get_communicator();

  // Set up the vectors for the transfer
  TrilinosWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);

  // Interpolate the solution at time and previous time
  solution_transfer.interpolate(tmp);

  // Distribute constraints
  nonzero_constraints.distribute(tmp);

  // Fix on the new mesh
  present_solution = tmp;

  // Transfer previous solutions
  for (unsigned int i = 0; i < previous_solutions.size(); ++i)
    {
      TrilinosWrappers::MPI::Vector tmp_previous_solution(locally_owned_dofs,
                                                          mpi_communicator);
      previous_solutions_transfer[i].interpolate(tmp_previous_solution);
      nonzero_constraints.distribute(tmp_previous_solution);
      previous_solutions[i] = tmp_previous_solution;
    }
}

template <int dim>
void
DGTracer<dim>::write_checkpoint()
{
  std::vector<const TrilinosWrappers::MPI::Vector *> sol_set_transfer;

  sol_set_transfer.push_back(&present_solution);
  for (unsigned int i = 0; i < previous_solutions.size(); ++i)
    {
      sol_set_transfer.push_back(&previous_solutions[i]);
    }
  solution_transfer.prepare_for_serialization(sol_set_transfer);
}

template <int dim>
void
DGTracer<dim>::read_checkpoint()
{
  auto mpi_communicator = triangulation->get_communicator();
  this->pcout << "Reading tracer checkpoint" << std::endl;

  std::vector<TrilinosWrappers::MPI::Vector *> input_vectors(
    1 + previous_solutions.size());
  TrilinosWrappers::MPI::Vector distributed_system(locally_owned_dofs,
                                                   mpi_communicator);
  input_vectors[0] = &distributed_system;


  std::vector<TrilinosWrappers::MPI::Vector> distributed_previous_solutions;
  distributed_previous_solutions.reserve(previous_solutions.size());
  for (unsigned int i = 0; i < previous_solutions.size(); ++i)
    {
      distributed_previous_solutions.emplace_back(
        TrilinosWrappers::MPI::Vector(locally_owned_dofs, mpi_communicator));
      input_vectors[i + 1] = &distributed_previous_solutions[i];
    }

  solution_transfer.deserialize(input_vectors);

  present_solution = distributed_system;
  for (unsigned int i = 0; i < previous_solutions.size(); ++i)
    {
      previous_solutions[i] = distributed_previous_solutions[i];
    }
}


template <int dim>
void
DGTracer<dim>::setup_dofs()
{
  dof_handler.distribute_dofs(*fe);
  DoFRenumbering::Cuthill_McKee(this->dof_handler);

  auto mpi_communicator = triangulation->get_communicator();


  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  present_solution.reinit(locally_owned_dofs,
                          locally_relevant_dofs,
                          mpi_communicator);

  // Previous solutions for transient schemes
  for (auto &solution : this->previous_solutions)
    {
      solution.reinit(locally_owned_dofs,
                      locally_relevant_dofs,
                      mpi_communicator);
    }

  system_rhs.reinit(locally_owned_dofs, mpi_communicator);

  newton_update.reinit(locally_owned_dofs, mpi_communicator);

  local_evaluation_point.reinit(this->locally_owned_dofs, mpi_communicator);

  {
    nonzero_constraints.clear();
    DoFTools::make_hanging_node_constraints(this->dof_handler,
                                            nonzero_constraints);

    for (unsigned int i_bc = 0;
         i_bc < this->simulation_parameters.boundary_conditions_tracer.size;
         ++i_bc)
      {
        // Dirichlet condition : imposed temperature at i_bc
        if (this->simulation_parameters.boundary_conditions_tracer.type[i_bc] ==
            BoundaryConditions::BoundaryType::tracer_dirichlet)
          {
            VectorTools::interpolate_boundary_values(
              this->dof_handler,
              this->simulation_parameters.boundary_conditions_tracer.id[i_bc],
              *this->simulation_parameters.boundary_conditions_tracer
                 .tracer[i_bc],
              nonzero_constraints);
          }
      }
  }
  nonzero_constraints.close();

  // Boundary conditions for Newton correction
  {
    zero_constraints.clear();
    DoFTools::make_hanging_node_constraints(this->dof_handler,
                                            zero_constraints);

    for (unsigned int i_bc = 0;
         i_bc < this->simulation_parameters.boundary_conditions_tracer.size;
         ++i_bc)
      {
        if (this->simulation_parameters.boundary_conditions_tracer.type[i_bc] ==
            BoundaryConditions::BoundaryType::tracer_dirichlet)
          {
            VectorTools::interpolate_boundary_values(
              this->dof_handler,
              this->simulation_parameters.boundary_conditions_tracer.id[i_bc],
              Functions::ZeroFunction<dim>(),
              zero_constraints);
          }
      }
  }
  zero_constraints.close();

  // Sparse matrices initialization
  DynamicSparsityPattern dsp(this->dof_handler.n_dofs());
  DoFTools::make_flux_sparsity_pattern(this->dof_handler,
                                       dsp,
                                       nonzero_constraints,
                                       /*keep_constrained_dofs = */ true);

  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             mpi_communicator,
                                             locally_relevant_dofs);
  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       mpi_communicator);

  this->pcout << "   Number of tracer degrees of freedom: "
              << dof_handler.n_dofs() << std::endl;

  // Provide the tracer dof_handler and present solution pointers to the
  // multiphysics interface
  multiphysics->set_dof_handler(PhysicsID::dg_tracer, &this->dof_handler);
  multiphysics->set_solution(PhysicsID::dg_tracer, &this->present_solution);
}

template <int dim>
void
DGTracer<dim>::set_initial_conditions()
{
  VectorTools::interpolate(*mapping,
                           dof_handler,
                           simulation_parameters.initial_condition->tracer,
                           newton_update);
  nonzero_constraints.distribute(newton_update);
  present_solution = newton_update;
  finish_time_step();
}

template <int dim>
void
DGTracer<dim>::solve_linear_system(const bool initial_step,
                                   const bool /*renewed_matrix*/)
{
  auto mpi_communicator = triangulation->get_communicator();

  const AffineConstraints<double> &constraints_used =
    initial_step ? nonzero_constraints : this->zero_constraints;

  const double absolute_residual =
    simulation_parameters.linear_solver.minimum_residual;
  const double relative_residual =
    simulation_parameters.linear_solver.relative_residual;

  const double linear_solver_tolerance =
    std::max(relative_residual * system_rhs.l2_norm(), absolute_residual);

  if (this->simulation_parameters.linear_solver.verbosity !=
      Parameters::Verbosity::quiet)
    {
      this->pcout << "  -Tolerance of iterative solver is : "
                  << linear_solver_tolerance << std::endl;
    }

  const double ilu_fill = simulation_parameters.linear_solver.ilu_precond_fill;
  const double ilu_atol = simulation_parameters.linear_solver.ilu_precond_atol;
  const double ilu_rtol = simulation_parameters.linear_solver.ilu_precond_rtol;
  TrilinosWrappers::PreconditionILU::AdditionalData preconditionerOptions(
    ilu_fill, ilu_atol, ilu_rtol, 0);

  TrilinosWrappers::PreconditionILU ilu_preconditioner;

  ilu_preconditioner.initialize(system_matrix, preconditionerOptions);

  TrilinosWrappers::MPI::Vector completely_distributed_solution(
    locally_owned_dofs, mpi_communicator);

  SolverControl solver_control(
    simulation_parameters.linear_solver.max_iterations,
    linear_solver_tolerance,
    true,
    true);

  TrilinosWrappers::SolverGMRES::AdditionalData solver_parameters(
    false, simulation_parameters.linear_solver.max_krylov_vectors);


  TrilinosWrappers::SolverGMRES solver(solver_control, solver_parameters);


  solver.solve(system_matrix,
               completely_distributed_solution,
               system_rhs,
               ilu_preconditioner);

  if (simulation_parameters.linear_solver.verbosity !=
      Parameters::Verbosity::quiet)
    {
      this->pcout << "  -Iterative solver took : " << solver_control.last_step()
                  << " steps " << std::endl;
    }

  constraints_used.distribute(completely_distributed_solution);
  newton_update = completely_distributed_solution;
}


template class DGTracer<2>;

template class DGTracer<3>;
