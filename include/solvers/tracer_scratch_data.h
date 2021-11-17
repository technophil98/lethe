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
 * Scratch data for the tracer auxiliary physics
 */

#include <core/bdf.h>
#include <core/multiphysics.h>
#include <core/sdirk.h>

#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/numerics/vector_tools.h>


#ifndef lethe_tracer_scratch_data_h
#  define lethe_tracer_scratch_data_h

using namespace dealii;


/**
 * @brief TracerScratchData class
 * stores the information required by the assembly procedure
 * for a Tracer advection-diffusion equation. Consequently, this class
 *calculates the tracer (values, gradients, laplacians) and the shape function
 * (values, gradients, laplacians) at all the gauss points for all degrees
 * of freedom and stores it into arrays. Additionnaly, the use can request
 * that this class gathers additional fields for physics which are coupled
 * to the Tracer equation, such as the velocity which is required. This class
 * serves as a seperation between the evaluation at the gauss point of the
 * variables of interest and their use in the assembly, which is carried out
 * by the assembler functions. For more information on this design, the reader
 * can consult deal.II step-9
 * "https://www.dealii.org/current/doxygen/deal.II/step_9.html". In this latter
 * example, the scratch is a struct instead of a templated class because of the
 * simplicity of step-9.
 *
 * @tparam dim An integer that denotes the dimension of the space in which
 * the flow is solved
 *  @ingroup solvers
 **/
template <int dim>
class TracerScratchData
{
public:
  /**
   * @brief Constructor. The constructor creates the fe_values that will be used
   * to fill the member variables of the scratch. It also allocated the
   * necessary memory for all member variables. However, it does not do any
   * evalution, since this needs to be done at the cell level.
   *
   * @param fe The FESystem used to solve the Navier-Stokes equations
   *
   * @param quadrature The quadrature to use for the assembly
   *
   * @param mapping The mapping of the domain in which the Navier-Stokes equations are solved
   *
   */
  TracerScratchData(const FiniteElement<dim> &fe_tracer,
                    const Quadrature<dim> &   quadrature,
                    const Mapping<dim> &      mapping,
                    const FiniteElement<dim> &fe_navier_stokes,
                    const UpdateFlags &       update_flags = update_values |
                                                      update_quadrature_points |
                                                      update_JxW_values |
                                                      update_gradients |
                                                      update_hessians)
    : fe_values_tracer(mapping, fe_tracer, quadrature, update_flags)
    , fe_values_navier_stokes(mapping,
                              fe_navier_stokes,
                              quadrature,
                              update_values)
  {
    allocate();
  }

  /**
   * @brief Copy Constructor. Same as the main constructor.
   *  This constructor only uses the other scratch to build the FeValues, it
   * does not copy the content of the other scratch into itself since, by
   * definition of the WorkStream mechanism it is assumed that the content of
   * the scratch will be reset on a cell basis.
   *
   * @param fe The FESystem used to solve the Navier-Stokes equations
   *
   * @param quadrature The quadrature to use for the assembly
   *
   * @param mapping The mapping of the domain in which the Navier-Stokes equations are solved
   */
  TracerScratchData(const TracerScratchData<dim> &sd)
    : fe_values_tracer(sd.fe_values_tracer.get_mapping(),
                       sd.fe_values_tracer.get_fe(),
                       sd.fe_values_tracer.get_quadrature(),
                       update_values | update_quadrature_points |
                         update_JxW_values | update_gradients | update_hessians)
    , fe_values_navier_stokes(sd.fe_values_navier_stokes.get_mapping(),
                              sd.fe_values_navier_stokes.get_fe(),
                              sd.fe_values_navier_stokes.get_quadrature(),
                              update_values)
  {
    allocate();
  }


  /** @brief Allocates the memory for the scratch
   *
   * This function allocates the necessary memory for all members of the scratch
   *
   */
  void
  allocate();

  /** @brief Reinitialize the content of the scratch
   *
   * Using the FeValues and the content ofthe solutions, previous solutions and
   * solutions stages, fills all of the class member of the scratch
   *
   * @param cell The cell over which the assembly is being carried.
   * This cell must be compatible with the fe which is used to fill the FeValues
   *
   * @param current_solution The present value of the solution for [u,p]
   *
   * @param previous_solutions The solutions at the previous time steps
   *
   * @param solution_stages The solution at the intermediary stages (for SDIRK methods)
   *
   * @param source_function The function describing the tracer source term
   *
   */

  template <typename VectorType>
  void
  reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
         const VectorType &                                    current_solution,
         const std::vector<VectorType> &previous_solutions,
         const std::vector<VectorType> &solution_stages,
         Function<dim> *                source_function)
  {
    this->fe_values_tracer.reinit(cell);

    quadrature_points = this->fe_values_tracer.get_quadrature_points();
    auto &fe_tracer   = this->fe_values_tracer.get_fe();

    source_function->value_list(quadrature_points, source);

    if (dim == 2)
      this->cell_size =
        std::sqrt(4. * cell->measure() / M_PI) / fe_tracer.degree;
    else if (dim == 3)
      this->cell_size =
        pow(6 * cell->measure() / M_PI, 1. / 3.) / fe_tracer.degree;

    // Gather tracer (values, gradient and laplacian)
    this->fe_values_tracer.get_function_values(current_solution,
                                               this->tracer_values);
    this->fe_values_tracer.get_function_gradients(current_solution,
                                                  this->tracer_gradients);
    this->fe_values_tracer.get_function_laplacians(current_solution,
                                                   this->tracer_laplacians);

    // Gather previous tracer values
    for (unsigned int p = 0; p < previous_solutions.size(); ++p)
      {
        this->fe_values_tracer.get_function_values(previous_solutions[p],
                                                   previous_tracer_values[p]);
      }

    // Gather tracer stages
    for (unsigned int s = 0; s < solution_stages.size(); ++s)
      {
        this->fe_values_tracer.get_function_values(solution_stages[s],
                                                   stages_tracer_values[s]);
      }


    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        this->JxW[q] = this->fe_values_tracer.JxW(q);

        for (unsigned int k = 0; k < n_dofs; ++k)
          {
            // Shape function
            this->phi[q][k]      = this->fe_values_tracer.shape_value(k, q);
            this->grad_phi[q][k] = this->fe_values_tracer.shape_grad(k, q);
            this->hess_phi[q][k] = this->fe_values_tracer.shape_hessian(k, q);
            this->laplacian_phi[q][k] = trace(this->hess_phi[q][k]);
          }
      }
  }

  template <typename VectorType>
  void
  reinit_velocity(const typename DoFHandler<dim>::active_cell_iterator &cell,
                  const VectorType &current_solution)
  {
    this->fe_values_navier_stokes.reinit(cell);

    this->fe_values_navier_stokes[velocities].get_function_values(
      current_solution, velocity_values);
  }


  // FEValues for the Tracer problem
  FEValues<dim> fe_values_tracer;
  unsigned int  n_dofs;
  unsigned int  n_q_points;
  double        cell_size;

  // Quadrature
  std::vector<double>     JxW;
  std::vector<Point<dim>> quadrature_points;

  // Tracer values
  std::vector<double>              tracer_values;
  std::vector<Tensor<1, dim>>      tracer_gradients;
  std::vector<double>              tracer_laplacians;
  std::vector<std::vector<double>> previous_tracer_values;
  std::vector<std::vector<double>> stages_tracer_values;

  // Source term
  std::vector<double> source;

  // Shape functions
  std::vector<std::vector<double>>         phi;
  std::vector<std::vector<Tensor<2, dim>>> hess_phi;
  std::vector<std::vector<double>>         laplacian_phi;
  std::vector<std::vector<Tensor<1, dim>>> grad_phi;


  /**
   * Scratch component for the Navier-Stokes component
   */
  FEValuesExtractors::Vector velocities;
  // This FEValues must mandatorily be instantiated for the velocity
  FEValues<dim>               fe_values_navier_stokes;
  std::vector<Tensor<1, dim>> velocity_values;
};


/**
 * @brief DGTracerScratchData class
 * stores the information required by the assembly procedure
 * for a Tracer advection-diffusion equation with DG formulation. Consequently,
 *this class calculates the tracer (values, gradients, laplacians) and the shape
 *function (values, gradients, laplacians) at all the gauss points for all
 *degrees of freedom and stores it into arrays. Additionnally, the user can
 *request that this class gathers additional fields for physics which are
 *coupled to the Tracer equation, such as the velocity which is required. This
 *class serves as a separation between the evaluation at the gauss point of the
 * variables of interest and their use in the assembly, which is carried out
 * by the assembler functions. For more information on this design, the reader
 * can consult deal.II step-9 and step-12
 * "https://www.dealii.org/current/doxygen/deal.II/step_9.html".
 * "https://www.dealii.org/current/doxygen/deal.II/step_12.html".
 * In these examples, the scratch is a struct instead of a templated class
 *because of their simplicity
 *
 * @tparam dim An integer that denotes the dimension of the space in which
 * the flow is solved
 *  @ingroup solvers
 **/
template <int dim>
class DGTracerScratchData
{
public:
  /**
   * @brief Constructor. The constructor creates the fe_values and fe_interface_values
   * that will be used to fill the member variables of the scratch. It also
   * allocated the necessary memory for all member variables. However, it does
   * not do any evalution, since this needs to be done at the cell level.
   *
   * @param fe The FESystem used to solve the Navier-Stokes equations
   *
   * @param quadrature The quadrature to use for the assembly
   *
   * @param mapping The mapping of the domain in which the Navier-Stokes equations are solved
   *
   */
  DGTracerScratchData(
    const FiniteElement<dim> & fe_tracer,
    const Quadrature<dim> &    cell_quadrature,
    const Quadrature<dim - 1> &face_quadrature,
    const Mapping<dim> &       mapping,
    const FiniteElement<dim> & fe_navier_stokes,
    const UpdateFlags &update_flags = update_values | update_quadrature_points |
                                      update_JxW_values | update_gradients |
                                      update_hessians,
    const UpdateFlags &interface_update_flags =
      update_values | update_quadrature_points | update_JxW_values |
      update_gradients | update_hessians | update_normal_vectors)
    : fe_values_tracer(mapping, fe_tracer, cell_quadrature, update_flags)
    , fe_interface_values_tracer(mapping,
                                 fe_tracer,
                                 face_quadrature,
                                 interface_update_flags)
    , fe_values_navier_stokes(mapping,
                              fe_navier_stokes,
                              cell_quadrature,
                              update_values)
    , fe_interface_values_navier_stokes(mapping,
                                        fe_navier_stokes,
                                        face_quadrature,
                                        interface_update_flags)
    , fe_tracer_degree(fe_tracer.degree)
  {
    allocate_cell();
  }

  /**
   * @brief Copy Constructor. Same as the main constructor.
   *  This constructor only uses the other scratch to build the FeValues, it
   * does not copy the content of the other scratch into itself since, by
   * definition of the WorkStream mechanism it is assumed that the content of
   * the scratch will be reset on a cell basis.
   *
   * @param fe The FESystem used to solve the Navier-Stokes equations
   *
   * @param quadrature The quadrature to use for the assembly
   *
   * @param mapping The mapping of the domain in which the Navier-Stokes equations are solved
   */
  DGTracerScratchData(const DGTracerScratchData<dim> &sd)
    : fe_values_tracer(sd.fe_values_tracer.get_mapping(),
                       sd.fe_values_tracer.get_fe(),
                       sd.fe_values_tracer.get_quadrature(),
                       sd.fe_values_tracer.get_update_flags())
    , fe_interface_values_tracer(
        sd.fe_interface_values_tracer.get_mapping(),
        sd.fe_interface_values_tracer.get_fe(),
        sd.fe_interface_values_tracer.get_quadrature(),
        sd.fe_interface_values_tracer.get_update_flags())
    , fe_values_navier_stokes(sd.fe_values_navier_stokes.get_mapping(),
                              sd.fe_values_navier_stokes.get_fe(),
                              sd.fe_values_navier_stokes.get_quadrature(),
                              update_values)
    , fe_interface_values_navier_stokes(
        sd.fe_interface_values_navier_stokes.get_mapping(),
        sd.fe_interface_values_navier_stokes.get_fe(),
        sd.fe_interface_values_navier_stokes.get_quadrature(),
        sd.fe_interface_values_navier_stokes.get_update_flags())
    , fe_tracer_degree(sd.fe_values_tracer.get_fe().degree)
  {
    allocate_cell();
  }


  /** @brief Allocates the memory for the scratch
   *
   * This function allocates the necessary memory for all cell members of the
   * scratch
   *
   */
  void
  allocate_cell();

  /** @brief Allocates the memory for the scratch
   *
   * This function allocates the necessary memory for all cell members of the
   * scratch
   *
   */
  void
  allocate_face(const unsigned int n_dofs, const unsigned int n_q_points);

  /** @brief Allocates the memory for the scratch
   *
   * This function allocates the necessary memory for all cell members of the
   * scratch
   *
   */
  void
  allocate_boundary(const unsigned int n_dofs, const unsigned int n_q_points);

  /** @brief Reinitialize the content of the scratch
   *
   * Using the FeValues and the content ofthe solutions, previous solutions and
   * solutions stages, fills all of the class member of the scratch
   *
   * @param cell The cell over which the assembly is being carried.
   * This cell must be compatible with the fe which is used to fill the FeValues
   *
   * @param current_solution The present value of the solution for [u,p]
   *
   * @param previous_solutions The solutions at the previous time steps
   *
   * @param solution_stages The solution at the intermediary stages (for SDIRK methods)
   *
   * @param source_function The function describing the tracer source term
   *
   */
  template <typename VectorType>
  void
  reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
         const VectorType &                                    current_solution,
         const std::vector<VectorType> &previous_solutions,
         const std::vector<VectorType> &solution_stages,
         Function<dim> *                source_function)
  {
    this->fe_values_tracer.reinit(cell);

    cell_quadrature_points = this->fe_values_tracer.get_quadrature_points();
    auto &fe_tracer        = this->fe_values_tracer.get_fe();

    source_function->value_list(cell_quadrature_points, cell_source);

    // Gather tracer (values, gradient and laplacian)
    this->fe_values_tracer.get_function_values(current_solution,
                                               this->cell_tracer_values);
    this->fe_values_tracer.get_function_gradients(current_solution,
                                                  this->cell_tracer_gradients);
    this->fe_values_tracer.get_function_laplacians(
      current_solution, this->cell_tracer_laplacians);

    // Gather previous tracer values
    for (unsigned int p = 0; p < previous_solutions.size(); ++p)
      {
        this->fe_values_tracer.get_function_values(
          previous_solutions[p], cell_previous_tracer_values[p]);
      }

    // Gather tracer stages
    for (unsigned int s = 0; s < solution_stages.size(); ++s)
      {
        this->fe_values_tracer.get_function_values(
          solution_stages[s], cell_stages_tracer_values[s]);
      }

    for (unsigned int q = 0; q < cell_n_q_points; ++q)
      {
        this->cell_JxW[q] = this->fe_values_tracer.JxW(q);

        for (unsigned int k = 0; k < cell_n_dofs; ++k)
          {
            // Shape function
            this->cell_phi[q][k] = this->fe_values_tracer.shape_value(k, q);
            this->cell_grad_phi[q][k] = this->fe_values_tracer.shape_grad(k, q);
            this->cell_hess_phi[q][k] =
              this->fe_values_tracer.shape_hessian(k, q);
            this->cell_laplacian_phi[q][k] = trace(this->cell_hess_phi[q][k]);
          }
      }
  }

  /** @brief Reinitialize the content of the scratch
   *
   * Using the FeValues and the content ofthe solutions, previous solutions and
   * solutions stages, fills all of the class member of the scratch
   *
   * @param cell The cell over which the assembly is being carried.
   * This cell must be compatible with the fe which is used to fill the FeValues
   *
   * @param current_solution The present value of the solution for [u,p]
   *
   * @param previous_solutions The solutions at the previous time steps
   *
   * @param solution_stages The solution at the intermediary stages (for SDIRK methods)
   *
   * @param source_function The function describing the tracer source term
   *
   */
  template <typename VectorType>
  void
  reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
         const unsigned int &                                  f,
         const unsigned int &                                  sf,
         const typename DoFHandler<dim>::active_cell_iterator &ncell,
         const unsigned int &                                  nf,
         const unsigned int &                                  nsf,
         const VectorType &                                    current_solution,
         const std::vector<VectorType> &previous_solutions,
         const std::vector<VectorType> &solution_stages,
         Function<dim> *                source_function)
  {
    this->fe_interface_values_tracer.reinit(cell, f, sf, ncell, nf, nsf);

    if (!this->face_allocated |
        this->fe_interface_values_tracer.n_current_interface_dofs() !=
          this->face_n_dofs |
        this->fe_interface_values_tracer.get_quadrature_points().size() !=
          this->face_n_q_points)
      {
        allocate_face(
          this->fe_interface_values_tracer.n_current_interface_dofs(),
          fe_interface_values_tracer.get_quadrature().size());
        this->face_allocated = true;
      }

    this->face_quadrature_points =
      this->fe_interface_values_tracer.get_quadrature_points();
    auto &fe_tracer = this->fe_values_tracer.get_fe();

    source_function->value_list(face_quadrature_points, face_source);

    face_size          = cell->measure() / cell->face(f)->measure();
    face_size_neighbor = ncell->measure() / ncell->face(nf)->measure();

    this->face_normals = this->fe_interface_values_tracer.get_normal_vectors();

    for (unsigned int q = 0; q < face_n_q_points; ++q)
      {
        this->face_JxW[q] = this->fe_values_tracer.JxW(q);

        for (unsigned int k = 0; k < face_n_dofs; ++k)
          {
            // Shape function
            this->face_phi_inflow[q][k] =
              this->fe_interface_values_tracer.shape_value(false, k, q);
            this->face_phi_outflow[q][k] =
              this->fe_interface_values_tracer.shape_value(true, k, q);

            this->face_tracer_jump[q][k] =
              this->fe_interface_values_tracer.jump(k, q);
            this->face_tracer_average_gradients[q][k] =
              this->fe_interface_values_tracer.average_gradient(k, q);
          }
      }
  }

  /** @brief Reinitialize the content of the scratch
   *
   * Using the FeValues and the content ofthe solutions, previous solutions and
   * solutions stages, fills all of the class member of the scratch
   *
   * @param cell The cell over which the assembly is being carried.
   * This cell must be compatible with the fe which is used to fill the FeValues
   *
   * @param current_solution The present value of the solution for [u,p]
   *
   * @param previous_solutions The solutions at the previous time steps
   *
   * @param solution_stages The solution at the intermediary stages (for SDIRK methods)
   *
   * @param source_function The function describing the tracer source term
   *
   */
  template <typename VectorType>
  void
  reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
         const unsigned int &                                  face_no,
         const VectorType &                                    current_solution,
         const std::vector<VectorType> &previous_solutions,
         const std::vector<VectorType> &solution_stages,
         Function<dim> *                source_function)
  {
    this->fe_interface_values_tracer.reinit(cell, face_no);
    const FEFaceValuesBase<dim> &fe_face_values_tracer =
      this->fe_interface_values_tracer.get_fe_face_values(0);
    this->boundary_quadrature_points =
      fe_face_values_tracer.get_quadrature_points();

    if (!this->boundary_allocated |
        fe_face_values_tracer.get_fe().n_dofs_per_cell() !=
          this->boundary_n_dofs |
        fe_face_values_tracer.get_quadrature_points().size() !=
          this->boundary_n_q_points)
      {
        allocate_boundary(fe_face_values_tracer.get_fe().n_dofs_per_cell(),
                          this->boundary_quadrature_points.size());
        this->boundary_allocated = true;
      }

    this->boundary_JxW     = fe_face_values_tracer.get_JxW_values();
    this->boundary_normals = fe_face_values_tracer.get_normal_vectors();

    boundary_size = cell->measure() / cell->face(face_no)->measure();

    auto &fe_tracer = this->fe_values_tracer.get_fe();

    // Forcing term array
    this->boundary_source = std::vector<double>(boundary_n_q_points);
    source_function->value_list(boundary_quadrature_points, boundary_source);

    // Dirichlet boundary condition array
    this->boundary_dirichlet = std::vector<double>(boundary_n_q_points);


    // Initialize vectors tracer
    this->boundary_tracer_values = std::vector<double>(boundary_n_q_points);
    this->boundary_tracer_gradients =
      std::vector<Tensor<1, dim>>(boundary_n_q_points);
    this->boundary_tracer_laplacians = std::vector<double>(boundary_n_q_points);

    this->fe_interface_values_tracer.get_fe_face_values(0).get_function_values(
      current_solution, this->boundary_tracer_values);
    this->fe_interface_values_tracer.get_fe_face_values(0)
      .get_function_gradients(current_solution,
                              this->boundary_tracer_gradients);
    this->fe_interface_values_tracer.get_fe_face_values(0)
      .get_function_laplacians(current_solution,
                               this->boundary_tracer_laplacians);

    for (unsigned int q = 0; q < boundary_n_q_points; ++q)
      {
        this->boundary_JxW[q] = this->fe_values_tracer.JxW(q);

        for (unsigned int k = 0; k < boundary_n_dofs; ++k)
          {
            // Shape function
            this->boundary_phi[q][k] = fe_face_values_tracer.shape_value(k, q);

            this->boundary_grad_phi[q][k] =
              fe_face_values_tracer.shape_grad(k, q);
          }
      }
  }

  void
  compute_dirichlet_values(const unsigned int i_bc,
                           Function<dim> &    dirichlet_function)
  {
    // Dirichlet boundary condition array
    this->boundary_dirichlet = std::vector<double>(this->boundary_n_q_points);
    dirichlet_function.value_list(boundary_quadrature_points,
                                  boundary_dirichlet);
  }

  template <typename VectorType>
  void
  reinit_velocity(const typename DoFHandler<dim>::active_cell_iterator &cell,
                  const VectorType &current_solution)
  {
    this->fe_values_navier_stokes.reinit(cell);

    this->fe_values_navier_stokes[velocities].get_function_values(
      current_solution, cell_velocity_values);
  }

  template <typename VectorType>
  void
  reinit_velocity(const typename DoFHandler<dim>::active_cell_iterator &cell,
                  const unsigned int &                                  face_no,
                  const VectorType &current_solution)
  {
    this->fe_interface_values_navier_stokes.reinit(cell, face_no);

    const FEFaceValuesBase<dim> &fe_face_values_navier_stokes =
      this->fe_interface_values_navier_stokes.get_fe_face_values(0);
    fe_face_values_navier_stokes[velocities].get_function_values(
      current_solution, face_velocity_values);
  }

  // FEValues for the Tracer problem
  FEValues<dim>          fe_values_tracer;
  FEInterfaceValues<dim> fe_interface_values_tracer;

  /**
   * Scratch component for the Navier-Stokes component
   */
  // This FEValues must mandatorily be instantiated for the velocity
  FEValues<dim>               fe_values_navier_stokes;
  FEInterfaceValues<dim>      fe_interface_values_navier_stokes;
  FEValuesExtractors::Vector  velocities;
  std::vector<Tensor<1, dim>> cell_velocity_values;
  std::vector<Tensor<1, dim>> face_velocity_values;
  std::vector<Tensor<1, dim>> boundary_velocity_values;

  unsigned int fe_tracer_degree;
  unsigned int cell_n_dofs;
  unsigned int cell_n_q_points;
  unsigned int face_n_dofs;
  unsigned int face_n_q_points;
  double       face_size;
  double       face_size_neighbor;
  unsigned int boundary_n_dofs;
  unsigned int boundary_n_q_points;
  double       boundary_size;

  // Quadrature
  std::vector<double>     cell_JxW;
  std::vector<Point<dim>> cell_quadrature_points;
  std::vector<double>     face_JxW;
  std::vector<Point<dim>> face_quadrature_points;
  std::vector<double>     boundary_JxW;
  std::vector<Point<dim>> boundary_quadrature_points;

  // Tracer values
  std::vector<double>              cell_tracer_values;
  std::vector<Tensor<1, dim>>      cell_tracer_gradients;
  std::vector<double>              cell_tracer_laplacians;
  std::vector<std::vector<double>> cell_previous_tracer_values;
  std::vector<std::vector<double>> cell_stages_tracer_values;

  std::vector<std::vector<double>>         face_tracer_jump;
  std::vector<std::vector<Tensor<1, dim>>> face_tracer_average_gradients;

  std::vector<double>         boundary_tracer_values;
  std::vector<Tensor<1, dim>> boundary_tracer_gradients;
  std::vector<double>         boundary_tracer_laplacians;

  // Source term
  std::vector<double> cell_source;
  std::vector<double> face_source;
  std::vector<double> boundary_source;

  // Dirichlet condition
  std::vector<double> boundary_dirichlet;

  // Shape functions
  std::vector<std::vector<double>>         cell_phi;
  std::vector<std::vector<Tensor<2, dim>>> cell_hess_phi;
  std::vector<std::vector<double>>         cell_laplacian_phi;
  std::vector<std::vector<Tensor<1, dim>>> cell_grad_phi;
  std::vector<std::vector<double>>         face_phi_outflow;
  std::vector<std::vector<double>>         face_phi_inflow;
  std::vector<std::vector<double>>         boundary_phi;
  std::vector<std::vector<Tensor<1, dim>>> boundary_grad_phi;

  std::vector<Tensor<1, dim>> face_normals;
  std::vector<Tensor<1, dim>> boundary_normals;

  bool face_allocated     = false;
  bool boundary_allocated = false;
};

#endif
