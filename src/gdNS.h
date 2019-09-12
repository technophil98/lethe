﻿/* ---------------------------------------------------------------------
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

 *
 * Author: Bruno Blais, Polytechnique Montreal, 2019-
 */

#ifndef LETHE_GDNS_H
#define LETHE_GDNS_H

// Dealii Includes

// Base
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

// Lac
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

// Lac - Trilinos includes
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_vector.h>

// Grid
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

// Dofs
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

// Fe
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

// Numerics
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

// Distributed
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

// Lethe Includes
#include "bdf.h"
#include "boundaryconditions.h"
#include "manifolds.h"
#include "navier_stokes_base.h"
#include "navierstokessolverparameters.h"
#include "parameters.h"
#include "postprocessors.h"
#include "pvdhandler.h"
#include "simulationcontrol.h"

// Std
#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * A solver class for the Steady-Sate  Navier-Stokes equation using Grad-Div
 * stabilization
 *
 * @tparam dim An integer that denotes the dimension of the space in which
 * the flow is solved
 *
 * @ingroup solvers
 * @author Bruno Blais, 2019
 */

template <int dim>
class GDNavierStokesSolver : public NavierStokesBase<dim>
{
public:
  GDNavierStokesSolver(NavierStokesSolverParameters<dim> &nsparam,
                       const unsigned int                 degreeVelocity,
                       const unsigned int                 degreePressure);
  ~GDNavierStokesSolver();

  void
  solve();

protected:
  void
  read_mesh();
  void
  create_manifolds();
  void
  refine_mesh();
  void
  setup_dofs();
  double
  calculate_L2_error();
  double
  calculate_average_KE();
  double
  calculate_average_enstrophy();
  void
  set_initial_condition(Parameters::InitialConditionType initial_condition_type,
                        bool                             restart = false);
  void
  postprocess(bool firstIter);
  void
  finish_simulation();

  void
  finish_time_step();
  void
  set_solution_vector(double value);
  void
  set_periodicity();
  void
  iterate(bool firstIteration);

  void
  make_cube_grid(int refinementLevel);

  Function<dim> *exact_solution;
  Function<dim> *forcing_function;

  // Force analysis
  std::vector<Tensor<1, dim>> forces_;
  std::vector<Tensor<1, 3>>   torques_;

  // Solver parameters
  NavierStokesSolverParameters<dim> nsparam;

  // A local copy of the simulation control is kept since this one constantly
  // modified
  SimulationControl simulationControl;

private:
  template <bool                                              assemble_matrix,
            Parameters::SimulationControl::TimeSteppingMethod scheme>
  void
  assembleGD();

  void
  newton_iteration(const bool is_initial_step);
  void
  assemble_L2_projection();
  void
  set_nodal_values();
  void
  refine_mesh_Kelly();
  void
  refine_mesh_uniform();

  void
  calculate_forces();
  void
  calculate_torques();
  double
  calculate_CFL();

  void
  assemble_system();
  void
  assemble_rhs();

  /**
   * Solver for the L2 Projection linear system
   */

  void
  solve_L2_system(bool   initial_step,
                  double relative_residual,
                  double minimum_residual);

  /**
   * Solver for the NS linear system of equations
   */

  void
  solve_NS_system(bool   initial_step,
                  double relative_residual,
                  double minimum_residual);

  /**
   * Checkpointing writer of the solutions vector of the GD solver
   */
  void
  write_checkpoint();

  /**
   * Checkpointing reader of the solutions vector of the GD solver
   */
  void
  read_checkpoint();

  void
  write_output_forces();
  void
  write_output_torques();

  /**
   * Post-processing as parallel VTU files
   */
  void
  write_output_results(const std::string  folder,
                       const std::string  solutionName,
                       const unsigned int cycle,
                       const double       time);

  /**
   * Members
   */

  std::vector<IndexSet> locally_owned_dofs;
  std::vector<IndexSet> locally_relevant_dofs;

  AffineConstraints<double> zero_constraints;
  AffineConstraints<double> nonzero_constraints;

  TrilinosWrappers::BlockSparsityPattern sparsity_pattern;
  TrilinosWrappers::BlockSparseMatrix    system_matrix;
  TrilinosWrappers::SparseMatrix         pressure_mass_matrix;


  std::vector<types::global_dof_index> dofs_per_block;


  TrilinosWrappers::MPI::BlockVector present_solution;
  TrilinosWrappers::MPI::BlockVector newton_update;
  TrilinosWrappers::MPI::BlockVector system_rhs;
  TrilinosWrappers::MPI::BlockVector evaluation_point;
  TrilinosWrappers::MPI::BlockVector local_evaluation_point;

  TrilinosWrappers::MPI::BlockVector solution_m1;
  TrilinosWrappers::MPI::BlockVector solution_m2;
  TrilinosWrappers::MPI::BlockVector solution_m3;

  double       globalVolume_;
  const double gamma      = 1;
  const bool   SUPG       = true;
  const double GD_u_scale = 1;
  PVDHandler   pvdhandler;

  // Force analysis
  std::vector<TableHandler> forces_tables;
  std::vector<TableHandler> torques_tables;



  // Convergence Analysis
  ConvergenceTable table;
};

template <class PreconditionerMp>
class BlockSchurPreconditioner : public Subscriptor
{
public:
  BlockSchurPreconditioner(double                                     gamma,
                           double                                     viscosity,
                           const TrilinosWrappers::BlockSparseMatrix &S,
                           const TrilinosWrappers::SparseMatrix &     P,
                           const PreconditionerMp &Mppreconditioner,
                           SolverControl &         A_parameters);

  void
  vmult(TrilinosWrappers::MPI::BlockVector &      dst,
        const TrilinosWrappers::MPI::BlockVector &src) const;

private:
  const double                               gamma;
  const double                               viscosity;
  const TrilinosWrappers::BlockSparseMatrix &stokes_matrix;
  const TrilinosWrappers::SparseMatrix &     pressure_mass_matrix;
  TrilinosWrappers::PreconditionILU          amat_preconditioner;
  TrilinosWrappers::PreconditionILU          pmass_preconditioner;

  const PreconditionerMp &                    mp_preconditioner;
  SolverFGMRES<TrilinosWrappers::MPI::Vector> A_inverse;
};

// We can notice that the initialization of the inverse of the matrix at the
// top left corner is completed in the constructor. If so, every application
// of the preconditioner then no longer requires the computation of the
// matrix factors.
template <class PreconditionerMp>
BlockSchurPreconditioner<PreconditionerMp>::BlockSchurPreconditioner(
  double                                     gamma,
  double                                     viscosity,
  const TrilinosWrappers::BlockSparseMatrix &S,
  const TrilinosWrappers::SparseMatrix &     P,
  const PreconditionerMp &                   Mppreconditioner,
  SolverControl &                            A_parameters)
  : gamma(gamma)
  , viscosity(viscosity)
  , stokes_matrix(S)
  , pressure_mass_matrix(P)
  , mp_preconditioner(Mppreconditioner)
  , A_inverse(A_parameters)
{
  TrilinosWrappers::PreconditionILU::AdditionalData amat_preconditionerOptions(
    0, 1e-10, 1, 0);
  amat_preconditioner.initialize(stokes_matrix.block(0, 0),
                                 amat_preconditionerOptions);

  TrilinosWrappers::PreconditionILU::AdditionalData pmass_preconditionerOptions(
    0, 1e-10, 1, 0);
  TrilinosWrappers::PreconditionILU pmass_preconditioner;
  pmass_preconditioner.initialize(pressure_mass_matrix,
                                  pmass_preconditionerOptions);
}

template <class PreconditionerMp>
void
BlockSchurPreconditioner<PreconditionerMp>::vmult(
  TrilinosWrappers::MPI::BlockVector &      dst,
  const TrilinosWrappers::MPI::BlockVector &src) const
{
  //  MPI_Comm   this->mpi_communicator(MPI_COMM_WORLD);
  //  ConditionalOStream pcout(std::cout,
  //  (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0));
  //  TimerOutput computing_timer(this->mpi_communicator,
  //                              pcout,
  //                              TimerOutput::summary,
  //                              TimerOutput::wall_times);

  TrilinosWrappers::MPI::Vector utmp(src.block(0));
  {
    //    computing_timer.enter_section("Pressure");
    SolverControl              solver_control(100000,
                                 std::max(1e-3 * src.block(0).l2_norm(),
                                          1e-12));
    TrilinosWrappers::SolverCG cg(solver_control);

    dst.block(1) = 0.0;
    cg.solve(pressure_mass_matrix,
             dst.block(1),
             src.block(1),
             pmass_preconditioner);
    dst.block(1) *= -(viscosity + gamma);
    //    computing_timer.exit_section("Pressure");
  }

  {
    //    computing_timer.enter_section("Operations");
    stokes_matrix.block(0, 1).vmult(utmp, dst.block(1));
    utmp *= -1.0;
    utmp += src.block(0);
    //    computing_timer.exit_section("Operations");
  }
  {
    //    computing_timer.enter_section("A Matrix");
    SolverControl solver_control(10000,
                                 std::max(1e-2 * src.block(0).l2_norm(),
                                          1e-12));



    TrilinosWrappers::SolverBicgstab solver(solver_control);

    // A_inverse.solve(stokes_matrix.block(0, 0),dst.block(0), utmp,
    // mp_preconditioner);
    solver.solve(stokes_matrix.block(0, 0),
                 dst.block(0),
                 utmp,
                 amat_preconditioner);
    //    computing_timer.exit_section("A Matrix");
  }
}

template <class PreconditionerMp>
class BlockCGPreconditioner : public Subscriptor
{
public:
  BlockCGPreconditioner(double                                     gamma,
                        double                                     viscosity,
                        const TrilinosWrappers::BlockSparseMatrix &S,
                        const PreconditionerMp &Mppreconditioner,
                        SolverControl &         A_parameters);

  void
  vmult(TrilinosWrappers::MPI::BlockVector &      dst,
        const TrilinosWrappers::MPI::BlockVector &src) const;

private:
  const double                                gamma;
  const double                                viscosity;
  const TrilinosWrappers::BlockSparseMatrix & stokes_matrix;
  TrilinosWrappers::PreconditionILU           amat_preconditioner;
  TrilinosWrappers::PreconditionILU           pmat_preconditioner;
  const PreconditionerMp &                    mp_preconditioner;
  SolverFGMRES<TrilinosWrappers::MPI::Vector> A_inverse;
};

template <class PreconditionerMp>
BlockCGPreconditioner<PreconditionerMp>::BlockCGPreconditioner(
  double                                     gamma,
  double                                     viscosity,
  const TrilinosWrappers::BlockSparseMatrix &S,
  const PreconditionerMp &                   Mppreconditioner,
  SolverControl &                            A_parameters)
  : gamma(gamma)
  , viscosity(viscosity)
  , stokes_matrix(S)
  , mp_preconditioner(Mppreconditioner)
  , A_inverse(A_parameters)
{
  TrilinosWrappers::PreconditionILU::AdditionalData preconditionerOptions(0,
                                                                          1e-12,
                                                                          1,
                                                                          0);
  amat_preconditioner.initialize(stokes_matrix.block(0, 0),
                                 preconditionerOptions);

  TrilinosWrappers::PreconditionILU::AdditionalData pmat_preconditionerOptions(
    0, 1e-12, 1, 0);
  TrilinosWrappers::PreconditionILU pmat_preconditioner;
  pmat_preconditioner.initialize(stokes_matrix.block(1, 1),
                                 pmat_preconditionerOptions);
}

template <class PreconditionerMp>
void
BlockCGPreconditioner<PreconditionerMp>::vmult(
  TrilinosWrappers::MPI::BlockVector &      dst,
  const TrilinosWrappers::MPI::BlockVector &src) const
{
  {
    SolverControl              solver_control(100000, 1e-12);
    TrilinosWrappers::SolverCG cg(solver_control);

    cg.solve(stokes_matrix.block(1, 1),
             dst.block(1),
             src.block(1),
             pmat_preconditioner);
  }
  {
    SolverControl solver_control(10000, 1e-12);

    TrilinosWrappers::SolverGMRES solver(solver_control);
    solver.solve(stokes_matrix.block(0, 0),
                 dst.block(0),
                 src.block(0),
                 amat_preconditioner);
  }
}


// Constructor for class GDNavierStokesSolver
template <int dim>
GDNavierStokesSolver<dim>::GDNavierStokesSolver(
  NavierStokesSolverParameters<dim> &p_nsparam,
  const unsigned int                 degreeVelocity,
  const unsigned int                 degreePressure)
  : NavierStokesBase<dim>(p_nsparam, degreeVelocity, degreePressure)

  , nsparam(p_nsparam)
{
  simulationControl = nsparam.simulationControl;

  // Pre-allocate the force tables to match the number of boundary conditions
  forces_.resize(nsparam.boundaryConditions.size);
  torques_.resize(nsparam.boundaryConditions.size);
  forces_tables.resize(nsparam.boundaryConditions.size);
  torques_tables.resize(nsparam.boundaryConditions.size);

  // Get the exact solution from the parser
  exact_solution = &nsparam.analyticalSolution->velocity;

  // If there is a forcing function, get it from the parser
  if (nsparam.sourceTerm->source_term())
    {
      forcing_function = &nsparam.sourceTerm->source;
    }
  else
    {
      forcing_function = new NoForce<dim>;
    }

  this->pcout << "Running on "
              << Utilities::MPI::n_mpi_processes(this->mpi_communicator)
              << " MPI rank(s)..." << std::endl;
}

template <int dim>
GDNavierStokesSolver<dim>::~GDNavierStokesSolver()
{
  this->dof_handler.clear();
}

template <int dim>
void
GDNavierStokesSolver<dim>::make_cube_grid(int refinementLevel)
{
  GridGenerator::hyper_cube(this->triangulation, -1, 1);
  this->triangulation.refine_global(refinementLevel);
}

template <int dim>
void
GDNavierStokesSolver<dim>::finish_time_step()
{
  if (simulationControl.getMethod() != Parameters::SimulationControl::steady)
    {
      solution_m3      = solution_m2;
      solution_m2      = solution_m1;
      solution_m1      = present_solution;
      const double CFL = calculate_CFL();
      simulationControl.setCFL(CFL);
    }
  if (nsparam.restartParameters.checkpoint &&
      simulationControl.getIter() % nsparam.restartParameters.frequency == 0)
    {
      write_checkpoint();
    }

  if (this->nsparam.timer.type == Parameters::Timer::iteration)
    {
      this->computing_timer.print_summary();
      this->computing_timer.reset();
    }
}

template <int dim>
void
GDNavierStokesSolver<dim>::set_solution_vector(double value)
{
  present_solution = value;
}

template <int dim>
void
GDNavierStokesSolver<dim>::set_periodicity()
{
  // Setup parallelism for periodic boundary conditions
  for (unsigned int i_bc = 0; i_bc < nsparam.boundaryConditions.size; ++i_bc)
    {
      if (nsparam.boundaryConditions.type[i_bc] == BoundaryConditions::periodic)
        {
          std::vector<GridTools::PeriodicFacePair<
            typename parallel::distributed::Triangulation<dim>::cell_iterator>>
            periodicity_vector;
          GridTools::collect_periodic_faces(
            this->triangulation,
            nsparam.boundaryConditions.id[i_bc],
            nsparam.boundaryConditions.periodic_id[i_bc],
            nsparam.boundaryConditions.periodic_direction[i_bc],
            periodicity_vector);
          this->triangulation.add_periodicity(periodicity_vector);
        }
    }
}

template <int dim>
void
GDNavierStokesSolver<dim>::setup_dofs()
{
  TimerOutput::Scope t(this->computing_timer, "setup_dofs");

  system_matrix.clear();

  this->dof_handler.distribute_dofs(this->fe);
  // DoFRenumbering::Cuthill_McKee(this->dof_handler);


  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise(this->dof_handler, block_component);
  dofs_per_block.resize(2);
  DoFTools::count_dofs_per_block(this->dof_handler,
                                 dofs_per_block,
                                 block_component);
  unsigned int dof_u = dofs_per_block[0];
  unsigned int dof_p = dofs_per_block[1];

  locally_owned_dofs.resize(2);
  locally_owned_dofs[0] =
    this->dof_handler.locally_owned_dofs().get_view(0, dof_u);
  locally_owned_dofs[1] =
    this->dof_handler.locally_owned_dofs().get_view(dof_u, dof_u + dof_p);

  IndexSet locally_relevant_dofs_acquisition;
  DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                          locally_relevant_dofs_acquisition);
  locally_relevant_dofs.resize(2);
  locally_relevant_dofs[0] =
    locally_relevant_dofs_acquisition.get_view(0, dof_u);
  locally_relevant_dofs[1] =
    locally_relevant_dofs_acquisition.get_view(dof_u, dof_u + dof_p);

  const MappingQ<dim>        mapping(this->degreeVelocity_,
                              nsparam.femParameters.qmapping_all);
  FEValuesExtractors::Vector velocities(0);

  // Non-zero constraints
  {
    nonzero_constraints.clear();

    DoFTools::make_hanging_node_constraints(this->dof_handler,
                                            nonzero_constraints);
    for (unsigned int i_bc = 0; i_bc < nsparam.boundaryConditions.size; ++i_bc)
      {
        if (nsparam.boundaryConditions.type[i_bc] == BoundaryConditions::noslip)
          {
            VectorTools::interpolate_boundary_values(
              mapping,
              this->dof_handler,
              nsparam.boundaryConditions.id[i_bc],
              ZeroFunction<dim>(dim + 1),
              nonzero_constraints,
              this->fe.component_mask(velocities));
          }
        else if (nsparam.boundaryConditions.type[i_bc] ==
                 BoundaryConditions::slip)
          {
            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert(
              nsparam.boundaryConditions.id[i_bc]);
            VectorTools::compute_no_normal_flux_constraints(
              this->dof_handler,
              0,
              no_normal_flux_boundaries,
              nonzero_constraints);
          }
        else if (nsparam.boundaryConditions.type[i_bc] ==
                 BoundaryConditions::function)
          {
            VectorTools::interpolate_boundary_values(
              mapping,
              this->dof_handler,
              nsparam.boundaryConditions.id[i_bc],
              FunctionDefined<dim>(
                &nsparam.boundaryConditions.bcFunctions[i_bc].u,
                &nsparam.boundaryConditions.bcFunctions[i_bc].v,
                &nsparam.boundaryConditions.bcFunctions[i_bc].w),
              nonzero_constraints,
              this->fe.component_mask(velocities));
          }

        else if (nsparam.boundaryConditions.type[i_bc] ==
                 BoundaryConditions::periodic)
          {
            DoFTools::make_periodicity_constraints<DoFHandler<dim>>(
              this->dof_handler,
              nsparam.boundaryConditions.id[i_bc],
              nsparam.boundaryConditions.periodic_id[i_bc],
              nsparam.boundaryConditions.periodic_direction[i_bc],
              nonzero_constraints);
          }
      }
  }
  nonzero_constraints.close();

  {
    zero_constraints.clear();
    DoFTools::make_hanging_node_constraints(this->dof_handler,
                                            zero_constraints);

    for (unsigned int i_bc = 0; i_bc < nsparam.boundaryConditions.size; ++i_bc)
      {
        if (nsparam.boundaryConditions.type[i_bc] == BoundaryConditions::slip)
          {
            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert(
              nsparam.boundaryConditions.id[i_bc]);
            VectorTools::compute_no_normal_flux_constraints(
              this->dof_handler,
              0,
              no_normal_flux_boundaries,
              zero_constraints);
          }
        else if (nsparam.boundaryConditions.type[i_bc] ==
                 BoundaryConditions::periodic)
          {
            DoFTools::make_periodicity_constraints<DoFHandler<dim>>(
              this->dof_handler,
              nsparam.boundaryConditions.id[i_bc],
              nsparam.boundaryConditions.periodic_id[i_bc],
              nsparam.boundaryConditions.periodic_direction[i_bc],
              zero_constraints);
          }
        else // if(nsparam.boundaryConditions.boundaries[i_bc].type==Parameters::noslip
             // || Parameters::function)
          {
            VectorTools::interpolate_boundary_values(
              mapping,
              this->dof_handler,
              nsparam.boundaryConditions.id[i_bc],
              ZeroFunction<dim>(dim + 1),
              zero_constraints,
              this->fe.component_mask(velocities));
          }
      }
  }
  zero_constraints.close();

  present_solution.reinit(locally_owned_dofs,
                          locally_relevant_dofs,
                          this->mpi_communicator);

  solution_m1.reinit(locally_owned_dofs,
                     locally_relevant_dofs,
                     this->mpi_communicator);
  solution_m2.reinit(locally_owned_dofs,
                     locally_relevant_dofs,
                     this->mpi_communicator);
  solution_m3.reinit(locally_owned_dofs,
                     locally_relevant_dofs,
                     this->mpi_communicator);

  newton_update.reinit(locally_owned_dofs, this->mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, this->mpi_communicator);
  local_evaluation_point.reinit(locally_owned_dofs, this->mpi_communicator);


  sparsity_pattern.reinit(locally_owned_dofs,
                          locally_owned_dofs,
                          locally_relevant_dofs,
                          MPI_COMM_WORLD);

  Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
  for (unsigned int c = 0; c < dim + 1; ++c)
    for (unsigned int d = 0; d < dim + 1; ++d)
      if (!((c == dim) && (d == dim)))
        coupling[c][d] = DoFTools::always;
      else
        coupling[c][d] = DoFTools::always;

  DoFTools::make_sparsity_pattern(this->dof_handler,
                                  coupling,
                                  sparsity_pattern,
                                  nonzero_constraints,
                                  true,
                                  Utilities::MPI::this_mpi_process(
                                    MPI_COMM_WORLD));

  sparsity_pattern.compress();

  system_matrix.reinit(sparsity_pattern);
  pressure_mass_matrix.reinit(sparsity_pattern.block(1, 1));


  globalVolume_ = GridTools::volume(this->triangulation);

  this->pcout << "   Number of active cells:       "
              << this->triangulation.n_global_active_cells() << std::endl
              << "   Number of degrees of freedom: "
              << this->dof_handler.n_dofs() << std::endl;
  this->pcout << "   Volume of triangulation:      " << globalVolume_
              << std::endl;
}

template <int dim>
template <bool                                              assemble_matrix,
          Parameters::SimulationControl::TimeSteppingMethod scheme>
void
GDNavierStokesSolver<dim>::assembleGD()
{
  double viscosity = nsparam.physicalProperties.viscosity;


  if (assemble_matrix)
    system_matrix = 0;

  system_rhs = 0;

  QGauss<dim>         quadrature_formula(this->degreeQuadrature_);
  const MappingQ<dim> mapping(this->degreeVelocity_,
                              nsparam.femParameters.qmapping_all);

  FEValues<dim> fe_values(mapping,
                          this->fe,
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values | update_gradients);

  const unsigned int dofs_per_cell = this->fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  FullMatrix<double>          local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>              local_rhs(dofs_per_cell);
  std::vector<Vector<double>> rhs_force(n_q_points, Vector<double>(dim + 1));


  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // For the linearized system, we create temporary storage for present
  // velocity and gradient, and present pressure. In practice, they are all
  // obtained through their shape functions at quadrature points.

  std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
  std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
  std::vector<double>         present_pressure_values(n_q_points);

  std::vector<double>         div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double>         phi_p(dofs_per_cell);

  Tensor<1, dim> force;

  // Get the BDF coefficients
  Vector<double> alpha_bdf;

  if (scheme == Parameters::SimulationControl::bdf1)
    alpha_bdf = bdf_coefficients(1, simulationControl.getTimeSteps());

  if (scheme == Parameters::SimulationControl::bdf2)
    alpha_bdf = bdf_coefficients(2, simulationControl.getTimeSteps());

  if (scheme == Parameters::SimulationControl::bdf3)
    alpha_bdf = bdf_coefficients(3, simulationControl.getTimeSteps());

  // Values at previous time step for backward Euler scheme
  std::vector<Tensor<1, dim>> p1_velocity_values(n_q_points);
  std::vector<Tensor<1, dim>> p2_velocity_values(n_q_points);
  std::vector<Tensor<1, dim>> p3_velocity_values(n_q_points);
  std::vector<Tensor<1, dim>> p4_velocity_values(n_q_points);


  for (const auto &cell : this->dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          local_matrix = 0;
          local_rhs    = 0;

          fe_values[velocities].get_function_values(evaluation_point,
                                                    present_velocity_values);

          fe_values[velocities].get_function_gradients(
            evaluation_point, present_velocity_gradients);

          fe_values[pressure].get_function_values(evaluation_point,
                                                  present_pressure_values);

          if (scheme != Parameters::SimulationControl::steady)
            fe_values[velocities].get_function_values(solution_m1,
                                                      p1_velocity_values);

          if (scheme == Parameters::SimulationControl::bdf2 ||
              scheme == Parameters::SimulationControl::bdf3)
            fe_values[velocities].get_function_values(solution_m2,
                                                      p2_velocity_values);

          if (scheme == Parameters::SimulationControl::bdf3)
            fe_values[velocities].get_function_values(solution_m3,
                                                      p3_velocity_values);

          if (forcing_function)
            forcing_function->vector_value_list(
              fe_values.get_quadrature_points(), rhs_force);

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              // Establish the force vector
              for (int i = 0; i < dim; ++i)
                {
                  const unsigned int component_i =
                    this->fe.system_to_component_index(i).first;
                  force[i] = rhs_force[q](component_i);
                }

              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  div_phi_u[k]  = fe_values[velocities].divergence(k, q);
                  grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                  phi_u[k]      = fe_values[velocities].value(k, q);
                  phi_p[k]      = fe_values[pressure].value(k, q);
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  if (assemble_matrix)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          local_matrix(i, j) +=
                            (viscosity *
                               scalar_product(grad_phi_u[j], grad_phi_u[i]) +
                             present_velocity_gradients[q] * phi_u[j] *
                               phi_u[i] +
                             grad_phi_u[j] * present_velocity_values[q] *
                               phi_u[i] -
                             div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j] +
                             gamma * div_phi_u[j] * div_phi_u[i] +
                             phi_p[i] * phi_p[j]) *
                            fe_values.JxW(q);

                          // Mass matrix
                          if (scheme == Parameters::SimulationControl::bdf1 ||
                              scheme == Parameters::SimulationControl::bdf2 ||
                              scheme == Parameters::SimulationControl::bdf3)
                            local_matrix(i, j) += phi_u[j] * phi_u[i] *
                                                  alpha_bdf[0] *
                                                  fe_values.JxW(q);
                        }
                    }

                  double present_velocity_divergence =
                    trace(present_velocity_gradients[q]);
                  local_rhs(i) +=
                    (-viscosity * scalar_product(present_velocity_gradients[q],
                                                 grad_phi_u[i]) -
                     present_velocity_gradients[q] *
                       present_velocity_values[q] * phi_u[i] +
                     present_pressure_values[q] * div_phi_u[i] +
                     present_velocity_divergence * phi_p[i] -
                     gamma * present_velocity_divergence * div_phi_u[i] +
                     force * phi_u[i]) *
                    fe_values.JxW(q);

                  if (scheme == Parameters::SimulationControl::bdf1)
                    local_rhs(i) -=
                      alpha_bdf[0] *
                      (present_velocity_values[q] - p1_velocity_values[q]) *
                      phi_u[i] * fe_values.JxW(q);

                  if (scheme == Parameters::SimulationControl::bdf2)
                    local_rhs(i) -=
                      (alpha_bdf[0] * (present_velocity_values[q] * phi_u[i]) +
                       alpha_bdf[1] * (p1_velocity_values[q] * phi_u[i]) +
                       alpha_bdf[2] * (p2_velocity_values[q] * phi_u[i])) *
                      fe_values.JxW(q);

                  if (scheme == Parameters::SimulationControl::bdf3)
                    local_rhs(i) -=
                      (alpha_bdf[0] * (present_velocity_values[q] * phi_u[i]) +
                       alpha_bdf[1] * (p1_velocity_values[q] * phi_u[i]) +
                       alpha_bdf[2] * (p2_velocity_values[q] * phi_u[i]) +
                       alpha_bdf[3] * (p3_velocity_values[q] * phi_u[i])) *
                      fe_values.JxW(q);
                }
            }

          cell->get_dof_indices(local_dof_indices);

          const AffineConstraints<double> &constraints_used = zero_constraints;

          if (assemble_matrix)
            {
              constraints_used.distribute_local_to_global(local_matrix,
                                                          local_rhs,
                                                          local_dof_indices,
                                                          system_matrix,
                                                          system_rhs);
            }
          else
            {
              constraints_used.distribute_local_to_global(local_rhs,
                                                          local_dof_indices,
                                                          system_rhs);
            }
        }
    }

  if (assemble_matrix)
    {
      system_matrix.compress(VectorOperation::add);

      // Finally we move pressure mass matrix into a separate matrix:
      pressure_mass_matrix.reinit(sparsity_pattern.block(1, 1));
      pressure_mass_matrix.copy_from(system_matrix.block(1, 1));

      // Note that settings this pressure block to zero is not identical to
      // not assembling anything in this block, because this operation here
      // will (incorrectly) delete diagonal entries that come in from
      // hanging node constraints for pressure DoFs. This means that our
      // whole system matrix will have rows that are completely
      // zero. Luckily, FGMRES handles these rows without any problem.
      system_matrix.block(1, 1) = 0;
    }
  system_rhs.compress(VectorOperation::add);
}

/**
 * Set the initial condition using a L2 or a viscous solver
 **/
template <int dim>
void
GDNavierStokesSolver<dim>::set_initial_condition(
  Parameters::InitialConditionType initial_condition_type,
  bool                             restart)
{
  if (restart)
    {
      this->pcout << "************************" << std::endl;
      this->pcout << "---> Simulation Restart " << std::endl;
      this->pcout << "************************" << std::endl;
      read_checkpoint();
    }
  else if (initial_condition_type ==
           Parameters::InitialConditionType::L2projection)
    {
      assemble_L2_projection();
      solve_L2_system(true, 1e-15, 1e-15);
      present_solution = newton_update;
      finish_time_step();
      postprocess(true);
    }
  else if (initial_condition_type == Parameters::InitialConditionType::nodal)
    {
      set_nodal_values();
      finish_time_step();
      postprocess(true);
    }

  else if (initial_condition_type == Parameters::InitialConditionType::viscous)
    {
      set_nodal_values();
      double viscosity = this->nsparam.physicalProperties.viscosity;
      this->nsparam.physicalProperties.viscosity =
        nsparam.initialCondition->viscosity;
      Parameters::SimulationControl::TimeSteppingMethod previousControl =
        simulationControl.getMethod();
      simulationControl.setMethod(Parameters::SimulationControl::steady);
      newton_iteration(false);
      simulationControl.setMethod(previousControl);
      finish_time_step();
      postprocess(true);
      simulationControl.setMethod(previousControl);
      this->nsparam.physicalProperties.viscosity = viscosity;
    }
  else
    {
      throw std::runtime_error("GDNS - Initial condition could not be set");
    }
}

// Do an iteration with the GD NavierStokes Solver
// Handles the fact that we may or may not be at a first
// iteration with the solver and sets the initial condition
template <int dim>
void
GDNavierStokesSolver<dim>::iterate(bool firstIteration)
{
  // Carry out the integration normally
  if (!firstIteration)
    {
      newton_iteration(false);
    }
  // This is the first iteration
  else
    {
      if (simulationControl.getMethod() ==
          Parameters::SimulationControl::steady)
        {
          newton_iteration(false);
        }
      else if (simulationControl.getMethod() ==
               Parameters::SimulationControl::bdf1)
        {
          newton_iteration(false);
        }
      else if (simulationControl.getMethod() ==
               Parameters::SimulationControl::bdf2)
        {
          Parameters::SimulationControl timeParameters =
            simulationControl.getParameters();

          // Start the BDF2 with a single Euler time step with a lower time step
          simulationControl.setTimeStep(
            timeParameters.dt * timeParameters.startup_timestep_scaling);
          simulationControl.setMethod(Parameters::SimulationControl::bdf1);
          newton_iteration(false);
          solution_m2 = solution_m1;
          solution_m1 = present_solution;

          // Reset the time step and do a bdf 2 newton iteration using the two
          // steps to complete the full step
          simulationControl.setMethod(Parameters::SimulationControl::bdf2);
          simulationControl.setTimeStep(
            timeParameters.dt * (1. - timeParameters.startup_timestep_scaling));
          newton_iteration(false);
        }

      else if (simulationControl.getMethod() ==
               Parameters::SimulationControl::bdf3)
        {
          Parameters::SimulationControl timeParameters =
            simulationControl.getParameters();

          // Start the BDF2 with a single Euler time step with a lower time step
          simulationControl.setTimeStep(
            timeParameters.dt * timeParameters.startup_timestep_scaling);
          simulationControl.setMethod(Parameters::SimulationControl::bdf1);
          newton_iteration(false);
          solution_m2 = solution_m1;
          solution_m1 = present_solution;

          // Reset the time step and do a bdf 2 newton iteration using the two
          // steps
          simulationControl.setMethod(Parameters::SimulationControl::bdf1);
          simulationControl.setTimeStep(
            timeParameters.dt * timeParameters.startup_timestep_scaling);
          newton_iteration(false);
          solution_m3 = solution_m2;
          solution_m2 = solution_m1;
          solution_m1 = present_solution;

          // Reset the time step and do a bdf 3 newton iteration using the two
          // steps to complete the full step
          simulationControl.setMethod(Parameters::SimulationControl::bdf3);
          simulationControl.setTimeStep(
            timeParameters.dt *
            (1. - 2. * timeParameters.startup_timestep_scaling));
          newton_iteration(false);
        }
    }
}

template <int dim>
void
GDNavierStokesSolver<dim>::assemble_L2_projection()
{
  system_matrix = 0;
  system_rhs    = 0;
  QGauss<dim>                 quadrature_formula(this->degreeQuadrature_);
  const MappingQ<dim>         mapping(this->degreeVelocity_,
                              nsparam.femParameters.qmapping_all);
  FEValues<dim>               fe_values(mapping,
                          this->fe,
                          quadrature_formula,
                          update_values | update_quadrature_points |
                            update_JxW_values);
  const unsigned int          dofs_per_cell = this->fe.dofs_per_cell;
  const unsigned int          n_q_points    = quadrature_formula.size();
  FullMatrix<double>          local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>              local_rhs(dofs_per_cell);
  std::vector<Vector<double>> initial_velocity(n_q_points,
                                               Vector<double>(dim + 1));
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const FEValuesExtractors::Vector     velocities(0);
  const FEValuesExtractors::Scalar     pressure(dim);

  Tensor<1, dim> rhs_initial_velocity_pressure;
  double         rhs_initial_pressure;

  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<double>         phi_p(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler
                                                          .begin_active(),
                                                 endc = this->dof_handler.end();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          local_matrix = 0;
          local_rhs    = 0;
          nsparam.initialCondition->uvwp.vector_value_list(
            fe_values.get_quadrature_points(), initial_velocity);
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  phi_p[k] = fe_values[pressure].value(k, q);
                  phi_u[k] = fe_values[velocities].value(k, q);
                }

              // Establish the rhs tensor operator
              for (int i = 0; i < dim; ++i)
                {
                  const unsigned int component_i =
                    this->fe.system_to_component_index(i).first;
                  rhs_initial_velocity_pressure[i] =
                    initial_velocity[q](component_i);
                }
              rhs_initial_pressure = initial_velocity[q](dim);

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  // Matrix assembly
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      local_matrix(i, j) +=
                        (phi_u[j] * phi_u[i]) * fe_values.JxW(q);
                      local_matrix(i, j) +=
                        (phi_p[j] * phi_p[i]) * fe_values.JxW(q);
                    }
                  local_rhs(i) += (phi_u[i] * rhs_initial_velocity_pressure +
                                   phi_p[i] * rhs_initial_pressure) *
                                  fe_values.JxW(q);
                }
            }

          cell->get_dof_indices(local_dof_indices);
          const AffineConstraints<double> &constraints_used =
            nonzero_constraints;
          constraints_used.distribute_local_to_global(local_matrix,
                                                      local_rhs,
                                                      local_dof_indices,
                                                      system_matrix,
                                                      system_rhs);
        }
    }
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}

template <int dim>
double
GDNavierStokesSolver<dim>::calculate_CFL()
{
  QGauss<dim>                          quadrature_formula(1);
  const MappingQ<dim>                  mapping(this->degreeVelocity_,
                              nsparam.femParameters.qmapping_all);
  FEValues<dim>                        fe_values(mapping,
                          this->fe,
                          quadrature_formula,
                          update_values | update_quadrature_points);
  const unsigned int                   dofs_per_cell = this->fe.dofs_per_cell;
  const unsigned int                   n_q_points = quadrature_formula.size();
  std::vector<Vector<double>>          initial_velocity(n_q_points,
                                                        Vector<double>(dim + 1));
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const FEValuesExtractors::Vector     velocities(0);
  const FEValuesExtractors::Scalar     pressure(dim);

  std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);

  // Element size
  double h;

  // Current time step
  double timeStep = simulationControl.getCurrentTimeStep();

  // CFL
  double CFL = 0;

  typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler
                                                          .begin_active(),
                                                 endc = this->dof_handler.end();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          if (dim == 2)
            h = std::sqrt(4. * cell->measure() / M_PI) / this->degreeVelocity_;
          else if (dim == 3)
            h =
              pow(6 * cell->measure() / M_PI, 1. / 3.) / this->degreeVelocity_;
          fe_values.reinit(cell);
          fe_values[velocities].get_function_values(present_solution,
                                                    present_velocity_values);
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              const double localCFL =
                present_velocity_values[q].norm() / h * timeStep;
              CFL = std::max(CFL, localCFL);
            }
        }
    }
  CFL = Utilities::MPI::max(CFL, this->mpi_communicator);

  return CFL;
}

template <int dim>
void
GDNavierStokesSolver<dim>::set_nodal_values()
{
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);
  const MappingQ<dim>              mapping(this->degreeVelocity_,
                              nsparam.femParameters.qmapping_all);
  VectorTools::interpolate(mapping,
                           this->dof_handler,
                           nsparam.initialCondition->uvwp,
                           newton_update,
                           this->fe.component_mask(velocities));
  VectorTools::interpolate(mapping,
                           this->dof_handler,
                           nsparam.initialCondition->uvwp,
                           newton_update,
                           this->fe.component_mask(pressure));
  nonzero_constraints.distribute(newton_update);
  present_solution = newton_update;
}

template <int dim>
void
GDNavierStokesSolver<dim>::assemble_system()
{
  TimerOutput::Scope t(this->computing_timer, "assemble_system");

  if (simulationControl.getMethod() == Parameters::SimulationControl::bdf1)
    assembleGD<true, Parameters::SimulationControl::bdf1>();
  else if (simulationControl.getMethod() == Parameters::SimulationControl::bdf2)
    assembleGD<true, Parameters::SimulationControl::bdf2>();
  else if (simulationControl.getMethod() == Parameters::SimulationControl::bdf3)
    assembleGD<true, Parameters::SimulationControl::bdf3>();
  else if (simulationControl.getMethod() ==
           Parameters::SimulationControl::steady)
    assembleGD<true, Parameters::SimulationControl::steady>();
}
template <int dim>
void
GDNavierStokesSolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(this->computing_timer, "assemble_rhs");

  if (simulationControl.getMethod() == Parameters::SimulationControl::bdf1)
    assembleGD<false, Parameters::SimulationControl::bdf1>();
  else if (simulationControl.getMethod() == Parameters::SimulationControl::bdf2)
    assembleGD<false, Parameters::SimulationControl::bdf2>();
  else if (simulationControl.getMethod() == Parameters::SimulationControl::bdf3)
    assembleGD<false, Parameters::SimulationControl::bdf3>();
  else if (simulationControl.getMethod() ==
           Parameters::SimulationControl::steady)
    assembleGD<false, Parameters::SimulationControl::steady>();
}

template <int dim>
void
GDNavierStokesSolver<dim>::solve_L2_system(const bool initial_step,
                                           double     absolute_residual,
                                           double     relative_residual)
{
  TimerOutput::Scope t(this->computing_timer, "solve_linear_system");
  const AffineConstraints<double> &constraints_used =
    initial_step ? nonzero_constraints : zero_constraints;
  const double linear_solver_tolerance =
    std::max(relative_residual * system_rhs.l2_norm(), absolute_residual);

  if (nsparam.linearSolver.verbosity != Parameters::quiet)
    {
      this->pcout << "  -Tolerance of iterative solver is : "
                  << std::setprecision(nsparam.linearSolver.residual_precision)
                  << linear_solver_tolerance << std::endl;
    }
  TrilinosWrappers::MPI::BlockVector completely_distributed_solution(
    locally_owned_dofs, this->mpi_communicator);

  SolverControl solver_control(nsparam.linearSolver.max_iterations,
                               linear_solver_tolerance,
                               true,
                               true);
  SolverFGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);
  TrilinosWrappers::PreconditionILU                pmass_preconditioner;

  //**********************************************
  // Trillinos Wrapper ILU Preconditioner
  //*********************************************
  const double ilu_fill = nsparam.linearSolver.ilu_precond_fill;
  const double ilu_atol = nsparam.linearSolver.ilu_precond_atol;
  const double ilu_rtol = nsparam.linearSolver.ilu_precond_rtol;
  TrilinosWrappers::PreconditionILU::AdditionalData preconditionerOptions(
    ilu_fill, ilu_atol, ilu_rtol, 0);
  pmass_preconditioner.initialize(pressure_mass_matrix, preconditionerOptions);


  const BlockCGPreconditioner<TrilinosWrappers::PreconditionILU> preconditioner(
    gamma,
    nsparam.physicalProperties.viscosity,
    system_matrix,
    pmass_preconditioner,
    solver_control);

  // preconditioner.initialize(system_matrix, preconditionerOptions);

  solver.solve(system_matrix,
               completely_distributed_solution,
               system_rhs,
               preconditioner);

  if (nsparam.linearSolver.verbosity != Parameters::quiet)
    {
      this->pcout << "  -Iterative solver took : " << solver_control.last_step()
                  << " steps " << std::endl;
    }

  constraints_used.distribute(completely_distributed_solution);
  newton_update = completely_distributed_solution;
}

template <int dim>
void
GDNavierStokesSolver<dim>::solve_NS_system(const bool initial_step,
                                           double     absolute_residual,
                                           double     relative_residual)
{
  TimerOutput::Scope t(this->computing_timer, "solve_linear_system");

  const AffineConstraints<double> &constraints_used =
    initial_step ? nonzero_constraints : zero_constraints;
  const double linear_solver_tolerance =
    std::max(relative_residual * system_rhs.l2_norm(), absolute_residual);

  if (nsparam.linearSolver.verbosity != Parameters::quiet)
    {
      this->pcout << "  -Tolerance of iterative solver is : "
                  << std::setprecision(nsparam.linearSolver.residual_precision)
                  << linear_solver_tolerance << std::endl;
    }
  TrilinosWrappers::MPI::BlockVector completely_distributed_solution(
    locally_owned_dofs, this->mpi_communicator);

  SolverControl solver_control(nsparam.linearSolver.max_iterations,
                               linear_solver_tolerance,
                               true,
                               true);

  SolverFGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);
  TrilinosWrappers::PreconditionILU                pmass_preconditioner;

  //**********************************************
  // Trillinos Wrapper ILU Preconditioner
  //*********************************************
  const double ilu_fill = nsparam.linearSolver.ilu_precond_fill;
  const double ilu_atol = nsparam.linearSolver.ilu_precond_atol;
  const double ilu_rtol = nsparam.linearSolver.ilu_precond_rtol;
  TrilinosWrappers::PreconditionILU::AdditionalData preconditionerOptions(
    ilu_fill, ilu_atol, ilu_rtol, 0);

  pmass_preconditioner.initialize(pressure_mass_matrix, preconditionerOptions);

  const BlockSchurPreconditioner<TrilinosWrappers::PreconditionILU>
    preconditioner(gamma,
                   nsparam.physicalProperties.viscosity,
                   system_matrix,
                   pressure_mass_matrix,
                   pmass_preconditioner,
                   solver_control);

  solver.solve(system_matrix, newton_update, system_rhs, preconditioner);
  if (nsparam.linearSolver.verbosity != Parameters::quiet)
    {
      this->pcout << "  -Iterative solver took : " << solver_control.last_step()
                  << " steps " << std::endl;
    }

  constraints_used.distribute(newton_update);
}

template <int dim>
void
GDNavierStokesSolver<dim>::refine_mesh()
{
  if (simulationControl.getIter() % nsparam.meshAdaptation.frequency == 0)
    {
      if (nsparam.meshAdaptation.type == nsparam.meshAdaptation.kelly)
        refine_mesh_Kelly();
      if (nsparam.meshAdaptation.type == nsparam.meshAdaptation.uniform)
        refine_mesh_uniform();
    }
}

template <int dim>
void
GDNavierStokesSolver<dim>::refine_mesh_Kelly()
{
  // Time monitoring
  TimerOutput::Scope t(this->computing_timer, "refine");

  Vector<float> estimated_error_per_cell(this->triangulation.n_active_cells());
  const MappingQ<dim>              mapping(this->degreeVelocity_,
                              nsparam.femParameters.qmapping_all);
  const FEValuesExtractors::Vector velocity(0);
  const FEValuesExtractors::Scalar pressure(dim);
  if (nsparam.meshAdaptation.variable == Parameters::MeshAdaptation::pressure)
    {
      KellyErrorEstimator<dim>::estimate(
        mapping,
        this->dof_handler,
        QGauss<dim - 1>(this->degreeQuadrature_ + 1),
        typename std::map<types::boundary_id, const Function<dim, double> *>(),
        present_solution,
        estimated_error_per_cell,
        this->fe.component_mask(pressure));
    }
  else if (nsparam.meshAdaptation.variable ==
           Parameters::MeshAdaptation::velocity)
    {
      KellyErrorEstimator<dim>::estimate(
        mapping,
        this->dof_handler,
        QGauss<dim - 1>(this->degreeQuadrature_ + 1),
        typename std::map<types::boundary_id, const Function<dim, double> *>(),
        present_solution,
        estimated_error_per_cell,
        this->fe.component_mask(velocity));
    }

  if (nsparam.meshAdaptation.fractionType == Parameters::MeshAdaptation::number)
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
      this->triangulation,
      estimated_error_per_cell,
      nsparam.meshAdaptation.fractionRefinement,
      nsparam.meshAdaptation.fractionCoarsening,
      nsparam.meshAdaptation.maxNbElements);

  else if (nsparam.meshAdaptation.fractionType ==
           Parameters::MeshAdaptation::fraction)
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
      this->triangulation,
      estimated_error_per_cell,
      nsparam.meshAdaptation.fractionRefinement,
      nsparam.meshAdaptation.fractionCoarsening);

  if (this->triangulation.n_levels() > nsparam.meshAdaptation.maxRefLevel)
    for (typename Triangulation<dim>::active_cell_iterator cell =
           this->triangulation.begin_active(nsparam.meshAdaptation.maxRefLevel);
         cell != this->triangulation.end();
         ++cell)
      cell->clear_refine_flag();
  for (typename Triangulation<dim>::active_cell_iterator cell =
         this->triangulation.begin_active(nsparam.meshAdaptation.minRefLevel);
       cell !=
       this->triangulation.end_active(nsparam.meshAdaptation.minRefLevel);
       ++cell)
    cell->clear_coarsen_flag();

  this->triangulation.prepare_coarsening_and_refinement();

  // Solution transfer objects for all the solutions
  parallel::distributed::SolutionTransfer<dim,
                                          TrilinosWrappers::MPI::BlockVector>
    solution_transfer(this->dof_handler);
  parallel::distributed::SolutionTransfer<dim,
                                          TrilinosWrappers::MPI::BlockVector>
    solution_transfer_m1(this->dof_handler);
  parallel::distributed::SolutionTransfer<dim,
                                          TrilinosWrappers::MPI::BlockVector>
    solution_transfer_m2(this->dof_handler);
  parallel::distributed::SolutionTransfer<dim,
                                          TrilinosWrappers::MPI::BlockVector>
    solution_transfer_m3(this->dof_handler);
  solution_transfer.prepare_for_coarsening_and_refinement(present_solution);
  solution_transfer_m1.prepare_for_coarsening_and_refinement(solution_m1);
  solution_transfer_m2.prepare_for_coarsening_and_refinement(solution_m2);
  solution_transfer_m3.prepare_for_coarsening_and_refinement(solution_m3);

  this->triangulation.execute_coarsening_and_refinement();
  setup_dofs();

  // Set up the vectors for the transfer
  TrilinosWrappers::MPI::BlockVector tmp(locally_owned_dofs,
                                         this->mpi_communicator);
  TrilinosWrappers::MPI::BlockVector tmp_m1(locally_owned_dofs,
                                            this->mpi_communicator);
  TrilinosWrappers::MPI::BlockVector tmp_m2(locally_owned_dofs,
                                            this->mpi_communicator);
  TrilinosWrappers::MPI::BlockVector tmp_m3(locally_owned_dofs,
                                            this->mpi_communicator);

  // Interpolate the solution at time and previous time
  solution_transfer.interpolate(tmp);
  solution_transfer_m1.interpolate(tmp_m1);
  solution_transfer_m2.interpolate(tmp_m2);
  solution_transfer_m3.interpolate(tmp_m3);

  // Distribute constraints
  nonzero_constraints.distribute(tmp);
  nonzero_constraints.distribute(tmp_m1);
  nonzero_constraints.distribute(tmp_m2);
  nonzero_constraints.distribute(tmp_m3);

  // Fix on the new mesh
  present_solution = tmp;
  solution_m1      = tmp_m1;
  solution_m2      = tmp_m2;
  solution_m3      = tmp_m3;
}

template <int dim>
void
GDNavierStokesSolver<dim>::refine_mesh_uniform()
{
  TimerOutput::Scope t(this->computing_timer, "refine");

  // Solution transfer objects for all the solutions
  parallel::distributed::SolutionTransfer<dim,
                                          TrilinosWrappers::MPI::BlockVector>
    solution_transfer(this->dof_handler);
  parallel::distributed::SolutionTransfer<dim,
                                          TrilinosWrappers::MPI::BlockVector>
    solution_transfer_m1(this->dof_handler);
  parallel::distributed::SolutionTransfer<dim,
                                          TrilinosWrappers::MPI::BlockVector>
    solution_transfer_m2(this->dof_handler);
  parallel::distributed::SolutionTransfer<dim,
                                          TrilinosWrappers::MPI::BlockVector>
    solution_transfer_m3(this->dof_handler);
  solution_transfer.prepare_for_coarsening_and_refinement(present_solution);
  solution_transfer_m1.prepare_for_coarsening_and_refinement(solution_m1);
  solution_transfer_m2.prepare_for_coarsening_and_refinement(solution_m2);
  solution_transfer_m3.prepare_for_coarsening_and_refinement(solution_m3);

  // Refine
  this->triangulation.refine_global(1);

  setup_dofs();

  // Set up the vectors for the transfer
  TrilinosWrappers::MPI::BlockVector tmp(locally_owned_dofs,
                                         this->mpi_communicator);
  TrilinosWrappers::MPI::BlockVector tmp_m1(locally_owned_dofs,
                                            this->mpi_communicator);
  TrilinosWrappers::MPI::BlockVector tmp_m2(locally_owned_dofs,
                                            this->mpi_communicator);
  TrilinosWrappers::MPI::BlockVector tmp_m3(locally_owned_dofs,
                                            this->mpi_communicator);

  // Interpolate the solution at time and previous time
  solution_transfer.interpolate(tmp);
  solution_transfer_m1.interpolate(tmp_m1);
  solution_transfer_m2.interpolate(tmp_m2);
  solution_transfer_m3.interpolate(tmp_m3);

  // Distribute constraints
  nonzero_constraints.distribute(tmp);
  nonzero_constraints.distribute(tmp_m1);
  nonzero_constraints.distribute(tmp_m2);
  nonzero_constraints.distribute(tmp_m3);

  // Fix on the new mesh
  present_solution = tmp;
  solution_m1      = tmp_m1;
  solution_m2      = tmp_m2;
  solution_m3      = tmp_m3;
}

template <int dim>
void
GDNavierStokesSolver<dim>::newton_iteration(const bool is_initial_step)
{
  double current_res;
  double last_res;
  bool   first_step = is_initial_step;
  {
    unsigned int outer_iteration = 0;
    last_res                     = 1.0;
    current_res                  = 1.0;
    while ((current_res > nsparam.nonLinearSolver.tolerance) &&
           outer_iteration < nsparam.nonLinearSolver.maxIterations)
      {
        evaluation_point = present_solution;
        assemble_system();
        if (outer_iteration == 0)
          {
            current_res = system_rhs.l2_norm();
            last_res    = current_res;
          }
        if (nsparam.nonLinearSolver.verbosity != Parameters::quiet)
          this->pcout << "Newton iteration: " << outer_iteration
                      << "  - Residual:  " << current_res << std::endl;
        solve_NS_system(first_step,
                        nsparam.linearSolver.relative_residual,
                        nsparam.linearSolver.minimum_residual);

        for (double alpha = 1.0; alpha > 1e-3; alpha *= 0.5)
          {
            local_evaluation_point = present_solution;
            local_evaluation_point.add(alpha, newton_update);
            nonzero_constraints.distribute(local_evaluation_point);
            evaluation_point = local_evaluation_point;
            assemble_rhs();
            current_res = system_rhs.l2_norm();
            if (nsparam.nonLinearSolver.verbosity != Parameters::quiet)
              this->pcout << "\t\talpha = " << std::setw(6) << alpha
                          << std::setw(0) << " res = "
                          << std::setprecision(
                               nsparam.nonLinearSolver.display_precision)
                          << current_res << std::endl;
            if (current_res < 0.9 * last_res ||
                last_res < nsparam.nonLinearSolver.tolerance)
              break;
          }
        present_solution = evaluation_point;
        last_res         = current_res;
        ++outer_iteration;
      }
  }
}



template <int dim>
void
GDNavierStokesSolver<dim>::write_checkpoint()
{
  TimerOutput::Scope timer(this->computing_timer, "write_checkpoint");
  std::string        prefix = nsparam.restartParameters.filename;
  if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
    simulationControl.save(prefix);
  if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
    pvdhandler.save(prefix);

  std::vector<const TrilinosWrappers::MPI::BlockVector *> sol_set_transfer;
  sol_set_transfer.push_back(&present_solution);
  sol_set_transfer.push_back(&solution_m1);
  sol_set_transfer.push_back(&solution_m2);
  sol_set_transfer.push_back(&solution_m3);
  parallel::distributed::SolutionTransfer<dim,
                                          TrilinosWrappers::MPI::BlockVector>
    system_trans_vectors(this->dof_handler);
  system_trans_vectors.prepare_for_serialization(sol_set_transfer);

  std::string triangulationName = prefix + ".triangulation";
  this->triangulation.save(prefix + ".triangulation");
}

template <int dim>
void
GDNavierStokesSolver<dim>::read_checkpoint()
{
  TimerOutput::Scope timer(this->computing_timer, "read_checkpoint");
  std::string        prefix = nsparam.restartParameters.filename;
  simulationControl.read(prefix);
  pvdhandler.read(prefix);

  const std::string filename = prefix + ".triangulation";
  std::ifstream     in(filename.c_str());
  if (!in)
    AssertThrow(false,
                ExcMessage(
                  std::string(
                    "You are trying to restart a previous computation, "
                    "but the restart file <") +
                  filename + "> does not appear to exist!"));

  try
    {
      this->triangulation.load(filename.c_str());
    }
  catch (...)
    {
      AssertThrow(false,
                  ExcMessage("Cannot open snapshot mesh file or read the "
                             "triangulation stored there."));
    }
  setup_dofs();
  std::vector<TrilinosWrappers::MPI::BlockVector *> x_system(4);

  TrilinosWrappers::MPI::BlockVector distributed_system(system_rhs);
  TrilinosWrappers::MPI::BlockVector distributed_system_m1(system_rhs);
  TrilinosWrappers::MPI::BlockVector distributed_system_m2(system_rhs);
  TrilinosWrappers::MPI::BlockVector distributed_system_m3(system_rhs);
  x_system[0] = &(distributed_system);
  x_system[1] = &(distributed_system_m1);
  x_system[2] = &(distributed_system_m2);
  x_system[3] = &(distributed_system_m3);
  parallel::distributed::SolutionTransfer<dim,
                                          TrilinosWrappers::MPI::BlockVector>
    system_trans_vectors(this->dof_handler);
  system_trans_vectors.deserialize(x_system);
  present_solution = distributed_system;
  solution_m1      = distributed_system_m1;
  solution_m2      = distributed_system_m2;
  solution_m3      = distributed_system_m3;
}

template <int dim>
void
GDNavierStokesSolver<dim>::write_output_results(const std::string  folder,
                                                const std::string  solutionName,
                                                const unsigned int iter,
                                                const double       time)
{
  TimerOutput::Scope            t(this->computing_timer, "output");
  const MappingQ<dim>           mapping(this->degreeVelocity_,
                              nsparam.femParameters.qmapping_all);
  vorticity_postprocessor<dim>  vorticity;
  qcriterion_postprocessor<dim> ob1;
  std::vector<std::string>      solution_names(dim, "velocity");
  solution_names.push_back("pressure");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(this->dof_handler);
  data_out.add_data_vector(present_solution,
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  data_out.add_data_vector(present_solution, vorticity);
  data_out.add_data_vector(present_solution, ob1);
  Vector<float> subdomain(this->triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = this->triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");
  // data_out.add_data_vector (rot_u,"vorticity");
  data_out.build_patches(mapping, simulationControl.getSubdivision());

  const std::string filename =
    (folder + solutionName + "." + Utilities::int_to_string(iter, 4) + "." +
     Utilities::int_to_string(this->triangulation.locally_owned_subdomain(),
                              4));
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(this->mpi_communicator);
           ++i)
        filenames.push_back(solutionName + "." +
                            Utilities::int_to_string(iter, 4) + "." +
                            Utilities::int_to_string(i, 4) + ".vtu");

      std::string pvtu_filename =
        (solutionName + "." + Utilities::int_to_string(iter, 4) + ".pvtu");
      std::ofstream master_output((folder + pvtu_filename).c_str());

      data_out.write_pvtu_record(master_output, filenames);

      const std::string pvdPrefix = (folder + solutionName);
      pvdhandler.append(time, pvtu_filename);
      std::ofstream pvd_output(pvdPrefix + ".pvd");
      DataOutBase::write_pvd_record(pvd_output, pvdhandler.times_and_names_);
    }
}

template <int dim>
void
GDNavierStokesSolver<dim>::write_output_forces()
{
  TimerOutput::Scope t(this->computing_timer, "output_forces");
  for (unsigned int boundary_id = 0;
       boundary_id < nsparam.boundaryConditions.size;
       ++boundary_id)
    {
      std::string filename = nsparam.forcesParameters.force_output_name + "." +
                             Utilities::int_to_string(boundary_id, 2) + ".dat";
      std::ofstream output(filename.c_str());

      forces_tables[boundary_id].write_text(output);
    }
}

template <int dim>
void
GDNavierStokesSolver<dim>::write_output_torques()
{
  TimerOutput::Scope t(this->computing_timer, "output_torques");
  for (unsigned int boundary_id = 0;
       boundary_id < nsparam.boundaryConditions.size;
       ++boundary_id)
    {
      std::string filename = nsparam.forcesParameters.torque_output_name + "." +
                             Utilities::int_to_string(boundary_id, 2) + ".dat";
      std::ofstream output(filename.c_str());

      torques_tables[boundary_id].write_text(output);
    }
}

// Find the l2 norm of the error between the finite element sol'n and the exact
// sol'n
template <int dim>
double
GDNavierStokesSolver<dim>::calculate_L2_error()
{
  TimerOutput::Scope t(this->computing_timer, "error");

  QGauss<dim>         quadrature_formula(this->degreeQuadrature_ + 1);
  const MappingQ<dim> mapping(this->degreeVelocity_,
                              nsparam.femParameters.qmapping_all);
  FEValues<dim>       fe_values(mapping,
                          this->fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  const unsigned int dofs_per_cell =
    this->fe.dofs_per_cell; // This gives you dofs per cell
  std::vector<types::global_dof_index> local_dof_indices(
    dofs_per_cell); //  Local connectivity

  const unsigned int n_q_points = quadrature_formula.size();

  std::vector<Vector<double>> q_exactSol(n_q_points, Vector<double>(dim + 1));

  std::vector<Tensor<1, dim>> local_velocity_values(n_q_points);
  std::vector<double>         local_pressure_values(n_q_points);


  double l2errorU = 0.;

  // loop over elements
  typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler
                                                          .begin_active(),
                                                 endc = this->dof_handler.end();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values[velocities].get_function_values(present_solution,
                                                    local_velocity_values);
          fe_values[pressure].get_function_values(present_solution,
                                                  local_pressure_values);

          // Retrieve the effective "connectivity matrix" for this element
          cell->get_dof_indices(local_dof_indices);

          // Get the exact solution at all gauss points
          exact_solution->vector_value_list(fe_values.get_quadrature_points(),
                                            q_exactSol);

          for (unsigned int q = 0; q < n_q_points; q++)
            {
              // Find the values of x and u_h (the finite element solution) at
              // the quadrature points
              double ux_sim   = local_velocity_values[q][0];
              double ux_exact = q_exactSol[q][0];

              double uy_sim   = local_velocity_values[q][1];
              double uy_exact = q_exactSol[q][1];

              l2errorU +=
                (ux_sim - ux_exact) * (ux_sim - ux_exact) * fe_values.JxW(q);
              l2errorU +=
                (uy_sim - uy_exact) * (uy_sim - uy_exact) * fe_values.JxW(q);

              if (dim == 3)
                {
                  double uz_sim   = local_velocity_values[q][2];
                  double uz_exact = q_exactSol[q][2];
                  l2errorU += (uz_sim - uz_exact) * (uz_sim - uz_exact) *
                              fe_values.JxW(q);
                }
            }
        }
    }
  l2errorU = Utilities::MPI::sum(l2errorU, this->mpi_communicator);
  return std::sqrt(l2errorU);
}

// kinetic energy calculation
template <int dim>
double
GDNavierStokesSolver<dim>::calculate_average_KE()
{
  TimerOutput::Scope t(this->computing_timer, "KE");

  QGauss<dim>         quadrature_formula(this->degreeQuadrature_ + 1);
  const MappingQ<dim> mapping(this->degreeVelocity_,
                              nsparam.femParameters.qmapping_all);
  FEValues<dim>       fe_values(mapping,
                          this->fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const FEValuesExtractors::Vector velocities(0);
  const unsigned int               n_q_points = quadrature_formula.size();

  std::vector<Tensor<1, dim>> local_velocity_values(n_q_points);
  double                      KEU = 0.0;

  // loop over elements
  typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler
                                                          .begin_active(),
                                                 endc = this->dof_handler.end();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values[velocities].get_function_values(present_solution,
                                                    local_velocity_values);

          for (unsigned int q = 0; q < n_q_points; q++)
            {
              // Find the values of x and u_h (the finite element solution) at
              // the quadrature points
              double ux_sim = local_velocity_values[q][0];
              double uy_sim = local_velocity_values[q][1];

              if (dim == 2)
                {
                  KEU += 0.5 * ((ux_sim) * (ux_sim)*fe_values.JxW(q)) *
                         (1 / globalVolume_);
                  KEU += 0.5 * ((uy_sim) * (uy_sim)*fe_values.JxW(q)) *
                         (1 / globalVolume_);
                }
              else
                {
                  double uz_sim = local_velocity_values[q][2];
                  KEU += 0.5 * ((ux_sim) * (ux_sim)*fe_values.JxW(q)) *
                         (1 / globalVolume_);
                  KEU += 0.5 * ((uy_sim) * (uy_sim)*fe_values.JxW(q)) *
                         (1 / globalVolume_);
                  KEU += 0.5 * ((uz_sim) * (uz_sim)*fe_values.JxW(q)) *
                         (1 / globalVolume_);
                }
            }
        }
    }
  KEU = Utilities::MPI::sum(KEU, this->mpi_communicator);
  return (KEU);
}

// enstrophy calculation
template <int dim>
double
GDNavierStokesSolver<dim>::calculate_average_enstrophy()
{
  TimerOutput::Scope t(this->computing_timer, "Entrosphy");

  QGauss<dim>         quadrature_formula(this->degreeQuadrature_ + 1);
  const MappingQ<dim> mapping(this->degreeVelocity_,
                              nsparam.femParameters.qmapping_all);
  FEValues<dim>       fe_values(mapping,
                          this->fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const FEValuesExtractors::Vector velocities(0);

  const unsigned int n_q_points = quadrature_formula.size();

  std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
  double                      en = 0.0;

  // loop over elements
  typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler
                                                          .begin_active(),
                                                 endc = this->dof_handler.end();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          fe_values[velocities].get_function_gradients(
            present_solution, present_velocity_gradients);

          for (unsigned int q = 0; q < n_q_points; q++)
            {
              // Find the values of gradient of ux and uy (the finite element
              // solution) at the quadrature points
              double ux_y = present_velocity_gradients[q][0][1];
              double uy_x = present_velocity_gradients[q][1][0];

              if (dim == 2)
                {
                  en += 0.5 * (uy_x - ux_y) * (uy_x - ux_y) * fe_values.JxW(q) *
                        (1 / globalVolume_);
                }
              else
                {
                  double uz_y = present_velocity_gradients[q][2][1];
                  double uy_z = present_velocity_gradients[q][1][2];
                  double ux_z = present_velocity_gradients[q][0][2];
                  double uz_x = present_velocity_gradients[q][2][0];
                  en += 0.5 * (uz_y - uy_z) * (uz_y - uy_z) * fe_values.JxW(q) *
                        (1 / globalVolume_);
                  en += 0.5 * (ux_z - uz_x) * (ux_z - uz_x) * fe_values.JxW(q) *
                        (1 / globalVolume_);
                  en += 0.5 * (uy_x - ux_y) * (uy_x - ux_y) * fe_values.JxW(q) *
                        (1 / globalVolume_);
                }
            }
        }
    }
  en = Utilities::MPI::sum(en, this->mpi_communicator);

  return (en);
}
// This is a primitive first implementation that could be greatly improved by
// doing a single pass instead of N boundary passes
template <int dim>
void
GDNavierStokesSolver<dim>::calculate_forces()
{
  TimerOutput::Scope t(this->computing_timer, "calculate_forces");
  double             viscosity = this->nsparam.physicalProperties.viscosity;

  QGauss<dim - 1>     face_quadrature_formula(this->degreeQuadrature_ + 1);
  const MappingQ<dim> mapping(this->degreeVelocity_,
                              nsparam.femParameters.qmapping_all);
  const int           n_q_points = face_quadrature_formula.size();
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);
  std::vector<double>              pressure_values(n_q_points);
  std::vector<Tensor<2, dim>>      velocity_gradients(n_q_points);
  Tensor<1, dim>                   normal_vector;
  Tensor<2, dim>                   fluid_stress;
  Tensor<2, dim>                   fluid_pressure;
  Tensor<1, dim>                   force;

  FEFaceValues<dim> fe_face_values(mapping,
                                   this->fe,
                                   face_quadrature_formula,
                                   update_values | update_quadrature_points |
                                     update_gradients | update_JxW_values |
                                     update_normal_vectors);

  for (unsigned int boundary_id = 0;
       boundary_id < nsparam.boundaryConditions.size;
       ++boundary_id)
    {
      force                                               = 0;
      typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler
                                                              .begin_active(),
                                                     endc =
                                                       this->dof_handler.end();
      for (; cell != endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              for (unsigned int face = 0;
                   face < GeometryInfo<dim>::faces_per_cell;
                   face++)
                {
                  if (cell->face(face)->at_boundary())
                    {
                      fe_face_values.reinit(cell, face);
                      if (cell->face(face)->boundary_id() == boundary_id)
                        {
                          std::vector<Point<dim>> q_points =
                            fe_face_values.get_quadrature_points();
                          fe_face_values[velocities].get_function_gradients(
                            present_solution, velocity_gradients);
                          fe_face_values[pressure].get_function_values(
                            present_solution, pressure_values);
                          for (int q = 0; q < n_q_points; q++)
                            {
                              normal_vector = -fe_face_values.normal_vector(q);
                              for (int d = 0; d < dim; ++d)
                                {
                                  fluid_pressure[d][d] = pressure_values[q];
                                }
                              fluid_stress =
                                viscosity * (velocity_gradients[q] +
                                             transpose(velocity_gradients[q])) -
                                fluid_pressure;
                              force += fluid_stress * normal_vector *
                                       fe_face_values.JxW(q);
                            }
                        }
                    }
                }
            }
        }
      forces_[boundary_id] = Utilities::MPI::sum(force, this->mpi_communicator);
    }

  if (nsparam.forcesParameters.verbosity == Parameters::verbose &&
      this->this_mpi_process == 0)
    {
      std::cout << std::endl;
      TableHandler table;

      for (unsigned int boundary_id = 0;
           boundary_id < nsparam.boundaryConditions.size;
           ++boundary_id)
        {
          table.add_value("Boundary ID", boundary_id);
          table.add_value("f_x", forces_[boundary_id][0]);
          table.add_value("f_y", forces_[boundary_id][1]);
          table.set_precision("f_x",
                              nsparam.forcesParameters.display_precision);
          table.set_precision("f_y",
                              nsparam.forcesParameters.display_precision);
          if (dim == 3)
            {
              table.add_value("f_z", forces_[boundary_id][2]);
              table.set_precision("f_z",
                                  nsparam.forcesParameters.display_precision);
            }
        }
      std::cout << "+------------------------------------------+" << std::endl;
      std::cout << "|  Force  summary                          |" << std::endl;
      std::cout << "+------------------------------------------+" << std::endl;
      table.write_text(std::cout);
    }

  for (unsigned int boundary_id = 0;
       boundary_id < nsparam.boundaryConditions.size;
       ++boundary_id)
    {
      forces_tables[boundary_id].add_value("time", simulationControl.getTime());
      forces_tables[boundary_id].add_value("f_x", forces_[boundary_id][0]);
      forces_tables[boundary_id].add_value("f_y", forces_[boundary_id][1]);
      if (dim == 3)
        forces_tables[boundary_id].add_value("f_z", forces_[boundary_id][2]);
      else
        forces_tables[boundary_id].add_value("f_z", 0.);

      // Precision
      forces_tables[boundary_id].set_precision(
        "f_x", nsparam.forcesParameters.output_precision);
      forces_tables[boundary_id].set_precision(
        "f_y", nsparam.forcesParameters.output_precision);
      forces_tables[boundary_id].set_precision(
        "f_z", nsparam.forcesParameters.output_precision);
      forces_tables[boundary_id].set_precision(
        "time", nsparam.forcesParameters.output_precision);
    }
}

template <int dim>
void
GDNavierStokesSolver<dim>::calculate_torques()
{
  TimerOutput::Scope t(this->computing_timer, "calculate_torques");
  double             viscosity = this->nsparam.physicalProperties.viscosity;

  QGauss<dim - 1>     face_quadrature_formula(this->degreeQuadrature_ + 1);
  const MappingQ<dim> mapping(this->degreeVelocity_,
                              nsparam.femParameters.qmapping_all);
  const int           n_q_points = face_quadrature_formula.size();
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);
  std::vector<double>              pressure_values(n_q_points);
  std::vector<Tensor<2, dim>>      velocity_gradients(n_q_points);
  Tensor<1, dim>                   normal_vector;
  Tensor<2, dim>                   fluid_stress;
  Tensor<2, dim>                   fluid_pressure;

  Tensor<1, dim> force;
  Tensor<1, dim> distance;
  // torque tensor had to be considered in 3D at all time...
  Tensor<1, 3> torque;

  FEFaceValues<dim> fe_face_values(mapping,
                                   this->fe,
                                   face_quadrature_formula,
                                   update_values | update_quadrature_points |
                                     update_gradients | update_JxW_values |
                                     update_normal_vectors);

  for (unsigned int boundary_id = 0;
       boundary_id < nsparam.boundaryConditions.size;
       ++boundary_id)
    {
      torque = 0;
      Point<dim> center_of_rotation =
        nsparam.boundaryConditions.bcFunctions[boundary_id].cor;
      typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler
                                                              .begin_active(),
                                                     endc =
                                                       this->dof_handler.end();
      for (; cell != endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              for (unsigned int face = 0;
                   face < GeometryInfo<dim>::faces_per_cell;
                   face++)
                {
                  if (cell->face(face)->at_boundary())
                    {
                      fe_face_values.reinit(cell, face);
                      if (cell->face(face)->boundary_id() == boundary_id)
                        {
                          std::vector<Point<dim>> q_points =
                            fe_face_values.get_quadrature_points();
                          fe_face_values[velocities].get_function_gradients(
                            present_solution, velocity_gradients);
                          fe_face_values[pressure].get_function_values(
                            present_solution, pressure_values);
                          for (int q = 0; q < n_q_points; q++)
                            {
                              normal_vector = -fe_face_values.normal_vector(q);
                              for (int d = 0; d < dim; ++d)
                                {
                                  fluid_pressure[d][d] = pressure_values[q];
                                }
                              fluid_stress =
                                viscosity * (velocity_gradients[q] +
                                             transpose(velocity_gradients[q])) -
                                fluid_pressure;
                              force = fluid_stress * normal_vector *
                                      fe_face_values.JxW(q);

                              distance = q_points[q] - center_of_rotation;
                              if (dim == 2)
                                {
                                  torque[0] = 0.;
                                  torque[1] = 0.;
                                  torque[2] += distance[0] * force[1] -
                                               distance[1] * force[0];
                                }
                              else if (dim == 3)
                                {
                                  torque[0] += distance[1] * force[2] -
                                               distance[2] * force[1];
                                  torque[1] += distance[2] * force[0] -
                                               distance[0] * force[2];
                                  torque[2] += distance[0] * force[1] -
                                               distance[1] * force[0];
                                }
                            }
                        }
                    }
                }
            }
        }
      torques_[boundary_id] =
        Utilities::MPI::sum(torque, this->mpi_communicator);
    }

  if (nsparam.forcesParameters.verbosity == Parameters::verbose &&
      this->this_mpi_process == 0)
    {
      this->pcout << std::endl;
      TableHandler table;

      for (unsigned int boundary_id = 0;
           boundary_id < nsparam.boundaryConditions.size;
           ++boundary_id)
        {
          table.add_value("Boundary ID", boundary_id);
          table.add_value("T_x", torques_[boundary_id][0]);
          table.add_value("T_y", torques_[boundary_id][1]);
          table.set_precision("T_x",
                              nsparam.forcesParameters.display_precision);
          table.set_precision("T_y",
                              nsparam.forcesParameters.display_precision);
          table.add_value("T_z", torques_[boundary_id][2]);
          table.set_precision("T_z",
                              nsparam.forcesParameters.display_precision);
        }

      std::cout << "+------------------------------------------+" << std::endl;
      std::cout << "|  Torque summary                          |" << std::endl;
      std::cout << "+------------------------------------------+" << std::endl;
      table.write_text(std::cout);
    }

  for (unsigned int boundary_id = 0;
       boundary_id < nsparam.boundaryConditions.size;
       ++boundary_id)
    {
      torques_tables[boundary_id].add_value("time",
                                            simulationControl.getTime());
      torques_tables[boundary_id].add_value("T_x", torques_[boundary_id][0]);
      torques_tables[boundary_id].add_value("T_y", torques_[boundary_id][1]);
      torques_tables[boundary_id].add_value("T_z", torques_[boundary_id][2]);

      // Precision
      torques_tables[boundary_id].set_precision(
        "T_x", nsparam.forcesParameters.output_precision);
      torques_tables[boundary_id].set_precision(
        "T_y", nsparam.forcesParameters.output_precision);
      torques_tables[boundary_id].set_precision(
        "T_z", nsparam.forcesParameters.output_precision);
      torques_tables[boundary_id].set_precision(
        "time", nsparam.forcesParameters.output_precision);
    }
}

/*
 * Reads a CFD Mesh from a GMSH file or generates a pre-defined primitive
 */
template <int dim>
void
GDNavierStokesSolver<dim>::read_mesh()
{
  // GMSH input
  if (nsparam.mesh.type == Parameters::Mesh::gmsh)
    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(this->triangulation);
      std::ifstream input_file(this->nsparam.mesh.fileName);
      grid_in.read_msh(input_file);
      this->set_periodicity();
    }
  // Primitive input
  else if (nsparam.mesh.type == Parameters::Mesh::primitive)
    {
      const int initialSize = this->nsparam.mesh.initialRefinement;

      if (nsparam.mesh.primitiveType == Parameters::Mesh::hyper_cube)
        {
          GridGenerator::hyper_cube(this->triangulation,
                                    this->nsparam.mesh.hc_left,
                                    this->nsparam.mesh.hc_right,
                                    this->nsparam.mesh.colorize);
        }
      else if (nsparam.mesh.primitiveType == Parameters::Mesh::hyper_shell)
        {
          Point<dim> circleCenter;
          if (dim == 2)
            circleCenter = Point<dim>(0, 0);

          if (dim == 3)
            circleCenter = Point<dim>(0, 0, 0);

          GridGenerator::hyper_shell(this->triangulation,
                                     circleCenter,
                                     this->nsparam.mesh.hs_inner_radius,
                                     this->nsparam.mesh.hs_outer_radius,
                                     4,
                                     this->nsparam.mesh.colorize);
        }
      else
        {
          throw std::runtime_error(
            "Unsupported primitive type - mesh will not be created");
        }
      this->set_periodicity();
      this->triangulation.refine_global(initialSize);
    }
  else
    throw std::runtime_error(
      "Unsupported mesh type - mesh will not be created");
}

/*
 * Attaches manifold to the boundaries of the mesh
 */
template <int dim>
void
GDNavierStokesSolver<dim>::create_manifolds()
{
  Parameters::Manifolds manifolds = nsparam.manifoldsParameters;

  for (unsigned int i = 0; i < manifolds.types.size(); ++i)
    {
      if (manifolds.types[i] == Parameters::Manifolds::spherical)
        {
          Point<dim> circleCenter;
          circleCenter = Point<dim>(manifolds.arg1[i], manifolds.arg2[i]);
          static const SphericalManifold<dim> manifold_description(
            circleCenter);
          this->triangulation.set_manifold(manifolds.id[i],
                                           manifold_description);
          this->triangulation.set_all_manifold_ids_on_boundary(manifolds.id[i],
                                                               manifolds.id[i]);
        }
      else if (manifolds.types[i] == Parameters::Manifolds::none)
        {}
      else
        throw std::runtime_error("Unsupported manifolds type");
    }
}


template <int dim>
void
GDNavierStokesSolver<dim>::postprocess(bool firstIter)
{
  if (simulationControl.isOutputIteration())
    write_output_results(simulationControl.getOutputFolder(),
                         simulationControl.getOuputName(),
                         simulationControl.getIter(),
                         simulationControl.getTime());

  if (nsparam.postProcessingParameters.calculate_enstrophy)
    {
      double enstrophy = this->calculate_average_enstrophy();
      this->enstrophy_table.add_value("time", simulationControl.getTime());
      this->enstrophy_table.add_value("enstrophy", enstrophy);
      if (nsparam.postProcessingParameters.verbosity == Parameters::verbose)
        {
          this->pcout << "Enstrophy  : " << enstrophy << std::endl;
        }
    }

  if (nsparam.postProcessingParameters.calculate_kinetic_energy)
    {
      double kE = this->calculate_average_KE();
      this->kinetic_energy_table.add_value("time", simulationControl.getTime());
      this->kinetic_energy_table.add_value("kinetic-energy", kE);
      if (nsparam.postProcessingParameters.verbosity == Parameters::verbose)
        {
          this->pcout << "Kinetic energy : " << kE << std::endl;
        }
    }

  if (!firstIter)
    {
      // Calculate forces on the boundary conditions
      if (nsparam.forcesParameters.calculate_force)
        {
          if (simulationControl.getIter() %
                nsparam.forcesParameters.calculation_frequency ==
              0)
            calculate_forces();
          if (simulationControl.getIter() %
                nsparam.forcesParameters.output_frequency ==
              0)
            write_output_forces();
        }

      // Calculate torques on the boundary conditions
      if (nsparam.forcesParameters.calculate_torque)
        {
          if (simulationControl.getIter() %
                nsparam.forcesParameters.calculation_frequency ==
              0)
            calculate_torques();
          if (simulationControl.getIter() %
                nsparam.forcesParameters.output_frequency ==
              0)
            write_output_torques();
        }

      // Calculate error with respect to analytical solution
      if (nsparam.analyticalSolution->calculate_error())
        {
          // Update the time of the exact solution to the actual time
          exact_solution->set_time(simulationControl.getTime());
          const double error = this->calculate_L2_error();
          if (simulationControl.getMethod() ==
              Parameters::SimulationControl::steady)
            {
              table.add_value("cells",
                              this->triangulation.n_global_active_cells());
              table.add_value("error", error);
            }
          else
            {
              table.add_value("time", simulationControl.getTime());
              table.add_value("error", error);
            }
          if (nsparam.analyticalSolution->verbosity == Parameters::verbose)
            {
              this->pcout << "L2 error : " << error << std::endl;
            }
        }
    }
}

template <int dim>
void
GDNavierStokesSolver<dim>::finish_simulation()
{
  if (nsparam.forcesParameters.calculate_force)
    {
      write_output_forces();
    }

  if (nsparam.forcesParameters.calculate_torque)
    {
      write_output_torques();
    }
  if (nsparam.analyticalSolution->calculate_error())
    {
      if (simulationControl.getMethod() ==
          Parameters::SimulationControl::steady)
        {
          table.omit_column_from_convergence_rate_evaluation("cells");
          table.evaluate_all_convergence_rates(
            ConvergenceTable::reduction_rate_log2);
        }
      table.set_scientific("error", true);

      if (this->this_mpi_process == 0)
        {
          table.write_text(std::cout);
          std::string filename =
            nsparam.analyticalSolution->get_filename() + ".dat";
          std::ofstream output(filename.c_str());
          table.write_text(output);
        }
    }
}

/*
 * Generic CFD Solver application
 * Handles the majority of the cases for the GD-NS solver
 */
template <int dim>
void
GDNavierStokesSolver<dim>::solve()
{
  this->read_mesh();
  this->create_manifolds();

  this->setup_dofs();
  this->set_initial_condition(this->nsparam.initialCondition->type,
                              this->nsparam.restartParameters.restart);

  while (this->simulationControl.integrate())
    {
      printTime(this->pcout, this->simulationControl);
      if (!this->simulationControl.firstIter())
        {
          this->refine_mesh();
        }
      this->iterate(this->simulationControl.firstIter());
      this->postprocess(false);
      this->finish_time_step();
    }

  this->finish_simulation();
}

#endif
