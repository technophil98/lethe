/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 -  by the Lethe authors
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
 * Author: Bruno Blais, Polytechnique Montreal, 2019 -
 */

#ifndef lethe_navier_stokes_solver_parameters_h
#define lethe_navier_stokes_solver_parameters_h

#include <core/boundary_conditions.h>
#include <core/manifolds.h>
#include <core/nitsche.h>
#include <core/parameters.h>
#include <core/parameters_multiphysics.h>

#include <solvers/analytical_solutions.h>
#include <solvers/initial_conditions.h>
#include <solvers/source_terms.h>

#include <dem/dem_solver_parameters.h>
#include <fem-dem/parameters_cfd_dem.h>

template <int dim>
class SimulationParameters
{
public:
  Parameters::Testing                               test;
  Parameters::LinearSolver                          linear_solver;
  Parameters::NonLinearSolver                       non_linear_solver;
  Parameters::MeshAdaptation                        mesh_adaptation;
  Parameters::Mesh                                  mesh;
  std::shared_ptr<Parameters::MeshBoxRefinement>    mesh_box_refinement;
  std::shared_ptr<Parameters::Nitsche<dim>>         nitsche;
  Parameters::PhysicalProperties                    physical_properties;
  Parameters::SimulationControl                     simulation_control;
  Parameters::Timer                                 timer;
  Parameters::FEM                                   fem_parameters;
  Parameters::Forces                                forces_parameters;
  Parameters::PostProcessing                        post_processing;
  Parameters::Restart                               restart_parameters;
  Parameters::Manifolds                             manifolds_parameters;
  BoundaryConditions::NSBoundaryConditions<dim>     boundary_conditions;
  BoundaryConditions::HTBoundaryConditions<dim>     boundary_conditions_ht;
  BoundaryConditions::TracerBoundaryConditions<dim> boundary_conditions_tracer;
  Parameters::InitialConditions<dim> *              initial_condition;
  AnalyticalSolutions::AnalyticalSolution<dim> *    analytical_solution;
  SourceTerms::SourceTerm<dim> *                    source_term;
  Parameters::VelocitySource                        velocity_sources;
  std::shared_ptr<Parameters::IBParticles<dim>>     particlesParameters;
  Parameters::DynamicFlowControl                    flow_control;
  Parameters::NonNewtonian                          non_newtonian;
  Parameters::InterfaceSharpening                   interface_sharpening;
  Parameters::Multiphysics                          multiphysics;

  void
  declare(ParameterHandler &prm)
  {
    Parameters::SimulationControl::declare_parameters(prm);
    physical_properties.declare_parameters(prm);
    Parameters::Mesh::declare_parameters(prm);
    nitsche = std::make_shared<Parameters::Nitsche<dim>>();
    nitsche->declare_parameters(prm);
    Parameters::Restart::declare_parameters(prm);
    boundary_conditions.declare_parameters(prm);
    boundary_conditions_ht.declare_parameters(prm);
    boundary_conditions_tracer.declare_parameters(prm);


    initial_condition = new Parameters::InitialConditions<dim>;
    initial_condition->declare_parameters(prm);

    Parameters::FEM::declare_parameters(prm);
    Parameters::Multiphysics::declare_parameters(prm);
    Parameters::Timer::declare_parameters(prm);
    Parameters::Forces::declare_parameters(prm);
    Parameters::MeshAdaptation::declare_parameters(prm);
    mesh_box_refinement = std::make_shared<Parameters::MeshBoxRefinement>();
    mesh_box_refinement->declare_parameters(prm);

    Parameters::NonLinearSolver::declare_parameters(prm);
    Parameters::LinearSolver::declare_parameters(prm);
    Parameters::PostProcessing::declare_parameters(prm);
    Parameters::DynamicFlowControl ::declare_parameters(prm);
    particlesParameters = std::make_shared<Parameters::IBParticles<dim>>();
    particlesParameters->declare_parameters(prm);
    manifolds_parameters.declare_parameters(prm);
    non_newtonian.declare_parameters(prm);
    interface_sharpening.declare_parameters(prm);

    analytical_solution = new AnalyticalSolutions::AnalyticalSolution<dim>;
    analytical_solution->declare_parameters(prm);
    source_term = new SourceTerms::SourceTerm<dim>;
    source_term->declare_parameters(prm);
    Parameters::Testing::declare_parameters(prm);

    Parameters::VelocitySource::declare_parameters(prm);

    multiphysics.declare_parameters(prm);
  }

  void
  parse(ParameterHandler &prm)
  {
    test.parse_parameters(prm);
    linear_solver.parse_parameters(prm);
    non_linear_solver.parse_parameters(prm);
    mesh_adaptation.parse_parameters(prm);
    mesh.parse_parameters(prm);
    mesh_box_refinement->parse_parameters(prm);
    nitsche->parse_parameters(prm);
    physical_properties.parse_parameters(prm);
    multiphysics.parse_parameters(prm);
    timer.parse_parameters(prm);
    fem_parameters.parse_parameters(prm);
    forces_parameters.parse_parameters(prm);
    post_processing.parse_parameters(prm);
    flow_control.parse_parameters(prm);
    non_newtonian.parse_parameters(prm);
    interface_sharpening.parse_parameters(prm);
    restart_parameters.parse_parameters(prm);
    boundary_conditions.parse_parameters(prm);
    boundary_conditions_ht.parse_parameters(prm);
    boundary_conditions_tracer.parse_parameters(prm);
    manifolds_parameters.parse_parameters(prm);
    initial_condition->parse_parameters(prm);
    analytical_solution->parse_parameters(prm);
    source_term->parse_parameters(prm);
    simulation_control.parse_parameters(prm);
    velocity_sources.parse_parameters(prm);
    particlesParameters->parse_parameters(prm);

    multiphysics.parse_parameters(prm);

    // Check consistency of parameters parsed in different subsections
    if (multiphysics.VOF && physical_properties.number_of_fluids != 2)
      {
        throw std::logic_error(
          "Inconsistency in .prm!\n with VOF = true\n use: number of fluids = 2");
      }
  }
};

#endif
