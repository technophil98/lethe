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
 * Author: Bruno Blais, Polytechnique Montreal, 2019-
 */

/*
 * This file defines the parameters in the parameter namespace
 * that pertain to multiphysics simulations
 */


#ifndef lethe_parameters_multiphysics_h
#define lethe_parameters_multiphysics_h

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

namespace Parameters
{
  struct Multiphysics
  {
    bool fluid_dynamics;
    bool heat_transfer;
    bool tracer;
    bool VOF;
    bool interface_sharpening;
    bool buoyancy_force;

    // subparameter for heat_transfer
    bool viscous_dissipation;

    // subparameter for free_surface
    bool conservation_monitoring;
    int  fluid_index;

    static void
    declare_parameters(ParameterHandler &prm);
    void
    parse_parameters(ParameterHandler &prm);
  }; // namespace Parameters

} // namespace Parameters
#endif
