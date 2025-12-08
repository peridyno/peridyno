cmake_minimum_required(VERSION 3.10)

option(PERIDYNO_LIBRARY_PARTICLESYSTEM "Enable binding the particle system library" ON)
option(PERIDYNO_LIBRARY_PERIDYNAMICS "Enable binding the peridynamics library" ON)
option(PERIDYNO_LIBRARY_RIGIDBODY "Enable binding the rigid body library" ON)
option(PERIDYNO_LIBRARY_VOLUME "Enable binding the volume library" ON)
option(PERIDYNO_LIBRARY_HEIGHTFIELD "Enable binding the Height Field library" ON)
option(PERIDYNO_LIBRARY_SEMIANALYTICALSCHEME "Enable binding the semi-analycial scheme library" ON)

option(PERIDYNO_LIBRARY_DUALPARTICLESYSTEM "Enable binding the dual-particle scheme library" ON)
option(PERIDYNO_LIBRARY_MULTIPHYSICS "Enable building the multiphysics library" ON)
option(PERIDYNO_LIBRARY_EULERFLUID "Enable binding the Euler fluid library" ON)

if(PERIDYNO_LIBRARY_PARTICLESYSTEM)
    set(LIB_DEPENDENCY Core IO Framework Topology Modeling GLRenderEngine)
    build_library(ParticleSystem LIB_DEPENDENCY)
endif()

if(PERIDYNO_LIBRARY_PERIDYNAMICS)
    set(LIB_DEPENDENCY IO Topology ParticleSystem)
    build_library(Peridynamics LIB_DEPENDENCY)
endif()

if(PERIDYNO_LIBRARY_RIGIDBODY)
    set(LIB_DEPENDENCY Core IO Framework Topology GLRenderEngine Modeling)
    build_library(RigidBody LIB_DEPENDENCY)
endif()

if(PERIDYNO_LIBRARY_VOLUME)
    set(LIB_DEPENDENCY Core IO Framework Topology Modeling GLRenderEngine)
    build_library(Volume LIB_DEPENDENCY)
endif()


if(PERIDYNO_LIBRARY_HEIGHTFIELD)
    set(LIB_DEPENDENCY Core IO Framework Topology RigidBody GLRenderEngine cufft)
    append_library("cufft")
    build_library(HeightField LIB_DEPENDENCY)
endif()

if(PERIDYNO_LIBRARY_SEMIANALYTICALSCHEME)
    set(LIB_DEPENDENCY ParticleSystem)
    build_library(SemiAnalyticalScheme LIB_DEPENDENCY)
endif()

if(PERIDYNO_LIBRARY_MULTIPHYSICS)
	set(LIB_DEPENDENCY ParticleSystem Peridynamics RigidBody Volume)
    build_library(Multiphysics LIB_DEPENDENCY)
endif()

if(PERIDYNO_LIBRARY_DUALPARTICLESYSTEM)
    set(LIB_DEPENDENCY SemiAnalyticalScheme)
    build_library(DualParticleSystem LIB_DEPENDENCY)
endif()

if(PERIDYNO_LIBRARY_EULERFLUID)
    set(LIB_DEPENDENCY Core IO Framework Topology Modeling GLRenderEngine Volume)
    build_library(EulerFluid LIB_DEPENDENCY)
endif()