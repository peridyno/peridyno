#ifndef SHAREDDATAINPARTICLESYSTEM_H
#define SHAREDDATAINPARTICLESYSTEM_H

struct FluidSolverParams
{
	float gravity;
	float dt;
};

struct HashGrid
{
	uint nx;
	uint ny;
	uint nz;
	uint total;

	float h;
};

#endif