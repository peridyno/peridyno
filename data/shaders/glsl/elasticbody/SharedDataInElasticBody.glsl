#ifndef ELASTICBODY_H
#define ELASTICBODY_H

struct SolverParams
{
	float damping;

	float dt;
	vec3 gravity;
};

#endif