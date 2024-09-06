#include "ParticleIntegrator.h"

namespace dyno
{
	IMPLEMENT_CLASS(ParticleIntegrator)


	struct FluidSolverParams
	{
		float gravity;
		float dt;
	};

	ParticleIntegrator::ParticleIntegrator()
		: Module()
	{
		this->addKernel(
			"ParticleIntegrator",
			std::make_shared<VkProgram>(
				BUFFER(Vec3f),    // inout: position
				BUFFER(Vec3f),    // inout: velocity
				UNIFORM(FluidSolverParams), // in: array info
				CONSTANT(uint)		// in: particle number
				)
		);
		kernel("ParticleIntegrator")->load(getAssetPath() + "shaders/glsl/particlesystem/ParticleIntegrator.comp.spv");
	}

	void ParticleIntegrator::updateImpl()
	{
		float dt = this->inTimeStep()->getValue();

		auto& pos = this->inPosition()->getData();
		auto& vel = this->inVelocity()->getData();

		uint num = pos.size();

		FluidSolverParams params;
		params.dt = dt;
		params.gravity = -9.8f;

		VkUniform<FluidSolverParams> uniform;
		uniform.setValue(params);

		VkConstant<uint> constNum(pos.size());

		kernel("ParticleIntegrator")->flush(
			vkDispatchSize(num, 64),
			pos.handle(),
			vel.handle(),
			&uniform,
			&constNum);
	}
}