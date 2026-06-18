#include "EulerianFluid3D.h"

#include "PhaseField/TwoPhaseFlow.h"

namespace dyno
{
	IMPLEMENT_TCLASS(EulerianFluid3D, TDataType)

	template<typename TDataType>
	EulerianFluid3D<TDataType>::EulerianFluid3D()
		: Node()
	{
		this->statePhaseField()->allocate();

		auto flowsolver = std::make_shared<TwoPhaseFlow<TDataType>>();
		this->stateTimeStep()->connect(flowsolver->inTimeStep());
		this->stateMass()->connect(flowsolver->inMass());
		this->stateVelocity()->connect(flowsolver->inVelocity());
		this->statePhaseField()->connect(flowsolver->inPhaseField());
		this->animationPipeline()->pushModule(flowsolver);
	}

	template<typename TDataType>
	EulerianFluid3D<TDataType>::~EulerianFluid3D()
	{
	}

	template<typename TDataType>
	void EulerianFluid3D<TDataType>::resetStates()
	{
		auto pf = this->statePhaseField()->getDataPtr();

		auto dim = this->varDimension()->getValue();

		this->stateVelocity()->resize(dim.x, dim.y, dim.z);
		this->stateVelocity()->reset();

		this->stateMass()->resize(dim.x, dim.y, dim.z);
		this->stateMass()->reset();

		CArray3D<Real> fraction(dim.x, dim.y, dim.z);
		CArray3D<Coord> initial_vel(dim.x, dim.y, dim.z);

		for(int i = 0; i < dim.x; i++)
			for (int j = 0; j <dim.y; j++)
			{
				for (int k = 0; k < dim.z; k++)
				{
					fraction(i, j, k) = (i < 32 && j < 32) ? 1.0f : 0.0f;
					initial_vel(i, j, k) = Coord(0, 0, 0);
				}
			}

		this->stateMass()->assign(fraction);
		this->stateVelocity()->assign(initial_vel);

		pf->initialize(dim.x, dim.y, dim.z);
		pf->volumeFraction().assign(this->stateMass()->constData());

		fraction.clear();
		initial_vel.clear();
	}

	template<typename TDataType>
	void EulerianFluid3D<TDataType>::postUpdateStates()
	{
		auto phase = this->statePhaseField()->getDataPtr();

		phase->volumeFraction().assign(this->stateMass()->constData());
	}

	DEFINE_CLASS(EulerianFluid3D);
}