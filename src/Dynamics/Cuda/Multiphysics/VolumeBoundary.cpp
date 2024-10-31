#include "VolumeBoundary.h"

#include "ParticleSystem/Module/BoundaryConstraint.h"

namespace dyno
{
	template<typename TDataType>
	VolumeBoundary<TDataType>::VolumeBoundary()
		: Node()
	{
		mBoundaryConstraint = std::make_shared<BoundaryConstraint<TDataType>>();

		this->varNormalFriction()->attach(
			std::make_shared<FCallBackFunc>(
				[=]() {
					mBoundaryConstraint->varNormalFriction()->setValue(this->varNormalFriction()->getValue());
				}));

		this->varTangentialFriction()->attach(
			std::make_shared<FCallBackFunc>(
				[=]() {
					mBoundaryConstraint->varTangentialFriction()->setValue(this->varTangentialFriction()->getValue());
				}));

		this->varNormalFriction()->setValue(0.95f);
		this->varTangentialFriction()->setValue(0.0f);
	}

	template<typename TDataType>
	VolumeBoundary<TDataType>::~VolumeBoundary()
	{
	}

	template<typename TDataType>
	void VolumeBoundary<TDataType>::updateVolume()
	{
		Real dt = this->stateTimeStep()->getValue();

		auto volumes = this->getVolumes();

		for (size_t t = 0; t < volumes.size(); t++)
		{
			auto levelset = volumes[t]->stateLevelSet()->getDataPtr();

			auto pSys = this->getParticleSystems();

			for (int i = 0; i < pSys.size(); i++)
			{
				auto posFd = pSys[i]->statePosition();
				auto velFd = pSys[i]->stateVelocity();

				if(!posFd->isEmpty() && !velFd->isEmpty())
					mBoundaryConstraint->constrain(posFd->getData(), velFd->getData(), levelset->getSDF(), dt);
			}

			auto triSys = this->getTriangularSystems();
			for (int i = 0; i < triSys.size(); i++)
			{
				auto posFd = triSys[i]->statePosition();
				auto velFd = triSys[i]->stateVelocity();

				if (!posFd->isEmpty() && !velFd->isEmpty())
					mBoundaryConstraint->constrain(posFd->getData(), velFd->getData(), levelset->getSDF(), dt);
			}

			auto tetSys = this->getTetrahedralSystems();
			for (int i = 0; i < tetSys.size(); i++)
			{
				auto posFd = tetSys[i]->statePosition();
				auto velFd = tetSys[i]->stateVelocity();

				if (!posFd->isEmpty() && !velFd->isEmpty())
					mBoundaryConstraint->constrain(posFd->getData(), velFd->getData(), levelset->getSDF(), dt);
			}
		}
	}

	template<typename TDataType>
	void VolumeBoundary<TDataType>::updateStates()
	{
		updateVolume();
	}

	DEFINE_CLASS(VolumeBoundary);
}