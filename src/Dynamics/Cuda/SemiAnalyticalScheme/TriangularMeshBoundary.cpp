#include "TriangularMeshBoundary.h"

#include "Collision/NeighborTriangleQuery.h"

#include "TriangularMeshConstraint.h"

#include "Auxiliary/DataSource.h"

namespace dyno
{
	IMPLEMENT_TCLASS(TriangularMeshBoundary, TDataType)

	template<typename TDataType>
	TriangularMeshBoundary<TDataType>::TriangularMeshBoundary()
		: Node()
	{
		auto smoothingLength = std::make_shared<FloatingNumber<TDataType>>();
		smoothingLength->setName("Smoothing Length");
		smoothingLength->varValue()->setValue(Real(0.012));
		this->animationPipeline()->pushModule(smoothingLength);

		//triangle neighbor
		auto nbrQueryTri = std::make_shared<NeighborTriangleQuery<TDataType>>();
		smoothingLength->outFloating()->connect(nbrQueryTri->inRadius());
		this->statePosition()->connect(nbrQueryTri->inPosition());
		this->inTriangleSet()->connect(nbrQueryTri->inTriangleSet());
		this->animationPipeline()->pushModule(nbrQueryTri);

		//mesh collision
		auto meshCollision = std::make_shared<TriangularMeshConstraint<TDataType>>();
		this->stateTimeStep()->connect(meshCollision->inTimeStep());
		this->statePosition()->connect(meshCollision->inPosition());
		this->stateVelocity()->connect(meshCollision->inVelocity());
		this->inTriangleSet()->connect(meshCollision->inTriangleSet());
		nbrQueryTri->outNeighborIds()->connect(meshCollision->inTriangleNeighborIds());
		this->animationPipeline()->pushModule(meshCollision);

		this->varNormalFriction()->attach(
			std::make_shared<FCallBackFunc>(
				[=]() {
					meshCollision->varNormalFriction()->setValue(this->varNormalFriction()->getValue());
				})
		);

		this->varTangentialFriction()->attach(
			std::make_shared<FCallBackFunc>(
				[=]() {
					meshCollision->varTangentialFriction()->setValue(this->varTangentialFriction()->getValue());
				})
		);

		this->varThickness()->attach(
			std::make_shared<FCallBackFunc>(
				[=]() {
					Real thickness = this->varThickness()->getValue();
					smoothingLength->varValue()->setValue(thickness);
					meshCollision->varThickness()->setValue(thickness);
				})
		);
	}

	template<typename TDataType>
	TriangularMeshBoundary<TDataType>::~TriangularMeshBoundary()
	{
	}

	template<typename TDataType>
	void TriangularMeshBoundary<TDataType>::preUpdateStates()
	{
		auto& particleSystems = this->getParticleSystems();

		if (particleSystems.size() == 0)
			return;

		int new_num = 0;
		for (int i = 0; i < particleSystems.size(); i++) {
			new_num += particleSystems[i]->statePosition()->size();
		}

		if (new_num <= 0)
			return;

		int cur_num = this->statePosition()->size();

		if (new_num != cur_num)
		{
			this->statePosition()->resize(new_num);
			this->stateVelocity()->resize(new_num);
		}

		auto& new_pos = this->statePosition()->getData();
		auto& new_vel = this->stateVelocity()->getData();

		int offset = 0;
		for (int i = 0; i < particleSystems.size(); i++)//update particle system
		{
			DArray<Coord>& points = particleSystems[i]->statePosition()->getData();
			DArray<Coord>& vels = particleSystems[i]->stateVelocity()->getData();
			int num = points.size();

			new_pos.assign(points, num, offset);
			new_vel.assign(vels, num, offset);

			offset += num;
		}
	}

	template<typename TDataType>
	void TriangularMeshBoundary<TDataType>::updateStates()
	{
		Node::updateStates();
	}

	template<typename TDataType>
	void TriangularMeshBoundary<TDataType>::postUpdateStates()
	{
		auto& particleSystems = this->getParticleSystems();

		if (particleSystems.size() <= 0 || this->statePosition()->size() <= 0)
			return;

		auto& new_pos = this->statePosition()->getData();
		auto& new_vel = this->stateVelocity()->getData();

		uint offset = 0;
		for (int i = 0; i < particleSystems.size(); i++)//extend current particles
		{
			DArray<Coord>& points = particleSystems[i]->statePosition()->getData();
			DArray<Coord>& vels = particleSystems[i]->stateVelocity()->getData();

			int num = points.size();

			points.assign(new_pos, num, 0, offset);
			vels.assign(new_vel, num, 0, offset);

			offset += num;
		}
	}

	DEFINE_CLASS(TriangularMeshBoundary);
}