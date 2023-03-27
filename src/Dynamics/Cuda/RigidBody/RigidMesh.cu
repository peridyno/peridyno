#include "RigidMesh.h"

namespace dyno
{
	template<typename TDataType>
	RigidMesh<TDataType>::RigidMesh()
		: RigidBody<TDataType>()
	{
		this->stateEnvelope()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->stateInitialEnvelope()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		this->stateMesh()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->stateInitialMesh()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
	}

	template<typename TDataType>
	RigidMesh<TDataType>::~RigidMesh()
	{

	}

	template<typename TDataType>
	void RigidMesh<TDataType>::resetStates()
	{
		Coord center(0);

		if (this->varEnvelopeName()->getDataPtr()->string() != "") {
			auto envlope = this->stateInitialEnvelope()->getDataPtr();

			envlope->loadObjFile(this->varEnvelopeName()->getDataPtr()->string());
			envlope->scale(this->varScale()->getData());

			auto points = envlope->getPoints();

			Reduction<Coord> reduce;
			Coord lo = reduce.minimum(points.begin(), points.size());
			Coord hi = reduce.maximum(points.begin(), points.size());

			Real lx = hi.x - lo.x;
			Real ly = hi.y - lo.y;
			Real lz = hi.z - lo.z;

			center = 0.5 * (lo + hi);

			Real rho = this->varDensity()->getData();
			Real mass = rho * lx * ly * lz;
			Matrix inertia = 1.0f / 12.0f * mass
				* Mat3f(ly * ly + lz * lz, 0, 0,
					0, lx * lx + lz * lz, 0,
					0, 0, lx * lx + ly * ly);

			this->stateMass()->setValue(mass);
			this->stateCenter()->setValue(center);
			this->stateInertia()->setValue(inertia);
			this->stateInitialInertia()->setValue(inertia);

			envlope->translate(-center);

			this->stateEnvelope()->getDataPtr()->copyFrom(*envlope);
			this->stateEnvelope()->getDataPtr()->translate(center);
		}

		if (this->varMeshName()->getDataPtr()->string() != "") {
			auto mesh = this->stateInitialMesh()->getDataPtr();

			mesh->loadObjFile(this->varMeshName()->getDataPtr()->string());
			mesh->scale(this->varScale()->getData());

			mesh->translate(-center);
			this->stateMesh()->getDataPtr()->copyFrom(*mesh);
			this->stateMesh()->getDataPtr()->translate(center);
		}

		RigidBody<TDataType>::resetStates();
	}

	template<typename TDataType>
	void RigidMesh<TDataType>::updateStates()
	{
		RigidBody<TDataType>::updateStates();

		auto center = this->stateCenter()->getData();
		auto quat = this->stateQuaternion()->getData();
 
		auto envlope = this->stateEnvelope()->getDataPtr();
		envlope->copyFrom(this->stateInitialEnvelope()->getData());
		envlope->rotate(quat);
		envlope->translate(center);

		auto mesh = this->stateMesh()->getDataPtr();
		mesh->copyFrom(this->stateInitialMesh()->getData());
		mesh->rotate(quat);
		mesh->translate(center);
	}

	DEFINE_CLASS(RigidMesh);
}