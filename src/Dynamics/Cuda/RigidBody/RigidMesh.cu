#include "RigidMesh.h"

#include <GLSurfaceVisualModule.h>

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

		this->varDensity()->setRange(1.0f, 10000.0f);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&RigidMesh<TDataType>::transform, this));
		this->varLocation()->attach(callback);
		this->varScale()->attach(callback);
		this->varRotation()->attach(callback);

		auto rigidMeshRender = std::make_shared<GLSurfaceVisualModule>();
		rigidMeshRender->setColor(Vec3f(0.8, 0.8, 0.8));
		this->stateMesh()->promoteOuput()->connect(rigidMeshRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(rigidMeshRender);
	}

	template<typename TDataType>
	RigidMesh<TDataType>::~RigidMesh()
	{

	}

	template<typename TDataType>
	void RigidMesh<TDataType>::resetStates()
	{
		Coord location = this->varLocation()->getData();
		Coord rot = this->varRotation()->getData();
		Coord scale = this->varScale()->getData();

		dyno::Quat<Real> quat = dyno::Quat<Real>(M_PI * rot[0] / 180, Coord(1, 0, 0))
			* dyno::Quat<Real>(M_PI * rot[1] / 180, Coord(0, 1, 0))
			* dyno::Quat<Real>(M_PI * rot[2] / 180, Coord(0, 0, 1));

		Coord center(0);

		if (this->varEnvelopeName()->getDataPtr()->string() != "") {
			auto initEnvlope = this->stateInitialEnvelope()->getDataPtr();

			initEnvlope->loadObjFile(this->varEnvelopeName()->getDataPtr()->string());

			auto points = initEnvlope->getPoints();

			Reduction<Coord> reduce;
			Coord lo = reduce.minimum(points.begin(), points.size());
			Coord hi = reduce.maximum(points.begin(), points.size());

			center = 0.5f * (hi + lo);

			initEnvlope->translate(-center);

			auto curEnvlope = this->stateEnvelope()->getDataPtr();
			curEnvlope->copyFrom(*initEnvlope);
			curEnvlope->scale(scale);
			curEnvlope->rotate(quat);
			curEnvlope->translate(location);

			Real lx = hi.x - lo.x;
			Real ly = hi.y - lo.y;
			Real lz = hi.z - lo.z;

			Real rho = this->varDensity()->getData();
			Real mass = rho * lx * ly * lz;
			Matrix inertia = 1.0f / 12.0f * mass
				* Mat3f(ly * ly + lz * lz, 0, 0,
					0, lx * lx + lz * lz, 0,
					0, 0, lx * lx + ly * ly);

			this->stateMass()->setValue(mass);
			this->stateCenter()->setValue(location);
			this->stateVelocity()->setValue(Vec3f(0));
			this->stateAngularVelocity()->setValue(Vec3f(0));
			this->stateInertia()->setValue(inertia);
			this->stateQuaternion()->setValue(quat);
			this->stateInitialInertia()->setValue(inertia);
		}

		if (this->varMeshName()->getDataPtr()->string() != "") {
			auto initMesh = this->stateInitialMesh()->getDataPtr();

			initMesh->loadObjFile(this->varMeshName()->getDataPtr()->string());
			initMesh->translate(-center);

			auto curMesh = this->stateMesh()->getDataPtr();
			curMesh->copyFrom(*initMesh);
			curMesh->scale(scale);
			curMesh->rotate(quat);
			curMesh->translate(location);
		}

		RigidBody<TDataType>::resetStates();
	}

	template<typename TDataType>
	void RigidMesh<TDataType>::updateStates()
	{
		RigidBody<TDataType>::updateStates();

		auto center = this->stateCenter()->getData();
		auto quat = this->stateQuaternion()->getData();
		auto scale = this->varScale()->getData();
 
		auto envlope = this->stateEnvelope()->getDataPtr();
		envlope->copyFrom(this->stateInitialEnvelope()->getData());
		envlope->rotate(quat);
		envlope->scale(scale);
		envlope->translate(center);

		auto mesh = this->stateMesh()->getDataPtr();
		mesh->copyFrom(this->stateInitialMesh()->getData());
		mesh->rotate(quat);
		mesh->scale(scale);
		mesh->translate(center);
	}

	template<typename TDataType>
	void RigidMesh<TDataType>::transform()
	{
		Coord location = this->varLocation()->getData();
		Coord rot = this->varRotation()->getData();
		Coord scale = this->varScale()->getData();

		dyno::Quat<Real> quat = dyno::Quat<Real>(M_PI * rot[0] / 180, Coord(1, 0, 0))
			* dyno::Quat<Real>(M_PI * rot[1] / 180, Coord(0, 1, 0))
			* dyno::Quat<Real>(M_PI * rot[2] / 180, Coord(0, 0, 1));

		if (this->varEnvelopeName()->getDataPtr()->string() != "") {
			auto initEnvlope = this->stateInitialEnvelope()->getDataPtr();

			auto curEnvlope = this->stateEnvelope()->getDataPtr();
			curEnvlope->copyFrom(*initEnvlope);
			curEnvlope->scale(scale);
			curEnvlope->rotate(quat);
			curEnvlope->translate(location);
		}

		if (this->varMeshName()->getDataPtr()->string() != "") {
			auto initMesh = this->stateInitialMesh()->getDataPtr();

			auto curMesh = this->stateMesh()->getDataPtr();
			curMesh->copyFrom(*initMesh);
			curMesh->scale(scale);
			curMesh->rotate(quat);
			curMesh->translate(location);
		}
	}


	DEFINE_CLASS(RigidMesh);
}