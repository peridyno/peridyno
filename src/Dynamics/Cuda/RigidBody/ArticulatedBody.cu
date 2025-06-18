#include "ArticulatedBody.h"

//Collision
#include "Collision/NeighborElementQuery.h"
#include "Collision/CollistionDetectionTriangleSet.h"

//RigidBody
#include "Module/ContactsUnion.h"
#include "Module/TJConstraintSolver.h"
#include "Module/InstanceTransform.h"
#include "Module/SharedFuncsForRigidBody.h"

//Rendering
#include "Module/GLPhotorealisticInstanceRender.h"

#include "GltfFunc.h"
#include "helpers/tinyobj_helper.h"

namespace dyno
{
	template<typename TDataType>
	ArticulatedBody<TDataType>::ArticulatedBody()
		: ParametricModel<TDataType>()
		, RigidBodySystem<TDataType>()
	{
		this->setAutoHidden(false);

		this->stateTextureMesh()->setDataPtr(std::make_shared<TextureMesh>());

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&ArticulatedBody<TDataType>::varChanged, this));
		this->varFilePath()->attach(callback);

		this->animationPipeline()->clear();

		auto transformer = std::make_shared<InstanceTransform<DataType3f>>();
		this->stateCenter()->connect(transformer->inCenter());
		this->stateRotationMatrix()->connect(transformer->inRotationMatrix());
		this->stateBindingPair()->connect(transformer->inBindingPair());
		this->stateBindingTag()->connect(transformer->inBindingTag());
		this->stateInstanceTransform()->connect(transformer->inInstanceTransform());
		this->graphicsPipeline()->pushModule(transformer);

		auto prRender = std::make_shared<GLPhotorealisticInstanceRender>();
		this->stateTextureMesh()->connect(prRender->inTextureMesh());
		transformer->outInstanceTransform()->connect(prRender->inTransform());
		this->graphicsPipeline()->pushModule(prRender);

		this->setForceUpdate(true);
	}

	template<typename TDataType>
	ArticulatedBody<TDataType>::~ArticulatedBody()
	{

	}

	template<typename TDataType>
	void ArticulatedBody<TDataType>::resetStates()
	{
		RigidBodySystem<TDataType>::resetStates();

		auto topo = this->stateTopology()->getDataPtr();

		int sizeOfRigids = this->stateCenter()->size();

		auto texMesh = this->stateTextureMesh()->constDataPtr();

		uint N = 0;
		if (texMesh != NULL);
		N = texMesh->shapes().size();

		CArrayList<Transform3f> tms;
		CArray<uint> instanceNum(N);
		instanceNum.reset();

		//Calculate instance number
		for (uint i = 0; i < mBindingPair.size(); i++)
		{
			instanceNum[mBindingPair[i].first]++;
		}

		if (instanceNum.size() > 0)
			tms.resize(instanceNum);

		//Initialize CArrayList
		for (uint i = 0; i < N; i++)
		{
			for (uint j = 0; j < instanceNum[i]; j++)
			{
				tms[i].insert(Transform3f());
			}
		}

		this->stateInstanceTransform()->assign(tms);

		auto deTopo = this->stateTopology()->constDataPtr();
		auto offset = deTopo->calculateElementOffset();

		std::vector<Pair<uint, uint>> bindingPair(sizeOfRigids);
		std::vector<int> tags(sizeOfRigids, 0);


		CArray<uint> instanceCount(N);
		instanceCount.reset();
		//Setup the mapping from shape id to rigid body id
		for (int i = 0; i < mBindingPair.size(); i++)
		{
			uint rId = mActors[i]->idx;
			uint sId = mBindingPair[i].first;
			bindingPair[rId] = Pair<uint, uint>(sId, instanceCount[sId]);
			tags[rId] = 1;
			instanceCount[sId]++;
		}

		this->stateBindingPair()->assign(bindingPair);
		this->stateBindingTag()->assign(tags);

		this->updateInstanceTransform();

		tms.clear();
		bindingPair.clear();
		instanceCount.clear();
		tags.clear();

		this->transform();

		topo->setPosition(this->stateCenter()->constData());
		topo->setRotation(this->stateRotationMatrix()->constData());
		topo->update();
	}

	template<typename TDataType>
	void ArticulatedBody<TDataType>::varChanged()
	{
		std::shared_ptr<TextureMesh> texMesh = this->stateTextureMesh()->getDataPtr();
		auto filepath = this->varFilePath()->getValue();

		auto ext = filepath.path().extension().string();
		auto name = filepath.string();

		if (ext == ".gltf")
		{
			loadGLTFTextureMesh(texMesh, name);
		}
		else if (ext == ".obj")
		{
			loadTextureMeshFromObj(texMesh, name);
		}
	}

	template<typename TDataType>
	void ArticulatedBody<TDataType>::transform()
	{
		////************************** initial mInitialRot *************************//

		CArray<Coord> hostCenter;
		hostCenter.assign(this->stateCenter()->constData());

		CArray<Quat<Real>> hostQuaternion;
		hostQuaternion.assign(this->stateQuaternion()->constData());

		CArray<Mat3f> hostRotation;
		hostRotation.assign(this->stateRotationMatrix()->constData());

		//for (size_t i = 0; i < vehicleNum; i++)
		{
			//get varTransform;
			auto quat = this->computeQuaternion();
			Coord location = this->varLocation()->getValue();

			for (uint i = 0; i < hostCenter.size(); i++)
			{
				hostCenter[i] = quat.rotate(hostCenter[i]) + location;
			}

			//***************************** Rotation *************************//

			for (uint i = 0; i < hostQuaternion.size(); i++)
			{
				hostQuaternion[i] = quat * hostQuaternion[i];
			}

			for (uint i = 0; i < hostRotation.size(); i++)
			{
				hostRotation[i] = quat.toMatrix3x3() * hostRotation[i];
			}
		}

		this->stateCenter()->assign(hostCenter);
		this->stateQuaternion()->assign(hostQuaternion);
		this->stateRotationMatrix()->assign(hostRotation);

		hostCenter.clear();
		hostQuaternion.clear();
		hostRotation.clear();
	}


	template<typename TDataType>
	void ArticulatedBody<TDataType>::updateStates()
	{
		RigidBodySystem<TDataType>::updateStates();
	}

	template<typename TDataType>
	void ArticulatedBody<TDataType>::updateInstanceTransform()
	{
		ApplyTransform(
			this->stateInstanceTransform()->getData(),
			this->stateCenter()->getData(),
			this->stateRotationMatrix()->getData(),
			this->stateBindingPair()->constData(),
			this->stateBindingTag()->constData());
	}

	template<typename TDataType>
	void ArticulatedBody<TDataType>::bindShape(std::shared_ptr<PdActor> actor, Pair<uint, uint> shapeId)
	{
		mActors.push_back(actor);
		mBindingPair.push_back(shapeId);
	}

	template<typename TDataType>
	void ArticulatedBody<TDataType>::clearVechicle()
	{
		mBindingPair.clear();
		mActors.clear();
	}

	DEFINE_CLASS(ArticulatedBody);
}