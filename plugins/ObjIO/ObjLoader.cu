#include "ObjLoader.h"

#include "Topology/TriangleSet.h"
#include <iostream>
#include <sys/stat.h>

#include "GLWireframeVisualModule.h"
#include "GLSurfaceVisualModule.h"
#include "GLPointVisualModule.h"


#include "tinyobjloader/tiny_obj_loader.h"


namespace dyno
{
	IMPLEMENT_TCLASS(ObjLoader, TDataType)

		template<typename TDataType>
	ObjLoader<TDataType>::ObjLoader()
		: Node()
	{
		auto triSet = std::make_shared<TriangleSet<TDataType>>();

		std::vector<Coord> vertList;
		std::vector<TopologyModule::Triangle> faceList;

		triSet->setPoints(vertList);
		triSet->setTriangles(faceList);

		this->stateTopology()->setDataPtr(triSet);
		this->outTriangleSet()->setDataPtr(triSet);

		auto callback = std::make_shared<FCallBackFunc>(std::bind(&ObjLoader<TDataType>::animationUpdate, this));

		this->varVelocity()->attach(callback);
		this->varSequence()->attach(callback);
		this->varAngularVelocity()->attach(callback);

		auto surfacerender = std::make_shared<GLSurfaceVisualModule>();
		surfacerender->setVisible(true);
		surfacerender->setColor(Color(0.8, 0.52, 0.25));

		this->stateTopology()->connect(surfacerender->inTriangleSet());
		this->graphicsPipeline()->pushModule(surfacerender);

	}

	template<typename TDataType>
	void ObjLoader<TDataType>::animationUpdate()
	{
		if (this->varSequence()->getValue() == true || this->varVelocity()->getValue() != Vec3f(0) || this->varAngularVelocity()->getValue() != Vec3f(0))
			this->setForceUpdate(true);
		else
			this->setForceUpdate(false);
	}

	template<typename TDataType>
	void ObjLoader<TDataType>::resetStates()
	{
		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->stateTopology()->getDataPtr());

		if (this->varFileName()->getValue().string() == "")
			return;
		std::string filename = this->varFileName()->getValue().string();
		loadObj(*triSet, filename);
		triSet->scale(this->varScale()->getValue());
		triSet->translate(this->varLocation()->getValue());
		triSet->rotate(this->varRotation()->getValue() * PI / 180);
		triSet->update();

		Node::resetStates();

		initPos.resize(triSet->getPoints().size());
		initPos.assign(triSet->getPoints());
		center = this->varCenter()->getData();
		centerInit = center;
	}

	template <typename Coord, typename Matrix>
	__global__ void K_InitKernelFunctionMesh(
		DArray<Coord> posArr,
		DArray<Coord> posInit,
		Coord center,
		Coord centerInit,
		Matrix rotation
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size())
			return;
		Coord pos;
		pos = posInit[pId] - centerInit;
		pos = rotation * pos;
		posArr[pId] = pos + center;

	}


	template<typename TDataType>
	void ObjLoader<TDataType>::updateStates()
	{

		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->stateTopology()->getDataPtr());

		if (this->varSequence()->getValue() == true)
		{
			std::string filename = this->varFileName()->getValue().string();
			int num_ = filename.rfind("_");

			filename.replace(num_ + 1, filename.length() - 4 - (num_ + 1), std::to_string(this->stateFrameNumber()->getData()));

			loadObj(*triSet, filename);
			triSet->scale(this->varScale()->getValue());
			triSet->translate(this->varLocation()->getValue());
			triSet->rotate(this->varRotation()->getValue() * PI / 180);
			triSet->update();


			initPos.resize(triSet->getPoints().size());
			initPos.assign(triSet->getPoints());
			center = this->varCenter()->getData();
			centerInit = center;
		}

		Coord velocity = this->varVelocity()->getData();
		Coord angularVelocity = this->varAngularVelocity()->getData();

		Real dt = 0.001f;
		//Real dt = this->stateTimeStep()->getData();
		rotQuat = rotQuat.normalize();
		rotQuat += dt * 0.5f *
			Quat<Real>(angularVelocity[0], angularVelocity[1], angularVelocity[2], 0.0) * (rotQuat);

		rotQuat = rotQuat.normalize();
		rotMat = rotQuat.toMatrix3x3();

		center += velocity * dt;

		if (!triSet->getTriangles().isEmpty() && !triSet->getVertex2Triangles().isEmpty())
		{
			cuExecute(triSet->getPoints().size(),
				K_InitKernelFunctionMesh,
				triSet->getPoints(),
				initPos,
				center,
				centerInit,
				rotMat
			);
		}

	}


	template<typename TDataType>
	void ObjLoader<TDataType>::loadObj(TriangleSet<TDataType>& Triangleset, std::string filename)
	{
		std::vector<Coord> vertList;
		std::vector<TopologyModule::Triangle> faceList;

		tinyobj::attrib_t myattrib;
		std::vector <tinyobj::shape_t> myshape;
		std::vector <tinyobj::material_t> mymat;
		std::string mywarn;
		std::string myerr;

		char* fname = (char*)filename.c_str();
		std::cout << fname << std::endl;
		tinyobj::LoadObj(&myattrib, &myshape, &mymat, &mywarn, &myerr, fname, nullptr, true, true);
		std::cout << mywarn << std::endl;
		std::cout << myerr << std::endl;
		std::cout << "************************    Loading : shapelod    ************************  " << std::endl << std::endl;
		std::cout << "                        " << "    shape size =" << myshape.size() << std::endl << std::endl;
		std::cout << "************************    Loading : v    ************************  " << std::endl << std::endl;
		std::cout << "                        " << "    point sizelod = " << myattrib.GetVertices().size() / 3 << std::endl << std::endl;

		if (myshape.size() == 0) { return; }

		for (int i = 0; i < myattrib.GetVertices().size() / 3; i++)
		{

			vertList.push_back(Coord(myattrib.GetVertices()[3 * i], myattrib.GetVertices()[3 * i + 1], myattrib.GetVertices()[3 * i + 2]));
		}
		std::cout << "************************    Loading : f    ************************  " << std::endl << std::endl;
		for (int i = 0; i < myshape.size(); i++)
		{
			std::cout << "                        " << "    Triangle " << i << " size =" << myshape[i].mesh.indices.size() / 3 << std::endl << std::endl;

			for (int s = 0; s < myshape[i].mesh.indices.size() / 3; s++)
			{
				//std::cout << myshape[i].mesh.indices[s].vertex_index <<"  " << std::endl;

				faceList.push_back(TopologyModule::Triangle(myshape[i].mesh.indices[3 * s].vertex_index, myshape[i].mesh.indices[3 * s + 1].vertex_index, myshape[i].mesh.indices[3 * s + 2].vertex_index));
			}
		}
		std::cout << "************************    Loading completed    **********************" << std::endl << std::endl;

		Triangleset.setPoints(vertList);
		Triangleset.setTriangles(faceList);
		//Triangleset.updateVertexNormal();
		Triangleset.update();

	}


	DEFINE_CLASS(ObjLoader);
}