#include "ObjPointLoader.h"

#include "Topology/TriangleSet.h"
#include <iostream>
#include <sys/stat.h>
#include "tinyobjloader/tiny_obj_loader.h"



namespace dyno
{
	IMPLEMENT_TCLASS(ObjPoint, TDataType)

		template<typename TDataType>
	ObjPoint<TDataType>::ObjPoint()
		: Node()
	{
		auto pointSet = std::make_shared<PointSet<TDataType>>();
		this->varRadius()->setRange(0,10);


		this->stateTopology()->setDataPtr(pointSet);

		this->outPointSet()->setDataPtr(pointSet);

		pointrender = new GLPointVisualModule;
		std::shared_ptr<GLPointVisualModule>ptrender(pointrender);
		
		pointrender->setVisible(true);
		pointrender->setColor(Color(1, 0, 0));
		pointrender->varPointSize()->setValue(this->varRadius()->getData());
		pointrender->varPointSize()->setValue(varRadius()->getData()/100);

		this->stateTopology()->connect(pointrender->inPointSet());
		this->graphicsPipeline()->pushModule(ptrender);

	}

	template<typename TDataType>
	void ObjPoint<TDataType>::resetStates()
	{
		auto pointSet = TypeInfo::cast<PointSet<TDataType>>(this->stateTopology()->getDataPtr());


		std::string filename = this->varFileName()->constDataPtr()->string();

		loadObj(*pointSet, filename);
		pointSet->scale(this->varScale()->getData());
		pointSet->translate(this->varLocation()->getData());
		pointSet->rotate(this->varRotation()->getData() * PI / 180);

		
		Node::resetStates();
		
		initPos.resize(pointSet->getPoints().size());
		initPos.assign(pointSet->getPoints());
		center = this->varCenter()->getData();
		centerInit = center;

		pointrender->varPointSize()->setValue(varRadius()->getData() / 100);

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
	void ObjPoint<TDataType>::updateStates()
	{
		auto pointSet = TypeInfo::cast<PointSet<TDataType>>(this->stateTopology()->getDataPtr());

		if (this->varSequence()->getData() == true)
		{
			std::string filename = this->varFileName()->constDataPtr()->string();
			int num_ = filename.rfind("_");

			filename.replace(num_ + 1, filename.length() - 4 - (num_ + 1), std::to_string(this->stateFrameNumber()->getData()));

				loadObj(*pointSet,filename);
				pointSet->scale(this->varScale()->getData());
				pointSet->translate(this->varLocation()->getData());
				pointSet->rotate(this->varRotation()->getData() * PI / 180);

				initPos.resize(pointSet->getPoints().size());
				initPos.assign(pointSet->getPoints());
				center = this->varCenter()->getData();
				centerInit = center;
		}

		Coord velocity = this->varVelocity()->getData();
		Coord angularVelocity = this->varAngularVelocity()->getData();

		Real dt = 0.001f;
		rotQuat = rotQuat.normalize();
		rotQuat += dt * 0.5f *
			Quat<Real>(angularVelocity[0], angularVelocity[1], angularVelocity[2], 0.0) * (rotQuat);

		rotQuat = rotQuat.normalize();
		rotMat = rotQuat.toMatrix3x3();

		center += velocity * dt;

		if (!pointSet->getPointSize() == 0)
		{
			cuExecute(pointSet->getPoints().size(),
				K_InitKernelFunctionMesh,
				pointSet->getPoints(),
				initPos,
				center,
				centerInit,
				rotMat
			);
		}


	}


	template<typename TDataType>
	void ObjPoint<TDataType>::loadObj(PointSet<TDataType>& pointset, std::string filename)
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

		if (myshape.size() == 0) 
		{ 
			if (myattrib.GetVertices().size() == 0)
			{
				return;
			}
		}

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
		int s = vertList.size();
		printf("%d", s);
		pointset.setPoints(vertList);
	}

	template<typename TDataType>
	void ObjPoint<TDataType>::convertData(TriangleSet<TDataType>& Triangleset, std::string filename, PointSet<TDataType>& PointSet)
	{

	}

	DEFINE_CLASS(ObjPoint);
}