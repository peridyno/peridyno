#include "StaticTriangularMesh.h"

#include "GLSurfaceVisualModule.h"

#include "Topology/TriangleSet.h"

namespace dyno
{
	IMPLEMENT_TCLASS(StaticTriangularMesh, TDataType)

	template<typename TDataType>
	StaticTriangularMesh<TDataType>::StaticTriangularMesh()
		: ParametricModel<TDataType>()
	{
		auto triSet = std::make_shared<TriangleSet<TDataType>>();
		this->stateTopology()->setDataPtr(triSet);

		this->stateInitialTopology()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());

		auto surfaceRender = std::make_shared<GLSurfaceVisualModule>();
		surfaceRender->setColor(Color(0.8f, 0.52f, 0.25f));
		surfaceRender->setVisible(true);
		this->stateTopology()->connect(surfaceRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(surfaceRender);

		auto callFileLoader = std::make_shared<FCallBackFunc>(
			[=]() {
				auto initTopo = this->stateInitialTopology()->getDataPtr();
				auto curTopo = this->stateTopology()->getDataPtr();

				std::string fileName = this->varFileName()->getDataPtr()->string();

				if (fileName != "")
				{
					initTopo->loadObjFile(fileName);
					curTopo->copyFrom(*initTopo);

					curTopo->scale(this->varScale()->getData());
					curTopo->rotate(this->varRotation()->getData() * M_PI / 180);
					curTopo->translate(this->varLocation()->getData());
				}
			}
		);
		this->varFileName()->attach(callFileLoader);

		auto transform = std::make_shared<FCallBackFunc>(
			[=]() {
				auto initTopo = this->stateInitialTopology()->getDataPtr();
				auto curTopo = this->stateTopology()->getDataPtr();

				curTopo->copyFrom(*initTopo);
				curTopo->scale(this->varScale()->getData());
				curTopo->rotate(this->varRotation()->getData() * M_PI / 180);
				curTopo->translate(this->varLocation()->getData());
			}
		);
		this->varLocation()->attach(transform);
		this->varScale()->attach(transform);
		this->varRotation()->attach(transform);

		this->outTriangleSet()->setDataPtr(triSet);

		this->inTriangleSet_IN()->setDataPtr(triSet);

		this->inTriangleSet_IN()->tagOptional(true);
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
	void StaticTriangularMesh<TDataType>::updateStates()
	{

		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->stateTopology()->getDataPtr());

		if (this->varSequence()->getData() == true)
		{


			std::string filename = this->varFileName()->getDataPtr()->string();
			int num_ = filename.rfind("_");

			filename.replace(num_ + 1, filename.length() - 4 - (num_ + 1), std::to_string(this->stateFrameNumber()->getData()));


			auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->stateTopology()->getDataPtr());

			if (this->varSequence()->getData() == true)
			{


				std::string filename = this->varFileName()->getDataPtr()->string();
				int num_ = filename.rfind("_");

				filename.replace(num_ + 1, filename.length() - 4 - (num_ + 1), std::to_string(this->stateFrameNumber()->getData()));


				struct stat buffer;
				bool isvaid = (stat(filename.c_str(), &buffer) == 0);

				if (isvaid)
				{
					triSet->loadObjFile(filename);

					triSet->scale(this->varScale()->getData());
					triSet->translate(this->varLocation()->getData());
					triSet->rotate(this->varRotation()->getData() * M_PI / 180);

					initPos.resize(triSet->getPoints().size());
					initPos.assign(triSet->getPoints());
					center = this->varCenter()->getData();
					centerInit = center;
				}


			}


			Coord velocity = this->varVelocity()->getData();
			Coord angularVelocity = this->varAngularVelocity()->getData();

			//printf("velocity = %.10lf %.10lf %.10lf\n", velocity[0], velocity[1], velocity[2]);

			Real dt = 0.001f;
			rotQuat = rotQuat.normalize();
			rotQuat += dt * 0.5f *
				Quat<Real>(angularVelocity[0], angularVelocity[1], angularVelocity[2], 0.0) * (rotQuat);

			rotQuat = rotQuat.normalize();
			rotMat = rotQuat.toMatrix3x3();

			center += velocity * dt;

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

	DEFINE_CLASS(StaticTriangularMesh);
}