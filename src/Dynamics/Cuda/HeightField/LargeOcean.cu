#include "LargeOcean.h"

#include "Module/ApplyBumpMap2TriangleSet.h"

#include "GLSurfaceVisualModule.h"

namespace dyno
{
	template<typename TDataType>
	LargeOcean<TDataType>::LargeOcean()
		: OceanBase<TDataType>()
	{
		auto ts = std::make_shared<TriangleSet<TDataType>>();
		this->stateTriangleSet()->setDataPtr(ts);

		//Set default mesh
		this->varFileName()->setValue(getAssetPath() + "ocean/OceanPlane.obj");

		auto mapper = std::make_shared<ApplyBumpMap2TriangleSet<DataType3f>>();
		this->stateTriangleSet()->connect(mapper->inTriangleSet());
		this->stateHeightField()->connect(mapper->inHeightField());
		this->graphicsPipeline()->pushModule(mapper);

		auto sRender = std::make_shared<GLSurfaceVisualModule>();
		sRender->setColor(Color(0, 0.2, 1.0));
		sRender->varUseVertexNormal()->setValue(true);
		mapper->outTriangleSet()->connect(sRender->inTriangleSet());
		this->graphicsPipeline()->pushModule(sRender);
	}

	template<typename TDataType>
	LargeOcean<TDataType>::~LargeOcean()
	{
	}

	template <typename Coord3D>
	__global__  void LO_CalculateTexCoord(
		DArray<Vec2f> texCoords,
		DArray<Coord3D> vertices)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= vertices.size()) return;

		Coord3D vert = vertices[tId];

		//TODO: expose the parameter
		float scale = 1 / 512.0f;

		vert *= scale;

		float u = vert.x - floor(vert.x);
		float v = vert.z - floor(vert.z);

		texCoords[tId] = Vec2f(u, v);
	}

	template <typename Coord3D, typename Coord4D>
	__global__ void LO_UpdateBumpMap(
		DArray2D<Coord4D> bumpMap,
		DArray2D<Coord3D> displacement)
	{
		unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (i < displacement.nx() && j < displacement.ny())
		{
			Coord3D disp_ij = displacement(i, j);

			Coord4D pixel;

			pixel.x = disp_ij.x;
			pixel.y = disp_ij.y;
			pixel.z = disp_ij.z;
			pixel.w = 0;

			bumpMap(i, j) = pixel;
		}
	}

	template<typename TDataType>
	void LargeOcean<TDataType>::resetStates()
	{
		auto patch = this->getOceanPatch();

		auto& topo = patch->stateHeightField()->constDataPtr();
		auto& disp = topo->getDisplacement();

		auto ts = this->stateTriangleSet()->constDataPtr();

		auto name = this->varFileName()->getValue().string();
		if (name != mFileName)
		{
			ts->loadObjFile(name);
			mFileName = name;
		}

		this->stateTexCoord()->resize(ts->getPointSize());
		this->stateBumpMap()->resize(disp.nx(), disp.ny());

		cuExecute(ts->getPointSize(),
			LO_CalculateTexCoord,
			this->stateTexCoord()->getData(),
			ts->getPoints());

		cuExecute2D(make_uint2(disp.nx(), disp.ny()),
			LO_UpdateBumpMap,
			this->stateBumpMap()->getData(),
			disp);

		this->stateHeightField()->setDataPtr(topo);
	}

	template<typename TDataType>
	void LargeOcean<TDataType>::updateStates()
	{
		this->stateHeightField()->tick();
	}

	DEFINE_CLASS(LargeOcean);
}

