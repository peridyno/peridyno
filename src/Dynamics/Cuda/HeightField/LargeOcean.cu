#include "LargeOcean.h"

namespace dyno
{
	template<typename TDataType>
	LargeOcean<TDataType>::LargeOcean()
		: Node()
	{
		auto ts = std::make_shared<TriangleSet<TDataType>>();

		ts->loadObjFile(getAssetPath() + "ocean/OceanPlane.obj");

		this->stateTriangleSet()->setDataPtr(ts);
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

		auto& disp = patch->stateDisplacement()->constData();

		auto ts = this->stateTriangleSet()->constDataPtr();

		this->stateTexCoord()->resize(ts->getPointSize());
		this->stateTexCoordIndex()->assign(ts->getTriangles());

		this->stateBumpMap()->resize(disp.nx(), disp.ny());

		cuExecute(ts->getPointSize(),
			LO_CalculateTexCoord,
			this->stateTexCoord()->getData(),
			ts->getPoints());

		cuExecute2D(make_uint2(disp.nx(), disp.ny()),
			LO_UpdateBumpMap,
			this->stateBumpMap()->getData(),
			disp);
	}

	template<typename TDataType>
	void LargeOcean<TDataType>::updateStates()
	{
		auto patch = this->getOceanPatch();

		auto& disp = patch->stateDisplacement()->constData();

		cuExecute2D(make_uint2(disp.nx(), disp.ny()),
			LO_UpdateBumpMap,
			this->stateBumpMap()->getData(),
			disp);
	}

	template<typename TDataType>
	bool LargeOcean<TDataType>::validateInputs()
	{
		return this->getOceanPatch() != nullptr;
	}

	DEFINE_CLASS(LargeOcean);
}

