#include "Coupling.h"
#include "../Core/Matrix/Matrix3x3.h"
namespace dyno
{
	static void printFloat(float* device, int n)
	{
		float* host = new float[n];
		cudaMemcpy(host, device, n * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < n; i++) {
			printf("a[%d] =%f\n", i, host[i]);
		}
		printf("end-----------------\n");
	}



	static void printfDArray2D(DArray2D<Vec4f> displacement) {
		int size = displacement.size();
		std::vector<Vec4f> host;
		host.resize(size);

		//host.assign(displacement);

		for (int i = 0; i < size; i++) {

		}

	}
	template<typename TDataType>
	Coupling<TDataType>::Coupling(std::string name)
		: Node(name)
	{
		//printf("Coupling\n");

		m_origin = Vec2f(0, 0);
		m_force_corrected = false;

		m_heightShift = 6.0f;
		m_eclipsedTime = 0.0f;

		m_prePos = Vec3f(0, 0, 0);

		m_name = name;
	}

	template<typename TDataType>
	Coupling<TDataType>::~Coupling()
	{
		cudaFree(m_forceX);
		cudaFree(m_forceY);
		cudaFree(m_forceZ);
		cudaFree(m_torqueX);
		cudaFree(m_torqueY);
		cudaFree(m_torqueZ);
		cudaFree(m_sample_heights);
	}

	template<typename TDataType>
	void Coupling<TDataType>::initialize()
	{
		auto boat = getRigidBodySystem();

		//int sizeInBytes = boat->getSamplingPointSize() * sizeof(Vec3f);
		int sizeInBytesF = boat->getSamplingPointSize() * sizeof(float);

		//m_reduce = Physika::Reduction<float>::Create(boat->getSamplingPointSize());

		cudaMalloc(&m_forceX, sizeInBytesF);
		cudaMalloc(&m_forceY, sizeInBytesF);
		cudaMalloc(&m_forceZ, sizeInBytesF);
		cudaMalloc(&m_torqueX, sizeInBytesF);
		cudaMalloc(&m_torqueY, sizeInBytesF);
		cudaMalloc(&m_torqueZ, sizeInBytesF);

		cudaMalloc(&m_sample_heights, sizeInBytesF);
		/*

		glm::vec3 center = boat->getCenter();


		float dg = m_trail->getRealGridSize();

		int nx = center.x / dg - m_trail->getGridSize() / 2;
		int ny = center.z / dg - m_trail->getGridSize() / 2;

		m_trail->setOriginX(nx);
		m_trail->setOriginY(ny);	*/
	}


	template<typename TDataType>
	void Coupling<TDataType>::resetStates()
	{


	}

	static int iDivUp(int a, int b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	template<typename TDataType>
	void Coupling<TDataType>::updateStates()
	{
		this->animate(0.016f);
	}

	__device__ int getDisplacement(Vec3f pos, DArray2D<Vec4f> ocean, Vec2f origin, float patchSize, int gridSize)
	{
		Vec2f uv_i = (Vec2f(pos.x, pos.z) - origin) / patchSize;
		float u = (uv_i.x - floor(uv_i.x)) * gridSize;
		float v = (uv_i.y - floor(uv_i.y)) * gridSize;
		int i = floor(u);
		int j = floor(v);
		float fx = u - i;
		float fy = v - j;
		if (i == gridSize - 1)
		{
			i = gridSize - 2;
			fx = 1.0f;
		}
		if (j == gridSize - 1)
		{
			j = gridSize - 2;
			fy = 1.0f;
		}
		int id = i + j * gridSize;
		Vec4f d00 = ocean[id];
		Vec4f d10 = ocean[id + 1];
		Vec4f d01 = ocean[id + gridSize];
		Vec4f d11 = ocean[id + gridSize + 1];

		return id;

	}


	template<typename Matrix, typename Coord>
	__global__ void C_ComputeForceAndTorque(
		float* forceX,
		float* forceY,
		float* forceZ,
		float* torqueX,
		float* torqueY,
		float* torqueZ,
		float* sampleHeights,
		DArray2D<Vec3f> normals,
		DArray2D<Vec3f> samples,
		DArray2D<Vec4f> ocean,
		DArray<Coord> boatCenter,
		DArray<Matrix> rotation,
		Vec2f origin,
		int numOfSamples,
		float patchSize,
		int gridSize
	)
	{
		int pId = threadIdx.x + blockIdx.x * blockDim.x;
		if (pId < numOfSamples)
		{
			Vec3f dir_i = samples[pId];
			Vec3f rotDir = rotation[pId] * Vec3f(dir_i.x, dir_i.y, dir_i.z);
			Vec3f pos_i = boatCenter[pId] + rotation[pId] * Vec3f(dir_i.x, dir_i.y, dir_i.z);

			//int index = getDisplacement(pos_i, ocean, origin, patchSize, gridSize);

			Vec2f uv_i = (Vec2f(pos_i.x, pos_i.z) - origin) / patchSize;
			float u = (uv_i.x - floor(uv_i.x)) * gridSize;
			float v = (uv_i.y - floor(uv_i.y)) * gridSize;
			int i = floor(u);
			int j = floor(v);
			float fx = u - i;
			float fy = v - j;

			if (i == gridSize - 1)
			{
				i = gridSize - 2;
				fx = 1.0f;
			}
			if (j == gridSize - 1)
			{
				j = gridSize - 2;
				fy = 1.0f;
			}

			int id = i + j * gridSize;
			Vec4f d00 = ocean[id];
			Vec4f d10 = ocean[id + 1];
			Vec4f d01 = ocean[id + gridSize];
			Vec4f d11 = ocean[id + gridSize + 1];



			Vec4f dis_i = d00 * (1 - fx) * (1 - fy) + d10 * fx * (1 - fy) + d01 * (1 - fx) * fy + d11 * fx * fy;

			dis_i.y *= 1.0f;
			Vec3f normal_i = normals[pId];
			Vec3f force_i(0, 0, 0);
			Vec3f torque_i(0, 0, 0);


			//if (pos_i.y < dis_i.y)
			//{
			//	force_i = 9800.0f * normal_i * (dis_i.y - pos_i.y);
			//	torque_i = Vec3f(0.0f, 9800.0f, 0.0f) * (dis_i.y - pos_i.y);
			//}
			//torque_i =  Vec3f(rotDir.x, rotDir.y, rotDir.z).cross(torque_i);
			////torque_i = cross(Vec3f(rotDir.x, rotDir.y, rotDir.z), torque_i);
			//forceX[pId] = force_i.x;
			//forceY[pId] = force_i.y;
			//forceZ[pId] = force_i.z;
			//torqueX[pId] = torque_i.x;
			//torqueY[pId] = torque_i.y;
			//torqueZ[pId] = torque_i.z;
			//sampleHeights[pId] = dis_i.y;

		}
	}

	template<typename TDataType>
	void Coupling<TDataType>::animate(float dt)
	{
		m_eclipsedTime += dt;

		//synchronCheck;

		auto m_boat = getRigidBodySystem();
		int num = m_boat->getSamplingPointSize();
		int pDims = iDivUp(m_boat->getSamplingPointSize(), 64);

		auto m_ocean_patch = getOcean()->getOceanPatch();

		DArray2D <Vec4f> displacement = m_ocean_patch->getDisplacement();

		//printfDArray2D(displacement);

		C_ComputeForceAndTorque << <pDims, 64 >> > (
			m_forceX,
			m_forceY,
			m_forceZ,
			m_torqueX,
			m_torqueY,
			m_torqueZ,
			m_sample_heights,
			m_boat->getNormals(),
			m_boat->getSamples(),
			m_ocean_patch->getDisplacement(),
			m_boat->stateCenter()->getData(),
			m_boat->stateRotationMatrix()->getData(),
			m_origin,
			m_boat->getSamplingPointSize(),
			m_ocean_patch->getPatchSize(),
			m_ocean_patch->getGridSize()
			);


		printf("Coupling<TDataType>::animate  \n");
	}

	DEFINE_CLASS(Coupling);
}
