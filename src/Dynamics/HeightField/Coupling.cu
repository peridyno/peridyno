#include "Coupling.h"
#include "../Core/Matrix/Matrix3x3.h"
#include "../Core/Array/Array.h"

namespace dyno
{
	static void printFloat(float* device, int n)
	{
		float* host = new float[n];
		cudaMemcpy(host, device, n * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < n; i++) {
			printf("a[%d] =%f\n", i, host[i]);
			if (host[i] > 0.005 || host[i] < -0.005) {
				continue;
			}
		}
		printf("end-----------------\n");
	}



	static void printfDArray2D(DArray2D<Vec4f> displacement) {
		int size = displacement.size();
		if (size == 0 ) {
			printf("size == 0");
			return;
		}
		CArray2D<Vec4f> host;
		host.resize(displacement.nx(),displacement.ny());

		host.assign(displacement);

		for (int i = 0; i < displacement.ny(); i++) {
			for(int j = 0; j < displacement.nx();j++)
			printf("[%d, %d]=%f %f %f %f\n", i,j, host[i,j].x, host[i, j].y, host[i, j].z, host[i, j].w);
		}

	}

	template<typename Coord>
	static void printfDArray(DArray<Coord> displacement) {
		int size = displacement.size();
		if (size == 0) {
			printf("size == 0");
			return;
		}
		CArray<Coord> host;
		host.resize(size);
		host.assign(displacement);
		for (int i = 0; i < size; i++) {
			printf("%d= %f %f %f\n",i,   host[i].x, host[i].y, host[i].z);
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
		//cudaFree(m_forceX);
		//cudaFree(m_forceY);
		//cudaFree(m_forceZ);
		//cudaFree(m_torqueX);
		//cudaFree(m_torqueY);
		//cudaFree(m_torqueZ);
		//cudaFree(m_sample_heights);
	}

	template<typename TDataType>
	void Coupling<TDataType>::initialize()
	{
	
		int sizeInBytesF = 8 * sizeof(float);

		m_reduce = Reduction<float>::Create(8);
		
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


	__device__ Vec4f getDisplacement2(Vec3f pos, DArray2D<Vec4f> oceanPatch, Vec2f origin, float patchSize, int gridSize)
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
		Vec4f d00 = oceanPatch[id];
		Vec4f d10 = oceanPatch[id + 1];
		Vec4f d01 = oceanPatch[id + gridSize];
		Vec4f d11 = oceanPatch[id + gridSize + 1];

		return d00 * (1 - fx) * (1 - fy) + d10 * fx * (1 - fy) + d01 * (1 - fx) * fy + d11 * fx * fy;
	}



	template<typename Matrix>
	__global__ void C_ComputeForceAndTorque(
		float* forceX,
		float* forceY,
		float* forceZ,
		float* torqueX,
		float* torqueY,
		float* torqueZ,
		float* sampleHeights,
		DArray<Vec3f> normals,
		DArray<Vec3f> samples,
		DArray2D<Vec4f> ocean,
		DArray<Vec3f> boatCenter,
		DArray<Matrix> rotation,
		Vec2f origin,
		int numOfSamples,
		float patchSize,
		int gridSize, 
		DArray<Vec3f> velocity
	)
	{
		int pId = threadIdx.x + blockIdx.x * blockDim.x;
		if (pId < numOfSamples)
		{
			Vec3f dir_i = samples[pId];
			Vec3f rotDir = rotation[0] * Vec3f(dir_i.x, dir_i.y, dir_i.z);
			//Vec3f pos_i = boatCenter[0] + rotation[0] * Vec3f(dir_i.x, dir_i.y, dir_i.z);
			Vec3f pos_i = samples[pId];


			//
			//判断顶点对应的海洋高度
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

			Vec4f dis_i =  d00 * (1 - fx) * (1 - fy) + d10 * fx * (1 - fy) + d01 * (1 - fx) * fy + d11 * fx * fy;
			//-------------------------------
			dis_i.y *= 1.0f;
			Vec3f normal_i = normals[pId];
			Vec3f force_i(0,0,0);
			Vec3f torque_i(0, 0, 0);

			dis_i.y += 0.2;

			float cha = pos_i.y - dis_i.y;

			if (pos_i.y < dis_i.y)
			{
				//force_i = 9.8f * normal_i * (dis_i.y - pos_i.y);
				force_i = Vec3f(0.0f, 0.098f, 0.0f) * (dis_i.y - pos_i.y);
				//torque_i = Vec3f(0.0f, 0.098f, 0.0f) * (dis_i.y - pos_i.y);
				Vec3f DampingForce(0,-0.1 * velocity[0].y,0);//--------------------------------------------
				force_i += DampingForce;
				//printf("?????????????\n");
				//printf("%d pointY = %f  OceanY = %f  %f\n ", pId, pos_i.y, dis_i.y, cha);
			}

			torque_i = rotDir.cross(torque_i);

			forceX[pId] = force_i.x;
			forceY[pId] = force_i.y;
			forceZ[pId] = force_i.z;
			torqueX[pId] = torque_i.x;
			torqueY[pId] = torque_i.y;
			torqueZ[pId] = torque_i.z;
			sampleHeights[pId] = dis_i.y;
		}
	}

	template<typename TDataType>
	void Coupling<TDataType>::animate(float dt)
	{
		m_eclipsedTime += dt;

		//synchronCheck;
	
		auto m_boat = getRigidBodySystem();
		int num =8;
		int pDims = iDivUp(8, 64);

		auto m_center = getRigidBodySystem()->stateCenter()->getData();
		auto rotation = getRigidBodySystem()->stateRotationMatrix();
			
		//CArray<Vec3f> CVertexNormal;
		//CVertexNormal.resize(center.size());
		//CVertexNormal.assign(center);
		
		//获取正方体的顶点
		DArray<Coord> point = inTriangleSet()->getDataPtr()->getPoints();
		inTriangleSet()->getDataPtr()->updateVertexNormal();
		//printfDArray(point);

		auto VertexNormal = inTriangleSet()->getDataPtr()->outVertexNormal()->getData();
		//CArray<Vec3f> CVertexNormalpoint;
		//CVertexNormal.resize(VertexNormal.size());
		//CVertexNormal.assign(VertexNormal);

		//获取海洋的高度
		DArray2D<Vec4f> Oceandisplacement = getOcean()->getOceanPatch()->getDisplacement();
		auto m_ocean_patch = getOcean()->getOceanPatch();
		//printfDArray2D(Oceandisplacement)
		/**/
		
		//计算力和力矩
		
		C_ComputeForceAndTorque << <pDims, 64 >> > (
			m_forceX,
			m_forceY,
			m_forceZ,
			m_torqueX,
			m_torqueY,
			m_torqueZ,
			m_sample_heights,
			VertexNormal,
			point,
			Oceandisplacement,
			m_boat->stateCenter()->getData(),//	m_boat->getCenter(), glm::vec3 boatCenter,   size == 1
			m_boat->stateRotationMatrix()->getData(),//m_boat->getOrientation(), glm::mat3 rotation,
			m_origin,
			8,
			m_ocean_patch->getPatchSize(),
			m_ocean_patch->getGridSize(),
			getRigidBodySystem()->stateVelocity()->getData()
		);
		
		//printFloat(m_forceX,8);
		//printFloat(m_forceY,8);
		//printFloat(m_forceZ,8);
		//printFloat(m_torqueX,8);
		//printFloat(m_torqueY,8);
		//printFloat(m_torqueZ,8);

		float fx = m_reduce->accumulate(m_forceX, 8);
		float fy = m_reduce->accumulate(m_forceY, 8);
		float fz = m_reduce->accumulate(m_forceZ, 8);

		float tx = m_reduce->accumulate(m_torqueX, 8);
		float ty = m_reduce->accumulate(m_torqueY, 8);
		float tz = m_reduce->accumulate(m_torqueZ, 8);

		float h = m_reduce->accumulate(m_sample_heights, 8);


		Vec3f force = Vec3f(fx / num, fy/num, fz / num);
		Vec3f torque = Vec3f(tx / num, ty / num, tz / num);
		if (!m_force_corrected)
		{
			m_force_corrector = force;
			m_torque_corrector = torque;
			m_force_corrected = true;
		}


	
		m_boat->updateVelocityAngule(force - m_force_corrector, torque - m_torque_corrector, dt);


		auto capillaryWaves = getOcean()->getCapillaryWaves();
		auto m_trail = capillaryWaves[0];

		int originX = m_trail->getOriginX();
		int originZ = m_trail->getOriginZ();

		float dg = m_trail->getRealGridSize();
		int gridSize = m_trail->getGridSize();
		

		CArray<Vec3f> center;
		center.resize(m_center.size());
		center.assign(m_center);

		int new_x = floor(center[0].x / dg) - gridSize / 2;
		int new_z = floor(center[0].z / dg) - gridSize / 2;

		
		if (abs(new_x - originX) > 20 || abs(new_z - originZ) > 20)
		{
			m_trail->setOriginX(new_x);
			m_trail->setOriginY(new_z);
		}
		else
			m_trail->moveDynamicRegion(new_x - originX, new_z - originZ);
		
		m_trail->moveDynamicRegion(0, 0);
		//m_trail->resetSource();
		m_trail->updateStates();
		(m_trail->getmSource()).reset();
		(m_trail->getWeight()).reset();

		C_ComputeTrail << <pDims, 64 >> > (
			m_trail->getmSource(),
			m_trail->getWeight(),//
			m_trail->getGridSize(),//
			m_trail->getOrigin(),//
			m_trail->getRealGridSize(),//
			point,
			point.size(),
			m_boat->stateVelocity()->getData(),
			m_boat->stateCenter()->getData(),
			m_boat->stateRotationMatrix()->getData(),//m_boat->getOrientation(), glm::mat3 rotation,
			m_eclipsedTime);

		
		uint2 extent;
		extent.x = m_trail->getGridSize();
		extent.y = m_trail->getGridSize();

		cuExecute2D(extent,
			C_NormalizeTrail,
			m_trail->getmSource(),
			m_trail->getWeight(),
			m_trail->getGridSize());



		printf("Coupling<TDataType>::animate  \n");

	}

	__global__ void C_NormalizeTrail(
		DArray2D<Vec2f> trails,
		DArray2D<float> weights,
		int trail_size)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < trail_size && j < trail_size)
		{
			int id = i + trail_size * j;
			float w = weights[id];
			if (w > 1.0f)
			{
				trails[id] /= w;
			}
		}
	}

	template<typename Matrix>
	__global__ void C_ComputeTrail(
		DArray2D<Vec2f> trails,//mSource
		DArray2D<float> weights,
		int trail_size,
		Vec2f trail_origin,
		float trail_grid_distance,
		DArray<Vec3f> samples,
		int sample_size,
		DArray<Vec3f> boat_velocity,
		DArray<Vec3f> boatCenter,
		DArray<Matrix> boat_rotation,
		float t)
	{
		int pId = threadIdx.x + blockIdx.x * blockDim.x;
		if (pId < sample_size)
		{
			Vec3f dir_i = samples[pId];
			//if (abs(dir_i.z) < 120.0f && abs(dir_i.x) < 30.0f)
			//{
				Vec3f pos_i = samples[pId]*100;
				Vec2f local_pi = (Vec2f(pos_i.x, pos_i.z) - trail_origin) / trail_grid_distance;
				int i = floor(local_pi.x);
				int j = floor(local_pi.y);
				

				printf("pos_i=(%f %f %f)  i=%d  j=%d  \n", pos_i.x, pos_i.y, pos_i.z,i ,j);
				printf("local_pi=(%f, %f) \n", local_pi.x, local_pi.y);
				printf("trail_origin=%f, trail_grid_distance = %f \n", trail_origin, trail_grid_distance);

				Matrix aniso(2.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.5f);
				aniso = boat_rotation[0] * aniso * boat_rotation[0].transpose();

				boat_velocity[0].x = 3;

				int r = 5;
				for (int s = i - r; s <= i + r; s++)
				{
					for (int t = j - r; t <= j + r; t++)
					{
						float dx = s - i;
						float dz = t - j;
						Vec3f rotated = Vec3f(dx, 0.0f, dz);


						float d = sqrt(rotated.x * rotated.x + rotated.z * rotated.z);
						if (d < r)
						{
							Vec2f dw = (1.0f - d / r) * 0.005f * Vec2f(boat_velocity[0].x, boat_velocity[0].z);
							atomicAdd(&trails[s + t * trail_size].x, dw.x);
							atomicAdd(&trails[s + t * trail_size].y, dw.y);
							atomicAdd(&weights[s + t * trail_size], 1.0f);

							//trails[s + t * trail_size].y = 0.4;
							//trails[s + t * trail_size].x = 0.4;
							//printf("dw= %f %f  \n", dw.x , dw.y);
							//trails[s + t*trail_size] = (1.0f-d/r)*0.03f*make_float2(boat_velocity.x, boat_velocity.y);
						}
					}
				}



				//printf("trails  %d, %d, %f %f\n", i, j, trails[i + j * trail_size].x, trails[i + j * trail_size].y);
			//}
		}
	}


	DEFINE_CLASS(Coupling);
}
