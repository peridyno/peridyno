#include "HeightFieldNode.h"
#include "Topology/HeightField.h"
#include "ShallowWaterEquationModel.h"

namespace dyno
{
	IMPLEMENT_CLASS_1(HeightFieldNode, TDataType)

	template<typename TDataType>
	HeightFieldNode<TDataType>::HeightFieldNode(std::string name)
		: Node(name)
	{
		auto swe = this->template setNumericalModel<ShallowWaterEquationModel<TDataType>>("swe");
		this->setNumericalModel(swe);
		SWEconnect();
		
		m_height_field = std::make_shared<HeightField<TDataType>>();
		this->setTopologyModule(m_height_field);
	}

	template<typename TDataType>
	void HeightFieldNode<TDataType>::SWEconnect()
	{
		auto swe = this->getModule<ShallowWaterEquationModel<TDataType>>("swe");
		//auto swe = this->template setNumericalModel<ShallowWaterEquationModel<TDataType>>("swe2");
		//this->setNumericalModel(swe);
		//std::shared_ptr<ShallowWaterEquationModel<TDataType>> swe = this->getNumericalModel();
		this->currentPosition()->connect(&(swe->m_position));
		
		this->currentVelocity()->connect(&(swe->m_velocity));
		//this->h.connect(swe->h);
		this->normal.connect(&(swe->normal));

		this->neighbors.connect(&(swe->neighborIndex));
		this->isBound.connect(&(swe->isBound));
		this->solid.connect(&(swe->solid));

		swe->setDistance(distance);
		swe->setRelax(relax);
	}

	template<typename TDataType>
	bool HeightFieldNode<TDataType>::initialize()
	{
		return Node::initialize();
	}

	//template<typename Real, typename Coord>
	__global__ void InitNeighbor(
		NeighborList<int> neighbors,
		int zcount,
		int xcount)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= neighbors.size()) return;
		if(i%zcount==0)
			neighbors.setElement(i, 0, - 1);
		else
			neighbors.setElement(i, 0, i - 1);
		if((i+1)%zcount == 0)
			neighbors.setElement(i, 1, -1);
		else
			neighbors.setElement(i, 1, i + 1);

		neighbors.setElement(i, 2, i - zcount);
		neighbors.setElement(i, 3, i + zcount);

	}

	template<typename TDataType>
	void HeightFieldNode<TDataType>::loadHeightFieldParticles(Coord lo, Coord hi, Real distance, Real slope)
	{
		std::vector<Coord> vertList;
		std::vector<Coord> normalList;

		float height, e = 2.71828;
		nx = (hi[0] - lo[0]) / distance;
		nz = (hi[2] - lo[0]) / distance;
		float xcenter = (hi[0] - lo[0]) / 2, zcenter = (hi[2] - lo[2]) / 2;
		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			for (Real z = lo[2]; z <= hi[2]; z += distance)
			{
				//if (pow(x - xcenter, 2) + pow(z - zcenter, 2) > 400*distance*distance)
					//height = slope * pow(e, -400*distance*distance*100);
				//	height = slope * pow(e, -400 * distance*distance * 10);
				//else
				height = 0.3 + slope * pow(e, -(pow(x - xcenter, 2) + pow(z - zcenter, 2)) * 100);
				//height = slope * pow(e, -(pow(x, 2) + pow(z, 2)) * 5);
				//height = 3*slope*(x + z - lo[0] - lo[2]);
				Coord p = Coord(x, 0, z);
				vertList.push_back(Coord(x, height + lo[1], z));
				normalList.push_back(Coord(0, 1, 0));
			}
		}

		this->currentPosition()->setElementCount(vertList.size());
		this->currentPosition()->getValue().assign(vertList);

		this->currentVelocity()->setElementCount(vertList.size());

		vertList.clear();
		normalList.clear();
	}

	template<typename TDataType>
	void HeightFieldNode<TDataType>::loadParticles(Coord lo, Coord hi, Real distance,Real slope, Real relax)
	{
		loadHeightFieldParticles(lo, hi, distance, slope);
		
		this->distance = distance;
		this->relax = relax;
		std::vector<Coord> solidList;
		std::vector<Coord> normals;
		std::vector<int>  isbound;
		float height;
		int xcount = 0;
		for (Real x = lo[0]; x <= hi[0]; x += distance)
		{
			xcount++;
			for (Real z = lo[2]; z <= hi[2]; z += distance)
			{
				height = 20 * (x + z - lo[0] - lo[2]);
				if (z + distance > hi[2] || x + distance > hi[0] || x == lo[0] || z == lo[2])
					isbound.push_back(1);
				else
					isbound.push_back(0);
				//	height = 
				solidList.push_back(Coord(x, lo[1], z));
				//wh.push_back(height - lo[1]);
				normals.push_back(Coord(0, 1, 0));
			}
		}

		solid.setElementCount(solidList.size());
		solid.getValue().assign(solidList);

		//h.setElementCount(solidList.size());
		//Function1Pt::copy(h.getValue(), wh);

		isBound.setElementCount(solidList.size());
		isBound.getValue().assign(isbound);

		normal.setElementCount(solidList.size());
		normal.getValue().assign(normals);
		//m_velocity.setElementCount(solidList.size());
		//neighbors.resize(solidList.size(), 4);
		neighbors.setElementCount(solidList.size(), 4);
		//add four neighbors:up down left right
		int zcount = solidList.size() / xcount;
		int num = solidList.size();
		printf("zcount is %d, xcount is %d\n", zcount, xcount);
		uint pDims = cudaGridSize(num, BLOCK_SIZE);
		InitNeighbor << < pDims, BLOCK_SIZE >> > (neighbors.getValue(), zcount, xcount);
		cuSynchronize();
		solidList.clear();
		
		isbound.clear();
		normals.clear();

		DeviceArrayField<Coord> pos = *(this->currentPosition());
		SWEconnect();

		this->updateTopology();
	}

	template<typename TDataType>
	HeightFieldNode<TDataType>::~HeightFieldNode()
	{
	}
	template<typename TDataType>
	void HeightFieldNode<TDataType>::advance(Real dt)
	{
		auto nModel = this->getNumericalModel();
		nModel->step(dt);
	}

	template<typename Real, typename Coord>
	__global__ void SetupHeights(
		GArray2D<Real> height, 
		GArray<Coord> pts)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;

		if (i < height.nx() && j < height.ny())
		{
			int id = i + j * (height.nx() + 1);

			height(i, j) = pts[id][1];
		}
	}

	template<typename TDataType>
	void HeightFieldNode<TDataType>::updateTopology()
	{
		if (!this->currentPosition()->isEmpty())
		{
			int num = this->currentPosition()->getElementCount();
			auto& pts = this->currentPosition()->getValue();

			m_height_field->setSpace(0.005, 0.005);
			auto& heights = m_height_field->getHeights();

			if (nx != heights.nx() || nz != heights.ny())
			{
				heights.resize(nx, nz);
			}

			uint3 total_size;
			total_size.x = nx;
			total_size.y = nz;
			total_size.z = 1;

			//ti++;

			cuExecute3D(total_size, SetupHeights,
				heights,
				pts);
		}
	}

	//template<typename TDataType>
	//void HeightField<TDataType>::loadObjFile(std::string filename)
	//{
	//	
	//}

}