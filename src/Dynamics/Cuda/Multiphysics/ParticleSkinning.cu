#include "ParticleSkinning.h"
#include "ComputeSurfaceLevelset.h"

namespace dyno
{
	IMPLEMENT_TCLASS(ParticleSkinning, TDataType)

	template<typename TDataType>
	ParticleSkinning<TDataType> ::ParticleSkinning()
		:Node()
	{
		this->stateLevelSet()->setDataPtr(std::make_shared<LevelSet<TDataType>>());
		this->stateTriangleSet()->setDataPtr(std::make_shared<TriangleSet<TDataType>>());
		this->statePoints()->allocate();

		auto iso = std::make_shared<ComputeSurfaceLevelset<TDataType>>();
		this->stateLevelSet()->connect(iso->inLevelSet());
		this->stateGridSpacing()->connect(iso->inGridSpacing());
		this->statePoints()->connect(iso->inPoints());
		this->animationPipeline()->pushModule(iso);

	}

	template<typename TDataType>
	void ParticleSkinning<TDataType>::resetStates() {
		this->updateLevelset();
	};

	template<typename TDataType>
	void ParticleSkinning<TDataType>::preUpdateStates() {
		this->updateLevelset();
	};

	template<typename Coord>
	__global__ void constrGridPosition(
		DArray<Coord> GridPositions,
		Coord lo,
		int nx,
		int ny,
		int nz,
		Coord h
	)
	{
		int gId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (gId >= GridPositions.size()) return;
		
		uint nk = gId / (nx * ny);
		uint nj = (gId % (nx * ny)) / ny;
		uint ni = (gId % ny);

		GridPositions[gId][0] = (float)(ni) * h[0];
		GridPositions[gId][1] = (float)(nj)*h[1];
		GridPositions[gId][2] = (float)(nk)*h[2];

		//if ((nk == 1) && (nj == 1))
		//{
		//	//printf("ijk: %f, %f, %f \r\n", GridPositions[gId][0], GridPositions[gId][1], GridPositions[gId][2]);
		//	printf("%f, %f, %f \r\n", h[0], h[1], h[2]);
		//}
		
	};

	template<typename TDataType>
	void ParticleSkinning<TDataType> ::constrGridPositionArray() {
	
		auto& sdf = this->stateLevelSet()->getDataPtr()->getSDF();
		auto& leveset = this->stateLevelSet()->getDataPtr()->getSDF().getMDistance();
	
		int num = leveset.size();
		std::cout <<"Grid number " << num << std::endl;

		if (this->stateGridPoistion()->size() == 0)
		{
			this->stateGridPoistion()->resize(num);
		}

		cuExecute(num, constrGridPosition,
			this->stateGridPoistion()->getData(),
			this->stateLevelSet()->getDataPtr()->getSDF().lowerBound(),
			this->stateLevelSet()->getDataPtr()->getSDF().nx(),
			this->stateLevelSet()->getDataPtr()->getSDF().ny(),
			this->stateLevelSet()->getDataPtr()->getSDF().nz(),
			this->stateLevelSet()->getDataPtr()->getSDF().getH()
		)


	};

	template<typename TDataType>
	void ParticleSkinning<TDataType>::updateLevelset() 
	{

		auto& sdf = this->stateLevelSet()->getDataPtr()->getSDF();


		this->statePoints()->assign(this->getParticleSystem()->statePosition()->getData());
		auto particles = this->statePoints()->getData();

		std::cout << "Pos number : " << particles.size() << std::endl;

		Reduction<Coord> reduce;
		Coord hiBound = reduce.maximum(particles.begin(), particles.size());
		Coord loBound = reduce.minimum(particles.begin(), particles.size());
	
		Real h = this->stateGridSpacing()->getValue();
		

		hiBound = h * Coord(
			(Real)((int)(hiBound[0] / h)), 
			(Real)((int)(hiBound[1] / h)), 
			(Real)((int)(hiBound[2] / h)));

		loBound = h * Coord(
			(Real)((int)(loBound[0] / h)),
			(Real)((int)(loBound[1] / h)),
			(Real)((int)(loBound[2] / h)));

		std::cout << hiBound[0] << "," << hiBound[1] << "," << hiBound[2] << "," << std::endl;
		std::cout << loBound[0] << "," << loBound[1] << "," << loBound[2] << "," << std::endl;




		hiBound += 8.0 * h;			
		loBound -= 8.0 * h;
		uint nx = (hiBound - loBound)[0] / h;
		uint ny = (hiBound - loBound)[1] / h;
		uint nz = (hiBound - loBound)[2] / h;

		sdf.setSpace(loBound, hiBound, nx, ny, nz);

		constrGridPositionArray();

		};


	DEFINE_CLASS(ParticleSkinning);
}