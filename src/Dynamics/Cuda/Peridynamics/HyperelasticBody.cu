#pragma once
#include "HyperelasticBody.h"

#include "Primitive/Primitive3D.h"
#include "Topology/TetrahedronSet.h"
#include "Mapping/PointSetToPointSet.h"

#include "ParticleSystem/Module/ParticleIntegrator.h"

#include "Primitive/Primitive3D.h"


#include "Topology/DistanceField3D.h"

#include "Module/SemiImplicitHyperelasticitySolver.h"
#include "Module/CalculateNormalSDF.h"

#include "Auxiliary/DataSource.h"

namespace dyno
{
	IMPLEMENT_TCLASS(HyperelasticBody, TDataType)

	template<typename TDataType>
	HyperelasticBody<TDataType>::HyperelasticBody()
		: TetrahedralSystem<TDataType>()
	{
		auto horizon = std::make_shared<FloatingNumber<TDataType>>();
		horizon->varValue()->setValue(0.0085f);
		this->animationPipeline()->pushModule(horizon);

		//四面体邻域大小，求解超弹性时需要访问的状态变量，不需要更新
		//this->varHorizon()->setValue(0.0085);
		//粒子时间积分器
		auto integrator = std::make_shared<ParticleIntegrator<TDataType>>();
		this->stateTimeStep()->connect(integrator->inTimeStep());
		this->statePosition()->connect(integrator->inPosition());
		this->stateVelocity()->connect(integrator->inVelocity());
		this->stateAttribute()->connect(integrator->inAttribute());
		this->stateForce()->connect(integrator->inForceDensity());
		this->animationPipeline()->pushModule(integrator);
		//超弹性求解模块
		auto hyperElasticity = std::make_shared<SemiImplicitHyperelasticitySolver<TDataType>>();
		horizon->outFloating()->connect(hyperElasticity->inHorizon());
		this->stateTimeStep()->connect(hyperElasticity->inTimeStep());
		this->varEnergyType()->connect(hyperElasticity->inEnergyType());
		this->varEnergyModel()->connect(hyperElasticity->inEnergyModels());
		this->stateRestPosition()->connect(hyperElasticity->inX());
		this->statePosition()->connect(hyperElasticity->inY());
		this->stateVelocity()->connect(hyperElasticity->inVelocity());
		this->stateVolume()->connect(hyperElasticity->inVolume());
		this->stateBonds()->connect(hyperElasticity->inBonds());
		this->stateAttribute()->connect(hyperElasticity->inAttribute());
		this->stateVolumePair()->connect(hyperElasticity->inVolumePair());
		this->varAlphaComputed()->connect(hyperElasticity->varIsAlphaComputed());
		this->animationPipeline()->pushModule(hyperElasticity);
		//弹性能量，求解超弹性时需要访问的状态变量，不需要更新
		EnergyModels<Real> funcs;
		funcs.linearModel = LinearModel<Real>(48000000, 12000000);
		funcs.neohookeanModel = NeoHookeanModel<Real>(48000000, 12000000);
		funcs.stvkModel = StVKModel<Real>(48000000, 12000000);
		funcs.xuModel = XuModel<Real>(12000000);
		this->varEnergyModel()->setValue(funcs);



		auto CalcSDF = std::make_shared<CalculateNormalSDF<TDataType>>();
		this->statePosition()->connect(CalcSDF->inPosition());
		this->stateTets()->connect(CalcSDF->inTets());
		this->stateDisranceSDF()->connect(CalcSDF->inDisranceSDF());
		this->stateNormalSDF()->connect(CalcSDF->inNormalSDF());
		this->animationPipeline()->pushModule(CalcSDF);


		Coord lo(0.0f);
		Coord hi(1.0f);
		m_cSDF = std::make_shared<DistanceField3D<DataType3f>>();
		m_cSDF->setSpace(lo - 0.025f, hi + 0.025f, 105, 105, 105);
		m_cSDF->loadBox(lo, hi, true);

		this->setDt(Real(0.001));
	}

	template<typename TDataType>
	HyperelasticBody<TDataType>::~HyperelasticBody()
	{
		m_cSDF->release();
	}

	template<typename TDataType>
	void HyperelasticBody<TDataType>::setEnergyModel(XuModel<Real> model)
	{
		this->varEnergyType()->setValue(Xuetal);
		auto models = this->varEnergyModel()->getValue();
		models.xuModel = model;

		this->varEnergyModel()->setValue(models);
	}

	template<typename TDataType>
	void HyperelasticBody<TDataType>::setEnergyModel(NeoHookeanModel<Real> model)
	{
		this->varEnergyType()->setValue(NeoHooekean);
		auto models = this->varEnergyModel()->getValue();
		models.neohookeanModel = model;

		this->varEnergyModel()->setValue(models);
	}

	template<typename TDataType>
	void HyperelasticBody<TDataType>::setEnergyModel(LinearModel<Real> model)
	{
		this->varEnergyType()->setValue(Linear);
		auto models = this->varEnergyModel()->getValue();
		models.linearModel = model;

		this->varEnergyModel()->setValue(models);
	}

	template<typename TDataType>
	void HyperelasticBody<TDataType>::setEnergyModel(StVKModel<Real> model)
	{
		this->varEnergyType()->setValue(StVK);

		auto models = this->varEnergyModel()->getValue();
		models.stvkModel = model;

		this->varEnergyModel()->setValue(models);
	}

	template<typename TDataType>
	bool HyperelasticBody<TDataType>::translate(Coord t)
	{
		auto triSet = this->stateTetrahedronSet()->getDataPtr();
		triSet->translate(t);

		return true;
	}

	template<typename TDataType>
	bool HyperelasticBody<TDataType>::scale(Real s)
	{
		auto triSet = this->stateTetrahedronSet()->getDataPtr();
		triSet->scale(s);

		this->varHorizon()->setValue(s * this->varHorizon()->getData());

		return true;
	}

	template<typename TDataType>
	bool HyperelasticBody<TDataType>::scale(Coord s)
	{
		auto triSet = this->stateTetrahedronSet()->getDataPtr();
		triSet->scale(s);

		this->varHorizon()->setValue((s[0] + s[1] + s[2]) / 3.0f * this->varHorizon()->getData());

		return true;
	}

	template<typename TDataType>
	bool HyperelasticBody<TDataType>::rotate(Coord angle)
	{
		auto triSet = this->stateTetrahedronSet()->getDataPtr();
		triSet->rotate(angle);

		//TypeInfo::cast<TriangleSet<TDataType>>(m_surfaceNode->stateTopology()->getDataPtr())->rotate(angle);

		return true;
	}

	template<typename TDataType>
	bool HyperelasticBody<TDataType>::rotate(Quat<Real> angle)
	{
		auto ptSet = this->stateTetrahedronSet()->getDataPtr();
		ptSet->rotate(angle);

		return true;
	}

	template<typename Matrix>
	__global__ void InitRotation(
		DArray<Matrix> rots)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= rots.size()) return;

		rots[pId] = Matrix::identityMatrix();
	}

	/*template<typename TDataType>
	void HyperelasticBody<TDataType>::updateStates()
	{
		std::cout << "update:" << this->varFileName()->getValue().string() << std::endl;
	}*/
	template<typename TDataType>
	void HyperelasticBody<TDataType>::resetStates()
	{
		std::cout << this->varFileName()->getValue().string() << std::endl;
		if (this->varFileName()->getValue().string().length() > 1)
		{
			this->loadVertexFromGmshFile(this->varFileName()->getValue().string());
			this->scale(this->varScale()->getValue());
			this->translate(this->varLocation()->getValue());
			//ParticleSystem::resetStates();
		}
		printf("inside reset States\n");
		auto tetSet = TypeInfo::cast<TetrahedronSet<TDataType>>(this->stateTetrahedronSet()->getDataPtr());
		if (tetSet == nullptr) return;
		printf("host attribute\n");

		CArray<Attribute> host_attribute;
		host_attribute.resize(tetSet->getPoints().size());
		for (int i = 0; i < tetSet->getPoints().size(); i++)
		{
			host_attribute[i] = Attribute();
		}

		CArray<Coord> tetPoints;
		CArray<TopologyModule::Tetrahedron> tetIds;
		
		tetPoints.resize(tetSet->getPoints().size());
		tetPoints.assign(tetSet->getPoints());

		tetIds.resize(tetSet->getTetrahedrons().size());
		tetIds.assign(tetSet->getTetrahedrons());


		this->stateTets()->resize(tetSet->getTetrahedrons().size());
		this->stateTets()->getData().assign(tetSet->getTetrahedrons());

		this->stateNormalSDF()->resize(tetSet->getTetrahedrons().size());

		float global_min = 1000000;
		float global_max = 0.0f;
		for (int i = 0; i < tetIds.size(); i++)
		{
			auto tet = tetIds[i];

			Vec3f v0 = tetPoints[tet[0]];
			Vec3f v1 = tetPoints[tet[1]];
			Vec3f v2 = tetPoints[tet[2]];
			Vec3f v3 = tetPoints[tet[3]];

			Vec3f min_v = v0.minimum(v1).minimum(v2.minimum(v3));
			Vec3f max_v = v0.maximum(v1).maximum(v2.maximum(v3));

			Vec3f bounding = max_v - min_v;

			float max_edge = maximum(maximum(bounding[0], bounding[1]), bounding[2]);

			global_min = max_edge < global_min ? max_edge : global_min;
			global_max = max_edge > global_max ? max_edge : global_max;
		}

		tetPoints.clear();
		tetIds.clear();
		this->varHorizon()->setValue(1.5 * global_max);

		int vNum = tetSet->getPoints().size();

		this->stateRestPosition()->resize(vNum);
		//Function1Pt::copy(this->stateRestPosition()->getData(), tetSet->getPoints());
		this->stateRestPosition()->getData().assign(tetSet->getPoints(), tetSet->getPoints().size());

		this->statePosition()->resize(vNum);
		this->statePosition()->getData().assign(tetSet->getPoints(), tetSet->getPoints().size());

		this->stateVelocity()->resize(vNum);
		this->stateVelocity()->getDataPtr()->reset();
		this->stateForce()->resize(vNum);


		this->stateAttribute()->resize(vNum);
		//Function1Pt::copy(this->currentAttribute()->getData(), host_attribute);
		this->stateAttribute()->getData().assign(host_attribute, host_attribute.size());

		this->stateVertexRotation()->resize(vNum);
		cuExecute(vNum,
			InitRotation,
			this->stateVertexRotation()->getData());

		host_attribute.clear();

		HyperelasticBody<TDataType>::updateVolume();
		HyperelasticBody<TDataType>::updateRestShape();
	}


	__global__ void SetSize(
		DArray<uint> index,
		DArrayList<int> lists)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= lists.size()) return;

		//printf("size_0 %d\n", lists[pId].size() + 1);
		index[pId] = lists[pId].size();
	}

	template<typename Coord, typename Bond, typename Tetrahedron>
	__global__ void SetRestShape(
		DArrayList<Bond> restShapes,
		DArrayList<Real> volume,
		DArrayList<int> ver2tet,
		DArrayList<int> lists,
		DArray<Coord> tetVertex,
		DArray<Tetrahedron> tetIndex)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= lists.size()) return;

		Coord v_i = tetVertex[pId];

		List<Bond>& list_i = restShapes[pId];
		List<Real>& vol_i = volume[pId];
		List<int>& tets_i = ver2tet[pId];

		List<int> list = lists[pId];
		for (auto it = list.begin(); it != list.end(); it++)
		{
			int j = *it;

			Real vol_ij = Real(0);
			for (auto ik = tets_i.begin(); ik != tets_i.end(); ik++)
			{
				int k = *ik;
				Tetrahedron t_k = tetIndex[k];

				if (t_k[0] == j || t_k[1] == j || t_k[2] == j || t_k[3] == j)
				{
					TTet3D<Real> tet(tetVertex[t_k[0]], tetVertex[t_k[1]], tetVertex[t_k[2]], tetVertex[t_k[3]]);

					vol_ij += abs(tet.volume());
				}
			}

			Real minVol = Real(0.00001);
			vol_ij = maximum(vol_ij, minVol); //0.000123;// 

			list_i.insert(Bond(j, tetVertex[j] - v_i));
			vol_i.insert(vol_ij);
		}
		
		int size_i = restShapes[pId].size();
	}

	template<typename TDataType>
	void HyperelasticBody<TDataType>::updateRestShape()
	{
		printf("updateRestShape1\n");
		auto tetSet = TypeInfo::cast<TetrahedronSet<TDataType>>(this->stateTetrahedronSet()->getDataPtr());
		if (tetSet == nullptr) return;

		DArrayList<int> neighbors;
		tetSet->requestPointNeighbors(neighbors);

		auto ver2tet = tetSet->getVer2Tet();
		printf("+++++++");
		auto& restPos = this->stateRestPosition()->getData();
		//printf("------");
		//this->stateRestShape()->resize(restPos.size());
		if(this->stateBonds()->isEmpty())
			this->stateBonds()->allocate();

		if (this->stateVolumePair()->isEmpty())
		{
			this->stateVolumePair()->allocate();
		}

		//printf("aaaaaaaa");
		auto nbrPtr = this->stateBonds()->getDataPtr();
		//printf("bbbbbbbb");
		nbrPtr->resize(restPos.size());

		auto index = this->stateBonds()->getData().index();
		auto elements = this->stateBonds()->getData().elements();

		DArray<uint> index_temp;
		index_temp.resize(index.size());

		cuExecute(neighbors.size(),
			SetSize,
			index_temp,
			neighbors);


		this->stateBonds()->getData().resize(index_temp);
		this->stateVolumePair()->getDataPtr()->resize(index_temp);

		cuExecute(neighbors.size(),
			SetRestShape,
			stateBonds()->getData(),
			stateVolumePair()->getData(),
			ver2tet,
			neighbors,
			restPos,
			tetSet->getTetrahedrons());

		neighbors.clear();
	}

	template<typename Real, typename Coord, typename Tetrahedron>
	__global__ void HB_CalculateVolume(
		DArray<Real> volume,
		DArray<Coord> restPos,
		DArray<Tetrahedron> tets,
		DArrayList<int> lists)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= volume.size()) return;

		Real vol_i = Real(0);

		List<int>& list_i = lists[pId];
		int nbSize = list_i.size();

		for (int i = 0; i < nbSize; i++)
		{
			int tetId = list_i[i];//lists.getElement(pId, i);
			Tetrahedron tetIndex = tets[tetId];

			TTet3D<Real> tet(restPos[tetIndex[0]], restPos[tetIndex[1]], restPos[tetIndex[2]], restPos[tetIndex[3]]);

			vol_i += abs(tet.volume());
		}

		Real minVol = Real(0.00001);
		volume[pId] = maximum(vol_i, minVol); //0.000123;// 

//		printf("%f \n", volume[pId]);
	}

	template<typename TDataType>
	void HyperelasticBody<TDataType>::updateVolume()
	{
		auto tetSet = TypeInfo::cast<TetrahedronSet<TDataType>>(this->stateTetrahedronSet()->getDataPtr());
		if (tetSet == nullptr) return;

		auto& ver2Tet = tetSet->getVer2Tet();

		auto& restPos = this->stateRestPosition()->getData();

		this->stateVolume()->resize(restPos.size());

		cuExecute(restPos.size(),
			HB_CalculateVolume,
			this->stateVolume()->getData(),
			restPos,
			tetSet->getTetrahedrons(),
			ver2Tet);

		auto& volume = this->stateVolume()->getData();

		Reduction<Real> reduce;
		Real max_vol = reduce.maximum(volume.begin(), volume.size());
		Real min_vol = reduce.minimum(volume.begin(), volume.size());

		printf("max vol: %f; min vol: %f \n", max_vol, min_vol);
	}


	template<typename Real, typename Coord, typename Tetrahedron, typename TDataType>
	__global__ void K_InitTetCenterSDF(
		DArray<Coord> posArr,
		DArray<Tetrahedron> tets,
		DistanceField3D<TDataType> df,
		DArray<Real> distanceTetCenter)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= tets.size()) return;

		Coord posCenter = (posArr[tets[pId][0]] + posArr[tets[pId][1]] + posArr[tets[pId][2]] + posArr[tets[pId][3]]) / 4.0f;
		Coord normal;
		Real dist;
		df.getDistance(posCenter, dist, normal);
		distanceTetCenter[pId] = dist;
		
	}

	template<typename Real, typename Coord, typename TDataType>
	__global__ void K_InitTetVertexSDF(
		DArray<Coord> posArr,
		DistanceField3D<TDataType> df,
		DArray<Real> distanceTetVertex)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Coord posCenter = posArr[pId];
		Coord normal;
		Real dist;
		df.getDistance(posCenter, dist, normal);
		distanceTetVertex[pId] = dist;

	}



	

	template<typename TDataType>
	void HyperelasticBody<TDataType>::loadSDF(std::string filename, bool inverted)
	{
		m_cSDF->loadSDF(filename, inverted);

		auto tetSet = TypeInfo::cast<TetrahedronSet<TDataType>>(this->stateTetrahedronSet()->getDataPtr());
		if (tetSet == nullptr) return;

		//auto& ver2Tet = tetSet->getVer2Tet();

		auto& restPos = tetSet->getPoints();

		initDistance.resize(restPos.size());

		cuExecute(restPos.size(),
			K_InitTetVertexSDF,
			restPos,
			*m_cSDF,
			initDistance);

		printf("tet size = %d\n", tetSet->getTetrahedrons().size());

		this->stateDisranceSDF()->resize(restPos.size());
		this->stateDisranceSDF()->getData().assign(initDistance);
		this->stateNormalSDF()->resize(tetSet->getTetrahedrons().size());
		this->stateNormalSDF()->getData().reset();
		this->varSDF()->setValue(true);

		//printf("max vol: %f; min vol: %f \n", max_vol, min_vol);
	}

	DEFINE_CLASS(HyperelasticBody);
}