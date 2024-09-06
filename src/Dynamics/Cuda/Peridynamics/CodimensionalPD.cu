#pragma once
#include "CodimensionalPD.h"
#include "ParticleSystem/Module/ParticleIntegrator.h"
#include "Primitive/Primitive3D.h"
#include "Topology/TriangleSet.h"
#include "Mapping/PointSetToPointSet.h"

#include "Module/CoSemiImplicitHyperelasticitySolver.h"
#include "Module/DamplingParticleIntegrator.h"


#include "Module/DragSurfaceInteraction.h"
#include "Module/DragVertexInteraction.h"

namespace dyno
{
	IMPLEMENT_TCLASS(CodimensionalPD, TDataType)

		template<typename TDataType>
	CodimensionalPD<TDataType>::CodimensionalPD()
		: TriangularSystem<TDataType>()
	{
		this->varHorizon()->setValue(0.0085);
		auto interaction = std::make_shared<DragVertexInteraction<TDataType>>();
		interaction->varCacheEvent()->setValue(false);
		this->stateTriangleSet()->connect(interaction->inInitialTriangleSet());
		this->statePosition()->connect(interaction->inPosition());
		this->stateVelocity()->connect(interaction->inVelocity());
		this->stateAttribute()->connect(interaction->inAttribute());
		this->stateTimeStep()->connect(interaction->inTimeStep());
		this->animationPipeline()->pushModule(interaction);

		auto integrator = std::make_shared<DamplingParticleIntegrator<TDataType>>();
		this->stateTimeStep()->connect(integrator->inTimeStep());
		this->statePosition()->connect(integrator->inPosition());
		this->stateVelocity()->connect(integrator->inVelocity());
		this->stateAttribute()->connect(integrator->inAttribute());
		this->stateForce()->connect(integrator->inForceDensity());
		this->stateDynamicForce()->connect(integrator->inDynamicForce());
		this->stateNorm()->connect(integrator->inNorm());
		integrator->inMu()->setValue(0.00);
		integrator->inAirDisspation()->setValue(1.0);
		this->stateContactForce()->connect(integrator->inContactForce());
		this->animationPipeline()->pushModule(integrator);

		auto hyperElasticity = std::make_shared<CoSemiImplicitHyperelasticitySolver<TDataType>>();
		this->mHyperElasticity = hyperElasticity;
		this->varHorizon()->connect(hyperElasticity->inHorizon());
		this->stateTimeStep()->connect(hyperElasticity->inTimeStep());
		this->varEnergyType()->connect(hyperElasticity->inEnergyType());
		this->varEnergyModel()->connect(hyperElasticity->inEnergyModels());
		this->stateRestPosition()->connect(hyperElasticity->inX());
		this->statePosition()->connect(hyperElasticity->inY());
		this->stateVelocity()->connect(hyperElasticity->inVelocity());
		this->stateRestNorm()->connect(hyperElasticity->inRestNorm());
		this->stateNorm()->connect(hyperElasticity->inNorm());
		this->stateRestShape()->connect(hyperElasticity->inBonds());
		this->stateAttribute()->connect(hyperElasticity->inAttribute());
		this->stateMaxLength()->connect(hyperElasticity->inUnit());
		this->stateOldPosition()->connect(hyperElasticity->inOldPosition());
		this->stateMarchPosition()->connect(hyperElasticity->inMarchPosition());
		this->stateDynamicForce()->connect(hyperElasticity->inDynamicForce());
		this->stateTriangleSet()->connect(hyperElasticity->inTriangularMesh());
		this->stateContactForce()->connect(hyperElasticity->inContactForce());

	
		hyperElasticity->setS(0.1);
		hyperElasticity->setXi(0.15);
		hyperElasticity->setE(2000.0);
		hyperElasticity->setK_bend(0.01);
		

		auto contact = hyperElasticity->getContactRulePtr();
		this->stateTriangleSet()->connect(contact->inTriangularMesh());
		this->stateOldPosition()->connect(contact->inOldPosition());
		this->stateVelocity()->connect(contact->inVelocity());
		this->stateTimeStep()->connect(contact->inTimeStep());
		this->stateMaxLength()->connect(contact->inUnit());
		this->stateMarchPosition()->connect(contact->inNewPosition());


		
		this->animationPipeline()->pushModule(hyperElasticity);
		
		EnergyModels<Real> funcs;
		Real s0 = hyperElasticity->getS0(2000,0.1);
		Real s1 = hyperElasticity->getS1(2000,0.1);
		funcs.linearModel = LinearModel<Real>(48000, 12000);
		funcs.neohookeanModel = NeoHookeanModel<Real>(s0,s1);
		funcs.stvkModel = StVKModel<Real>(1, 12000);
		funcs.xuModel = XuModel<Real>(hyperElasticity->getE());
		funcs.fiberModel = FiberModel<Real>(1, 1, 1);

		this->varEnergyModel()->setValue(funcs);

		this->setDt(0.001f);
	}

	template<typename TDataType>
	CodimensionalPD<TDataType>::CodimensionalPD(Real Xi_IN, Real E_IN, Real kb_IN, Real timeStep = 1e-3,
		std::string name = "default"): TriangularSystem<TDataType>(){

		this->varHorizon()->setValue(0.0085);
		auto interaction = std::make_shared<DragVertexInteraction<TDataType>>();
		interaction->varCacheEvent()->setValue(false);
		this->stateTriangleSet()->connect(interaction->inInitialTriangleSet());
		this->statePosition()->connect(interaction->inPosition());
		this->stateVelocity()->connect(interaction->inVelocity());
		this->stateAttribute()->connect(interaction->inAttribute());
		this->stateTimeStep()->connect(interaction->inTimeStep());
		this->animationPipeline()->pushModule(interaction);
		this->stateTimeStep()->setValue(timeStep);
		this->setDt(timeStep);

		auto integrator = std::make_shared<DamplingParticleIntegrator<TDataType>>();
		this->stateTimeStep()->connect(integrator->inTimeStep());
		this->statePosition()->connect(integrator->inPosition());
		this->stateVelocity()->connect(integrator->inVelocity());
		this->stateAttribute()->connect(integrator->inAttribute());
		this->stateForce()->connect(integrator->inForceDensity());
		this->stateDynamicForce()->connect(integrator->inDynamicForce());
		this->stateNorm()->connect(integrator->inNorm());
		integrator->inMu()->setValue(0.00);
		integrator->inAirDisspation()->setValue(0.999);
		this->stateContactForce()->connect(integrator->inContactForce());
		this->animationPipeline()->pushModule(integrator);

		auto hyperElasticity = std::make_shared<CoSemiImplicitHyperelasticitySolver<TDataType>>();
		this->mHyperElasticity = hyperElasticity;
		this->varHorizon()->connect(hyperElasticity->inHorizon());
		this->stateTimeStep()->connect(hyperElasticity->inTimeStep());
		this->varEnergyType()->connect(hyperElasticity->inEnergyType());
		this->varEnergyModel()->connect(hyperElasticity->inEnergyModels());
		this->stateRestPosition()->connect(hyperElasticity->inX());
		this->statePosition()->connect(hyperElasticity->inY());
		this->stateVelocity()->connect(hyperElasticity->inVelocity());
		this->stateRestNorm()->connect(hyperElasticity->inRestNorm());
		this->stateNorm()->connect(hyperElasticity->inNorm());
		this->stateRestShape()->connect(hyperElasticity->inBonds());
		this->stateAttribute()->connect(hyperElasticity->inAttribute());
		this->stateMaxLength()->connect(hyperElasticity->inUnit());
		this->stateOldPosition()->connect(hyperElasticity->inOldPosition());
		this->stateMarchPosition()->connect(hyperElasticity->inMarchPosition());
		this->stateDynamicForce()->connect(hyperElasticity->inDynamicForce());
		this->stateTriangleSet()->connect(hyperElasticity->inTriangularMesh());
		this->stateContactForce()->connect(hyperElasticity->inContactForce());


		hyperElasticity->setS(0.1);
		hyperElasticity->setXi(Xi_IN);
		hyperElasticity->setE(E_IN);
		hyperElasticity->setK_bend(kb_IN);


		auto contact = hyperElasticity->getContactRulePtr();
		this->stateTriangleSet()->connect(contact->inTriangularMesh());
		this->stateOldPosition()->connect(contact->inOldPosition());
		this->stateVelocity()->connect(contact->inVelocity());
		this->stateTimeStep()->connect(contact->inTimeStep());
		this->stateMaxLength()->connect(contact->inUnit());
		this->stateMarchPosition()->connect(contact->inNewPosition());



		this->animationPipeline()->pushModule(hyperElasticity);

		EnergyModels<Real> funcs;
		Real s0 = hyperElasticity->getS0(E_IN, kb_IN);
		Real s1 = hyperElasticity->getS1(E_IN, kb_IN);
		funcs.linearModel = LinearModel<Real>(48000, 12000);
		funcs.neohookeanModel = NeoHookeanModel<Real>(s0,s1);
		funcs.stvkModel = StVKModel<Real>(1, 12000);
		funcs.xuModel = XuModel<Real>(hyperElasticity->getE());
		funcs.fiberModel = FiberModel<Real>(1, 1, 1);

		this->varEnergyModel()->setValue(funcs);
	}

	template<typename TDataType>
	CodimensionalPD<TDataType>::~CodimensionalPD()
	{

	}

	

	template<typename TDataType>
	void CodimensionalPD<TDataType>::updateTopology()
	{
		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->stateTriangleSet()->getDataPtr());
		triSet->getPoints().assign(this->statePosition()->getData());
		triSet->updateAngleWeightedVertexNormal(this->stateNorm()->getData());
	}
	

	template<typename TDataType>
	void CodimensionalPD<TDataType>::loadSurface(std::string filename)
	{
		auto triSetPtr = TypeInfo::cast<TriangleSet<TDataType>>(this->stateTriangleSet()->getDataPtr());
		triSetPtr ->loadObjFile(filename);
		triSetPtr ->update();
	}
	

	template<typename TDataType>
	void CodimensionalPD<TDataType>::setEnergyModel(XuModel<Real> model)
	{
		this->varEnergyType()->setValue(Xuetal);

		auto models = this->varEnergyModel()->getValue();
		models.xuModel = model;

		this->varEnergyModel()->setValue(models);
	}

	template<typename TDataType>
	void CodimensionalPD<TDataType>::setEnergyModel(NeoHookeanModel<Real> model)
	{
		this->varEnergyType()->setValue(NeoHooekean);

		auto models = this->varEnergyModel()->getValue();
		models.neohookeanModel = model;

		this->varEnergyModel()->setValue(models);
	}

	template<typename TDataType>
	void CodimensionalPD<TDataType>::setEnergyModel(LinearModel<Real> model)
	{
		this->varEnergyType()->setValue(Linear);

		auto models = this->varEnergyModel()->getValue();
		models.linearModel = model;

		this->varEnergyModel()->setValue(models);
	}

	template<typename TDataType>
	void CodimensionalPD<TDataType>::setEnergyModel(StVKModel<Real> model)
	{
		this->varEnergyType()->setValue(StVK);

		auto models = this->varEnergyModel()->getValue();
		models.stvkModel = model;

		this->varEnergyModel()->setValue(models);
	}

	template<typename TDataType>
	void CodimensionalPD<TDataType>::setEnergyModel(FiberModel<Real> model)
	{
		this->varEnergyType()->setValue(Fiber);

		auto models = this->varEnergyModel()->getValue();
		models.fiberModel = model;

		this->varEnergyModel()->setValue(models);
	}

	template<typename TDataType>
	bool CodimensionalPD<TDataType>::translate(Coord t)
	{
		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->stateTriangleSet()->getDataPtr());
		triSet->translate(t);

		return true;
	}

	template<typename TDataType>
	bool CodimensionalPD<TDataType>::scale(Real s)
	{
		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->stateTriangleSet()->getDataPtr());
		triSet->scale(s);

		this->varHorizon()->setValue(s * this->varHorizon()->getData());

		return true;
	}

	template<typename TDataType>
	bool CodimensionalPD<TDataType>::scale(Coord s)
	{
		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->stateTriangleSet()->getDataPtr());
		triSet->scale(s);

		this->varHorizon()->setValue((s[0] + s[1] + s[2]) / 3.0f * this->varHorizon()->getData());

		return true;
	}

	template<typename TDataType>
	void CodimensionalPD<TDataType>::preUpdateStates()
	{
		
		auto& posOld = this->stateOldPosition()->getData();
		auto& posNew = this->statePosition()->getData();
		posOld.assign(posNew);
		
	}

	template<typename Matrix>
	__global__ void InitRotation(
		DArray<Matrix> rots)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= rots.size()) return;

		rots[pId] = Matrix::identityMatrix();
	}

	template<typename TDataType>
	void CodimensionalPD<TDataType>::resetStates()
	{
       TriangularSystem<TDataType>::resetStates();
		                           
		printf("inside reset States\n");
		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->stateTriangleSet()->getDataPtr());
		if (triSet == nullptr) return;
		printf("host attribute\n");

		CArray<Attribute> host_attribute;
		host_attribute.resize(triSet->getPoints().size());
		for (int i = 0; i < triSet->getPoints().size(); i++)
		{
			host_attribute[i] = Attribute();
		}

		CArray<Coord> triPoints;
		CArray<TopologyModule::Triangle> triIds;

		triPoints.resize(triSet->getPoints().size());
		triPoints.assign(triSet->getPoints());

		triIds.resize(triSet->getTriangles().size());
		triIds.assign(triSet->getTriangles());


		float global_min = 1000000;
		float global_max = 0.0f;
		for (int i = 0; i < triIds.size(); i++)
		{
			auto tri = triIds[i];

			Vec3f v0 = triPoints[tri[0]];
			Vec3f v1 = triPoints[tri[1]];
			Vec3f v2 = triPoints[tri[2]];
			

			Vec3f min_v = v0.minimum(v1).minimum(v2);
			Vec3f max_v = v0.maximum(v1).maximum(v2);

			Vec3f bounding = max_v - min_v;

			float max_edge = maximum(maximum(bounding[0], bounding[1]), bounding[2]);

			global_min = max_edge < global_min ? max_edge : global_min;
			global_max = max_edge > global_max ? max_edge : global_max;
		}

	
	
		this->varHorizon()->setValue(1.5 * global_max);

		printf("host attr size %d\n", host_attribute.size());

		int vNum = triSet->getPoints().size();
		
		
		this->stateRestNorm()->resize(vNum);
		triSet->updateAngleWeightedVertexNormal(this->stateRestNorm()->getData());

	
		this->stateNorm()->resize(vNum);
		this->stateNorm()->getData().assign(this->stateRestNorm()->getData(), this->stateRestNorm()->getData().size());

		
		this->stateRestPosition()->resize(vNum);
		this->stateRestPosition()->assign(triSet->getPoints());

		this->stateVerticesRef()->resize(vNum);
		this->stateVerticesRef()->getData().assign(triSet->getPoints(), triSet->getPoints().size());
	
		this->stateOldPosition()->resize(vNum);
		this->stateOldPosition()->getData().assign(triSet->getPoints(), triSet->getPoints().size());

		this->stateMarchPosition()->resize(vNum);
		this->stateMarchPosition()->getData().assign(triSet->getPoints(), triSet->getPoints().size());

		this->stateAttribute()->resize(vNum);
		this->stateAttribute()->getData().assign(host_attribute, host_attribute.size());


		this->stateVertexRotation()->resize(vNum);

		this->stateDynamicForce()->resize(vNum);

		this->stateContactForce()->resize(vNum);

		cuExecute(vNum,
			InitRotation,
			this->stateVertexRotation()->getData());

		host_attribute.clear();

		triIds.clear();
		triPoints.clear();

		this->updateVolume();
		this->updateRestShape();
		printf("end of reset States\n");
	}


	__global__ void CPD_SetSize(
		DArray<uint> index,
		DArrayList<int> lists)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= lists.size()) return;
		index[pId] = lists[pId].size() + 1;
	}


	template<typename Coord, typename Bond>
	__global__ void SetRestShape(
		DArray<Coord> restPos,
		DArrayList<Bond> restShapes,
		DArrayList<int> lists)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= lists.size()) return;


		List<Bond>& list_i = restShapes[pId];

		List<int> list = lists[pId];

		list_i.insert(Bond(pId, restPos[pId]));

		for (auto it = list.begin(); it != list.end(); it++)
		{
			int j = *it;

			list_i.insert(Bond(j, restPos[j]));
		}


		int size_i = restShapes[pId].size();
	}

	template<typename TDataType>
	void CodimensionalPD<TDataType>::updateRestShape()
	{
		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->stateTriangleSet()->getDataPtr());
		if (triSet == nullptr) return;

		DArrayList<int> neighbors;
		triSet->requestPointNeighbors(neighbors);

		auto& restPos = this->stateRestPosition()->getData();

		this->stateRestShape()->resize(restPos.size());

		auto index = this->stateRestShape()->getData().index();
		auto elements = this->stateRestShape()->getData().elements();

		DArray<uint> index_temp;
		index_temp.resize(index.size());

		cuExecute(neighbors.size(),
			CPD_SetSize,
			index_temp,
			neighbors);

		this->stateRestShape()->getData().resize(index_temp);

		cuExecute(neighbors.size(),
			SetRestShape,
			restPos,
			stateRestShape()->getData(),
			neighbors);

		neighbors.clear();
	}

	template<typename Real, typename Coord, typename Triangle>
	__global__ void HB_UpdateUnit(
		DArray<Coord> restPos,
		DArray<Triangle> tris,
		DArray<Real> Min,
		DArray<Real> Max) {

		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= tris.size()) return;

		auto Coordnorm = [](Coord res) {
			Real c = Real(0);
			Real tmp = res[0] * res[0] + res[1] * res[1] + res[2] * res[2];
			c = sqrt(tmp);
			return c;
		};

		Triangle triIndex = tris[tId];
		Real length1 = Coordnorm(restPos[triIndex[0]] - restPos[triIndex[1]]);
		Real length2 = Coordnorm(restPos[triIndex[1]] - restPos[triIndex[2]]);
		Real length3 = Coordnorm(restPos[triIndex[2]] - restPos[triIndex[0]]);
		auto tmpMax = maximum(maximum(length3, Real(1e-9)), maximum(length1, length2));
		auto tmpMin = minimum(maximum(length3, Real(1e-9)), minimum(length1, length2));
		Min[tId] = tmpMin;
		Max[tId] = tmpMax;
	}

	template<typename Real, typename Coord, typename Triangle>
	__global__ void HB_CalculateVolume(
		DArray<Real> volume,
		DArray<Coord> restPos,
		DArray<Triangle> tris,
		DArrayList<int> lists)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= volume.size()) return;

		Real vol_i = Real(0);

		List<int>& list_i = lists[pId];
		int nbSize = list_i.size();

		for (int i = 0; i < nbSize; i++)
		{
			int triId = list_i[i];//lists.getElement(pId, i);
			Triangle triIndex = tris[triId];

			TTriangle3D<Real> tri(restPos[triIndex[0]], restPos[triIndex[1]], restPos[triIndex[2]]);

			vol_i += abs(tri.area());
		}

		//volume[pId] = samplingDistance * samplingDistance * samplingDistance;//1.0f;//maximum(vol_i, Real(0.0000000000001));
		volume[pId] = maximum(vol_i, Real(0.0000000000001));
		
	}

	template<typename TDataType>
	void CodimensionalPD<TDataType>::updateVolume()
	{
		auto triSet = TypeInfo::cast<TriangleSet<TDataType>>(this->stateTriangleSet()->getDataPtr());
		if (triSet == nullptr) return;

		auto& ver2Tri = triSet->getVertex2Triangles();

		auto& restPos = this->stateRestPosition()->getData();
		
// 		if (this->stateMaxLength()->isEmpty()) {
// 			this->stateMaxLength()->allocate();
// 		}
// 		if (this->stateMinLength()->isEmpty()) {
// 			this->stateMinLength()->allocate();
// 		}

		DArray<Real> maxL;
		maxL.resize(triSet->getTriangles().size());
		DArray<Real> minL;
		minL.resize(triSet->getTriangles().size());

		cuExecute(triSet->getTriangles().size(),
			HB_UpdateUnit,
			restPos,
			triSet->getTriangles(),
			minL,
			maxL
			);
		
		this->stateVolume()->resize(restPos.size());

		cuExecute(restPos.size(),
			HB_CalculateVolume,
			this->stateVolume()->getData(),
			restPos,
			triSet->getTriangles(),
			ver2Tri);

		auto& volume = this->stateVolume()->getData();

		Reduction<Real> reduce;
		Real max_vol = reduce.maximum(volume.begin(), volume.size());
		Real min_vol = reduce.minimum(volume.begin(), volume.size());
		Real min_L = reduce.minimum(minL.begin(), minL.size());
		Real max_L = reduce.maximum(maxL.begin(), maxL.size());
		this->stateMaxLength()->setValue(max_L);
		this->stateMinLength()->setValue(min_L);
		printf("max vol: %f; min vol: %f \n", max_vol, min_vol);
		printf("max length: %f; min length: %f \n",this->stateMaxLength()->getData() , this->stateMinLength()->getData());
	}

	DEFINE_CLASS(CodimensionalPD);
}