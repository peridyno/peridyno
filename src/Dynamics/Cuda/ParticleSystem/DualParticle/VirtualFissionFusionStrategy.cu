#include "VirtualFissionFusionStrategy.h"
#include "Node.h"
#include "Matrix/MatrixFunc.h"
#include <thrust/sort.h>

namespace dyno
{

	IMPLEMENT_TCLASS(VirtualFissionFusionStrategy, TDataType)

	template<typename TDataType>
	VirtualFissionFusionStrategy<TDataType>::VirtualFissionFusionStrategy()
		: VirtualParticleGenerator<TDataType>()
	{
		this->varRestDensity()->setValue(Real(1000));
		mSummation = std::make_shared<SummationDensity<TDataType>>();
		this->varRestDensity()->quote(mSummation->varRestDensity());
		this->inSmoothingLength()->connect(mSummation->inSmoothingLength());
		this->inSamplingDistance()->connect(mSummation->inSamplingDistance());
		this->inRPosition()->connect(mSummation->inPosition());
		this->inNeighborIds()->connect(mSummation->inNeighborIds());
		mSummation->outDensity()->connect(this->outDensity());
	}

	template<typename TDataType>
	VirtualFissionFusionStrategy<TDataType>::~VirtualFissionFusionStrategy()
	{
		mFissionParticles.clear();
		mFussionParticles.clear();
		mDivergence.clear();					
		mCurrentParticleStates.clear();		
		mPreParticleStates.clear();			
		mAnchorPoints.clear();
		mAnchorPointCodes.clear();
		mNonRepeatedCount.clear();
		mCandidateCodes.clear();
		mFissionParticleIds.clear();
		mFussionParticleIds.clear();
		mFissionVirtualParicles.clear();
		mFussionVirtualParticles.clear();
		fissions.clear();
		fussions.clear();
		ArrayPointer.clear();
		mEigens.clear();
		mThinSheets.clear();
	}

	template <typename Real, typename Coord>
	__global__ void GridFission_RealParticleDivergence(
		DArray<Real> divergences,
		DArray<Real> densities,
		DArray<Coord> positions,
		DArray<Coord> velocities,
		DArray<uint> particleStates,
		DArray<uint> PreStates,
		DArrayList<int> neighbors,
		Real transition,
		Real mass,
		Real smoothingLength,
		CubicKernel<Real> kernel
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= positions.size()) return;

		Coord& pos_i = positions[pId];

		divergences[pId] = 0.0f;
		Real div = 0.0f;
		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Real r = (pos_i - positions[j]).norm();
			if (r > EPSILON)
			{
				Coord gw = kernel.Gradient(r, smoothingLength) * (pos_i - positions[j]) / r;
				div += (velocities[j]- velocities[pId]).dot(gw);
			}
		}
		divergences[pId] = div * mass / densities[pId];
	}

	template <typename Real, typename Coord>
	__global__ void GridFission_StateJudge(
		DArray<uint> particleStates,
		DArray<Real> divergences,
		DArray<Real> densities,
		DArray<bool> thinSheets,
		DArray<Coord> positions,
		DArray<uint> PreStates,
		Real RestDensity,
		int Criteria,
		Real transition,
		Real mass,
		Real smoothingLength,
		CubicKernel<Real> kernel
	) {
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= particleStates.size()) return;

		if (Criteria == 0)
		{
			Real theta = transition;
			if (densities[pId] > 0.99 * RestDensity) theta = 100.0 * transition;
			if ((divergences[pId] < theta) && (divergences[pId] >= 0.0f))
			{
				//Unclear 
				particleStates[pId] = 0;
			}
			else if (divergences[pId] < 0.0f)
			{
				//Compressed
				particleStates[pId] = 1;
			}
			else
			{
				//Stretched
				particleStates[pId] = 2;
			}
		}
		else if (Criteria == 1)		//thin sheet
		{
			if (thinSheets[pId])
			{
				particleStates[pId] = 2;
			}
			else {
				particleStates[pId] = 1;
			}
		}
		else //if (Criteria == 2)
		{
			if (thinSheets[pId])
			{
				particleStates[pId] = 2;
			}
			else {
				Real theta = transition;
				if (densities[pId] > 0.99 * RestDensity) theta = 100.0 * transition;
				if ((divergences[pId] < theta) && (divergences[pId] >= 0.0f))
				{
					//Unclear 
					particleStates[pId] = 0;
				}
				else if (divergences[pId] < 0.0f)
				{
					//Compressed
					particleStates[pId] = 1;
				}
				else
				{
					//Stretched
					particleStates[pId] = 2;
				}
			}
		}
	}

	template <typename Real, typename Coord>
	__global__ void GridFission_StatesCorrection(
		DArray<uint> particleStates,
		DArray<uint> PreStates,
		DArray<Coord> positions,
		DArrayList<int> neighbors,
		Real smoothingLength
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= particleStates.size()) return;

		if (particleStates[pId] == 0)
		{
			if (PreStates[pId] != 0)
			{
				particleStates[pId] = PreStates[pId];
			}
			else
			{
				particleStates[pId] = 1;
			}
		}
		if (neighbors[pId].size() <= 1 )
		{
			particleStates[pId] = 1;
		}

	}

	__global__ void GridFission_intArrayReset(
		DArray<uint> arr,
		uint value
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= arr.size()) return;
		arr[pId] = value;
	}

	template <typename Coord>
	__global__ void GridFission_FissionVirtualCorrect
	(
		DArray<Coord> virtualParticles,
		DArray<Coord> RPos,
		DArray<bool> new_types,
		DArray<uint> new_VirtualToRealIds
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= virtualParticles.size()) return;

		if (new_types[id] == true)
		{
			virtualParticles[id] = RPos[new_VirtualToRealIds[id]];
		}
	}

	template <typename Coord>
	__global__ void GridFission_FissionAndFusionCounter
	(
		DArray<Coord> ParticleStates,
		DArray<Coord> fissions,
		DArray<Coord> fussions
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= ParticleStates.size()) return;

		fissions[pId] = 0;
		fussions[pId] = 0;

		if (ParticleStates[pId] == 1)
		{
			fussions[pId] = 1;
		}
		else if (ParticleStates[pId] == 2)
		{
			fissions[pId] = 1;
		}
		else
		{
			fussions[pId] = 1;
		}
	}

	template <typename Coord>
	__global__ void GridFission_PickUpParticles
	(
		DArray<Coord> PickedParticles,
		DArray<uint> PickedParticleIds,
		DArray<Coord> particles,
		DArray<uint> id_pointer,
		DArray<uint> flags
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= particles.size()) return;

		if (flags[pId] == 1)
		{
			PickedParticles[id_pointer[pId]] = particles[pId];
			PickedParticleIds[id_pointer[pId]] = pId;
		}
	}


	/*
	*@brief	Virtual particles' candinate points
	*		Every particle has 8 neighbors.
	*/
	template<typename Real, typename Coord>
	__global__ void GridFission_AnchorNeighbor_8
	(
		DArray<Coord> anchorPoint,
		DArray<Coord> pos,
		Coord origin,
		Real dh
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= pos.size()) return;

		Coord pos_ref = pos[id] - origin + Coord(dh / 4.0);
		Coord a(0);

		a[0] = (Real)((int)(floor(pos_ref[0] / dh))) * dh;
		a[1] = (Real)((int)(floor(pos_ref[1] / dh))) * dh;
		a[2] = (Real)((int)(floor(pos_ref[2] / dh))) * dh;

		anchorPoint[id * 8] = Coord(a[0], a[1], a[2]);
		anchorPoint[id * 8 + 1] = Coord(a[0] + dh, a[1], a[2]);
		anchorPoint[id * 8 + 2] = Coord(a[0] + dh, a[1] + dh, a[2]);
		anchorPoint[id * 8 + 3] = Coord(a[0] + dh, a[1] + dh, a[2] + dh);
		anchorPoint[id * 8 + 4] = Coord(a[0], a[1] + dh, a[2]);
		anchorPoint[id * 8 + 5] = Coord(a[0], a[1] + dh, a[2] + dh);
		anchorPoint[id * 8 + 6] = Coord(a[0], a[1], a[2] + dh);
		anchorPoint[id * 8 + 7] = Coord(a[0] + dh, a[1], a[2] + dh);

	}

	/*
	*@brief	Virtual particles' candinate points
	*		Every particle has 27 neighbors.
	*/
	template<typename Real, typename Coord>
	__global__ void GridFission_AnchorNeighbor_27
	(
		DArray<Coord> anchorPoint,
		DArray<Coord> pos,
		Coord origin,
		Real dh
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= pos.size()) return;

		Coord pos_ref = pos[id] - origin + dh / 2;

		for (int i = 0; i < 27; i++)
		{
			anchorPoint[id * 27 + i] = Coord(
				(Real)((int)(floor(pos_ref[0] / dh))) * dh + dh * Real(diff_v[i][0]),
				(Real)((int)(floor(pos_ref[1] / dh))) * dh + dh * Real(diff_v[i][1]),
				(Real)((int)(floor(pos_ref[2] / dh))) * dh + dh * Real(diff_v[i][2])
			);
		}

	}

	/*
	*@brief	Virtual particles' candinate points
	*		Every particle has 33 neighbors.
	*/
	template<typename Real, typename Coord>
	__global__ void GridFission_AnchorNeighbor_33
	(
		DArray<Coord> anchorPoint,
		DArray<Coord> pos,
		Coord origin,
		Real dh
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= pos.size()) return;

		//Coord pos_ref = pos[id] - origin + dh / 2;
		Coord pos_ref = pos[id] - origin + dh / 2;
		Coord a(0);

		a[0] = (Real)((int)(floor(pos_ref[0] / dh))) * dh;
		a[1] = (Real)((int)(floor(pos_ref[1] / dh))) * dh;
		a[2] = (Real)((int)(floor(pos_ref[2] / dh))) * dh;


		for (int i = 0; i < 33; i++)
		{
			anchorPoint[id * 33 + i] = Coord(a[0] + dh * Real(diff_v_33[i][0]),
				a[1] + dh * Real(diff_v_33[i][1]),
				a[2] + dh * Real(diff_v_33[i][2])
			);
		}

	}

	/*
	*@brief	Virtual particles' candinate points
	*		Every particle has 125 neighbors.
	*/
	template<typename Real, typename Coord>
	__global__ void GridFission_AnchorNeighbor_125
	(
		DArray<Coord> anchorPoint,
		DArray<Coord> pos,
		Coord origin,
		Real dh
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= pos.size()) return;

		Coord pos_ref = pos[id] - origin + dh / 2;
		Coord a(0);

		a[0] = (Real)((int)(floor(pos_ref[0] / dh))) * dh;
		a[1] = (Real)((int)(floor(pos_ref[1] / dh))) * dh;
		a[2] = (Real)((int)(floor(pos_ref[2] / dh))) * dh;

		for (int i = 0; i < 125; i++)
		{
			anchorPoint[id * 125 + i] = Coord(
				a[0] + dh * Real(diff_v_125[i][0]),
				a[1] + dh * Real(diff_v_125[i][1]),
				a[2] + dh * Real(diff_v_125[i][2])
			);
		}
	}

	/*
	*@brief	Positions(3-dimension) convert to Morton Codes
	*/
	template< typename Coord>
	__global__ void GridFission_PositionToMortonCode
	(
		DArray<uint32_t>  mortonCode,
		DArray<Coord> gridPoint,
		Real dh
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= gridPoint.size()) return;

		uint32_t  x = (uint32_t)(gridPoint[id][0] / dh + 100 * EPSILON);
		uint32_t  y = (uint32_t)(gridPoint[id][1] / dh + 100 * EPSILON);
		uint32_t  z = (uint32_t)(gridPoint[id][2] / dh + 100 * EPSILON);

		uint32_t  xi = x;
		uint32_t  yi = y;
		uint32_t  zi = z;

		uint32_t  key = 0;

		xi = (xi | (xi << 16)) & 0x030000FF;
		xi = (xi | (xi << 8)) & 0x0300F00F;
		xi = (xi | (xi << 4)) & 0x030C30C3;
		xi = (xi | (xi << 2)) & 0x09249249;
		yi = (yi | (yi << 16)) & 0x030000FF;
		yi = (yi | (yi << 8)) & 0x0300F00F;
		yi = (yi | (yi << 4)) & 0x030C30C3;
		yi = (yi | (yi << 2)) & 0x09249249;
		zi = (zi | (zi << 16)) & 0x030000FF;
		zi = (zi | (zi << 8)) & 0x0300F00F;
		zi = (zi | (zi << 4)) & 0x030C30C3;
		zi = (zi | (zi << 2)) & 0x09249249;
		key = xi | (yi << 1) | (zi << 2);

		mortonCode[id] = key;
	}

	/*
	*@brief	Search for adjacent elements with the same value
	*/
	__global__ void GridFission_RepeatedElementSearch
	(
		DArray<uint32_t>  morton,
		DArray<uint32_t>  counter
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= morton.size()) return;

		if (id == 0 || morton[id] != morton[id - 1])
		{
			counter[id] = 1;
		}
		else
			counter[id] = 0;
	}

	/*
	*@brief	Morton Codes convert to Positions(3-dimension)
	*/
	template< typename Coord>
	__global__ void GridFission_MortonCodeToPosition
	(
		DArray<Coord> gridPoint,
		DArray<uint32_t>  mortonCode,
		Real dh
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= mortonCode.size()) return;

		uint32_t key = mortonCode[id];

		uint32_t  xv = key & 0x09249249;
		uint32_t  yv = (key >> 1) & 0x09249249;
		uint32_t  zv = (key >> 2) & 0x09249249;

		xv = ((xv >> 2) | xv) & 0x030C30C3;
		xv = ((xv >> 4) | xv) & 0x0300F00F;
		xv = ((xv >> 8) | xv) & 0x030000FF;
		xv = ((xv >> 16) | xv) & 0x000003FF;

		yv = ((yv >> 2) | yv) & 0x030C30C3;
		yv = ((yv >> 4) | yv) & 0x0300F00F;
		yv = ((yv >> 8) | yv) & 0x030000FF;
		yv = ((yv >> 16) | yv) & 0x000003FF;

		zv = ((zv >> 2) | zv) & 0x030C30C3;
		zv = ((zv >> 4) | zv) & 0x0300F00F;
		zv = ((zv >> 8) | zv) & 0x030000FF;
		zv = ((zv >> 16) | zv) & 0x000003FF;

		Real x = (float)(xv)*dh;
		Real y = (float)(yv)*dh;
		Real z = (float)(zv)*dh;

		gridPoint[id] = Coord(x, y, z);
	}

	/*
	*@brief	Accumulate the number of non-repeating elements
	*/
	__global__ void GridFission_nonRepeatedElementsCal
	(
		DArray<uint32_t>  non_repeated_elements,
		DArray<uint32_t>  post_elements,
		DArray<uint32_t> counter
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= post_elements.size()) return;

		if (id == 0 || post_elements[id] != post_elements[id - 1])
		{
			non_repeated_elements[counter[id]] = post_elements[id];
		}
	}

	template<typename Coord>
	__global__ void GridFission_CopyToVpos(
		DArray<Coord> vpos,
		Coord origin
	)
	{
		int id = threadIdx.x + (blockIdx.x * blockDim.x);
		if (id >= vpos.size()) return;
		vpos[id] = vpos[id] + origin;
	}

	//__global__ void GridFission_VirtualTypeSet(
	//	DArray<bool> types,
	//	uint fussionNum
	//)
	//{
	//	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	//	if (id >= types.size()) return;
	//	if (id < fussionNum)
	//	{
	//		types[id] = true;
	//	}
	//	else
	//	{
	//		types[id] = false;
	//	}
	//}

	template <typename Real, typename Coord, typename Matrix>
	__global__ void GridFission_ThinSheetJudge(
		DArray<bool> thinSheetFlag,
		DArray<Coord> Eigens,
		DArray<Coord> posArr,
		DArrayList<int> neighbors,
		Real threshold,
		Real rho_0,
		Real smoothingLength,
		Matrix mat)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		List<int>& list_i = neighbors[pId];
		int nbSize = list_i.size();

		Real total_weight = 0.0f;
		Matrix tm(0.0f);
		Real wij = 1.0f;

		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = list_i[ne];
			Coord& X_j = posArr[j];
			Coord& X_i = posArr[pId];
			Coord xij = (X_j - X_i) / smoothingLength;
			Real rij = (X_j - X_i).norm();

			if ((rij > EPSILON) || (pId != j))
			{
				//Real wij = weight(rij, smoothingLength, scale);
				total_weight += wij;

				tm(0, 0) += xij[0] * xij[0] * wij;	tm(0, 1) += xij[0] * xij[1] * wij;	tm(0, 2) += xij[0] * xij[2] * wij;
				tm(1, 0) += xij[1] * xij[0] * wij;	tm(1, 1) += xij[1] * xij[1] * wij;	tm(1, 2) += xij[1] * xij[2] * wij;
				tm(2, 0) += xij[2] * xij[0] * wij;	tm(2, 1) += xij[2] * xij[1] * wij;	tm(2, 2) += xij[2] * xij[2] * wij;
			}
		}

		if (total_weight > EPSILON)
		{
			tm *= (1.0 / total_weight);
		}
		else
		{
			tm = Matrix::identityMatrix();
		}


		Matrix R, U, D, V;
		polarDecomposition(tm, R, U, D, V);

		Eigens[pId][0] = D(0, 0);
		Eigens[pId][1] = D(1, 1);
		Eigens[pId][2] = D(2, 2);

		Coord& Eigen_i = Eigens[pId];

		if ((Eigen_i[0] < threshold) || (Eigen_i[1] < threshold) || (Eigen_i[2] < threshold))
		{
			thinSheetFlag[pId] = true;
		}
		else
		{
			thinSheetFlag[pId] = false;
		}
	}

	template<typename TDataType>
	void VirtualFissionFusionStrategy<TDataType>::resizeArrays(int num)
	{
		if (num != this->outVirtualParticles()->size())
		{
			this->outVirtualParticles()->resize(num);
		}
		if (num != mCurrentParticleStates.size())
		{
			mCurrentParticleStates.resize(num);
		}
		if (num != mDivergence.size())
		{
			mDivergence.resize(num);
		}
		if (mCurrentParticleStates.size() == !num)
		{
			mCurrentParticleStates.resize(num);
		}
		if (mEigens.size() == !num)
		{
			mEigens.resize(num);
		}
		if (mThinSheets.size() == !num)
		{
			mThinSheets.resize(num);
		}
	}

	/*
	* @brief	Determine a particle whether locate in a thin-layer of a fluid.
	*			Output: DArray<bool> mThinSheets
	*/
	template<typename TDataType>
	void VirtualFissionFusionStrategy<TDataType>::thinFeatureCompute()
	{
		cuExecute(this->inRPosition()->size(),
			GridFission_ThinSheetJudge,
			mThinSheets,
			mEigens,
			this->inRPosition()->getData(),
			this->inNeighborIds()->getData(),
			this->varTransitionRegionThreshold()->getValue(),
			this->varRestDensity()->getValue(),
			this->inSmoothingLength()->getValue(),
			Matrix(0.0f)
		);
	}

	/*
	* @brief	Determine whether particle is going to fission.
	*			Output: DArray<uint> mCurrentParticleStates
	*/
	template<typename TDataType>
	void VirtualFissionFusionStrategy<TDataType>::fissionJudger()
	{

		int r_num = this->inRPosition()->size();

		if (mPreParticleStates.size() == !r_num)
		{
			/*
			* note: PreParticleState cannot work with the partilce-emitter.
			*/
			mPreParticleStates.resize(r_num);

			cuExecute(mPreParticleStates.size(),
				GridFission_intArrayReset,
				mPreParticleStates,
				1);
		}

		if (this->inFrameNumber()->getValue() != 0)
		{
			mPreParticleStates.assign(mCurrentParticleStates);
		}
 
		mSummation->update();
		Real restDensity = this->varRestDensity()->getValue();
		Real h = this->inSmoothingLength()->getValue();
		Real d = this->inSamplingDistance()->getValue();
		Real mass = d * d * d * restDensity;

		cuExecute(r_num, GridFission_RealParticleDivergence,
			mDivergence,
			this->outDensity()->getData(),
			this->inRPosition()->getData(),
			this->inRVelocity()->getData(),
			mCurrentParticleStates,
			mPreParticleStates,
			this->inNeighborIds()->getData(),
			this->varTransitionRegionThreshold()->getData(),
			mass,
			h,
			kernel
		);

		cuExecute(r_num, GridFission_StateJudge,
			mCurrentParticleStates,
			mDivergence,
			this->outDensity()->getData(),
			mThinSheets, //this->inThinSheet()->getData(),
			//this->inThinFeature()->getData(),
			this->inRPosition()->getData(),
			mPreParticleStates,
			this->varRestDensity()->getValue(),
			this->varStretchedRegionCriteria()->getDataPtr()->currentKey(),
			this->varTransitionRegionThreshold()->getData(),
			mass,
			h,
			kernel
		);

		//std::cout <<"ThinFeature: " << this->inThinFeature()->size() << ", ThinSheet " << this->inThinSheet()->size() << std::endl;

		cuExecute(r_num, GridFission_StatesCorrection,
			mCurrentParticleStates,
			mPreParticleStates,
			this->inRPosition()->getData(),
			this->inNeighborIds()->getData(),
			h
		);
	}

	/*
	* @brief	Particle positions is divided into two groupus (separate arrays): 1. Fission particles; 2. Fussion particles.
	*			Output: DArray<Coord> mFissionParticles, mFussionParticles.
	*/
	template<typename TDataType>
	void VirtualFissionFusionStrategy<TDataType>::splitParticleArray()
	{
		int r_num = this->inRPosition()->size();

		if(fissions.size() != r_num)
			fissions.resize(r_num);

		if (fussions.size() != r_num)
			fussions.resize(r_num);

		if (ArrayPointer.size() != r_num)
			ArrayPointer.resize(r_num);

		uint fission_counter = 0;
		uint fussion_counter = 0;

		cuExecute(r_num, GridFission_FissionAndFusionCounter,
			mCurrentParticleStates,
			fissions,
			fussions
		);

		Reduction<uint> uint_rnum_reduce;
		fission_counter = uint_rnum_reduce.accumulate(fissions.begin(), fissions.size());
		fussion_counter = uint_rnum_reduce.accumulate(fussions.begin(), fussions.size());

		std::cout << "*DUAL-ISPH::Fission-Fusion Strategy::" <<  "Fission-" << fission_counter << " + Fussion-" << fussion_counter << std::endl;

		mFissionParticles.resize(fission_counter);
		mFissionParticleIds.resize(fission_counter);
		mFussionParticles.resize(fussion_counter);
		mFussionParticleIds.resize(fussion_counter);

		thrust::exclusive_scan(thrust::device, fissions.begin(), fissions.begin() + fissions.size(), ArrayPointer.begin());
		cuExecute(r_num, GridFission_PickUpParticles,
			mFissionParticles,
			mFissionParticleIds,
			this->inRPosition()->getData(),
			ArrayPointer,
			fissions
		);

		thrust::exclusive_scan(thrust::device, fussions.begin(), fussions.begin() + fussions.size(), ArrayPointer.begin());
		cuExecute(r_num, GridFission_PickUpParticles,
			mFussionParticles,
			mFussionParticleIds,
			this->inRPosition()->getData(),
			ArrayPointer,
			fussions
		);	
	};

	/*
	* @brief	Generate the virtual particles for the fission real particles.
	*			Output: DArray<Coord> mFissionVirtualParicles
	*/
	template<typename TDataType>
	void VirtualFissionFusionStrategy<TDataType>::constructFissionVirtualParticles()
	{
		Real gridSize = this->inSamplingDistance()->getData();
		int fission_num = mFissionParticles.size();
		if (fission_num == 0) return;

		int node_num = fission_num * (int)(this->varCandidatePointCount()->getDataPtr()->currentKey());

		if (mAnchorPoints.size() != node_num)
		{
			mAnchorPoints.resize(node_num);
		}

		if (mAnchorPointCodes.size() != node_num)
		{
			mAnchorPointCodes.resize(node_num);
		}

		if (mNonRepeatedCount.size() != node_num)
		{
			mNonRepeatedCount.resize(node_num);
		}

		Reduction<Coord> reduce;
		Coord hiBound = reduce.maximum(this->inRPosition()->getData().begin(), this->inRPosition()->getData().size());
		Coord loBound = reduce.minimum(this->inRPosition()->getData().begin(), this->inRPosition()->getData().size());

		int padding = 2;
		hiBound += Coord((padding + 1) * gridSize);
		loBound -= Coord(padding * gridSize);

		loBound[0] = (Real)((int)(floor(loBound[0] / gridSize))) * gridSize;
		loBound[1] = (Real)((int)(floor(loBound[1] / gridSize))) * gridSize;
		loBound[2] = (Real)((int)(floor(loBound[2] / gridSize))) * gridSize;

		Coord origin = Coord(loBound[0], loBound[1], loBound[2]);

		if (this->varCandidatePointCount()->getValue() == CandidatePointCount::neighbors_8) {
			cuExecute(fission_num,
				GridFission_AnchorNeighbor_8,
				mAnchorPoints,
				mFissionParticles,
				origin,
				gridSize
			);
		}
		else if (this->varCandidatePointCount()->getValue() == CandidatePointCount::neighbors_27) {
			cuExecute(fission_num,
				GridFission_AnchorNeighbor_27,
				mAnchorPoints,
				mFissionParticles,
				origin,
				gridSize
			);
		}
		else if (this->varCandidatePointCount()->getValue() == CandidatePointCount::neighbors_33) {
			cuExecute(fission_num,
				GridFission_AnchorNeighbor_33,
				mAnchorPoints,
				mFissionParticles,
				origin,
				gridSize
			);
		}
		else if (this->varCandidatePointCount()->getValue() == CandidatePointCount::neighbors_125) {
			cuExecute(fission_num,
				GridFission_AnchorNeighbor_125,
				mAnchorPoints,
				mFissionParticles,
				origin,
				gridSize
			);
		}
		else
		{
			std::cout << "*VIRTUAL PARTICLE:: ERROR!!!!!" << std::endl;
		}

		cuExecute(mAnchorPointCodes.size(),
			GridFission_PositionToMortonCode,
			mAnchorPointCodes,
			mAnchorPoints,
			gridSize
		);

		thrust::sort(thrust::device, mAnchorPointCodes.begin(), mAnchorPointCodes.begin() + mAnchorPointCodes.size());

		cuExecute(mAnchorPointCodes.size(),
			GridFission_RepeatedElementSearch,
			mAnchorPointCodes,
			mNonRepeatedCount
		);

		int candidatePoint_num = thrust::reduce(thrust::device, mNonRepeatedCount.begin(), mNonRepeatedCount.begin() + mNonRepeatedCount.size(), (int)0, thrust::plus<uint32_t>());

		thrust::exclusive_scan(thrust::device, mNonRepeatedCount.begin(), mNonRepeatedCount.begin() + mNonRepeatedCount.size(), mNonRepeatedCount.begin());

		mCandidateCodes.resize(candidatePoint_num);

		cuExecute(mAnchorPointCodes.size(),
			GridFission_nonRepeatedElementsCal,
			mCandidateCodes,
			mAnchorPointCodes,
			mNonRepeatedCount
		);
		//cuSynchronize();

		mFissionVirtualParicles.resize(mCandidateCodes.size());

		cuExecute(mCandidateCodes.size(),
			GridFission_MortonCodeToPosition,
			mFissionVirtualParicles,
			mCandidateCodes,
			gridSize
		);

		cuExecute(mFissionVirtualParicles.size(),
			GridFission_CopyToVpos,
			mFissionVirtualParicles,
			origin
		);
	};

	/*
	* @brief	1. Generate the virtual particles for the fussion real particles; 2. Merge the virtual particle position arrays
	*			Output: DArray<Coord> mVirtualPoints
	*/
	template<typename TDataType>
	void VirtualFissionFusionStrategy<TDataType>::mergeVirtualParticles()
	{
		
		mFussionVirtualParticles.assign(mFussionParticles);

		uint fussionVirtualCount = mFussionVirtualParticles.size();
		uint fissionVirtualCount = mFissionVirtualParicles.size();

		this->outVirtualParticles()->resize(fussionVirtualCount + fissionVirtualCount);
		auto& t_VirtualPoints = this->outVirtualParticles()->getData();

		t_VirtualPoints.assign(mFussionVirtualParticles, mFussionVirtualParticles.size(), 0, 0);
		t_VirtualPoints.assign(mFissionVirtualParicles, mFissionVirtualParicles.size(), mFussionVirtualParticles.size(), 0);

		//if (this->outVirtualPointType()->size() != t_VirtualPoints.size())
		//{
		//	this->outVirtualPointType()->resize(t_VirtualPoints.size());
		//}
		//cuExecute(t_VirtualPoints.size(),
		//	GridFission_VirtualTypeSet,
		//	this->outVirtualPointType()->getData(),
		//	fussionVirtualCount);
	};

	template<typename TDataType>
	void VirtualFissionFusionStrategy<TDataType>::constrain()
	{
		this->resizeArrays(this->inRPosition()->size());

		this->thinFeatureCompute();
		
		this->fissionJudger();

		this->splitParticleArray();

		this->constructFissionVirtualParticles();

		this->mergeVirtualParticles();
	}

	DEFINE_CLASS(VirtualFissionFusionStrategy);

}