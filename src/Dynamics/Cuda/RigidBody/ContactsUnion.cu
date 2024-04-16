#include "ContactsUnion.h"

namespace dyno 
{
	IMPLEMENT_TCLASS(ContactsUnion, TDataType)

	
	template<typename Joint>
	__global__ void setUpMapMatrix(
		DArray<int> mapMatrix,
		DArray<Joint> joints,
		int bodynum
	)
	{
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId >= joints.size())
			return;

		int idx1 = joints[tId].bodyId1;
		int idx2 = joints[tId].bodyId2;

		int index_1 = bodynum * idx1 + idx2;
		int index_2 = bodynum * idx2 + idx1;

		atomicAdd(&mapMatrix[index_1], 1);
		atomicAdd(&mapMatrix[index_2], 1);
	}
	template<typename TDataType>
	void ContactsUnion<TDataType>::filterArray(CArray<int>& mapMatrix, CArray<ContactPair>& contactsA, CArray<ContactPair>& contactsB, CArray<ContactPair>&contactsC, int bodynum)
	{
		for (int i = 0; i < contactsA.size(); i++)
		{
			int idx2 = contactsA[i].bodyId2;
			int idx1 = contactsA[i].bodyId1;
			if (idx2 != INVALID)
			{
				if (mapMatrix[idx1 * bodynum + idx2] == 0)
				{
					contactsC.pushBack(contactsA[i]);
				}
			}
			else
			{
				contactsC.pushBack(contactsA[i]);
			}
		}
		for (int i = 0; i < contactsB.size(); i++)
		{
			int idx2 = contactsB[i].bodyId2;
			int idx1 = contactsB[i].bodyId1;
			
			if (idx2 != INVALID)
			{
				if (mapMatrix[idx1 * bodynum + idx2] == 0)
				{
					contactsC.pushBack(contactsB[i]);
				}
			}
			else
			{
				contactsC.pushBack(contactsB[i]);
			}
		}
	}

	template<typename TDataType>
	void ContactsUnion<TDataType>::compute()
	{
		auto inDataA = this->inContactsA()->getDataPtr();
		auto inDataB = this->inContactsB()->getDataPtr();
		
		int bodyNum = this->inMass()->size();
		
		int ballAndSocketJoint_size = this->inBallAndSocketJoints()->size();
		int sliderJoint_size = this->inSliderJoints()->size();
		int hingeJoint_size = this->inHingeJoints()->size();
		int fixedJoint_size = this->inFixedJoints()->size();

		DArray<int> mapMatrix;
		mapMatrix.resize(bodyNum * bodyNum);
		mapMatrix.reset();


		if (ballAndSocketJoint_size != 0)
		{
			cuExecute(ballAndSocketJoint_size,
				setUpMapMatrix,
				mapMatrix,
				this->inBallAndSocketJoints()->getData(),
				bodyNum);
		}

		if (sliderJoint_size != 0)
		{
			cuExecute(sliderJoint_size,
				setUpMapMatrix,
				mapMatrix,
				this->inSliderJoints()->getData(),
				bodyNum);
		}

		if (hingeJoint_size != 0)
		{
			cuExecute(hingeJoint_size,
				setUpMapMatrix,
				mapMatrix,
				this->inHingeJoints()->getData(),
				bodyNum);
		}

		if (fixedJoint_size != 0)
		{
			cuExecute(fixedJoint_size,
				setUpMapMatrix,
				mapMatrix,
				this->inFixedJoints()->getData(),
				bodyNum);
		}

		

		CArray<int> mapMatrixCPU;
		mapMatrixCPU.assign(mapMatrix);
		CArray<ContactPair> contactsA;
		CArray<ContactPair> contactsB;
		CArray<ContactPair> contactsC;

		if (inDataA != nullptr)
			contactsA.assign(*inDataA);

		if (inDataB != nullptr)
			contactsB.assign(*inDataB);

		
		filterArray(mapMatrixCPU, contactsA, contactsB, contactsC, bodyNum);

		

		int total_size = contactsC.size();

		if (total_size == 0)
			return;

		if (this->outContacts()->size() != total_size)
			this->outContacts()->resize(total_size);

		auto& outData = this->outContacts()->getData();
		outData.assign(contactsC);


	}

	template<typename TDataType>
	bool ContactsUnion<TDataType>::validateInputs()
	{
		bool ret = this->inContactsA()->isEmpty() && this->inContactsB()->isEmpty();

		return !ret;
	}

	DEFINE_CLASS(ContactsUnion);
}