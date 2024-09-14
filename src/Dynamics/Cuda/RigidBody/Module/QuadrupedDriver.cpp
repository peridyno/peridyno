#include "QuadrupedDriver.h"

namespace dyno
{
	IMPLEMENT_TCLASS(QuadrupedDriver, TDataType);

	template<typename TDataType>
	QuadrupedDriver<TDataType>::QuadrupedDriver()
		: KeyboardInputModule()
	{
		this->varCacheEvent()->setValue(false);
	}

	template<typename TDataType>
	void QuadrupedDriver<TDataType>::onEvent(PKeyboardEvent event)
	{
	
		// QuadrupedDriver
		{
			move += this->inDeltaTime()->getValue() * this->varSpeed()->getValue();

			auto topo = TypeInfo::cast<DiscreteElements<DataType3f>>(this->inTopology()->getDataPtr());

			auto& d_hinge = topo->hingeJoints();
			CArray<HingeJoint> c_hinge;
			c_hinge.assign(d_hinge);
		

			float currentFrameinAnim = move - floor(move);


			//printf("move : %f, weight : %f\n",move,weight);

			std::vector <float> animTime = { 0,0.25,0.5,1 };

			std::vector<float> animLeftFrontUpAngle = { 57,52,-24,38,57 };
			std::vector<float> animLeftFrontDownAngle = { -49,-120,-28,-77,-49 };

			std::vector<float> animLeftBackUpAngle = { -6,42,54,55,-6 };
			std::vector<float> animLeftBackDownAngle = { -51,-77,-35,-112,-51 };

			std::vector<float> animRightFrontUpAngle = { -24,38,57,52,-24 };
			std::vector<float> animRightFrontDownAngle = { -28,-77,-49,-120,-28 };

			std::vector<float> animRightBackUpAngle = { 54,55,-6,42,54 };
			std::vector<float> animRightBackDownAngle = { -35,-112, -51,-77,-35 };


			//std::vector <float> animTime = { 0,0.5,1 };

			//std::vector<float> animLeftFrontUpAngle = { 57,-24,57 };
			//std::vector<float> animLeftFrontDownAngle = { -49,-28,-49 };

			//std::vector<float> animLeftBackUpAngle = { -6,54,-6 };
			//std::vector<float> animLeftBackDownAngle = { -51,-35,-51 };

			//std::vector<float> animRightFrontUpAngle = { -24,57,-24 };
			//std::vector<float> animRightFrontDownAngle = { -28,-49,-28 };

			//std::vector<float> animRightBackUpAngle = { 54,-6,54 };
			//std::vector<float> animRightBackDownAngle = { -35, -51,-35 };

			////std::vector<float> animRightFrontUpAngle = { 57,-24,57 };
			////std::vector<float> animRightFrontDownAngle = { -49,-28,-49 };

			////std::vector<float> animRightBackUpAngle = { -6,54,-6 };
			////std::vector<float> animRightBackDownAngle = { -51,-35,-51 };

			float weight = 0;
			int keyFrame = -1;
			for (size_t i = 0; i < animTime.size() - 1; i++)
			{
				if (currentFrameinAnim > animTime[i] && currentFrameinAnim <= animTime[i + 1]) 
				{
					float v = (currentFrameinAnim - animTime[i]) / (animTime[i + 1] - animTime[i]);
					weight = 1 - (cos(v * M_PI) / 2) - 0.5;
					keyFrame = i;
					break;
				}
			}

			float angleDivede = -1.0;

			Real epsilonAngle = M_PI / 360;

			switch (event.key)
			{
			case PKeyboardType::PKEY_W:	

				angleDivede = -1.0;
				break;

			case PKeyboardType::PKEY_S:

				angleDivede = 1.0;

				break;
			default:
				return;
				break;
				
			}


			Real angleFUL = lerp(animLeftFrontUpAngle[keyFrame], animLeftFrontUpAngle[keyFrame + 1], weight) * M_PI / 180 / angleDivede;
			Real angleFUR = lerp(animRightFrontUpAngle[keyFrame], animRightFrontUpAngle[keyFrame + 1], weight) * M_PI / 180 / angleDivede;
			Real angleFDL = lerp(animLeftFrontDownAngle[keyFrame], animLeftFrontDownAngle[keyFrame + 1], weight) * M_PI / 180 / angleDivede;
			Real angleFDR = lerp(animRightFrontDownAngle[keyFrame], animRightFrontDownAngle[keyFrame + 1], weight) * M_PI / 180 / angleDivede;

			Real angleBUL = lerp(animLeftBackUpAngle[keyFrame], animLeftBackUpAngle[keyFrame + 1], weight) * M_PI / 180 / angleDivede;
			Real angleBUR = lerp(animRightBackUpAngle[keyFrame], animRightBackUpAngle[keyFrame + 1], weight) * M_PI / 180 / angleDivede;
			Real angleBDL = lerp(animLeftBackDownAngle[keyFrame], animLeftBackDownAngle[keyFrame + 1], weight) * M_PI / 180 / angleDivede;
			Real angleBDR = lerp(animRightBackDownAngle[keyFrame], animRightBackDownAngle[keyFrame + 1], weight) * M_PI / 180 / angleDivede;


			c_hinge[0].setRange(angleFUL, angleFUL + epsilonAngle);
			c_hinge[1].setRange(angleFDL, angleFDL + epsilonAngle);

			c_hinge[2].setRange(angleBUL, angleBUL + epsilonAngle);
			c_hinge[3].setRange(angleBDL, angleBDL + epsilonAngle);

			c_hinge[4].setRange(angleFUR, angleFUR + epsilonAngle);
			c_hinge[5].setRange(angleFDR, angleFDR + epsilonAngle);

			c_hinge[6].setRange(angleBUR, angleBUR + epsilonAngle);
			c_hinge[7].setRange(angleBDR, angleBDR + epsilonAngle);

			//
			c_hinge[8].setRange(angleFUL, angleFUL + epsilonAngle);
			c_hinge[9].setRange(angleFDL, angleFDL + epsilonAngle);

			c_hinge[10].setRange(angleBUL, angleBUL + epsilonAngle);
			c_hinge[11].setRange(angleBDL, angleBDL + epsilonAngle);

			c_hinge[12].setRange(angleFUR, angleFUR + epsilonAngle);
			c_hinge[13].setRange(angleFDR, angleFDR + epsilonAngle);

			c_hinge[14].setRange(angleBUR, angleBUR + epsilonAngle);
			c_hinge[15].setRange(angleBDR, angleBDR + epsilonAngle);

			d_hinge.assign(c_hinge);
			
		}
	}


	float lerp(float a, float b, float t) {
		return a + t * (b - a);
	}

	
	DEFINE_CLASS(QuadrupedDriver);
}