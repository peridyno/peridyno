#pragma once
#include "Module/TopologyModule.h"

#include "AnimationCurve.h"
#include <iterator>
#include <random>

#include "Quat.h"
namespace dyno
{

    struct JCapsule
    {
        typedef typename DataType3f::Coord Coord;
        int id_joint;
        int id_cap;
        Coord v0, v1;

        DYN_FUNC JCapsule(){};
        DYN_FUNC JCapsule(int id1, int id2, Coord a0, Coord a1): 
        id_joint(id1), id_cap(id2), v0(a0), v1(a1){};
        DYN_FUNC ~JCapsule(){};
    };

    /*!
	*	\class	JointTree
	*	\brief	A JointTree(Skeleton) represents a hierarchical tree structure of joints
	*/    
    template<typename TDataType>
	class JointTree : public TopologyModule
	{ 
        DECLARE_TCLASS(JointTree, TDataType)
    private:
        std::default_random_engine generator;
        std::normal_distribution<double> dist;
        

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix; 
		typedef typename ::dyno::Mat4f Mat;
        
        JointTree();
        ~JointTree();

        void copyFrom(JointTree<TDataType> jointTree);

        void scale(Real s);
        void translate(Coord t);
        void getGlobalTransform();
        Mat getTransform(Coord & T, Coord& R, Coord& S);
		void getQuat(Coord &T, Coord &R, float &S);
        void getGlobalQuat();
		Coord getCoordByMatrix(Coord X);
		Coord getCoordByQuat(Coord X);
        
        void getGlobalCoord();

        void setAnimTranslation(std::shared_ptr<AnimationCurve<TDataType>> t) {
            AnimTranslation = t;
            AnimTranslation->setInitVal(LclTranslation);
        }
        void setAnimRotation(std::shared_ptr<AnimationCurve<TDataType>> r) {
            AnimRotation = r;
            AnimTranslation->setInitVal(LclRotation);
        }
        void setAnimScaling(std::shared_ptr<AnimationCurve<TDataType>> s) {
            AnimScaling = s;
            AnimScaling->setInitVal(LclScaling);
        }

        // 更新动画
        void applyAnimationByOne(Coord& init, Coord& cur, std::shared_ptr<AnimationCurve<TDataType>>& anim, Real ptime);
        void applyAnimationAll(Real ptime);

        unsigned long long id;
        // 用于对整个模型进行调整
        Coord PreRotation;
        Coord PreScaling;
		Coord PreTranslation;

		Coord tmp;

        Coord LclTranslation;   // Local Joint's coord
        Coord LclRotation;
        Coord LclScaling; 

        std::shared_ptr<AnimationCurve<TDataType>> AnimTranslation;    // Local Animation
        std::shared_ptr<AnimationCurve<TDataType>> AnimRotation;
        std::shared_ptr<AnimationCurve<TDataType>> AnimScaling;

        Coord CurTranslation;   // Current Joint's Transform (Animation)
        Coord CurRotation;
        Coord CurScaling; 

        Coord GlCoord;          // Global Joint's coord 
        Coord LastCoord;
        Mat GlobalTransform;
        bool RotationActive;

        Quat<Real> GlT;
        Quat<Real> GlR;
        Real GlS;
        std::vector<std::shared_ptr<JointTree>> children;
        std::shared_ptr<JointTree> parent;
    };
}