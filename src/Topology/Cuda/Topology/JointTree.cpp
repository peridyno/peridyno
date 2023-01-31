#pragma once
#include "JointTree.h"

#define cos_angle(angle) cos(double(angle * 3.1415926f / 180.0))
#define sin_angle(angle) sin(double(angle * 3.1415926f / 180.0))
#define radian(x) x * 3.1415926f / 180.0

namespace dyno
{
    IMPLEMENT_TCLASS(JointTree, TDataType)
    
    template<typename TDataType>
    JointTree<TDataType>::JointTree()
    {
        id = -1;
        RotationActive = true;
        PreRotation = Coord(0);
        PreTranslation = Coord(0);
        PreScaling = Coord(1); 

        LclTranslation = Coord(0);
        LclRotation = Coord(0);
        LclScaling = Coord(1);
		
        tmp = Coord(0, 0, 0);

        GlT = Quat<Real>(0, 0, 0, 0);
        GlR = Quat<Real>(0, 0, 0, 0);
        GlS = 1.f;
    }
    
    template<typename TDataType>
    JointTree<TDataType>::~JointTree()
    {

    }


	template<typename TDataType>
	typename JointTree<TDataType>::Mat JointTree<TDataType>::getTransform(Coord & T, Coord& R, Coord& S)
    {
        Mat translation = Mat(
            1, 0, 0, T[0],
            0, 1, 0, T[1],
            0, 0, 1, T[2],
            0, 0, 0, 1);
        
        // R[X,Y,Z] -> [Z,X,Y]轴

        double X = R[0];
        Mat rotation_x = Mat(
            1, 0, 0, 0,
            0, cos_angle(X), -sin_angle(X), 0,
            0, sin_angle(X), cos_angle(X), 0,
            0, 0, 0, 1);

		double Y = R[1];
        Mat rotation_y = Mat(
            cos_angle(Y), 0, sin_angle(Y), 0,
            0, 1, 0, 0,
            -sin_angle(Y), 0, cos_angle(Y), 0,
            0, 0, 0, 1);

		double Z = R[2];
        Mat rotation_z = Mat(
            cos_angle(Z), -sin_angle(Z), 0, 0,
            sin_angle(Z), cos_angle(Z), 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);

        Mat scaling= Mat(
            S[0], 0, 0, 0,
            0, S[1], 0, 0,
            0, 0, S[2], 0,
            0, 0, 0, 1);

        return  translation * scaling * rotation_x * rotation_y * rotation_z;
    }

    // 遍历关节层次时，顺便更新
	template<typename TDataType>
    void JointTree<TDataType>::getGlobalTransform()
    {
        // 注意顺序
        this->GlobalTransform = this->getTransform(this->CurTranslation, this->CurRotation, this->CurScaling);
        if(this->parent != nullptr)
            this->GlobalTransform = this->parent->GlobalTransform * this->GlobalTransform;
        else
        {
            //Pre
            this->GlobalTransform = this->getTransform(this->PreTranslation, this->PreRotation, this->PreScaling) * this->GlobalTransform;
        }
        //DEBUG
		// printf("Mat:\n");
        // for (int i = 0; i < 4; ++i)
        // {
        //     auto v = this->GlobalTransform.row(i);
        //     printf("[%f, %f, %f, %f]\n", v[0], v[1], v[2], v[3]);
        // }        
    }

    template<typename TDataType>
    void JointTree<TDataType>::getQuat(Coord &T, Coord &R, float &S)
    {
        Quat<Real> t(T[0], T[1], T[2], 0.f);
        // Rotate
        Quat<Real> q_x(radian(R[0]), Vec3f(1.f, 0.f, 0.f));
        Quat<Real> q_y(radian(R[1]), Vec3f(0.f, 1.f, 0.f));
        Quat<Real> q_z(radian(R[2]), Vec3f(0.f, 0.f, 1.f));
        
        this->GlT = this->GlT + this->GlS * this->GlR * t * this->GlR.conjugate();
        this->GlS = this->GlS * S;
        this->GlR = this->GlR * (q_x * q_y * q_z);

		// printf("Quat:\n");
        // printf("T: [%f, %f, %f, %f]\n", this->GlT.x, this->GlT.y, this->GlT.z, this->GlT.w);
        // printf("R: [%f, %f, %f, %f]\n", this->GlR.x, this->GlR.y, this->GlR.z, this->GlR.w);
        // printf("S: [%f]\n", this->GlS);
        // for (int i = 0; i < 4; ++i)
        // {
        //     auto v = this->GlobalTransform.row(i);
        //     printf("[%f, %f, %f, %f]\n", v[0], v[1], v[2], v[3]);
        // TODO:
        // GlR.normalize(); 
    }

    // 遍历关节层次时，顺便更新
	template<typename TDataType>
    void JointTree<TDataType>::getGlobalQuat()
    {
        if(this->parent != nullptr)
        {
            this->GlT = this->parent->GlT;
            this->GlS = this->parent->GlS;
            this->GlR = this->parent->GlR;
            getQuat(this->CurTranslation, this->CurRotation, this->CurScaling[0]);
        }else
        {
            this->GlT = Quat<Real>(0.f, 0.f, 0.f, 0.f);
            this->GlS = 1.f;
            this->GlR = Quat<Real>(0.f, 0.f, 0.f, 1.f);
            getQuat(this->PreTranslation, this->PreRotation, this->PreScaling[0]);
            getQuat(this->CurTranslation, this->CurRotation, this->CurScaling[0]);

        }
            //DEBUG
            // printf("[Root] Cur QuatR: (%f)\n", this->GlR.w);
            // printf("[Root] T: (%f, %f, %f)\n", this->CurTranslation[0], this->CurTranslation[1], this->CurTranslation[2]);
            // printf("[Root] R: (%f, %f, %f)\n", this->CurRotation[0], this->CurRotation[1], this->CurRotation[2]);
            // printf("[Root] S: (%f)\n", this->CurScaling[0]);
    }

    template<typename TDataType>
	typename JointTree<TDataType>::Coord JointTree<TDataType>::getCoordByMatrix(Coord X)
	{
        Vec4f tmp = this->GlobalTransform * Vec4f(X[0], X[1], X[2], 1) ;
		return Coord(tmp[0] / tmp[3], tmp[1] / tmp[3], tmp[2] / tmp[3]);
    }

    template<typename TDataType>
	typename JointTree<TDataType>::Coord JointTree<TDataType>::getCoordByQuat(Coord X)
	{
        Quat<Real> tmp(X[0], X[1], X[2], 1) ;
        tmp = this->GlS * this->GlR * tmp *  this->GlR.conjugate() + this->GlT;
		return Coord(tmp.x, tmp.y, tmp.z);
    }

    template<typename TDataType>
	void JointTree<TDataType>::getGlobalCoord()
    {
        this->LastCoord = this->GlCoord;
        this->GlCoord = this->getCoordByQuat(Coord(0, 0, 0));
    }

    template<typename TDataType>
    void JointTree<TDataType>::copyFrom(JointTree<TDataType> jointTree)
    {
        this->id = jointTree.id;
        this->PreRotation = jointTree.PreRotation;
        this->LclTranslation = jointTree.LclTranslation;
        this->LclRotation = jointTree.LclRotation;
        this->LclScaling = jointTree.LclScaling;
        this->AnimTranslation = jointTree.AnimTranslation;
        this->AnimRotation = jointTree.AnimRotation;
        this->AnimScaling = jointTree.AnimScaling;
        this->GlCoord = jointTree.GlCoord;
        this->GlobalTransform = jointTree.GlobalTransform;
        this->GlT = jointTree.GlT;
        this->GlR = jointTree.GlR;
        this->GlS = jointTree.GlS;
        this->RotationActive = jointTree.RotationActive;
        this->children.assign(jointTree.children.begin(), jointTree.children.end());
        this->parent = jointTree.parent;
    }

    template<typename TDataType>
    void JointTree<TDataType>::scale(Real s)
    {
		PreScaling *= s;
    }

    template<typename TDataType>
    void JointTree<TDataType>::translate(Coord t)
    {
        PreTranslation += t;
    }

    template<typename TDataType>
    void JointTree<TDataType>::applyAnimationByOne(Coord& init, Coord& cur, std::shared_ptr<AnimationCurve<TDataType>>& anim, Real ptime)
    {
        cur = anim->getCurveValueAll(ptime);
        // cur = anim->getCurveValueCycle(ptime);
    }

    template<typename TDataType>
    void JointTree<TDataType>::applyAnimationAll(Real ptime)
    {
        if (AnimTranslation != nullptr)
            applyAnimationByOne(LclTranslation, CurTranslation, AnimTranslation, ptime);
        if (AnimRotation != nullptr)
        {
            applyAnimationByOne(LclRotation, CurRotation, AnimRotation, ptime);
            // if (AnimRotation->m_maxSize > 1 && CurRotation[2] < 90)
                // CurRotation += Coord(0,0, 0.5);
        }
        if (AnimScaling != nullptr)
            applyAnimationByOne(LclScaling, CurScaling, AnimScaling, ptime);
    }
    
#ifdef PRECISION_FLOAT
	template class JointTree<DataType3f>;
#else
	template class JointTree<DataType3d>;
#endif    
}