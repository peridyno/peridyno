#pragma once
#include "Array/ArrayList.h"

typedef long long LongLong;

namespace dyno{

    #define FBXTIME 46186158000L
    #define BEFORE -1
    #define ANIM_SPEED 1.f

	template<typename TDataType>
	class AnimationCurve
	{
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename ::dyno::Mat4f Mat;

		AnimationCurve();
        AnimationCurve(int size, Real dx, Real dy, Real dz);
		~AnimationCurve();

        void set(int index, const std::vector<LongLong>& tim, const std::vector<Real>& val);

        // 转换为FBXTIME
        Real fbxTimeToSeconds(LongLong value){return Real(value) / FBXTIME * ANIM_SPEED;}
        LongLong secondsToFbxTime(Real value){return LongLong(value * ANIM_SPEED * FBXTIME);}

        void setInitVal(Coord init){
            m_initVal[0] = init[0];
            m_initVal[1] = init[1];
            m_initVal[2] = init[2];
        }
        
        // 获取递增时间下该时刻的曲线值
        Coord getCurveValueAlong(Real ptime);
        // 获取任意时刻的曲线值
        Coord getCurveValueAll(Real ptime);
        // 循环获取曲线值
        Coord getCurveValueCycle(Real ptime);
        
	public:
        int m_maxSize;
        // (X,Y,Z)
        int m_cur[3];
        Real m_initVal[3];
        Real m_endVal[3];
        std::vector<LongLong> m_times[3]; // 若不存在曲线，则只有一个 [0]
        std::vector<Real> m_values[3]; 
	};
}
