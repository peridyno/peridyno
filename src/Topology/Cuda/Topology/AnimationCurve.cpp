#include "AnimationCurve.h"

#include "Object.h"
#include "DataTypes.h"

namespace dyno 
{

	template<typename TDataType>
	AnimationCurve<TDataType>::AnimationCurve()
	{
        m_maxSize = 0;
        m_cur[0] = BEFORE;
        m_cur[1] = BEFORE;
        m_cur[2] = BEFORE;
	}

    template<typename TDataType>
	AnimationCurve<TDataType>::AnimationCurve(int size, Real dx, Real dy, Real dz)
    {
        m_maxSize = size;
        m_initVal[0] = dx;
        m_initVal[1] = dy;
        m_initVal[2] = dz;
        m_endVal[0] = dx;
        m_endVal[1] = dy;
        m_endVal[2] = dz;
    }

	template<typename TDataType>
	AnimationCurve<TDataType>::~AnimationCurve()
	{
	}

    template<typename TDataType>
	void AnimationCurve<TDataType>::set(int index, const std::vector<long long>& tim, const std::vector<Real>& val)
    {
        // for(auto t : tim)
        //     this->m_times[index].push_back((double)t / FBXTIME);
        this->m_times[index].assign(tim.begin(), tim.end());
        this->m_values[index].assign(val.begin(), val.end());
    }

    template<typename TDataType>
	typename AnimationCurve<TDataType>::Coord AnimationCurve<TDataType>::getCurveValueCycle(Real ptime)
    {
        long long fbx_time = secondsToFbxTime(ptime);
        auto getReal = [=](
            Real init, 
            std::vector<long long>& times, 
            std::vector<Real>& vals, 
            long long fbx_time) mutable -> Real
        {
            int cur = BEFORE;
            int size = times.size();
            if (fbx_time < 0.f) return init;
            if (times[size - 1] == 0) return vals[size - 1];

            if (fbx_time >= times[size - 1]) {
                fbx_time %= times[size - 1];
                // fbx_time %= (times[size - 1] * 2);
                // if (fbx_time >= times[size - 1])
                    // fbx_time = 2 * times[size - 1] - fbx_time;
                // return vals[size - 1];
            }

            while(cur + 1 < size && times[cur + 1] <= fbx_time) 
                ++cur;

            assert(cur < size - 1);

            long long left_time, right_time;
            left_time = (cur == BEFORE) ? 0 : times[cur];
            right_time = times[cur + 1];
            Real left_val, right_val;
            left_val = (cur == BEFORE) ? init : vals[cur];
            right_val = vals[cur + 1];

            // Lerp
            Real t = Real(double(fbx_time - left_time) / double(right_time - left_time));
            return left_val * (1 - t) + right_val * t;
		};

		return Coord(getReal(m_initVal[0], m_times[0], m_values[0], fbx_time),
					getReal(m_initVal[1], m_times[1], m_values[1], fbx_time),
					getReal(m_initVal[2], m_times[2], m_values[2], fbx_time));
    }

    template<typename TDataType>
	typename AnimationCurve<TDataType>::Coord AnimationCurve<TDataType>::getCurveValueAlong(Real ptime)
    {
        long long fbx_time = secondsToFbxTime(ptime);
        auto getReal = [=](
            Real init, 
            std::vector<long long>& times, 
            std::vector<Real>& vals, 
            long long fbx_time, 
            int& cur) mutable -> Real
        {
            int size = times.size();
            if (fbx_time < 0.f) return init;
            // if (fbx_time >= times[size - 1]) return init * vals[size - 1];
            if (fbx_time >= times[size - 1]) return vals[size - 1];

            while(cur + 1 < size && times[cur + 1] <= fbx_time) 
                ++cur;

            assert(cur < size - 1);

            long long left_time, right_time;
            left_time = (cur == BEFORE) ? 0 : times[cur];
            right_time = times[cur + 1];
            Real left_val, right_val;
            left_val = (cur == BEFORE) ? init : vals[cur];
            right_val = vals[cur + 1];

            // Lerp
            Real t = Real(double(fbx_time - left_time) / double(right_time - left_time));
            return left_val * (1 - t) + right_val * t;
		};

		return Coord(getReal(m_initVal[0], m_times[0], m_values[0], fbx_time, m_cur[0]),
					getReal(m_initVal[1], m_times[1], m_values[1], fbx_time, m_cur[1]),
					getReal(m_initVal[2], m_times[2], m_values[2], fbx_time, m_cur[2]));
    }


    template<typename TDataType>
	typename AnimationCurve<TDataType>::Coord AnimationCurve<TDataType>::getCurveValueAll(Real ptime)
    {
        long long fbx_time = secondsToFbxTime(ptime);
        auto getReal = [&](
            Real init, 
            std::vector<long long>& times, 
            std::vector<Real>& vals, 
            long long fbx_time) mutable -> Real
        {
            int cur = BEFORE;
            int size = times.size();
            if (fbx_time < 0.f) return init;
            // if (fbx_time >= times[size - 1]) return init * vals[size - 1];
            if (fbx_time >= times[size - 1]) 
				return vals[size - 1];

            while(cur + 1 < size && times[cur + 1] <= fbx_time) 
                ++cur;

            assert(cur < size - 1);

            long long left_time, right_time;
            left_time = (cur == BEFORE) ? 0 : times[cur];
            right_time = times[cur + 1];
            Real left_val, right_val;
            left_val = (cur == BEFORE) ? init : vals[cur];
            right_val = vals[cur + 1];

            // Lerp
			Real t = Real(double(fbx_time - left_time) / double(right_time - left_time));
            return left_val * (1 - t) + right_val * t;
		};
        
		return Coord(getReal(m_initVal[0], m_times[0], m_values[0], fbx_time),
					getReal(m_initVal[1], m_times[1], m_values[1], fbx_time),
					getReal(m_initVal[2], m_times[2], m_values[2], fbx_time));
    }
	DEFINE_CLASS(AnimationCurve);
}