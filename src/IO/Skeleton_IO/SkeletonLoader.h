/**
 * Copyright 2022 Xukun Luo
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "Node.h"

#include "Topology/JointTree.h"
#include "FilePath.h"

#include "ofbx.h"



namespace dyno
{
	/*!
	*	\class	SkeletonLoader
	*	\brief	Load a Skeleton 
	*/
	template<typename TDataType>
	class SkeletonLoader : public Node
	{
		DECLARE_TCLASS(SkeletonLoader, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

        typedef std::vector<std::shared_ptr<JointTree<typename TDataType>>> JointList;

		SkeletonLoader();
		virtual ~SkeletonLoader();

        void setJointMap(JointList &jointMap) { m_jointMap = jointMap; }
		JointList getJointMap() { return m_jointMap; }
		void getCenterQuat(Coord v0, Coord v1, Quat<Real> &T, Quat<Real> &R);
		// DEF_VAR(std::string, FileName, "", "");

		bool translate(Coord t);
		bool scale(Real s);

		void loadFBX();
		bool initFBX(const char* filepath);
		void getNodes(const ofbx::IScene& scene);

		void getModelProperties(const ofbx::Object& object, std::shared_ptr<JointTree<TDataType>> cur);
		void getAnimationCurve(const ofbx::Object& object, std::shared_ptr<JointTree<TDataType>> parent);
		void getLimbNode(const ofbx::Object& object, std::shared_ptr<JointTree<TDataType>> parent);
		
		// 需要根据模型与骨骼的坐标关系来改变
		void copyVec(Coord &dest, ofbx::Vec3 src){dest = Coord(src.x, src.y, src.z);}
		void copyVecR(Coord &dest, ofbx::Vec3 src){dest = Coord(src.x, src.y, src.z);}
		void copyVecT(Coord &dest, ofbx::Vec3 src){dest = Coord(src.x, src.y, src.z);}

	public:
        DEF_ARRAY_OUT(JCapsule, Capsule, DeviceType::GPU, "Capsule <V, U> Detail");

        // DEF_ARRAY_OUT(Coord, Velocity, DeviceType::GPU, "Capsule Velocity");
        // DEF_ARRAY_OUT(Coord, AngularVelocity, DeviceType::GPU, "Capsule AngularVelocity");
		DEF_ARRAY_OUT(Quat<Real>, Rotate, DeviceType::GPU, "Capsule Rotate");
		DEF_ARRAY_OUT(Quat<Real>, Translate, DeviceType::GPU, "Capsule Translate");

		// 动画插值的骨骼位置
		DEF_ARRAY_OUT(Coord, PosV, DeviceType::GPU, "Capsule <V, U> V position");
		DEF_ARRAY_OUT(Coord, PosU, DeviceType::GPU, "Capsule <V, U> U position");

		/**
		* @brief FBX file
		*/
		DEF_VAR(FilePath, FileName, "", "");

        JointList m_jointMap;

		std::vector<JCapsule> m_capLists;
		std::vector<Quat<Real>> m_T;
		std::vector<Quat<Real>> m_R;

        int m_numCaps = 0;
		int m_numjoints = 0;
		
	protected:
		void resetStates() override;
        void updateTopology() override;

		ofbx::IScene* g_scene = nullptr;
	};
}