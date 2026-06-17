/**
 * Copyright 2024 Yuzhong Guo
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
#include "RigidBody/ArticulatedBody.h"
#include "Module/KeyboardInputModule.h"

#include "STL/Pair.h"
#include "Field/FilePath.h"
#include "Topology/EdgeSet.h"
#include "Topology/DiscreteElements.h"

#include <algorithm>

namespace dyno
{
	template<typename TDataType>
	class J15Operator : public KeyboardInputModule
	{
		DECLARE_TCLASS(J15Operator, TDataType);

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;
		typedef typename ::dyno::Quat<Real> TQuat;

		struct FlightInput
		{
			Real Elevator = Real(0);
			Real Aileron = Real(0);
			Real Rudder = Real(0);
			Real Stabilator = Real(0);
			Real Throttle = Real(0);
		};

		struct FlightAttitude
		{
			Real Roll = Real(0);
			Real Pitch = Real(0);
			Real Yaw = Real(0);
		};

		struct FlightState
		{
			FlightAttitude Attitude;
			Coord LinearVelocity = Coord(0);
			Coord LinearVelocityIncrement = Coord(0);
		};

		struct FlightConstants
		{
			Real ThrottleGain = Real(45);
			Real DragAcceleration = Real(0);
			Real LiftGain = Real(30);
			Real StabilatorWeight = Real(1);

			Real RollGain = Real(1.6);
			Real PitchGain = Real(1.2);
			Real YawGain = Real(0.8);
			Real VelocityDirectionCorrectionGain = Real(4);

			Real MaxAbsRoll = Real(1.57);
			Real MaxAbsPitch = Real(1.22);
		};

		J15Operator();
		~J15Operator() override {};

		void reset();

	public:
		DEF_VAR_IN(Real, TimeStep, "");

		DEF_ARRAY_IN(Coord, Center, DeviceType::GPU, "");
		DEF_ARRAY_IN(Coord, Velocity, DeviceType::GPU, "");
		DEF_ARRAY_IN(Coord, AngularVelocity, DeviceType::GPU, "");
		DEF_ARRAY_IN(TQuat, Quaternion, DeviceType::GPU, "");
		DEF_ARRAY_IN(Matrix, RotationMatrix, DeviceType::GPU, "");
		DEF_ARRAY_IN(Matrix, Inertia, DeviceType::GPU, "");
		DEF_ARRAY_IN(Matrix, InitialInertia, DeviceType::GPU, "");
		DEF_INSTANCE_IN(DiscreteElements<TDataType>, Topology, "");

		void setFlightInput(const FlightInput& input) {
			mInput = input;
			mExternalInput = true;
		}
		void setFlightInput(Real elev, Real ail, Real rud, Real stab, Real thr) {
			mInput.Elevator = elev;
			mInput.Aileron = ail;
			mInput.Rudder = rud;
			mInput.Stabilator = stab;
			mInput.Throttle = thr;
			mExternalInput = true;
		}

		void step(Real elev, Real ail, Real rud, Real stab, Real thr, Real dt) {
			setFlightInput(elev, ail, rud, stab, thr);
			stepFlightModel(dt);
			applyFlightState();
		}

		void setUseExternalInput(bool b) { mExternalInput = b; }
		bool isExternalInput() const { return mExternalInput; }

		const FlightState& getFlightState() const { return mState; }
		const FlightInput& getFlightInput() const { return mInput; }
		const FlightConstants& getFlightConstants() const { return mConstants; }

	protected:
		void onEvent(PKeyboardEvent event) override;
		void postprocess() override;

		void updateInput();
		void stepFlightModel(Real dt);
		void applyFlightState();

	private:
		void setKeyState(PKeyboardType key, bool pressed);
		static Real clamp(Real value, Real lo, Real hi);

	private:
		bool mKeyA = false;
		bool mKeyS = false;
		bool mKeyD = false;
		bool mKeyW = false;
		bool mKeyQ = false;
		bool mKeyE = false;
		bool mKeyZ = false;
		bool mKeyX = false;
		bool mKeyJ = false;
		bool mKeyK = false;
		bool mKeyV = false;

		bool mExternalInput = false;
		bool mInitialized = false;
		bool mResetRequested = false;
		bool mHasAttitudeIncrement = false;
		bool mHasVelocityIncrement = false;
		uint mBodyId = 6;
		TQuat mInitialQuaternion = TQuat(Real(0), Real(0), Real(0), Real(1));
		Coord mBodyVelocityIncrement = Coord(0);
		Real mRollIncrement = Real(0);
		Real mPitchIncrement = Real(0);
		Real mElevatorPitchIncrement = Real(0);
		Real mYawIncrement = Real(0);

		FlightInput mInput;
		FlightState mState;
		FlightConstants mConstants;
	};

	template<typename TDataType>
	class J15 : virtual public ArticulatedBody<TDataType>
	{
		DECLARE_TCLASS(J15, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		J15();
		~J15() override;

		std::shared_ptr<J15Operator<TDataType>> getOperator() { return aircraftOperator; }

	protected:
		void resetStates() override;

	private:
		std::shared_ptr<J15Operator<TDataType>> aircraftOperator;
	};
}

