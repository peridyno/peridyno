/**
 * Copyright 2021 Xiaowei He
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
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include "FBase.h"

#include "Array/Array.h"
#include "Array/Array2D.h"
#include "Array/Array3D.h"
#include "Array/ArrayList.h"

namespace dyno {
	/*!
	*	\class	Variable
	*	\brief	Variables of build-in data types.
	*/
	template<typename T>
	class FVar : public FBase
	{
	public:
		typedef T				VarType;
		typedef T				DataType;
		typedef FVar<T>			FieldType;

		FVar() : FBase("", "") {}
		FVar(std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent)
			: FBase(name, description, fieldType, parent) {}

		FVar(T value, std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent);
		~FVar() override;

		const std::string getTemplateName() override { return std::string(typeid(VarType).name()); }
		const std::string getClassName() override { return "FVar"; }

		uint size() override { return 1; }

		/**
		 * @brief set the value
		 * 
		 * notify: call the callback function when the value is true 
		 */
		void setValue(T val, bool notify = true);
		T getValue();

		inline std::string serialize() override { return "Unknown"; }
		inline bool deserialize(const std::string& str) override { return false; }

		bool isEmpty() override {
			return this->constDataPtr() == nullptr;
		}

		bool connect(FieldType* dst)
		{
			this->connectField(dst);
			return true;
		}

		bool connect(FBase* dst) override {
			FieldType* derived = dynamic_cast<FieldType*>(dst);
			if (derived == nullptr) return false;
			return this->connect(derived);
		}

		DataType getData() {
			auto dataPtr = this->constDataPtr();
			assert(dataPtr != nullptr);
			return *dataPtr;
		}

		std::shared_ptr<DataType>& constDataPtr()
		{
			FBase* topField = this->getTopField();
			FieldType* derived = dynamic_cast<FieldType*>(topField);
			return derived->m_data;
		}

	private:
		std::shared_ptr<DataType>& getDataPtr()
		{
			FBase* topField = this->getTopField();
			FieldType* derived = dynamic_cast<FieldType*>(topField);
			return derived->m_data;
		}

		std::shared_ptr<DataType> m_data = nullptr;
	};

	template<typename T>
	FVar<T>::FVar(T value, std::string name, std::string description, FieldTypeEnum fieldType, OBase* parent)
		: FBase(name, description, fieldType, parent)
	{
		this->setValue(value);
	}

	template<typename T>
	FVar<T>::~FVar()
	{
	};

	template<typename T>
	void FVar<T>::setValue(T val, bool notify)
	{
		std::shared_ptr<T>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<T>(val);
		}
		else
		{
			*data = val;
		}

		if(notify && isActive())
			this->update();

		this->tick();
	}


	template<typename T>
	T FVar<T>::getValue()
	{
		std::shared_ptr<T>& data = this->constDataPtr();

		return *data;
	}


	template<typename T>
	using HostVarField = FVar<T>;

	template<typename T>
	using DeviceVarField = FVar<T>;

	/**
 * Define field for Array
 */
	template<typename T, DeviceType deviceType>
	class FArray : public FBase
	{
	public:
		typedef T							VarType;
		typedef Array<T, deviceType>		DataType;
		typedef FArray<T, deviceType>	FieldType;

		DEFINE_FIELD_FUNC(FieldType, DataType, FArray);

		~FArray() override;

		inline uint size() override {
			auto ref = this->constDataPtr();
			return ref == nullptr ? 0 : ref->size();
		}

		void resize(uint num);
		void reset();

		void clear();

		void assign(const T& val);
		void assign(const std::vector<T>& vals);
#ifndef NO_BACKEND
		void assign(const DArray<T>& vals);
#endif
		void assign(const CArray<T>& vals);

		bool isEmpty() override {
			return this->size() == 0;
		}
	};

	template<typename T, DeviceType deviceType>
	FArray<T, deviceType>::~FArray()
	{
		if (m_data.use_count() == 1)
		{
			m_data->clear();
		}
	}

	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::resize(uint num)
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr) {
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->resize(num);
	}

	template<typename T, DeviceType deviceType>
	void dyno::FArray<T, deviceType>::assign(const T& val)
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->assign(val);
	}

	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::assign(const std::vector<T>& vals)
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->assign(vals);
	}

	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::assign(const CArray<T>& vals)
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->assign(vals);
	}

#ifndef NO_BACKEND
	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::assign(const DArray<T>& vals)
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->assign(vals);
	}
#endif

	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::reset()
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->reset();
	}

	template<typename T, DeviceType deviceType>
	void FArray<T, deviceType>::clear()
	{
		std::shared_ptr<Array<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array<T, deviceType>>();
		}

		data->clear();
	}

	template<typename T>
	using HostArrayField = FArray<T, DeviceType::CPU>;

	template<typename T>
	using DeviceArrayField = FArray<T, DeviceType::GPU>;

	/**
	 * Define field for Array2D
	 */
	template<typename T, DeviceType deviceType>
	class FArray2D : public FBase
	{
	public:
		typedef T								VarType;
		typedef Array2D<T, deviceType>			DataType;
		typedef FArray2D<T, deviceType>			FieldType;

		DEFINE_FIELD_FUNC(FieldType, DataType, FArray2D);

		~FArray2D() override;

		inline uint size() override {
			auto ref = this->constDataPtr();
			return ref == nullptr ? 0 : ref->size();
		}

		void resize(uint nx, uint ny);

		void reset();

		void clear();

		void assign(const CArray2D<T>& vals);

#ifndef NO_BACKEND
		void assign(const DArray2D<T>& vals);
#endif

		bool isEmpty() override {
			return this->constDataPtr() == nullptr;
		}
	};

	template<typename T, DeviceType deviceType>
	FArray2D<T, deviceType>::~FArray2D()
	{
		if (m_data.use_count() == 1)
		{
			m_data->clear();
		}
	}

#ifndef NO_BACKEND
	template<typename T, DeviceType deviceType>
	void FArray2D<T, deviceType>::assign(const DArray2D<T>& vals)
	{
		std::shared_ptr<Array2D<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array2D<T, deviceType>>();
		}

		data->assign(vals);
	}
#endif

	template<typename T, DeviceType deviceType>
	void FArray2D<T, deviceType>::assign(const CArray2D<T>& vals)
	{
		std::shared_ptr<Array2D<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array2D<T, deviceType>>();
		}

		data->assign(vals);
	}

	template<typename T, DeviceType deviceType>
	void FArray2D<T, deviceType>::reset()
	{
		std::shared_ptr<Array2D<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array2D<T, deviceType>>();
		}

		data->reset();
	}

	template<typename T, DeviceType deviceType>
	void FArray2D<T, deviceType>::clear()
	{
		std::shared_ptr<Array2D<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<Array2D<T, deviceType>>();
		}

		data->clear();
	}

	template<typename T, DeviceType deviceType>
	void FArray2D<T, deviceType>::resize(uint nx, uint ny)
	{
		std::shared_ptr<Array2D<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr) {
			data = std::make_shared<Array2D<T, deviceType>>();
		}

		data->resize(nx, ny);
	}

	/**
	 * Define field for Array3D
	 */
	template<typename T, DeviceType deviceType>
	class FArray3D : public FBase
	{
	public:
		typedef T								VarType;
		typedef Array3D<T, deviceType>			DataType;
		typedef FArray3D<T, deviceType>			FieldType;

		DEFINE_FIELD_FUNC(FieldType, DataType, FArray3D);

		~FArray3D() override;

		inline uint size() override {
			auto ref = this->constDataPtr();
			return ref == nullptr ? 0 : ref->size();
		}

		void resize(const uint nx, const uint ny, const uint nz);

		void reset();

		void clear();

		void assign(const CArray3D<T>& vals);

#ifndef NO_BACKEND
		void assign(const DArray3D<T>& vals);
#endif

		bool isEmpty() override {
			return this->constDataPtr() == nullptr;
		}
	};

	template<typename T, DeviceType deviceType>
	FArray3D<T, deviceType>::~FArray3D()
	{
		if (m_data.use_count() == 1)
		{
			m_data->clear();
		}
	}

#ifndef NO_BACKEND
	template<typename T, DeviceType deviceType>
	void FArray3D<T, deviceType>::assign(const DArray3D<T>& vals)
	{
		std::shared_ptr<Array3D<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr)
			data = std::make_shared<Array3D<T, deviceType>>();

		data->assign(vals);
	}
#endif

	template<typename T, DeviceType deviceType>
	void FArray3D<T, deviceType>::assign(const CArray3D<T>& vals)
	{
		std::shared_ptr<Array3D<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr)
			data = std::make_shared<Array3D<T, deviceType>>();

		data->assign(vals);
	}

	template<typename T, DeviceType deviceType>
	void FArray3D<T, deviceType>::reset()
	{
		std::shared_ptr<Array3D<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr)
			data = std::make_shared<Array3D<T, deviceType>>();

		data->reset();
	}

	template<typename T, DeviceType deviceType>
	void FArray3D<T, deviceType>::clear()
	{
		std::shared_ptr<FArray3D<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<FArray3D<T, deviceType>>();
		}

		data->clear();
	}

	template<typename T, DeviceType deviceType>
	void FArray3D<T, deviceType>::resize(const uint nx, const uint ny, const uint nz)
	{
		std::shared_ptr<Array3D<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr)
			data = std::make_shared<Array3D<T, deviceType>>();

		data->resize(nx, ny, nz);
	}

#ifdef CUDA_BACKEND
	/**
	 * Define field for Array
	 */
	template<typename T, DeviceType deviceType>
	class FArrayList : public FBase
	{
	public:
		typedef T								VarType;
		typedef ArrayList<T, deviceType>		DataType;
		typedef FArrayList<T, deviceType>	FieldType;

		DEFINE_FIELD_FUNC(FieldType, DataType, FArrayList);

		~FArrayList() override;

		inline uint size() override {
			auto ref = this->constDataPtr();
			return ref == nullptr ? 0 : ref->size();
		}

		void resize(uint num);

		void resize(const Array<int, deviceType>& arr);

		void clear();

		void assign(const ArrayList<T, DeviceType::CPU>& src);
		void assign(const ArrayList<T, DeviceType::GPU>& src);

		bool isEmpty() override {
			return this->constDataPtr() == nullptr;
		}
	};

	template<typename T, DeviceType deviceType>
	FArrayList<T, deviceType>::~FArrayList()
	{
		if (m_data.use_count() == 1)
		{
			m_data->clear();
		}
	}

	template<typename T, DeviceType deviceType>
	void FArrayList<T, deviceType>::assign(const ArrayList<T, DeviceType::CPU>& src)
	{
		std::shared_ptr<ArrayList<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr)
			data = std::make_shared<ArrayList<T, deviceType>>();

		data->assign(src);
	}

	template<typename T, DeviceType deviceType>
	void FArrayList<T, deviceType>::assign(const ArrayList<T, DeviceType::GPU>& src)
	{
		std::shared_ptr<ArrayList<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr)
			data = std::make_shared<ArrayList<T, deviceType>>();

		data->assign(src);
	}

	template<typename T, DeviceType deviceType>
	void FArrayList<T, deviceType>::resize(uint num)
	{
		std::shared_ptr<ArrayList<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr)
			data = std::make_shared<ArrayList<T, deviceType>>();

		data->resize(num);
	}

	template<typename T, DeviceType deviceType>
	void FArrayList<T, deviceType>::resize(const Array<int, deviceType>& arr)
	{
		std::shared_ptr<ArrayList<T, deviceType>>& data = this->getDataPtr();

		if (data == nullptr)
			data = std::make_shared<ArrayList<T, deviceType>>();

		data->resize(arr);
	}

	template<typename T, DeviceType deviceType>
	void FArrayList<T, deviceType>::clear()
	{
		std::shared_ptr<ArrayList<T, deviceType>>& data = this->getDataPtr();
		if (data == nullptr)
		{
			data = std::make_shared<ArrayList<T, deviceType>>();
		}

		data->clear();
	}
#endif

#ifdef VK_BACKEND
// 	/**
// 	 * Define field for Array
// 	 */
// 	template<typename T, DeviceType deviceType>
// 	class FArray : public FBase
// 	{
// 	public:
// 		typedef T							VarType;
// 		typedef Array<T, deviceType>		DataType;
// 		typedef FArray<T, deviceType>		FieldType;
// 
// 		DEFINE_FIELD_FUNC(FieldType, DataType, FArray);
// 
// 		~FArray() override;
// 
// 		inline uint size() override {
// 			auto ref = this->getDataPtr();
// 			return ref == nullptr ? 0 : ref->size();
// 		}
// 
// 		void resize(uint num);
// 		void reset();
// 		void clear();
// 
// 		bool isEmpty() override {
// 			return this->size() == 0;
// 		}
// 	};
// 
// 	template<typename T, DeviceType deviceType>
// 	FArray<T, deviceType>::~FArray()
// 	{
// 		if (m_data.use_count() == 1)
// 		{
// 			m_data->clear();
// 		}
// 	}
// 
// 	template<typename T, DeviceType deviceType>
// 	void FArray<T, deviceType>::resize(uint num)
// 	{
// 		std::shared_ptr<DataType>& data = this->getDataPtr();
// 		if (data == nullptr) {
// 			data = std::make_shared<DataType>();
// 		}
// 
// 		data->resize(num);
// 	}
// 
// 	template<typename T, DeviceType deviceType>
// 	void FArray<T, deviceType>::reset()
// 	{
// 		std::shared_ptr<DataType>& data = this->getDataPtr();
// 		if (data == nullptr)
// 		{
// 			data = std::make_shared<DataType>();
// 		}
// 
// 		data->reset();
// 	}
// 
// 	template<typename T, DeviceType deviceType>
// 	void FArray<T, deviceType>::clear()
// 	{
// 		std::shared_ptr<DataType>& data = this->getDataPtr();
// 		if (data == nullptr) {
// 			data = std::make_shared<DataType>();
// 		}
// 
// 		data->clear();
// 	}


	/**
	 * Define field for Array
	 */
	template<typename T, DeviceType deviceType>
	class FArrayList : public FBase
	{
	public:
		typedef T								VarType;
		typedef ArrayList<T, deviceType>		DataType;
		typedef FArrayList<T, deviceType>	FieldType;

		DEFINE_FIELD_FUNC(FieldType, DataType, FArrayList);

		inline uint size() override {
			auto ref = this->constDataPtr();
			return ref == nullptr ? 0 : ref->size();
		}

// 		void resize(uint num);
// 
// 		void resize(const Array<int, deviceType>& arr);
// 
// 		void assign(const ArrayList<T, DeviceType::CPU>& src);
// 		void assign(const ArrayList<T, DeviceType::GPU>& src);

		bool isEmpty() override {
			return this->constDataPtr() == nullptr;
		}
	};

// 	template<typename T, DeviceType deviceType>
// 	void FArrayList<T, deviceType>::assign(const ArrayList<T, DeviceType::CPU>& src)
// 	{
// 		std::shared_ptr<ArrayList<T, deviceType>>& data = this->getDataPtr();
// 
// 		if (data == nullptr)
// 			data = std::make_shared<ArrayList<T, deviceType>>();
// 
// 		data->assign(src);
// 	}
// 
// 	template<typename T, DeviceType deviceType>
// 	void FArrayList<T, deviceType>::assign(const ArrayList<T, DeviceType::GPU>& src)
// 	{
// 		std::shared_ptr<ArrayList<T, deviceType>>& data = this->getDataPtr();
// 
// 		if (data == nullptr)
// 			data = std::make_shared<ArrayList<T, deviceType>>();
// 
// 		data->assign(src);
// 	}
// 
// 	template<typename T, DeviceType deviceType>
// 	void FArrayList<T, deviceType>::resize(uint num)
// 	{
// 		std::shared_ptr<ArrayList<T, deviceType>>& data = this->getDataPtr();
// 
// 		if (data == nullptr)
// 			data = std::make_shared<ArrayList<T, deviceType>>();
// 
// 		data->resize(num);
// 
// 		this->tick();
// 	}
// 
// 	template<typename T, DeviceType deviceType>
// 	void FArrayList<T, deviceType>::resize(const Array<int, deviceType>& arr)
// 	{
// 		std::shared_ptr<ArrayList<T, deviceType>>& data = this->getDataPtr();
// 
// 		if (data == nullptr)
// 			data = std::make_shared<ArrayList<T, deviceType>>();
// 
// 		data->resize(arr);
// 	}

#endif
}

#include "FSerialization.inl"
