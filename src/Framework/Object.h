/**
 * Copyright 2017-2021 Xiaowei He
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
#include <string>
#include <atomic>
#include <map>

namespace dyno
{
class Object;
class ClassInfo;


typedef Object* (*ObjectConstructorFn)(void);
bool Register(ClassInfo* ci);

class ClassInfo
{
public:
	ClassInfo(const std::string className, ObjectConstructorFn ctor)
		:m_className(className), m_objectConstructor(ctor)
	{
		Register(this);
	}
	virtual ~ClassInfo() {}

	inline bool operator==(const ClassInfo &ci) const
	{

		if (this->m_className == ci.m_className)
			return true;
		return false;
	}

	Object* createObject()const 
	{ 
		return m_objectConstructor ? (*m_objectConstructor)() : 0; 
	}

	bool isDynamic()const 
	{ 
		return NULL != m_objectConstructor; 
	}
	const std::string getClassName()const { return m_className; }

	ObjectConstructorFn getConstructor()const { return m_objectConstructor; }
public:
	std::string m_className;
	ObjectConstructorFn m_objectConstructor;
};

#define DECLARE_CLASS(name) \
public: \
    static const ClassInfo* ms_classinfo; \
public:  \
    virtual const ClassInfo* getClassInfo() const; \
    static Object* createObject();


#define IMPLEMENT_CLASS_COMMON(name,func) \
const ClassInfo* name::ms_classinfo = new ClassInfo((#name), (ObjectConstructorFn) func); \
                        \
const ClassInfo* name::getClassInfo() const \
    {return name::ms_classinfo;}

#define IMPLEMENT_CLASS(name)            \
IMPLEMENT_CLASS_COMMON(name,name::createObject) \
Object* name::createObject()                   \
    { return new name;}


#define _STR(s) #s

#define DECLARE_TCLASS(name, T1) \
public: \
    static const ClassInfo ms_classinfo; \
public:  \
    virtual const ClassInfo* getClassInfo() const; \
    static Object* createObject();

#define IMPLEMENT_CLASS_COMMON_1(name, T1, func) \
template<typename T1>		\
const ClassInfo name<T1>::ms_classinfo(std::string(_STR(name))+std::string("<")+T1::getName()+std::string(">"), \
            (ObjectConstructorFn) func); \
							\
template<typename T1>		\
const ClassInfo* name<T1>::getClassInfo() const \
    {return &name<T1>::ms_classinfo;}

#define IMPLEMENT_TCLASS(name, T1)            \
IMPLEMENT_CLASS_COMMON_1(name, T1, name<T1>::createObject) \
							\
template<typename T1>		\
Object* name<T1>::createObject()                   \
    { return new name<T1>;}

typedef uint32_t ObjectId;

class Object
{
	DECLARE_CLASS(Object)
public:
	Object();
	virtual ~Object() {};
	static bool registerClass(ClassInfo* ci);
	static Object* createObject(std::string name);
	static std::map< std::string, ClassInfo*>* getClassMap();

	/**
	 * @brief Base Id
	 * 
	 * @return All objects will be countered starting from base id. 
	 */
	static ObjectId baseId();

	ObjectId objectId() { return id; }
private:
	ObjectId id;

	static uint32_t generateObjectId();
	static std::atomic_uint32_t cId;
};

#ifdef PRECISION_FLOAT
#define DEFINE_CLASS(name) template class name<DataType3f>;
#else
#define DEFINE_CLASS(name) template class name<DataType3d>;
#endif

#define DEFINE_UNIQUE_CLASS(name, type) template class name<type>;
}