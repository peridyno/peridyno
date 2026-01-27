#include "guid.hpp"
#include <cstring>
#ifdef _WIN32
#include <objbase.h>
#else
#include <uuid/uuid.h>
#endif // _WIN32

namespace Wt
{
	Guid::Guid(const std::array<unsigned char, 16>& bytes) : _bytes(bytes) { }

	Guid::Guid(std::array<unsigned char, 16>&& bytes) : _bytes(std::move(bytes)) { }

	Guid::Guid() : _bytes{ {0} } { }

	bool Guid::operator==(const Guid& other) const
	{
		return _bytes == other._bytes;
	}

	bool Guid::operator!=(const Guid& other) const
	{
		return !((*this) == other);
	}

	std::string Guid::str() const
	{
		char one[10], two[6], three[6], four[6], five[14];

		snprintf(one, 10, "%02x%02x%02x%02x",
			_bytes[0], _bytes[1], _bytes[2], _bytes[3]);
		snprintf(two, 6, "%02x%02x",
			_bytes[4], _bytes[5]);
		snprintf(three, 6, "%02x%02x",
			_bytes[6], _bytes[7]);
		snprintf(four, 6, "%02x%02x",
			_bytes[8], _bytes[9]);
		snprintf(five, 14, "%02x%02x%02x%02x%02x%02x",
			_bytes[10], _bytes[11], _bytes[12], _bytes[13], _bytes[14], _bytes[15]);
		const std::string sep("-");
		std::string out(one);

		out += sep + two;
		out += sep + three;
		out += sep + four;
		out += sep + five;

		return out;
	}

	// conversion operator for std::string
	Guid::operator std::string() const
	{
		return str();
	}

	const std::array<unsigned char, 16>& Guid::bytes() const
	{
		return _bytes;
	}

	void Guid::swap(Guid& other)
	{
		_bytes.swap(other._bytes);
	}

	bool Guid::isValid() const
	{
		Guid empty;
		return *this != empty;
	}

	std::ostream& operator<<(std::ostream& s, const Guid& guid)
	{
		std::ios_base::fmtflags f(s.flags()); // politely don't leave the ostream in hex mode
		s << std::hex << std::setfill('0')
			<< std::setw(2) << (int)guid._bytes[0]
			<< std::setw(2) << (int)guid._bytes[1]
			<< std::setw(2) << (int)guid._bytes[2]
			<< std::setw(2) << (int)guid._bytes[3]
			<< "-"
			<< std::setw(2) << (int)guid._bytes[4]
			<< std::setw(2) << (int)guid._bytes[5]
			<< "-"
			<< std::setw(2) << (int)guid._bytes[6]
			<< std::setw(2) << (int)guid._bytes[7]
			<< "-"
			<< std::setw(2) << (int)guid._bytes[8]
			<< std::setw(2) << (int)guid._bytes[9]
			<< "-"
			<< std::setw(2) << (int)guid._bytes[10]
			<< std::setw(2) << (int)guid._bytes[11]
			<< std::setw(2) << (int)guid._bytes[12]
			<< std::setw(2) << (int)guid._bytes[13]
			<< std::setw(2) << (int)guid._bytes[14]
			<< std::setw(2) << (int)guid._bytes[15];
		s.flags(f);
		return s;
	}

	bool operator<(const Wt::Guid& lhs, const Wt::Guid& rhs)
	{
		return lhs.bytes() < rhs.bytes();
	}

	Guid newGuid()
	{
		std::array<unsigned char, 16> bytes;
#ifdef _WIN32
		GUID newId;
		CoCreateGuid(&newId);

		bytes =
		{
			(unsigned char)((newId.Data1 >> 24) & 0xFF),
			(unsigned char)((newId.Data1 >> 16) & 0xFF),
			(unsigned char)((newId.Data1 >> 8) & 0xFF),
			(unsigned char)((newId.Data1) & 0xff),

			(unsigned char)((newId.Data2 >> 8) & 0xFF),
			(unsigned char)((newId.Data2) & 0xff),

			(unsigned char)((newId.Data3 >> 8) & 0xFF),
			(unsigned char)((newId.Data3) & 0xFF),

			(unsigned char)newId.Data4[0],
			(unsigned char)newId.Data4[1],
			(unsigned char)newId.Data4[2],
			(unsigned char)newId.Data4[3],
			(unsigned char)newId.Data4[4],
			(unsigned char)newId.Data4[5],
			(unsigned char)newId.Data4[6],
			(unsigned char)newId.Data4[7]
		};
#else

		uuid_t uuid;
		uuid_generate(uuid);
		std::memcpy(bytes.data(), uuid, 16);

#endif // _WIN32

		return Guid{ std::move(bytes) };
	}
}

namespace std
{
	template <>
	void swap(Wt::Guid& lhs, Wt::Guid& rhs) noexcept
	{
		lhs.swap(rhs);
	}
}