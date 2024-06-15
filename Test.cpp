#include "nBuffer.hpp"
#include <memory>
#include <iostream>

template<typename data_t>
struct strip_arr { typedef data_t *type; };

template<typename data_t>
struct strip_arr<data_t[]> { typedef data_t *type; };

template<typename data_t>
using strip_arr_t = strip_arr<data_t>::type;

template<typename data_t, class deleter>
struct comp_pair : deleter
{
	strip_arr_t<data_t> ptr;
	auto get_del(void) const noexcept { return *this; }
};

template<typename data_t, typename deleter>
struct comp_pair<data_t, deleter*>
{
	deleter del;
	strip_arr_t<data_t> ptr;
	auto get_del(void) const noexcept { return del; };
};

template<typename data_t, typename deleter>
requires std::is_trivially_copyable_v<deleter> && std::is_default_constructible_v<deleter>
class std::atomic<std::unique_ptr<data_t, deleter>> : private std::atomic<comp_pair<data_t, deleter>>
{
	typedef comp_pair<data_t, deleter> _mypair;
	typedef std::atomic<_mypair> _mybase;
public:
	typedef std::unique_ptr<data_t, deleter> obj_t;
private:
	constexpr static auto strip(obj_t x)
	{
		return _mypair{ x.get_deleter(), x.release() };
	}
public:
	using atomic::atomic::is_lock_free;
	using atomic::atomic::is_always_lock_free;
	constexpr atomic(void) noexcept : _mybase{ { {}, nullptr } } {}
	atomic(obj_t value) noexcept : _mybase{ strip(std::move(value)) } {}
	[[nodiscard]] auto exchange(obj_t replacement, std::memory_order mo = std::memory_order_seq_cst)
		volatile noexcept
	{
		auto stripped{ strip(std::move(replacement)) };
		stripped = this->_mybase::exchange(stripped, mo);
		return obj_t{ stripped.ptr, stripped.get_del() };
	}
	void store(obj_t replacement, std::memory_order mo = std::memory_order_seq_cst) volatile noexcept
	{
		(void)exchange(std::move(replacement), mo);
	}
	[[nodiscard]] auto load(std::memory_order mo = std::memory_order_seq_cst) volatile noexcept
		requires std::is_default_constructible_v<obj_t>
	{
		return exchange({}, mo);
	}
	//Erases current value; DO NOT USE BLINDLY
	[[nodiscard]] explicit operator obj_t(void) volatile noexcept
		requires std::is_default_constructible_v<obj_t>
	{
		return load();
	}
	~atomic(void) { (void)load(std::memory_order::relaxed); }
};

multireader_work_buffer<std::unique_ptr<int>, 1 << 4> x;
work_list y;

void deleter(int *x)
{
	std::cout << "deleting " << x << std::endl;
}

int _cdecl main()
{
	x.insert(std::make_unique<int>(0));
	(void)x.extract();
	x.insert(std::make_unique<int>(1));
	x.insert(std::make_unique<int>(2));
	(void)x.extract();
	x.insert(std::make_unique<int>(3));
	(void)x.extract();
	(void)x.extract();

	//This would cause UB:
	//    y.insert(new node_base);
	//    delete y.extract();
	//*Cannot* insert after deleting most recent node
	y.insert(new node_base);
	y.insert(new node_base);
	delete y.extract();
	y.insert(new node_base);
	delete y.extract();
	delete y.extract();
}