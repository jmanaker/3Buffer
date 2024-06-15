#pragma once
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <atomic>
#include <bit>
#include <type_traits>

#define COMMON_ASSIGNMENT_OPERATOR \
    template<typename rhs_t> \
    constexpr decltype(auto) operator=(rhs_t &&rhs) & \
    { \
        typedef std::decay_t<decltype(*this)> _myt; \
        static_assert(std::is_convertible_v<rhs_t, _myt>, \
            "Assignment requires impossible conversion"); \
	    if /*constexpr*/ (std::is_same_v<std::decay_t<rhs_t>, _myt> && \
            /*rvalue reference=>temporary=>not aliased*/ \
            !std::is_rvalue_reference_v<decltype(rhs)>) \
            \
		    if (static_cast<void *>(this) == static_cast<void *>(&rhs)) \
			    return *this; \
        /*else*/ \
	    this->~_myt(); \
	    return *(new(this) _myt(std::forward<rhs_t>(rhs))); \
    }

namespace tweaks
{
	template<std::size_t bits>
	struct fl_uint;
	template<> struct fl_uint<8> { typedef std::uint8_t type; };
	template<> struct fl_uint<16> { typedef std::uint16_t type; };
	template<> struct fl_uint<32> { typedef std::uint32_t type; };
	template<> struct fl_uint<64> { typedef std::uint64_t type; };

	template<std::size_t bits>
	using uint_t = fl_uint<bits>::type;

	using std::div;
	template<typename T>
	struct div_t
	{
		T rem, quot;
	};

	auto div(auto lhs, auto rhs)
	{
		typedef std::common_type_t<decltype(lhs), decltype(rhs)> shared_t;
		return div_t<shared_t>{
			lhs %rhs, lhs / rhs
		};
	}
}

namespace atomics
{
	using enum std::memory_order;
	template<typename T>
	[[nodiscard]] bool compare(std::atomic<T> volatile &lhs, T rhs) noexcept
	{
		return lhs.compare_exchange_strong(rhs, rhs, relaxed);
	}

	//If data_t is fancy, want to use exchange as analogue of std::move
	//(This requires using a specialization of std::atomic, so is unlikely.)
	template<typename T>
	static [[nodiscard]] auto load_move(std::atomic<T> volatile &sync)
		noexcept(std::is_nothrow_default_constructible_v<T>)
	{
		return sync.exchange({}, relaxed);
	}
	//If data_t is trivially-copyable (this is likely), then 
	//exchange requires an extra computation (I think?)
	//so just read the bits instead.
	//
	//(Separate template function so move-only types w/out atomic::load 
	// (which will end up calling other overload) cause SFINAE).
	template<typename T> requires std::is_trivially_copyable_v<T>
		static [[nodiscard]] auto load_move(std::atomic<T> volatile &sync) noexcept
		{
			return sync.load(relaxed);
		}
}

//Fixed-size (size len) buffer for objects of type data_t
// * If buffer is full, `insert` overwrites (drops) last element.
// * If buffer is empty, `extract` blocks until an element is added.  
// * Assumes only one reader and only one writer; needs additional synchronization logic otherwise.  
//If you specialize std::atomic<data_t>, you don't need to define all the members, just 
//`wait`, `notify_one`, `load`, and `store`
//
//Remarks on multi-reader/multi-writer case:
//Without a state variable for each array element, can't publish reads to writers except in order. 
//A lockless variant for that case would effectively spinlock in `extract`;
//(OTOH, if moving data_t objects is expensive, that algorithm puts the movement outside the lock...)
//so I might as well just require the caller to lock.  
//Likewise for `insert`, I suspect, although the semantics of dropped entries are tricky.  
template<typename data_t, std::size_t len> requires requires { std::atomic<data_t>{}; }
class work_buffer
{
protected:
	//Ensure smooth rollover
	//(Want to require !((SIZE_MAX + 1) % len), but SIZE_MAX + 1 rolls over to 0, 
	//which is always divisible by len)
	static_assert(0 == (SIZE_MAX % len + 1) % len);
	using enum std::memory_order;
	//Needs to be atomic, because if we overwrite the last element in the queue,
	//we've already "published" it, and so if the reader catches up, we might tear
	std::atomic<data_t> storage[len];
	std::atomic_size_t read{ 0 }, write{ 0 };
public:
	virtual [[nodiscard]] data_t extract(void) volatile
		noexcept(std::is_nothrow_default_constructible_v<data_t>)
	{
		do
			if (auto const index{ read.load(acquire) };
				write.load(acquire) != index) [[likely]]
			{ //Fast path; have waiting data
				//Can't use if constexpr so separate fct. out-of-class.
				//(If "?: constexpr" existed, would use that instead.)
				auto temp{ atomics::load_move(storage[index % len]) };
				read.fetch_add(1, release);
				return temp;
			}
			else write.wait(index, relaxed);
		while (true);
	}
	virtual void insert(data_t obj) volatile
		noexcept(std::is_nothrow_move_constructible_v<data_t>)
	{
		auto next{ write.load(acquire) };
		if (next - read.load(acquire) < len) [[likely]]
			++next;
		storage[(next - 1) % len].store(std::move(obj), relaxed);
		write.store(next, release);
		write.notify_one();
	}
};

template<typename data_t, std::size_t len>
class multireader_work_buffer : public virtual work_buffer<data_t, len>
{
private:
	using enum std::memory_order;
	typedef tweaks::uint_t<std::min<std::size_t>(
#ifdef WIN32
	32
#else
	64
#endif
		, len)> bitfield_t;
	static_assert(2 == std::numeric_limits<bitfield_t>::radix);
	static constexpr auto const dig_ct{ std::numeric_limits<bitfield_t>::digits };
	//TODO: merge these two
	std::atomic_size_t read_start;
	std::atomic<bitfield_t> bitfields;
	void mark_reduce(bitfield_t mark) volatile
	{
		bitfield_t init(bitfields.fetch_or(mark, acq_rel) | mark), todo;
		//countr_one returns int, so...
		unsigned int chunk;
		do
		{
			//Yes, this risks false positives in countr below;
			//Assumes that, even if start is stale, we did not
			//loop around and stall threads in the exact same bitfield pattern.
			//(But yes, this could happen!  Need to test in production before confident.
			// If violated, probably need a lock.)
			//rotr takes int-sized shift, so we need to convert sometime; 
			//dig_ct<=64, so know that start fits in a ushort
			//(Also happens to give more compiler optimizations on rotr/rotl)
			unsigned short int const start(this->read.load(acquire) % dig_ct);
			//How many reads are free?
			chunk = std::countr_one(std::rotr(init, start));
			//If we can't publish anything, don't hog the bus; they'll clear us anyways
			if (!chunk)
				//No point in checking read again; we've made no read/writes.
				//Reversing publishing/flag-clearing order at end DOES NOT HELP;
				//Trades early drops for accidental flag-clearing and concomitant collapse
				//on looparound.  
				//(As mentioned above, we still have looparound issues, but we're less vulnerable to them.)
				return;
			//Otherwise prep for publishing/flag clearing
			bitfield_t const flags_up((bitfield_t{1} << chunk) - 1);
			todo = init & std::rotl(static_cast<bitfield_t>(~flags_up), start);
			//Try to clear the flags...
		} while (!bitfields.compare_exchange_weak(init, todo, release, acquire));
		//...and publish (release to ensure after cmpxchg bitfield)
		this->read.fetch_add(chunk, release);
	}
public:
	virtual [[nodiscard]] data_t extract(void) volatile
		noexcept(std::is_nothrow_default_constructible_v<data_t>)
	{
		do
		{
			auto index{ read_start.load(relaxed) };
			while (this->write.load(acquire) != index) [[likely]] //(*)
			{
				//Fast path; have waiting data
				auto const next{ index + 1 };
				//Acquire object to read
				//Can't use fetch_add b/c might violate condition (*)
				if (read_start.compare_exchange_weak(index, next, release, acquire))
				{
					//Can't use if constexpr so use separate fct. out-of-class.
					//(If "?: constexpr" existed, would use that instead.)
					auto temp{ atomics::load_move(this->storage[index % len]) };
					//Publish read
					//Ideally, we're the only reader right now; let's optimize for that case
					auto index_cpy{ index };
					if (this->read.compare_exchange_weak(index_cpy, next, release, relaxed)) [[likely]]
						; //Success!
					else
					{
						//Hard case: mark bitfields and reduce
						auto const offset{ index % dig_ct };
						mark_reduce(bitfield_t{ 1 } << offset);
					}
					return temp;
				}
			}
			this->write.wait(index, relaxed);
		} while (true);
	}
};

template<typename data_t, std::size_t len>
class multiwriter_work_buffer : public virtual work_buffer<data_t, len>
{
	using enum std::memory_order;
	std::atomic_size_t write_sync;
public:
	//TODO
	virtual void insert(data_t obj) volatile
		noexcept(std::is_nothrow_move_constructible_v<data_t>) abstract;
};

template<typename data_t, std::size_t len>
class multithread_work_buffer
	: public multiwriter_work_buffer<data_t, len>, multireader_work_buffer<data_t, len>
{};

struct node_base : private std::atomic<node_base volatile *>
{
	constexpr node_base(void) noexcept = default;
	//Not default b/c atomic is uncopyable
	constexpr node_base(node_base const volatile &other) noexcept : node_base() {}
	COMMON_ASSIGNMENT_OPERATOR;
	virtual ~node_base(void) = default;
	friend class work_list;
};

//Arbitrary-length list of node_base elements. (To add your own data, derive from node_base).
// * Relative order is preserved: elements added first are extracted first.
// * Does not assume single reader/writer.
// * If work_list is empty, extract will block until an element is added.  
// * Extracted nodes may be freely deleted with the following caveat:
//       It is UB to insert a new node after the immediately-previous node (if any)
//       has been deleted/free'd.  
//   One way to handle this is to have the consumer thread store node_bases in a "smart ptr".
//   Extraction replaces the value in the smart ptr, triggering deletion...but we know 
//   that the deleted object cannot be immediately-previous to an inserted node, 
//   because another has already been inserted and extracted.  
class work_list : private node_base
{
	std::atomic<node_base volatile *> last{ this };
	using enum std::memory_order;
public:
	void insert(node_base volatile *next) volatile noexcept
	{
		auto appendage{ last.exchange(next, acq_rel) };
		appendage->store(next, release);
		this->notify_one();
	}
	[[nodiscard]] auto extract(void) volatile noexcept
	{
		do
		{
			while (auto cur{ this->load(relaxed) }) [[likely]]
				//Make sure another reader hasn't stolen cur out from under us
				if (auto const replacement{ cur->load(acquire) };
					this->compare_exchange_weak(cur, replacement, acq_rel)
					) [[likely]]
					return cur;

			wait(nullptr, relaxed);
		} while (true);
	}
	[[nodiscard]] auto extract(void) noexcept
	{
		return const_cast<node_base *>(static_cast<volatile work_list*>(this)->extract());
	}
};