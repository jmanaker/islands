#pragma once
#include <cassert>
#include <chrono>
#include <concepts>
#include <iomanip>
#include <iostream>
#include <vector>

#define CRTP_ACCESS(CRTP_name) \
    constexpr [[nodiscard]] auto _virtual(void) noexcept {return static_cast<CRTP_name*>(this);} \
    constexpr [[nodiscard]] auto _virtual(void) const noexcept {return static_cast<CRTP_name const *>(this);}

namespace utils
{
    template<std::bidirectional_iterator Iter>
    [[nodiscard]] auto invert_injection(Iter start, Iter stop)
    {
        auto count{ stop - start };
        auto rstart{ std::make_reverse_iterator(stop) }, rstop{ std::make_reverse_iterator(start) };
        typedef std::iter_value_t<Iter> table_t;
        std::unique_ptr<table_t[]> lookup_table;
        if(count)
            lookup_table = std::make_unique_for_overwrite<table_t[]>(*rstart + 1);
        while(rstop != rstart)
            lookup_table[*rstart++] = --count;
        return lookup_table;
    }

    template<std::random_access_iterator Iter>
    auto uniquify(Iter start, Iter stop)
    {
        std::sort(start, stop);
        return std::unique(start, stop);
    }

    template<std::ios_base::fmtflags flags>
    class scoped_streamstate final
    {
    public:
        std::ios_base &stream;
    private:
        std::ios_base::fmtflags const state{ stream.flags() };
    public:
        scoped_streamstate(std::ios_base &stream) : stream{ stream } {}
        ~scoped_streamstate(void) noexcept(false) { stream.setf(state, flags); }
    };
}

namespace test
{
    class timer
    {
        typedef std::chrono::steady_clock clock;
        std::ostream &outpt;
        char const *const prefix;
        std::chrono::time_point<clock> const start;
    public:
        timer(std::ostream &outpt, char const *prefix = nullptr) :
            outpt{ outpt }, start{ clock::now() }, prefix{ prefix }
        {}
        ~timer(void) noexcept(false)
        {
            auto duration{ (clock::now() - start).count() };
            outpt <<
                prefix <<
                std::setw(std::numeric_limits<decltype(duration)>::digits10) << duration << "ns" <<
                std::endl;
        }
    };
}

namespace containers
{
    template<typename inheritor>
    struct wrap_iter
    {
    private:
        CRTP_ACCESS(inheritor);
    public:
        constexpr auto &operator+=(std::ptrdiff_t shift) &{ return _virtual()->op_shift(shift); }
        constexpr auto &operator-=(std::ptrdiff_t shift) &{ return *this += -shift; }
        friend constexpr [[nodiscard]] auto operator+(inheritor lhs, std::ptrdiff_t rhs)
        {
            return std::move(lhs += rhs);
        }
        friend constexpr [[nodiscard]] auto operator+(std::ptrdiff_t lhs, inheritor rhs)
        {
            return std::move(rhs += lhs);
        }
        friend constexpr [[nodiscard]] auto operator-(inheritor lhs, std::ptrdiff_t rhs)
        {
            return std::move(lhs -= rhs);
        }
    private:
        /*virtual*/ constexpr auto &op_inc(void) &{ return *this += 1; }
        /*virtual*/ constexpr auto &op_dec(void) &{ return *this -= 1; }
    public:
        constexpr [[nodiscard]] auto operator[](std::ptrdiff_t shift) const
        {
            return **(*this + shift)->_virtual();
        }
        constexpr auto &operator++(void) &{ return _virtual()->op_inc(); }
        constexpr auto &operator--(void) &{ return _virtual()->op_dec(); }
        constexpr auto operator++(int) &
        {
            auto copy{ *_virtual() };
            ++ *this;
            return copy;
        }
        constexpr auto operator--(int) &
        {
            auto copy{ *_virtual() };
            -- *this;
            return copy;
        }
    };

    typedef std::size_t treenode;
    class tree : protected std::vector<treenode>
    {
        static_assert(std::is_same_v<treenode, size_type>);
    public:
        typedef treenode node; //Convenience
        using vector::size_type;
        using vector::begin, vector::cbegin, vector::end, vector::cend;
        explicit tree(size_type sz = 0)
        {
            reserve(sz);
            while (sz--)
                add_new();
        }
        size_type add_new(void)
        {
            auto insertion{ size() };
            emplace_back(insertion);
            return insertion;
        }
        [[nodiscard]] auto count_roots(void) const noexcept
        {
            size_type count{ 0 }, index{ 0 };
            for(auto const &node : *this)
                count += (index++ == node);
            return count;
        }
        [[nodiscard]] auto trace_root(treenode k) const //not noexcept: Lakos rule
        {
            assert(k < size());
            treenode prev;
            do prev = std::exchange(k, (*this)[k]);
            while (k != prev);
            return k;
        }
        void coalesce_nocheck(treenode left, treenode above)
        {
            for (auto const key : {&left, &above})
                *key = trace_root(*key);
            (*this)[above] = (*this)[left] = std::min(above, left);
        }
        void coalesce(tree::node left, tree::node above)
        {
            //Tracing roots is expensive; skip it if possible
            if (left != above)
                coalesce_nocheck(left, above);
        }
    };
}